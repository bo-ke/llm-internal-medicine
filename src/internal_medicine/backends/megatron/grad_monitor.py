"""Activation-gradient health monitor for Megatron/Torch.

Answers "where does the gradient grow or die on the way back" with one L2 norm
per module output per layer. Three positions per layer -- the attention branch
output, the MLP/MoE branch output, and the decoder layer output (the residual
stream itself) -- times the three magnitudes in ``grad_metrics``.

Collection is a ``forward_hook`` that registers a tensor gradient hook on the
module's output. The forward hook measures nothing itself: it exists only to get
hold of the tensor whose gradient is wanted, so the forward path pays one
``register_hook`` per module and the reductions all happen during backward.

**AMP.** Megatron trains in bf16 by default, so activation gradients are *not*
loss-scaled and the numbers are already correct. Under fp16 a gradient hook sees
the loss-scaled gradient; ``finalize_scaled_grad_metrics(scaler)`` divides the
scale back out once per step and must be wired from the trainer (before
``optimizer.step()``, while the scaler still holds this step's value). Without
that wiring fp16 curves jump by 2x on every scaler halving; bf16 needs nothing.

**Sharding.** No collective runs in a hook (hot-path discipline: see
``.claude/skills/monitor-hook-perf-rules``); the exact cross-rank reduction
happens once per step at flush time. Under TP / SP / CP each rank measures its
own shard: ``rms`` is the true global value when shards are equal-sized, while
``norm`` is the per-shard norm.

**Recompute.** ``_should_monitor`` is False while grad is disabled, so the
discarded first forward of a recomputed block registers nothing and the replayed
forward -- the one whose graph actually carries the backward -- is hooked.

This is a port of ``backends/paddlefleet/grad_monitor.py`` with the PaddleFleet
concepts Megatron-Core does not have removed: no MTP layers and no per-layer
``attn_type`` tag (Megatron ``qk_stats`` does not tag either, so the metric key
layout stays ``layer_{i}/{position}_{metric}``).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ...core.grad_reduce import (
    DATA_CANDIDATES,
    SHARD_CANDIDATES,
    GroupInfo,
    describe,
    plan_total_groups,
)
from .base import TorchProbe
from .grad_metrics import (
    GLOBAL_METRICS,
    MAX_AGGREGATED,
    METRICS,
    SCALE_INVARIANT,
    TOKEN_METRICS,
    grad_square_and_stats,
)
from .layer_discovery import find_transformer_layers

logger = logging.getLogger(__name__)


def _output_tensor(outputs):
    """First tensor of a Megatron layer/module forward result, or ``None``.

    Megatron layers return ``(hidden_states, context)`` and its sub-modules
    ``(output, bias)``; both put the tensor whose gradient we want first, so only
    the leading element is followed -- scanning the whole tuple for "any tensor"
    would silently measure a bias gradient if the output were ever ``None``.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        return outputs.get("hidden_states")
    if isinstance(outputs, tuple | list) and outputs:
        return _output_tensor(outputs[0])
    return None


def _branch_modules(layer):
    """``[(position, module)]`` for the branches this layer actually has.

    Only positions that exist are returned, and the same list drives both the
    schema and the hooks -- so a layer without an ``mlp`` never declares a key it
    could not fill. Megatron names the attention ``self_attention`` and the FFN
    ``mlp`` (an MoE layer's ``mlp`` is the MoE block itself).
    """
    modules = []
    attn = getattr(layer, "self_attention", None)
    if attn is not None:
        modules.append(("attn_out", attn))
    ffn = getattr(layer, "mlp", None)
    if ffn is not None:
        modules.append(("ffn_or_moe_out", ffn))
    # The layer itself is the residual stream leaving this block.
    modules.append(("layer_out", layer))
    return modules


class ExactNormReducer:
    """Turns per-rank sums of squares into exact global norms, once per step.

    Why a collective at all: ``norm`` / ``rms`` are recorded per shard and then
    *averaged* across ranks, which is only an approximation of the global value
    (the arithmetic mean of per-rank RMS understates the true RMS). Summing
    squares instead is exact, but sums do not commute with sharding for free --
    hence this.

    Where the collective goes: **flush time, one batched reduction per group**,
    never in a hook. That is the exact failure mode ``monitor-hook-perf-rules``
    was written about.

    Two independent sums come off the *same* local value:

    - **shard sum** over the groups that split one tensor -- ``cp`` always, and
      ``tp`` only under ``sequence_parallel`` (without SP the very same tensors
      are *replicated* across TP, so summing would multiply every squared norm by
      ``tp_size``). Feeds ``norm_mb``.
    - **total sum** over a group set that covers every participating rank
      *exactly once*. Feeds ``rms_global`` / ``norm_global``.

    Membership is *checked*, not assumed (see ``core.grad_reduce``): two mesh
    groups through this rank are orthogonal iff they intersect in exactly this
    rank, and the product of the chosen sizes must equal the number of ranks that
    hold *distinct* samples (see ``_distinct_data_ranks``). When it does not,
    ``norm_global`` is suppressed rather than emitted wrong; ``rms_global`` stays
    an unbiased estimate on any subset and keeps reporting.

    ``pp`` is never summed: its ranks own different layers, so their key sets
    differ and a reduction across them would deadlock on mismatched shapes.
    """

    def __init__(self, sequence_parallel: bool, verbose: bool = False):
        self.shard_groups: list[GroupInfo] = []  # same tensor, split across ranks
        self.total_groups: list[GroupInfo] = []  # covers every rank exactly once
        self.shard_factor = 1
        self.total_factor = 1
        self.enabled = False
        self.exact_total = False  # False -> norm_global is suppressed
        self.plan = "no distributed context"
        self._init_groups(sequence_parallel, verbose)

    def _init_groups(self, sequence_parallel: bool, verbose: bool) -> None:
        try:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                return
        except Exception:
            return
        try:
            from megatron.core import parallel_state
        except Exception as exc:  # pragma: no cover - depends on backend version
            self.plan = f"parallel_state unavailable ({exc})"
            return

        world = int(dist.get_world_size())
        found: dict[str, GroupInfo] = {}
        for name, getter in self._group_getters(parallel_state).items():
            info = self._resolve(name, getter)
            if info is not None:
                found[name] = info

        try:
            pp_size = int(parallel_state.get_pipeline_model_parallel_world_size())
        except Exception:
            pp_size = 1
        tp = found.get("tp")
        tp_size = tp.size if tp is not None else 1
        target = self._distinct_data_ranks(world, pp_size, tp_size, sequence_parallel)

        for name in SHARD_CANDIDATES:
            if name == "tp" and not sequence_parallel:
                continue
            info = found.get(name)
            if info is not None and info.size > 1:
                self.shard_groups.append(info)
                self.shard_factor *= info.size

        pool = [found[name] for name in DATA_CANDIDATES if name in found] + list(self.shard_groups)
        self.total_groups, self.total_factor = plan_total_groups(pool, target)
        self.exact_total = self.total_factor == target
        self.enabled = True
        self.plan = (
            f"shard[{describe(self.shard_groups)}] total[{describe(self.total_groups)}]"
            f" covering {self.total_factor}/{target}"
        )
        seen = ", ".join(f"{name}={info.size}" for name, info in found.items()) or "none"
        if verbose or not self.exact_total:
            emit = logger.info if self.exact_total else logger.warning
            emit(f"[GradMonitor] exact-norm reduction {self.plan}; world={world}; groups seen: {seen}")
        if not self.exact_total:
            logger.warning(
                "[GradMonitor] chosen groups cover %d of %d ranks, so norm_global is suppressed "
                "-- a partial cover would make it a sub-batch norm. rms_global is unbiased on any "
                "subset and keeps being reported.",
                self.total_factor,
                target,
            )

    @staticmethod
    def _distinct_data_ranks(world: int, pp_size: int, tp_size: int, sequence_parallel: bool) -> int:
        """How many ranks the whole global batch is spread over, counted once each.

        ``pp`` ranks own different layers, so they never enter the sum. Without
        sequence parallel the TP group holds *replicated* activations, so those
        ranks add no new samples either -- counting them would make the cover
        unreachable and suppress ``norm_global`` on every TP-without-SP run.
        """
        replicated = 1 if sequence_parallel else max(1, tp_size)
        return max(1, world // (max(1, pp_size) * replicated))

    @staticmethod
    def _group_getters(parallel_state):
        """Map the planner's candidate names to Megatron process-group getters.

        ``dp`` is the pure data-parallel group; ``cp_dp`` is data-parallel *with*
        context parallel folded in, so the planner can cover ``dp`` and ``cp`` in
        one orthogonal group when the topology nests them that way.

        ``get_data_parallel_group`` is called with its default ``with_gtp_remat=True``
        on purpose: that is the full distinct-data group, while the replicate group
        would miss the gtp_remat peers, which hold their own micro-batches.
        """
        return {
            "cp": lambda: parallel_state.get_context_parallel_group(),
            "tp": lambda: parallel_state.get_tensor_model_parallel_group(),
            "dp": lambda: parallel_state.get_data_parallel_group(with_context_parallel=False),
            "cp_dp": lambda: parallel_state.get_data_parallel_group(with_context_parallel=True),
            "ep": lambda: parallel_state.get_expert_model_parallel_group(),
        }

    @staticmethod
    def _resolve(name, getter) -> GroupInfo | None:
        """One named process group as a ``GroupInfo``, or ``None`` if absent.

        Membership comes from ``get_process_group_ranks`` because the plan needs
        the rank *sets* to test orthogonality -- sizes alone cannot tell a nested
        group from a disjoint one.
        """
        try:
            import torch.distributed as dist

            group = getter()
            if group is None:
                return None
            ranks = dist.get_process_group_ranks(group)
            return GroupInfo(name, group, dist.get_world_size(group), ranks)
        except Exception:
            return None

    def reduce(self, sq_values, counts):
        """``(shard_sq, total_sq, total_count)`` from this rank's running sums.

        The two sums are **independent**, both taken from the same local value on
        its own copy; chaining them would double count whenever a data group spans
        a shard group. The total travels as one concatenated tensor, so the whole
        schema costs one reduction per group rather than one per key; the shard sum
        does not need the counts at all, so it sends half as much.

        With nothing to reduce (single rank, or a layout that shards nothing) this
        is the identity, so the exact metrics stay well defined and equal to their
        local counterparts rather than disappearing.
        """
        if not self.enabled or (not self.shard_groups and not self.total_groups):
            return sq_values, sq_values, counts
        import torch.distributed as dist

        n = int(sq_values.shape[0])

        shard = sq_values.clone()
        for info in self.shard_groups:
            dist.all_reduce(shard, group=info.group)

        total = torch.cat([sq_values, counts])
        for info in self.total_groups:
            dist.all_reduce(total, group=info.group)

        return shard, total[:n], total[n:]


class GradHealthMonitor(TorchProbe):
    """L2 norm / RMS / abs-max of the activation gradient, per layer per module."""

    METRIC_PREFIX = "grad_health"
    MAX_AGGREGATED = set(MAX_AGGREGATED)
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        sample_layers: list[int] | None = None,
        hook_timing_enabled: bool = False,
        exclude_families=None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
            exclude_families=exclude_families,
        )
        self.sample_layers = set(sample_layers) if sample_layers else None
        self._failed: set[tuple[int, str]] = set()
        # AMP: latched so the scale is divided out exactly once per step, and so a
        # step whose backward recorded nothing is left alone.
        self._grad_metrics_finalized = False
        # Exact-norm path. Keyed by ``(layer_idx, position)``: a running sum of
        # squares on GPU plus the element and microbatch counts as plain Python
        # ints, so the hot path adds one ``add_`` and two integer increments and
        # still never syncs.
        self._sq_acc: dict[tuple[int, str], torch.Tensor] = {}
        self._sq_numel: dict[tuple[int, str], int] = {}
        self._sq_micro: dict[tuple[int, str], int] = {}
        self._reducer: ExactNormReducer | None = None
        self._exact_warned = False

    # ------------------------------------------------------------------
    # Setup: discover -> declare -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self) -> None:
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def _find_transformer_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        return find_transformer_layers(model)

    def _prepare_layers(self, model: nn.Module, layer_offset: int = 0):
        """``[(layer_idx, position, module)]`` for every hooked module; declare keys."""
        layers = self._find_transformer_layers(model)
        if not layers:
            return []

        targets: list[tuple[int, str, nn.Module]] = []
        for local_idx, layer in layers:
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers), layer_offset)
            if self.sample_layers and global_idx not in self.sample_layers:
                continue
            for position, module in _branch_modules(layer):
                targets.append((global_idx, position, module))

        for layer_idx, position, _module in targets:
            for metric in METRICS + GLOBAL_METRICS + TOKEN_METRICS:
                self.declare_layer_metric(layer_idx, f"{position}_{metric}")
        return targets

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        self._init_parallel_state()
        targets = self._prepare_layers(model, layer_offset=layer_offset)
        if not targets:
            logger.info("[GradMonitor] No transformer layers found; skipping.")
            return
        device = next((p.device for p in model.parameters()), None)
        assert device is not None, "model has no parameters; cannot pick a device"
        self.allocate_buffers(device)
        self._build_exact_norm_state([(model, targets)], device)
        self._attach_hooks(targets)

    def _build_exact_norm_state(self, chunk_targets, device) -> None:
        """Build the reducer and per-slot accumulators after ``allocate_buffers``.

        ``device`` is passed in rather than read back off the metric buffers: those
        are empty when every key is switched off, and a CPU accumulator would then
        raise on each ``add_`` inside the hook, where the error is swallowed.

        ``sequence_parallel`` decides whether the TP group is summed at all, and
        getting it wrong scales every squared norm by ``tp_size``; read it off any
        chunk's config rather than guessing.
        """
        sp_on = any(self._detect_sp(model) for model, _ in chunk_targets)
        self._reducer = ExactNormReducer(sequence_parallel=sp_on, verbose=self.verbose)
        for _model, targets in chunk_targets:
            for layer_idx, position, _module in targets:
                self._init_sq_accum(layer_idx, position, device)

    def _attach_hooks(self, targets) -> None:
        for layer_idx, position, module in targets:
            hook = module.register_forward_hook(
                self.timed_hook("grad_capture", self._make_output_hook(layer_idx, position))
            )
            self.hooks.append(hook)
        layer_count = len({layer_idx for layer_idx, _p, _m in targets})
        logger.info(f"[GradMonitor] Registered {len(self.hooks)} hooks across {layer_count} layers.")

    # ------------------------------------------------------------------
    # Hooks (the hot path)
    # ------------------------------------------------------------------

    def _log_failure(self, layer_idx: int, position: str, exc: Exception) -> None:
        if self.verbose and (layer_idx, position) not in self._failed:
            logger.error(f"[GradMonitor] Error at layer {layer_idx}/{position}: {exc}")
            self._failed.add((layer_idx, position))

    def _make_output_hook(self, layer_idx: int, position: str):
        """Forward hook: attach a gradient hook to this module's output tensor.

        Outputs that do not require grad are skipped rather than guarded later: a
        tensor outside the graph will never call the hook, so registering one would
        only cost a closure per microbatch (and ``register_hook`` errors on it).
        """

        def hook_fn(module, args, output):
            if not self._should_monitor():
                return None
            try:
                tensor = _output_tensor(output)
                if tensor is None or not tensor.requires_grad:
                    return None
                tensor.register_hook(self._make_grad_recorder(layer_idx, position))
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return None

        return hook_fn

    def _make_grad_recorder(self, layer_idx: int, position: str):
        """Gradient hook: reduce ``grad`` into the accumulators, leave it unchanged."""
        slot = (layer_idx, position)

        def record(grad):
            try:
                with torch.no_grad():
                    stats, sum_sq, numel = grad_square_and_stats(grad)
                    for metric, value in stats.items():
                        self.record_layer_metric(layer_idx, f"{position}_{metric}", value)
                    buf = self._sq_acc.get(slot)
                    if buf is not None:
                        buf.add_(sum_sq)
                        self._sq_numel[slot] += numel
                        self._sq_micro[slot] += 1
                self._grad_metrics_finalized = False
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return None

        return record

    # ------------------------------------------------------------------
    # Exact global norms (cold path, one batched collective per step)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sp(model) -> bool:
        """Whether activations are sequence-sharded inside the TP group.

        Megatron chunks (GPTModel, or GPTModel under Float16Module) always carry a
        ``TransformerConfig`` at ``.config``, so one read suffices.
        """
        if hasattr(model, "module"):
            model = model.module
        return bool(getattr(getattr(model, "config", None), "sequence_parallel", False))

    def _init_sq_accum(self, layer_idx: int, position: str, device) -> None:
        slot = (layer_idx, position)
        self._sq_acc[slot] = torch.zeros((), dtype=torch.float32, device=device)
        self._sq_numel[slot] = 0
        self._sq_micro[slot] = 0

    def _reset_sq_accum(self) -> None:
        for slot, buf in self._sq_acc.items():
            buf.zero_()
            self._sq_numel[slot] = 0
            self._sq_micro[slot] = 0

    # ------------------------------------------------------------------
    # AMP de-scaling (cold path, once per step)
    # ------------------------------------------------------------------

    def finalize_scaled_grad_metrics(self, scaler=None) -> None:
        """Divide this step's AMP loss scale out of every degree-1 accumulator.

        The eight magnitude metrics are degree-1 homogeneous in the gradient, so
        one division fixes each, and it is valid on the raw accumulator because
        both aggregations commute with a positive scale. The three ratio/fraction
        metrics in ``SCALE_INVARIANT`` are degree-0 -- already correct as recorded
        -- so the division skips them; dividing would make them ``scale`` too small.

        No-op under bf16 (Megatron's default), which does not loss-scale. Under
        fp16 the trainer passes ``optimizer.grad_scaler`` here before
        ``optimizer.step()``; ``_flush_buffers`` is the fallback read point for
        direct users and non-AMP runs. Idempotent within a step.
        """
        if self._grad_metrics_finalized:
            return
        # Before the division, so the exact norms get de-scaled with everything
        # else -- they are sqrt of a sum of squares, i.e. also degree-1 in g.
        self._emit_exact_norms()
        scale = getattr(scaler, "_scale", None) if scaler is not None else None
        if scale is not None:
            # Megatron's grad scaler holds ``_scale`` as a shape-[1] tensor while the
            # accumulators are 0-dim, and an in-place divide cannot broadcast its own
            # output -- reshape to a scalar or every fp16 step raises.
            scale = scale.detach().float().reshape(()) if isinstance(scale, torch.Tensor) else float(scale)
            for key in self._mean_keys | self._max_keys:
                if self._gpu_cnt.get(key, 0) > 0 and not key.endswith(SCALE_INVARIANT):
                    self._gpu_acc[key].div_(scale)
        self._grad_metrics_finalized = True

    def _emit_exact_norms(self) -> None:
        """One batched collective, then write the three exact series.

        Recorded through ``record_layer_metric`` like any other value, so family
        filtering and the ``log_per_layer`` / global derivation keep working. Each
        key is written once, so the mean accumulator divides by a count of 1 and
        the number survives the flush unchanged; and because every rank inside the
        reduction ends up with the same value, the cross-rank mean is a no-op for
        the two global series.
        """
        slots = [slot for slot in sorted(self._sq_acc, key=str) if self._sq_micro[slot] > 0]
        if not slots or self._reducer is None:
            self._reset_sq_accum()
            return
        try:
            with torch.no_grad():
                device = self._sq_acc[slots[0]].device
                local_sq = torch.stack([self._sq_acc[slot] for slot in slots])
                counts = torch.tensor(
                    [float(self._sq_numel[slot]) for slot in slots], dtype=torch.float32, device=device
                )
                shard_sq, total_sq, total_count = self._reducer.reduce(local_sq, counts)
                micro = torch.tensor(
                    [float(self._sq_micro[slot]) for slot in slots], dtype=torch.float32, device=device
                )
                # norm_mb: complete tensor of one microbatch, quadratic-mean over
                # the microbatches this rank saw (the accumulator holds their sum).
                norm_mb = torch.sqrt(shard_sq / micro.clamp(min=1.0))
                rms_global = torch.sqrt(total_sq / total_count.clamp(min=1.0))
                norm_global = torch.sqrt(total_sq)
                # ``norm_global`` only means "the whole batch" when the reduction
                # covered every rank exactly once; on a partial cover it would be a
                # sub-batch norm, so it is dropped rather than logged wrong.
                emit_total_norm = self._reducer.exact_total or not self._reducer.enabled
                for i, (layer_idx, position) in enumerate(slots):
                    series = [("norm_mb", norm_mb[i]), ("rms_global", rms_global[i])]
                    if emit_total_norm:
                        series.append(("norm_global", norm_global[i]))
                    for metric, value in series:
                        self.record_layer_metric(layer_idx, f"{position}_{metric}", value)
        except Exception as exc:
            if self.verbose and not self._exact_warned:
                logger.error(f"[GradMonitor] exact-norm reduction failed: {exc}")
                self._exact_warned = True
        finally:
            self._reset_sq_accum()

    def _flush_buffers(self) -> None:
        self.finalize_scaled_grad_metrics()
        super()._flush_buffers()
        self._grad_metrics_finalized = False


def setup_grad_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_layers: list[int] | None = None,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = GradHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
        hook_timing_enabled=hook_timing_enabled,
        exclude_families=exclude_families,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_targets = []
    layer_offset = 0
    for m in models:
        targets = monitor._prepare_layers(m, layer_offset=layer_offset)
        chunk_targets.append((m, targets))
        layer_offset += len(monitor._find_transformer_layers(m))
    if any(targets for _, targets in chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        monitor._build_exact_norm_state(chunk_targets, device)
        for _, targets in chunk_targets:
            monitor._attach_hooks(targets)
    logger.info(f"[GradMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["grad_health"] = monitor
    return model
