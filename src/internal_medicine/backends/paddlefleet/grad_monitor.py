"""Activation-gradient health monitor for PaddleFleet.

Answers "where does the gradient grow or die on the way back" with one L2 norm
per module output per layer. Three positions per layer -- the attention branch
output, the MLP/MoE branch output, and the decoder layer output (the residual
stream itself) -- times the three magnitudes in ``grad_metrics``.

Collection is a ``forward_post_hook`` that registers a tensor gradient hook on
the module's output. The forward hook measures nothing itself: it exists only to
get hold of the tensor whose gradient is wanted, so the forward path pays one
``register_hook`` per module and the reductions all happen during backward.

**AMP.** A gradient hook sees the *loss-scaled* activation gradient, so the raw
numbers are the scale (order 1e4, and it moves on every overflow) times the
quantity of interest. ``finalize_scaled_grad_metrics`` divides the scale back out
once per step from ``on_optimizer_begin`` -- before ``optimizer.step()``, while
``scaler._scale`` still holds the value this step's backward actually used.
Without that step the curves would jump by 2x every time the scaler halves.

**Sharding.** No collective runs here (hot-path discipline: see
``.claude/skills/monitor-hook-perf-rules``); the cross-rank reduction happens at
flush time in ``gather.py``. Under TP / SP / CP each rank therefore measures its
own shard: ``rms`` is the true global value when shards are equal-sized, while
``norm`` is the per-shard norm -- a fixed ``1/sqrt(num_shards)`` of the global
one, so it tracks the same trend but is not the number a global grad-norm clip
would report.

**Recompute.** ``_should_monitor`` is False while grad is disabled, so the
discarded first forward of a recomputed block registers nothing and the replayed
forward -- the one whose graph actually carries the backward -- is what gets
hooked.
"""

from __future__ import annotations

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .grad_metrics import MAX_AGGREGATED, METRICS, grad_magnitude_stats
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)


def _is_transformer_layer(layer) -> bool:
    """Same predicate ``massive_act`` uses, so both monitors see one layer set."""
    return hasattr(layer, "self_attn") or hasattr(layer, "self_attention") or hasattr(layer, "input_layernorm")


def _output_tensor(outputs):
    """First tensor of a PaddleFleet forward result, or ``None``."""
    if isinstance(outputs, paddle.Tensor):
        return outputs
    if isinstance(outputs, dict):
        return outputs.get("hidden_states")
    if isinstance(outputs, (tuple, list)) and outputs:
        return _output_tensor(outputs[0])
    return None


def _branch_modules(layer):
    """``[(position, module)]`` for the branches this layer actually has.

    Only positions that exist are returned, and the same list drives both the
    schema and the hooks -- so a layer without an ``mlp`` never declares a key it
    could not fill.
    """
    modules = []
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "self_attention", None)
    if attn is not None:
        modules.append(("attn_out", attn))
    ffn = getattr(layer, "mlp", None)
    if ffn is None:
        ffn = getattr(layer, "moe", None)
    if ffn is not None:
        modules.append(("ffn_or_moe_out", ffn))
    # The layer itself is the residual stream leaving this block.
    modules.append(("layer_out", layer))
    return modules


class PaddleGradHealthMonitor(PaddleProbe):
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
        exclude_families=None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            exclude_families=exclude_families,
        )
        self.sample_layers = set(sample_layers) if sample_layers else None
        self._failed: set[tuple[int, str]] = set()
        # AMP: latched so the scale is divided out exactly once per step, and so a
        # step whose backward recorded nothing is left alone.
        self._grad_metrics_finalized = False

    # ------------------------------------------------------------------
    # Setup: discover -> declare -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self) -> None:
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

    def _find_targets(self, model):
        """``[(layer_idx, position, module, attn_type)]`` for every hooked module."""
        layers = get_decoder_layers(model)
        if not layers:
            return []

        monitor_layers = iter_monitor_layers(layers, _is_transformer_layer, pp_rank=self.pp_rank)
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)

        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            for position, module in _branch_modules(item.layer):
                targets.append((item.idx, position, module, item.attn_type))
        return targets

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        targets = self._find_targets(model)
        if not targets:
            logger.info("[PaddleGradMonitor] No transformer layers found; skipping.")
            return

        for layer_idx, position, _module, attn_type in targets:
            for metric in METRICS:
                self.declare_layer_metric(layer_idx, f"{position}_{metric}", attn_type=attn_type)

        self.allocate_buffers()

        for layer_idx, position, module, attn_type in targets:
            self.hooks.append(module.register_forward_post_hook(self._make_output_hook(layer_idx, position, attn_type)))

        layer_count = len({layer_idx for layer_idx, _p, _m, _a in targets})
        logger.info(f"[PaddleGradMonitor] Registered {len(self.hooks)} hooks across {layer_count} layers.")

    # ------------------------------------------------------------------
    # Hooks (the hot path)
    # ------------------------------------------------------------------

    def _log_failure(self, layer_idx: int, position: str, exc: Exception) -> None:
        if self.verbose and (layer_idx, position) not in self._failed:
            logger.error(f"[PaddleGradMonitor] Error at layer {layer_idx}/{position}: {exc}")
            self._failed.add((layer_idx, position))

    def _make_output_hook(self, layer_idx: int, position: str, attn_type: str | None):
        """Forward hook: attach a gradient hook to this module's output tensor.

        ``stop_gradient`` outputs are skipped rather than guarded later: a tensor
        outside the graph will never call the hook, so registering one would only
        cost a closure per microbatch.
        """

        def hook_fn(module, _inputs, outputs):
            if not module.training or not self._should_monitor():
                return None
            try:
                tensor = _output_tensor(outputs)
                if tensor is None or tensor.stop_gradient:
                    return None
                tensor.register_hook(self._make_grad_recorder(layer_idx, position, attn_type))
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return None

        return hook_fn

    def _make_grad_recorder(self, layer_idx: int, position: str, attn_type: str | None):
        """Gradient hook: reduce ``grad`` into the accumulators, return it unchanged."""

        def record(grad):
            try:
                with paddle.no_grad():
                    for metric, value in grad_magnitude_stats(grad).items():
                        self.record_layer_metric(layer_idx, f"{position}_{metric}", value, attn_type=attn_type)
                self._grad_metrics_finalized = False
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return grad

        return record

    # ------------------------------------------------------------------
    # AMP de-scaling (cold path, once per step)
    # ------------------------------------------------------------------

    def finalize_scaled_grad_metrics(self, scaler=None) -> None:
        """Divide this step's AMP loss scale out of every accumulator.

        Every metric this monitor owns is degree-1 homogeneous in the gradient, so
        one division fixes all of them -- and it is valid on the raw accumulator
        because both aggregations commute with a positive scale: ``sum(g_i)/S`` is
        the mean of ``g_i/S``, and ``max(g_i)/S`` is the max of ``g_i/S``. The
        scaler updates ``_scale`` in its own ``step``/``update``, i.e. once per
        optimizer step, so every microbatch folded into these sums shared it.

        Idempotent within a step: called from ``on_optimizer_begin`` when the
        trainer provides a scaler, with ``_flush_buffers`` as the fallback read
        point for direct users and non-AMP runs.
        """
        if self._grad_metrics_finalized:
            return
        scale = getattr(scaler, "_scale", None) if scaler is not None else None
        if scale is not None:
            scale = paddle.assign(scale).detach().astype("float32")
            for key in self._mean_keys | self._max_keys:
                if self._gpu_cnt.get(key, 0) > 0:
                    self._gpu_acc[key].divide_(scale)
        self._grad_metrics_finalized = True

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
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = PaddleGradHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
        exclude_families=exclude_families,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["grad_health"] = monitor
    return model
