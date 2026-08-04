"""Log-Alignment Ratio (LAR) online monitor for the Megatron backend.

Definition (arXiv:2605.28975): for a linear map ``y = W x``,

    LAR = log_n( ||W X||_rms / (||W||_rms * ||X||_rms) )

where ``n`` = input dim of ``W`` (log base), ``||.||_rms = ||.||_F / sqrt(numel)``.
This monitor computes LAR online at two families of sites without SVD:

1. ``lm_head`` (output projection) on the last PP stage — hidden and logits are
   already materialised in the forward, so all three norms are one ``mean(x**2)``
   each. Uses ``labels`` (Megatron's model-forward kwarg) to build a loss mask.

2. Every MoE ``router`` — hidden is the router input, weight is ``module.weight``,
   logits are recomputed locally with a small ``F.linear`` (the router's forward
   returns ``(probs, routing_map)``, not the raw gating logits).

All accumulation stays as GPU 0-dim tensors (perf-rules: no host sync in hooks;
distributed reductions and the final divisions happen in ``_flush_buffers``).
"""

import logging
import weakref
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...core.training_logs import training_logs
from .base import TorchProbe

logger = logging.getLogger(__name__)

_IGNORE_INDEX_DEFAULT = -100  # Megatron convention for ``labels`` padding


def _sum_of_squares(t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 sum-of-squares of ``t`` (optionally over ``mask``-selected rows only),
    without materialising any full-size temporary.

    Two allocation traps this avoids, both of which bite hard on the lm_head logits
    (a ``[4096, 200019]`` bf16 tensor is 1.53 GiB, and the cost scales with
    ``vocab x seq``):

    1. ``t.float().pow(2).sum()`` allocates TWO full fp32 temporaries — the cast, then
       the squares — i.e. ~4x the bf16 input, measured 781 MiB against a 195 MiB input.
       ``vector_norm(dtype=torch.float32)`` accumulates in fp32 while reading the source
       dtype in place: measured 0 extra bytes, and squaring the resulting 0-dim scalar is
       exact, so the value is bit-identical to the naive form.
    2. ``t[mask]`` materialises a masked COPY (~valid_frac x the tensor). Instead take
       per-row fp32 norms — ``[T]``, negligible — square them, and select there. Summing
       squared row norms is algebraically the same sum; measured agreement is fp32
       round-off (~1e-7 relative).

    ``mask`` is a boolean over ``t``'s first dim; ``t`` must be 2-D when it is given.
    """
    if mask is None:
        return torch.linalg.vector_norm(t, dtype=torch.float32).square()
    row_sq = torch.linalg.vector_norm(t, dim=-1, dtype=torch.float32).square()
    return row_sq[mask].sum()


def _masked_numel(t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Element count of ``t`` restricted to ``mask``'s rows, as a 0-dim GPU tensor.

    ``mask.sum()`` stays a tensor (no ``.item()``) so the hot path never syncs; the
    trailing dim comes from a shape read, which is free.

    fp64, not fp32: the logits count is ``T * vocab`` (~8.2e8 for seq 4096 x 200k vocab),
    well past fp32's exact-integer limit of 2**24, so an fp32 count silently rounds (a
    measured 79207520 for a true 79207524). The error is ~1e-7 relative and harmless for
    ``lar``, but the counts are all-reduced as fp64 downstream anyway, so there is no
    reason to introduce rounding here.
    """
    if mask is None:
        return torch.as_tensor(float(t.numel()), device=t.device, dtype=torch.float64)
    return mask.sum().to(torch.float64) * float(t.shape[-1])


class LARMonitor(TorchProbe):
    """Log-Alignment Ratio for lm_head + MoE routers.

    Emits exactly one metric per site (``lm_head`` and ``router_{L}``) per
    monitored step: ``lar``. The underlying ``rms_w`` / ``rms_x`` / ``rms_z`` are
    flush-time intermediates and deliberately not logged — only their combination
    is meaningful, and raw activation scale is already covered by ``massive_act``.
    Two globals: ``global_lm_head_lar`` (trivial — equals the single lm_head site)
    and ``global_router_lar`` (mean over routers).
    """

    METRIC_PREFIX = "lar"

    def __init__(
        self,
        hook_lm_head: bool = True,
        hook_moe_router: bool = True,
        apply_loss_mask: bool = True,
        label_ignore_index: int = _IGNORE_INDEX_DEFAULT,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
    ):
        super().__init__(
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        self.hook_lm_head = hook_lm_head
        self.hook_moe_router = hook_moe_router
        self.apply_loss_mask = apply_loss_mask
        self.label_ignore_index = label_ignore_index

        # site key -> {"H": int, "ss": {"W","X","Z"} -> 0-dim gpu tensor,
        #              "n": {"W","X","Z"} -> 0-dim gpu tensor,
        #              "w_done": bool}
        self._sites: dict[str, dict[str, Any]] = {}
        self._router_site_keys: list[str] = []  # ordered list for global_router
        self._captured_labels: torch.Tensor | None = None
        self._label_capture_models: list = []
        self._sequence_parallel = False
        self._warned_router_sp = False
        self._tp_group = None
        self._dp_group = None
        self._parallel_state_ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        """Single-chunk path. Multi-chunk (VPP) goes through ``setup_lar_monitor``."""
        self._init_parallel_state()
        self._attach(model, layer_offset=layer_offset)

    def _init_parallel_state(self):
        if self._parallel_state_ready:
            return
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self._tp_group = parallel_state.get_tensor_model_parallel_group()
                try:
                    self._dp_group = parallel_state.get_data_parallel_group()
                except Exception:
                    self._dp_group = None
        except ImportError:
            pass
        self._parallel_state_ready = True

    def _attach(self, model: nn.Module, layer_offset: int = 0):
        base = model.module if hasattr(model, "module") else model
        # Sequence-parallel flag drives the router-under-SP fallback (labels can't be
        # aligned cheaply to seq-sharded hidden without more slicing logic).
        cfg = getattr(base, "config", None)
        if cfg is not None:
            self._sequence_parallel = self._sequence_parallel or bool(getattr(cfg, "sequence_parallel", False))

        # Label capture: one model-level pre-hook per unique root model. Fires before
        # the layer hooks so ``self._captured_labels`` is set for that forward.
        if self.apply_loss_mask and not any(m is model for m in self._label_capture_models):
            self._label_capture_models.append(model)
            hook = model.register_forward_pre_hook(self._make_label_capture_hook(), with_kwargs=True)
            self.hooks.append(hook)

        # LM head: last PP stage only (owns ``output_layer``).
        if self.hook_lm_head:
            output_layer = getattr(base, "output_layer", None)
            weight = self._resolve_lm_head_weight(base, output_layer)
            if output_layer is not None and weight is not None:
                self._register_site("lm_head", H=int(weight.shape[1]))
                hook = output_layer.register_forward_hook(
                    self.timed_hook("lar_lm_head", self._make_lm_head_hook(weight)),
                    with_kwargs=False,
                )
                self.hooks.append(hook)
            elif self.verbose and output_layer is not None:
                logger.warning("[LARMonitor] output_layer has no resolvable weight; skipping lm_head site.")

        # Routers: every MoE layer.
        if self.hook_moe_router:
            for layer_idx, router in self._find_routers(base, layer_offset=layer_offset):
                site = f"router_{layer_idx}"
                weight = getattr(router, "weight", None)
                if weight is None:
                    continue
                self._register_site(site, H=int(weight.shape[1]))
                self._router_site_keys.append(site)
                hook = router.register_forward_hook(
                    self.timed_hook(site, self._make_router_hook(site, weakref.ref(router))),
                    with_kwargs=False,
                )
                self.hooks.append(hook)

    def _register_site(self, key: str, H: int) -> None:
        if key in self._sites:
            return
        self._sites[key] = {"H": H, "ss": {}, "n": {}, "w_done": False}

    def _resolve_lm_head_weight(self, base: nn.Module, output_layer):
        """Return the tied ``[V, H]`` LM-head weight, or ``None``.

        Tries ``shared_embedding_or_output_weight()`` first — this is the safe
        accessor when ``share_embeddings_and_output_weights=True`` (Megatron
        sets ``output_layer.weight = None`` and passes the shared tensor at
        forward time). Falls back to ``output_layer.weight`` for untied heads.
        """
        getter = getattr(base, "shared_embedding_or_output_weight", None)
        if callable(getter):
            try:
                w = getter()
                if w is not None:
                    return w
            except Exception:
                pass
        return getattr(output_layer, "weight", None) if output_layer is not None else None

    def _find_routers(self, model: nn.Module, layer_offset: int = 0) -> list[tuple[int, nn.Module]]:
        """Discover MoE routers via the same layer walk MoEMonitor uses."""
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            found = []
            for _, module in model.named_modules():
                if module.__class__.__name__ in ("MoELayer", "BaseMoELayer") and hasattr(module, "router"):
                    found.append((len(found), module.router))
            return found
        found: list[tuple[int, nn.Module]] = []
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers), layer_offset)
            router = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                router = layer.mlp.router
            elif hasattr(layer, "moe") and hasattr(layer.moe, "router"):
                router = layer.moe.router
            elif hasattr(layer, "router"):
                router = layer.router
            if router is not None:
                found.append((global_idx, router))
        return found

    # ------------------------------------------------------------------
    # Label capture (model-level pre-hook)
    # ------------------------------------------------------------------

    def _make_label_capture_hook(self):
        def hook_fn(module, args, kwargs):
            self._captured_labels = kwargs.get("labels") if kwargs else None
            return None

        return hook_fn

    def _aligned_labels(self, seq: int, batch: int) -> torch.Tensor | None:
        """Return labels flattened to ``[seq*batch]`` in seq-major order, else ``None``."""
        labels = self._captured_labels
        if labels is None:
            return None
        if labels.dim() == 2 and labels.shape == (batch, seq):
            return labels.transpose(0, 1).reshape(-1)
        if labels.dim() == 2 and labels.shape == (seq, batch):
            return labels.reshape(-1)
        if labels.dim() == 1 and labels.numel() == seq * batch:
            return labels.reshape(-1)
        return None

    def _valid_mask(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """Boolean mask over flattened tokens; ``None`` = "keep all"."""
        if not self.apply_loss_mask or hidden.dim() < 2:
            return None
        if hidden.dim() >= 3:
            seq, batch = hidden.shape[0], hidden.shape[1]
        else:
            return None
        labels_flat = self._aligned_labels(seq, batch)
        if labels_flat is None:
            return None
        return labels_flat != self.label_ignore_index

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def _accumulate(
        self,
        site: str,
        X: torch.Tensor,
        W: torch.Tensor,
        Z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Update per-site (ss, n) with the SUM-of-squares + numel of X, W, Z.

        ``mask`` (boolean over the token dim) restricts X and Z to the valid tokens
        WITHOUT the caller having to index them first — indexing would copy a
        ``[T, vocab]`` logits tensor. W is token-independent so the mask never applies.

        Sums go through ``_sum_of_squares``, which accumulates in fp32 without casting
        the input — important because ``Z`` is the ``[T, vocab]`` logits and an fp32 copy
        of it would dwarf the tensor itself.

        ``W``'s stats are computed once per step per site (weight is constant within
        a step; sums are data-independent) to avoid redundant kernels across
        microbatches on a potentially large ``[V,H]`` tensor.
        """
        s = self._sites[site]
        s_ssX = _sum_of_squares(X, mask)
        s_ssZ = _sum_of_squares(Z, mask)
        nX = _masked_numel(X, mask)
        nZ = _masked_numel(Z, mask)
        s["ss"]["X"] = s["ss"].get("X", torch.zeros_like(s_ssX)) + s_ssX
        s["ss"]["Z"] = s["ss"].get("Z", torch.zeros_like(s_ssZ)) + s_ssZ
        s["n"]["X"] = s["n"].get("X", torch.zeros_like(nX)) + nX
        s["n"]["Z"] = s["n"].get("Z", torch.zeros_like(nZ)) + nZ
        if not s["w_done"]:
            s["ss"]["W"] = _sum_of_squares(W)
            s["n"]["W"] = _masked_numel(W, None)
            s["w_done"] = True

    def _make_lm_head_hook(self, weight: torch.Tensor):
        def hook_fn(module, inputs, output):
            if not self._should_monitor():
                return
            try:
                x = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else None
                if not isinstance(x, torch.Tensor):
                    return
                # output_layer may return (logits, bias); take the first tensor.
                z = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(z, torch.Tensor):
                    return
                with torch.no_grad():
                    x_flat = x.reshape(-1, x.shape[-1])
                    z_flat = z.reshape(-1, z.shape[-1])
                    mask = self._valid_mask(x)
                    if mask is not None and mask.shape[0] != x_flat.shape[0]:
                        mask = None
                    self._accumulate("lm_head", x_flat, weight, z_flat, mask=mask)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[LARMonitor] lm_head hook error: {e}")

        return hook_fn

    def _make_router_hook(self, site: str, router_ref):
        def hook_fn(module, inputs, output):
            if not self._should_monitor():
                return
            try:
                x = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else None
                if not isinstance(x, torch.Tensor):
                    return
                w = getattr(module, "weight", None)
                if w is None:
                    return
                with torch.no_grad():
                    x_flat = x.reshape(-1, x.shape[-1])
                    # SP fallback: labels are [b, s_full], but hidden is seq-sharded.
                    # Skip masking on routers under SP (documented gotcha).
                    mask = None if self._sequence_parallel else self._valid_mask(x)
                    if mask is not None and mask.shape[0] == x_flat.shape[0]:
                        x_flat = x_flat[mask]
                    elif self._sequence_parallel and not self._warned_router_sp and self.verbose:
                        logger.warning(
                            "[LARMonitor] sequence_parallel=True: router LAR uses all tokens (mask skipped)."
                        )
                        self._warned_router_sp = True
                    x_fp32 = x_flat.float()
                    w_fp32 = w.float()
                    z_flat = F.linear(x_fp32, w_fp32)
                    self._accumulate(site, x_fp32, w, z_flat)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[LARMonitor] {site} hook error: {e}")

        return hook_fn

    # ------------------------------------------------------------------
    # Cold path: SUM-reduce + LAR computation (runs at step end)
    # ------------------------------------------------------------------

    def _reduce_pair(self, ss: torch.Tensor, n: torch.Tensor, group) -> tuple[torch.Tensor, torch.Tensor]:
        """SUM-all-reduce a (sum_of_squares, count) pair on the given group.

        Grouped as a length-2 stack so it's one collective per pair. Cold-path
        collective — see monitor-hook-perf-rules skill: only justified when
        correctness requires it (LAR is nonlinear in the sums, so per-rank LARs
        cannot be averaged; the sums themselves must be pooled globally).
        """
        if group is None or not dist.is_available() or not dist.is_initialized():
            return ss, n
        pair = torch.stack([ss.detach(), n.detach()]).to(dtype=torch.float64)
        dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=group)
        return pair[0], pair[1]

    def _finalise_site(self, site: str, key: str) -> dict[str, torch.Tensor] | None:
        s = self._sites[site]
        if "X" not in s["ss"] or "Z" not in s["ss"] or "W" not in s["ss"]:
            return None  # no data this step
        H = float(s["H"])
        if site == "lm_head":
            ssW, nW = self._reduce_pair(s["ss"]["W"], s["n"]["W"], self._tp_group)
            ssX, nX = self._reduce_pair(s["ss"]["X"], s["n"]["X"], self._dp_group)
            ssZ_tp, nZ_tp = self._reduce_pair(s["ss"]["Z"], s["n"]["Z"], self._tp_group)
            ssZ, nZ = self._reduce_pair(ssZ_tp, nZ_tp, self._dp_group)
        else:
            ssW, nW = s["ss"]["W"], s["n"]["W"]  # router weight replicated across TP
            ssX, nX = self._reduce_pair(s["ss"]["X"], s["n"]["X"], self._dp_group)
            ssZ, nZ = self._reduce_pair(s["ss"]["Z"], s["n"]["Z"], self._dp_group)
        eps = 1e-12
        rms_w = (ssW / nW.clamp_min(1)).clamp_min(eps).sqrt()
        rms_x = (ssX / nX.clamp_min(1)).clamp_min(eps).sqrt()
        rms_z = (ssZ / nZ.clamp_min(1)).clamp_min(eps).sqrt()
        ratio = rms_z / (rms_w * rms_x).clamp_min(eps)
        lar = ratio.log() / torch.log(torch.tensor(H, device=ratio.device, dtype=ratio.dtype))
        return {f"{key}/lar": lar}

    def _flush_buffers(self) -> None:
        super()._flush_buffers()  # harmless: no scalar keys declared via declare_*
        if not self._sites:
            return

        log_dict: dict[str, torch.Tensor] = {}

        # lm_head
        if "lm_head" in self._sites and self._sites["lm_head"]["ss"]:
            metrics = self._finalise_site("lm_head", "lm_head")
            if metrics is not None:
                log_dict.update({f"{self.METRIC_PREFIX}/{k}": v for k, v in metrics.items()})
                log_dict[f"{self.METRIC_PREFIX}/global_lm_head_lar"] = metrics["lm_head/lar"]

        # routers
        router_lars: list[torch.Tensor] = []
        for site in self._router_site_keys:
            if not self._sites[site]["ss"]:
                continue
            metrics = self._finalise_site(site, site)
            if metrics is None:
                continue
            log_dict.update({f"{self.METRIC_PREFIX}/{k}": v for k, v in metrics.items()})
            router_lars.append(metrics[f"{site}/lar"])
        if router_lars:
            log_dict[f"{self.METRIC_PREFIX}/global_router_lar"] = torch.stack(router_lars).mean()

        if log_dict:
            training_logs.update(**log_dict)

        # Reset per-step accumulators (keep site keys + ``H`` + hook wiring).
        for s in self._sites.values():
            s["ss"].clear()
            s["n"].clear()
            s["w_done"] = False
        self._captured_labels = None


def setup_lar_monitor(
    model,
    hook_lm_head: bool = True,
    hook_moe_router: bool = True,
    apply_loss_mask: bool = True,
    label_ignore_index: int = _IGNORE_INDEX_DEFAULT,
    monitor_interval: int = 1,
    verbose: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
):
    """Build LARMonitor and attach hooks across every model chunk (VPP-safe)."""
    monitor = LARMonitor(
        hook_lm_head=hook_lm_head,
        hook_moe_router=hook_moe_router,
        apply_loss_mask=apply_loss_mask,
        label_ignore_index=label_ignore_index,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    for chunk in models:
        monitor._attach(chunk)
    if monitor_dict is not None:
        monitor_dict["lar"] = monitor
    return monitor
