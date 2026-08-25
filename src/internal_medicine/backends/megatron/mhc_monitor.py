"""mHC Health Monitor for Megatron-Bridge.

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 29
scalar series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_h_post_stream_concentration  {attn,mlp}_h_post_token_std
    {attn,mlp}_branch_residual_share  {attn,mlp}_branch_residual_share_max
    {attn,mlp}_amax_gain_fwd  {attn,mlp}_amax_gain_bwd
    {attn,mlp}_h_res_logits_min  {attn,mlp}_h_res_logits_max
    {attn,mlp}_h_res_logits_grad_min  {attn,mlp}_h_res_logits_grad_max
    {attn,mlp}_composite_amax_gain_fwd_max  {attn,mlp}_composite_amax_gain_bwd_max
    {attn,mlp}_h_pre_logits_min   {attn,mlp}_h_pre_logits_max
    {attn,mlp}_h_post_logits_min  {attn,mlp}_h_post_logits_max
    {attn,mlp}_alpha_pre  {attn,mlp}_alpha_post  {attn,mlp}_alpha_res
    {attn,mlp}_bias_pre_mean   {attn,mlp}_bias_pre_abs_max
    {attn,mlp}_bias_post_mean  {attn,mlp}_bias_post_abs_max
    {attn,mlp}_bias_res_mean   {attn,mlp}_bias_res_abs_max

plus a set of per-element mapping series (``n^2 + 2n`` per hc module) that expand
``h_res`` / ``h_pre`` / ``h_post`` cell by cell.

The last 13 scalars cover paper Eq. (7) — the ``alpha`` / ``bias`` /
pre-sigmoid-logit terms that feed Eq. (8). The nine parameter series are
step-level and are read once per flush, off the hot path.

The schema, the metric semantics and the aggregation choices are kept identical
to ``backends/paddlefleet/mhc_monitor.py`` on purpose: the two backends train
the same mHC model, so the curves have to be comparable across them.

Forward hooks are not enough, so four bound methods are wrapped instead (see
``mhc_metrics`` for what each metric means):

- ``compute_mappings`` — ``h_pre`` is not in ``forward``'s return value.
- ``_sinkhorn_op``     — the mixing logits only exist before the Sinkhorn
  projection, and this is the one call site that consumes them. ``_compute_h``
  is *not* a usable hook point here: unlike PaddleFleet, Megatron's
  ``compute_mappings`` skips ``_compute_h`` entirely when
  ``config.use_fused_mhc`` selects ``fused_proj_rms_compute_h``, which produces
  the logits inside the kernel instead. Both paths converge on the
  ``_sinkhorn_op`` argument, so wrapping it records the logits the model
  actually consumes on either path.
- ``_compute_h`` — the *only* route to the pre-sigmoid ``h_pre`` / ``h_post``
  logits, which need its ``(proj, r)`` arguments. See the caveat below.
- ``fused_h_res_h_post_bda`` — the only point where both update terms are
  visible.

Two fused-path caveats, both specific to a Megatron whose ``use_fused_mhc``
folds ``compute_h`` into the kernel:

1. The kernel computes the residual-mixing logits itself, so on that path their
   values carry the kernel's numerics rather than the native reference's. The
   four ``h_res_logits*`` series are therefore trend-comparable, but not
   value-for-value comparable, against a backend whose logits come from a native
   ``_compute_h``.
2. ``fused_proj_rms_compute_h`` returns ``(h_pre, h_post, h_res, r)`` and never
   exposes ``proj``, so the four ``h_{pre,post}_logits_*`` series cannot be
   rebuilt on that path without repeating the projection matmul on the hot path.
   Their accumulators simply stay empty and are not emitted; the other 25
   scalars and all the per-element series are unaffected.

Strict cross-backend alignment runs should disable ``use_fused_mhc`` on both
sides.

Hot-path discipline (no D2H sync, no hook-time collectives, schema fixed at
registration): see ``.claude/skills/monitor-hook-perf-rules``.
"""

import logging

import torch
import torch.nn as nn

from .base import TorchProbe
from .mhc_metrics import (
    amax_gain,
    branch_residual_share,
    gate_logits_extrema,
    gate_stats,
    h_post_structure_stats,
    h_res_logits_extrema,
    mapping_param_stats,
)

logger = logging.getLogger(__name__)


# mHC classes are optional: this monitor must be a no-op when they are absent
# (non-mHC model, or a Megatron build without hyper-connections). Bind to None
# on import failure and gate every code path on that.
try:
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
except Exception:  # pragma: no cover - environment dependent
    HyperConnectionModule = None
    HyperConnectionTransformerLayer = None


_METRIC_NAMES = (
    "h_pre_mean",
    "h_pre_std",
    "h_post_mean",
    "h_post_std",
    "h_post_stream_concentration",
    "h_post_token_std",
    "branch_residual_share",
    "branch_residual_share_max",
    "amax_gain_fwd",
    "amax_gain_bwd",
    "h_res_logits_min",
    "h_res_logits_max",
    "h_res_logits_grad_min",
    "h_res_logits_grad_max",
    "composite_amax_gain_fwd_max",
    "composite_amax_gain_bwd_max",
    "h_pre_logits_min",
    "h_pre_logits_max",
    "h_post_logits_min",
    "h_post_logits_max",
    "alpha_pre",
    "alpha_post",
    "alpha_res",
    "bias_pre_mean",
    "bias_pre_abs_max",
    "bias_post_mean",
    "bias_post_abs_max",
    "bias_res_mean",
    "bias_res_abs_max",
)

# Read once per flush from the module parameters, not from the hot path.
_PARAM_METRICS = (
    "alpha_pre",
    "alpha_post",
    "alpha_res",
    "bias_pre_mean",
    "bias_pre_abs_max",
    "bias_post_mean",
    "bias_post_abs_max",
    "bias_res_mean",
    "bias_res_abs_max",
)


def _vector_metric_specs(n: int) -> tuple[tuple[str, str, int], ...]:
    """``(metric_name, elem_tag, size)`` for the per-element mapping series."""
    return (("h_res", "cell", n * n), ("h_pre", "idx", n), ("h_post", "idx", n))


# Extrema, not means: a worst case that a mean over all layers would bury. The
# `_max` / `_min` suffixes keep training_logs' classifier in agreement on the
# full key too.
_MAX_METRICS = frozenset(
    {
        "branch_residual_share_max",
        "h_res_logits_max",
        "h_res_logits_grad_max",
        "composite_amax_gain_fwd_max",
        "composite_amax_gain_bwd_max",
        "h_pre_logits_max",
        "h_post_logits_max",
        "bias_pre_abs_max",
        "bias_post_abs_max",
        "bias_res_abs_max",
    }
)
_MIN_METRICS = frozenset(
    {
        "h_res_logits_min",
        "h_res_logits_grad_min",
        "h_pre_logits_min",
        "h_post_logits_min",
    }
)

# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)
_COMPONENT_ORDER = {component: order for order, (component, _) in enumerate(_COMPONENTS)}


class MHCHealthMonitor(TorchProbe):
    """Monitor the h_pre / h_post / h_res mappings of mHC hyper-connection layers."""

    METRIC_PREFIX = "mhc_health"
    # Populated in __init__ with component-prefixed names, because
    # record_layer_metric classifies on the name it is handed.
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
    ):
        self.MAX_AGGREGATED = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MAX_METRICS}
        self.MIN_AGGREGATED = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MIN_METRICS}
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        # (module, attribute name, original bound method) triples, for remove_hooks().
        self._wrapped: list[tuple[nn.Module, str, object]] = []
        # (layer_idx, component) -> token-mean of the *operator* h_res^T [n, n],
        # for the composite product. Keyed rather than appended so the product is
        # built in layer order, not call order (see _record_composite).
        self._h_res_snapshot: dict[tuple[int, str], torch.Tensor] = {}
        # (layer_idx, component, module) for the step-level parameter series.
        self._param_targets: list[tuple[int, str, nn.Module]] = []
        # Set by the capture wrapper: whether this step's forward was monitored.
        # A plain bool assignment, so it costs nothing on the hot path, and it
        # keeps the parameter series on exactly the steps the rest are on
        # (`_should_monitor()` cannot answer that at flush time — `step()` has
        # already incremented `step_count` by then).
        self._captured_this_step = False
        # Gradient hooks see AMP-scaled activation gradients. The callback
        # finalizes these accumulators with this step's scale before flush.
        self._grad_metrics_finalized = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_layer_stack(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                return lm.decoder.layers
        return None

    def _find_hc_modules(self, model: nn.Module, layer_offset: int = 0):
        """Return ``[(global_idx, component, hc_module)]`` for every mHC hc module.

        Empty (auto-skip) when the mHC classes are unavailable or the model has no
        ``HyperConnectionTransformerLayer``. Uses ``isinstance`` against the real
        classes rather than duck-typing, so a plain ``TransformerLayer`` (or an
        ``IdentityOp`` placeholder) is never matched.
        """
        if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
            return []
        layers = self._find_layer_stack(model)
        if layers is None:
            return []

        entries = []
        num_local = len(layers)
        for local_idx, layer in enumerate(layers):
            if not isinstance(layer, HyperConnectionTransformerLayer):
                continue
            global_idx = self._resolve_layer_idx(layer, local_idx, num_local, layer_offset)
            for comp, attr in _COMPONENTS:
                mod = getattr(layer, attr, None)
                if isinstance(mod, HyperConnectionModule):
                    entries.append((global_idx, comp, mod))
        return entries

    # ------------------------------------------------------------------
    # Three-phase setup: prepare (declare) -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def _prepare_layers(self, model: nn.Module, layer_offset: int = 0):
        """Discover one chunk's hc modules and declare their metric schema.

        Buffers are still unallocated at this point, so every
        ``declare_layer_metric`` call remains legal.
        """
        entries = self._find_hc_modules(model, layer_offset=layer_offset)
        for global_idx, comp, mod in entries:
            for name in _METRIC_NAMES:
                self.declare_layer_metric(global_idx, f"{comp}_{name}")
            # `n` is static module metadata, so reading it at declare time costs
            # no hot-path sync and keeps the schema fixed (Rule 3).
            for name, tag, size in _vector_metric_specs(int(getattr(mod, "n", 0) or 0)):
                if size > 0:
                    self.declare_layer_vector(global_idx, f"{comp}_{name}", size, elem_tag=tag)
        return entries

    def _attach_hooks(self, targets):
        for layer_idx, comp, mod in targets:
            orig = mod.compute_mappings
            mod.compute_mappings = self._make_capture(orig, layer_idx, comp)
            self._wrapped.append((mod, "compute_mappings", orig))
            self._param_targets.append((layer_idx, comp, mod))
            orig_sinkhorn = getattr(mod, "_sinkhorn_op", None)
            if orig_sinkhorn is not None:
                mod._sinkhorn_op = self._make_sinkhorn_input_capture(orig_sinkhorn, layer_idx, comp)
                self._wrapped.append((mod, "_sinkhorn_op", orig_sinkhorn))
            orig_h = getattr(mod, "_compute_h", None)
            if orig_h is not None:
                mod._compute_h = self._make_gate_logits_capture(orig_h, layer_idx, comp, mod)
                self._wrapped.append((mod, "_compute_h", orig_h))
            orig_bda = getattr(mod, "fused_h_res_h_post_bda", None)
            if orig_bda is None:
                continue
            mod.fused_h_res_h_post_bda = self._make_bda_capture(orig_bda, layer_idx, comp)
            self._wrapped.append((mod, "fused_h_res_h_post_bda", orig_bda))
        logger.info(f"[MHCMonitor] Wrapped {len(self._wrapped)} hyper-connection methods.")

    def register_hooks(self, model: nn.Module):
        """Single-chunk convenience path. Prefer ``setup_mhc_monitor`` for VPP."""
        self._init_parallel_state()
        targets = self._prepare_layers(model)
        if not targets:
            logger.info("[MHCMonitor] No hyper-connection layers found; skipping.")
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def remove_hooks(self):
        # Restore the original bound methods so the monitor holds no module
        # references after teardown. ``delattr`` removes the instance attribute
        # and falls back to the class method; if that fails, we re-bind the
        # captured original.
        for mod, attr, orig in self._wrapped:
            if not hasattr(type(mod), attr):
                # ``_sinkhorn_op`` is assigned in ``__init__`` and exists only on
                # the instance, so ``delattr`` would remove it outright instead of
                # exposing a class-level fallback. Restore by assignment.
                setattr(mod, attr, orig)
                continue
            try:
                delattr(mod, attr)
            except AttributeError:
                setattr(mod, attr, orig)
        self._wrapped = []
        self._param_targets = []
        self._h_res_snapshot.clear()
        super().remove_hooks()

    # ------------------------------------------------------------------
    # Composite product (flush time, not hot path)
    # ------------------------------------------------------------------

    def _record_composite(self) -> None:
        """Record the gains of the layer-ordered cumulative product of h_res^T.

        The snapshots hold ``h_res^T``, i.e. the matrix ``apply_h_res`` actually
        applies to the stream vector (``mixed = h_res^T @ residual``), so the
        product ``A_k = h_res_k^T ... h_res_0^T`` is the real composite operator
        from the first layer up to layer ``k``. ``_fwd`` is its max-abs row sum
        (worst forward signal gain), ``_bwd`` its max-abs column sum.

        Scope is the layers this rank owns, i.e. one pipeline stage / VPP chunk
        set under PP. Unlike PaddleFleet there is no MTP filter: Megatron keeps
        MTP blocks outside ``decoder.layers``, so discovery never reaches them.
        """
        branches = sorted(
            ((idx, component, mat) for (idx, component), mat in self._h_res_snapshot.items()),
            key=lambda item: (item[0], _COMPONENT_ORDER[item[1]]),
        )

        prefix = None
        for layer_idx, component, mat in branches:
            prefix = mat if prefix is None else torch.matmul(mat, prefix)
            self.record_layer_metric(
                layer_idx,
                f"{component}_composite_amax_gain_fwd_max",
                amax_gain(prefix, dim=-1),
            )

        suffix = None
        for layer_idx, component, mat in reversed(branches):
            suffix = mat if suffix is None else torch.matmul(suffix, mat)
            self.record_layer_metric(
                layer_idx,
                f"{component}_composite_amax_gain_bwd_max",
                amax_gain(suffix, dim=-2),
            )

    def finalize_composite_microbatch(self) -> None:
        """Record and release one microbatch's detached composite snapshots."""
        if not self._h_res_snapshot:
            return
        try:
            with torch.no_grad():
                self._record_composite()
        except Exception as e:
            if self.verbose:
                logger.error(f"[MHCMonitor] Error composite: {e}")
        finally:
            self._h_res_snapshot.clear()

    def finalize_scaled_grad_metrics(self, scaler=None) -> None:
        """Remove AMP loss scaling from this step's logits-gradient extrema.

        A no-op under bf16 (Megatron runs it unscaled) and whenever the caller
        does not hand us the scaler. Under fp16 pass the optimizer's
        ``grad_scaler``; both the Megatron (``scale``) and torch
        (``_scale``) attribute names are accepted.
        """
        # This is the last hook before ``optimizer_step()``, so it is also where
        # the Eq. (7) parameters still hold the values this step's forward
        # actually used. Called outside the guard below and idempotent within a
        # step, so the ``_flush_buffers`` fallback stays harmless.
        self.finalize_param_metrics()
        if self._grad_metrics_finalized:
            return
        scale = None
        if scaler is not None:
            scale = getattr(scaler, "scale", None)
            if scale is None:
                scale = getattr(scaler, "_scale", None)
        if scale is not None:
            for key in self._max_keys | self._min_keys:
                if "_h_res_logits_grad_" in key and self._gpu_cnt[key] > 0:
                    acc = self._gpu_acc[key]
                    acc.div_(torch.as_tensor(scale, device=acc.device, dtype=acc.dtype))
        self._grad_metrics_finalized = True

    def finalize_param_metrics(self) -> None:
        """Record the step-level ``alpha`` / ``bias`` series from the parameters.

        Cold path: once per step, so nine tiny reductions per module instead of
        per microbatch. Skipped entirely when this step's forward was not
        monitored, which keeps these series sampled on the same steps as the
        activation ones under ``monitor_interval > 1`` — ``_should_monitor()``
        cannot answer that here, because ``step()`` increments ``step_count``
        before ``_flush_buffers`` runs.

        Read point is ``finalize_scaled_grad_metrics``, i.e. *before*
        ``optimizer.step()``, so the values line up with the forward that
        produced this step's activation metrics. A caller that only drives
        ``step()`` falls back to the flush path and therefore logs the
        post-update value — one update later than the rest of the step's series.

        The parameters are replicated across TP ranks (``HyperConnectionModule``
        uses a plain ``nn.Linear``), so no collective is needed — every rank
        holds the same values.
        """
        if not self._captured_this_step:
            return
        try:
            for layer_idx, component, mod in self._param_targets:
                try:
                    with torch.no_grad():
                        stats = mapping_param_stats(
                            mod.alpha_pre,
                            mod.alpha_post,
                            mod.alpha_res,
                            mod.bias,
                            int(mod.n),
                        )
                    for name, value in stats.items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)
                except Exception as e:
                    # Per module, so one module without the Eq. (7) parameters
                    # cannot silence the series for every other layer.
                    if self.verbose:
                        logger.error(f"[MHCMonitor] Error mapping params {layer_idx}/{component}: {e}")
        finally:
            self._captured_this_step = False

    def _flush_buffers(self) -> None:
        # Direct users and non-AMP trainers never call finalize_* themselves.
        self.finalize_scaled_grad_metrics()
        self.finalize_composite_microbatch()
        super()._flush_buffers()
        self._grad_metrics_finalized = False

    # ------------------------------------------------------------------
    # Capture wrappers (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd graph;
        we ``.detach()`` them and do all metric math under ``no_grad`` so no
        stored tensor pins the graph through backward. The only cross-call state
        is one token-meaned ``[n, n]`` snapshot per (layer, component), released
        by ``finalize_composite_microbatch``.
        """

        def wrapped(x):
            out = orig(x)  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                h_pre, h_post, h_res = out
                self._captured_this_step = True
                with torch.no_grad():
                    h_pre = h_pre.detach()
                    h_post = h_post.detach()
                    h_res = h_res.detach()

                    pre_mean, pre_std = gate_stats(h_pre)
                    post_mean, post_std = gate_stats(h_post)
                    self.record_layer_metric(layer_idx, f"{component}_h_pre_mean", pre_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_pre_std", pre_std)
                    self.record_layer_metric(layer_idx, f"{component}_h_post_mean", post_mean)
                    self.record_layer_metric(layer_idx, f"{component}_h_post_std", post_std)
                    for name, value in h_post_structure_stats(h_post).items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)

                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_fwd", amax_gain(h_res, dim=-2))
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_bwd", amax_gain(h_res, dim=-1))

                    # Token-mean of every mapping element. One reduce feeds both
                    # the per-cell series and the composite product's snapshot.
                    n = int(h_res.shape[-1])
                    res_mean = h_res.float().reshape(-1, n, n).mean(dim=0)
                    self.record_layer_vector(layer_idx, f"{component}_h_res", res_mean.reshape(-1))
                    self.record_layer_vector(layer_idx, f"{component}_h_pre", h_pre.float().reshape(-1, n).mean(dim=0))
                    self.record_layer_vector(
                        layer_idx, f"{component}_h_post", h_post.float().reshape(-1, n).mean(dim=0)
                    )

                    # Token-mean snapshot of the *operator* for the composite product.
                    self._h_res_snapshot[(layer_idx, component)] = res_mean.transpose(0, 1)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Error layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_sinkhorn_input_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``_sinkhorn_op`` to record the saturation sentinel from its input.

        The first argument is the pre-projection mixing logits, i.e. the same
        tensor ``_compute_h`` returns on the unfused path (modulo the ``view`` to
        [s, b, n, n], which leaves element-wise extrema and their gradients
        unchanged) and the kernel output on the fused path.
        """

        def wrapped(h_res_logits, *args, **kwargs):
            if self._should_monitor():
                try:
                    if h_res_logits.requires_grad:

                        def record_grad_extrema(grad):
                            try:
                                grad_fp32 = grad.detach().float()
                                self.record_layer_metric(
                                    layer_idx,
                                    f"{component}_h_res_logits_grad_min",
                                    grad_fp32.min(),
                                )
                                self.record_layer_metric(
                                    layer_idx,
                                    f"{component}_h_res_logits_grad_max",
                                    grad_fp32.max(),
                                )
                                self._grad_metrics_finalized = False
                            except Exception as e:
                                if self.verbose:
                                    logger.error(
                                        f"[MHCMonitor] Error logits gradient layer {layer_idx}/{component}: {e}"
                                    )
                            # Returning None leaves the autograd gradient untouched.
                            return None

                        h_res_logits.register_hook(record_grad_extrema)
                    with torch.no_grad():
                        for name, value in h_res_logits_extrema(h_res_logits).items():
                            self.record_layer_metric(layer_idx, f"{component}_{name}", value)
                except Exception as e:
                    if self.verbose:
                        logger.error(f"[MHCMonitor] Error logits layer {layer_idx}/{component}: {e}")
            # The real doubly-stochastic matrix the model consumes — unchanged.
            return orig(h_res_logits, *args, **kwargs)

        return wrapped

    def _make_gate_logits_capture(self, orig, layer_idx: int, component: str, mod):
        """Wrap ``_compute_h`` for the pre-sigmoid ``h_pre`` / ``h_post`` logits.

        ``_compute_h`` does not return them, so they are rebuilt from its own
        ``(proj, r)`` arguments plus ``mod``'s ``alpha`` / ``bias`` — see
        ``gate_logits_extrema``. The residual-mixing logits are *not* read here;
        ``_sinkhorn_op`` covers those on both the fused and the unfused path.

        Under ``use_fused_mhc`` this wrapper never runs (``compute_mappings``
        skips ``_compute_h``) and ``fused_proj_rms_compute_h`` does not expose
        ``proj``, so these four accumulators stay empty and are not emitted
        rather than being filled from a different quantity.
        """

        def wrapped(proj, r):
            out = orig(proj, r)  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                if proj is not None and r is not None:
                    with torch.no_grad():
                        gate_logits = gate_logits_extrema(
                            proj,
                            r,
                            mod.alpha_pre,
                            mod.alpha_post,
                            mod.bias,
                            int(mod.n),
                        )
                    for name, value in gate_logits.items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Error gate logits layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_bda_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``fused_h_res_h_post_bda`` to record the branch/residual ratio.

        The only place both mHC update terms are available: the call receives
        ``h_res`` + ``original_residual`` and ``layer_output_with_bias``. Metrics
        come from the **arguments**, so the fused fast path, the checkpointed
        path and the sequential dropout path are covered identically. Arguments
        are read by keyword with a positional fallback — Megatron's
        ``TransformerLayer`` passes them positionally today, but relying on that
        alone would silently stop recording if it switched.
        """

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            if not self._should_monitor():
                return out
            try:

                def pick(name, pos):
                    if name in kwargs:
                        return kwargs[name]
                    return args[pos] if len(args) > pos else None

                h_res = pick("h_res", 0)
                original_residual = pick("original_residual", 1)
                h_post = pick("h_post", 2)
                layer_output_with_bias = pick("layer_output_with_bias", 3)
                if h_res is None or original_residual is None or h_post is None:
                    return out
                if isinstance(layer_output_with_bias, tuple):
                    layer_output, bias = layer_output_with_bias
                else:
                    layer_output, bias = layer_output_with_bias, None
                if layer_output is None:
                    return out

                with torch.no_grad():
                    stats = branch_residual_share(h_res, original_residual, h_post, layer_output, bias)
                    for name, value in stats.items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MHCMonitor] Error bda layer {layer_idx}/{component}: {e}")
            return out

        return wrapped


def setup_mhc_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
):
    """Enable the mHC health monitor. No-op on any non-mHC model.

    Multi-chunk (VPP / interleaved 1F1B) safe: declares the schema across all
    chunks before ``allocate_buffers`` locks it, then attaches. Layer ids are
    offset per chunk, so the composite product keyed on them stays ordered and
    a later chunk's layers never contaminate an earlier chunk's product.
    """
    # No-op guarantee #1: mHC classes unavailable -> touch nothing.
    if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
        logger.info("[MHCMonitor] Hyper-connection classes unavailable; skipping.")
        return model

    monitor = MHCHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()

    chunk_targets = []
    layer_offset = 0
    for m in models:
        chunk_targets.append(monitor._prepare_layers(m, layer_offset=layer_offset))
        stack = monitor._find_layer_stack(m)
        layer_offset += len(stack) if stack is not None else 0

    if any(chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        for targets in chunk_targets:
            monitor._attach_hooks(targets)
        if monitor_dict is not None:
            monitor_dict["mhc_health"] = monitor
    else:
        # No-op guarantee #2: no HyperConnectionTransformerLayer found.
        logger.info("[MHCMonitor] No hyper-connection layers found; skipping.")

    logger.info(f"[MHCMonitor] Setup complete. Monitoring {len(monitor._wrapped)} hc methods.")
    return model
