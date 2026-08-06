"""mHC Health Monitor for PaddleFleet.

Paddle port of ``backends/megatron/mhc_monitor.py``. Monitors the three
per-token mappings of every ``HyperConnectionModule`` in an mHC
(Manifold-Constrained Hyper-Connections) model:

- ``h_pre`` / ``h_post`` — the aggregation / expansion gates: mean and std, plus
  the stream concentration and token sensitivity of ``h_post``.
- ``h_res``              — the doubly-stochastic residual-mixing matrix: the
  paper's ``amax_gain`` (max-abs row sum), kept as a NaN / Sinkhorn-divergence
  sentinel rather than as a diagnostic (see the note below).
- the **two update terms** of ``x_{l+1} = H_resᵀ x_l + H_postᵀ F(H_pre x_l)``:
  their relative magnitude, i.e. how much this layer still writes into the
  residual streams.

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 9
series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_h_post_stream_concentration  {attn,mlp}_h_post_token_std
    {attn,mlp}_branch_residual_share  {attn,mlp}_branch_residual_share_max
    {attn,mlp}_amax_gain_fwd

``h_pre`` is not part of ``HyperConnectionModule.forward``'s return, so we
cannot use a forward hook. Instead we wrap the module's ``compute_mappings``
bound method to capture its real ``(h_pre, h_post, h_res)`` — no recompute.
The branch/residual ratio needs the sublayer output too, which only exists where
the two terms are combined, so ``fused_h_res_h_post_bda`` is wrapped as well.
Everything is detached and computed under ``no_grad`` (see the VRAM-safety
notes on the hook).

Retired metrics (do not re-add without a new argument):

- ``composite_amax_gain_fwd`` / ``_bwd`` — the cumulative product of ``h_res``
  across layers. A product of doubly-stochastic matrices is itself doubly
  stochastic, so the composite gain is **1 by construction**; measured range was
  0.99999 ~ 0.999975, i.e. pure Sinkhorn ``eps`` accumulation. Worse, the value
  is order-dependent: it was built in execution order, which under recompute is
  the reverse-layer backward replay (measured: per-layer factor counts came out
  ``1, 25.6, 23.6, ..., 2.9`` instead of ``1, 3, 5, ..., 25.8``). Making it
  order-proof needs the per-layer ``h_res`` resident for the whole step
  (~45MB at 43 layers), which is not worth paying for a constant.
- ``amax_gain_bwd`` (max-abs **column** sum) — Sinkhorn's last step is a column
  normalization, so columns are pinned to 1 by construction. Measured bit-equal
  to ``amax_gain_fwd`` on every layer of four runs.

To actually watch for residual-stream amplification, monitor Sinkhorn
convergence residual (row sums vs 1) or the spectrum of ``h_res`` — neither is
implemented yet.

The monitor is a hard no-op unless the model actually uses the mHC layer: if
the mHC classes cannot be imported, or no ``HyperConnectionTransformerLayer``
is found, ``setup_mhc_monitor`` attaches nothing and registers no metrics.

Hot-path discipline (no D2H sync, no hook-time collectives, schema fixed at
registration): see ``.claude/skills/monitor-hook-perf-rules``.
"""

import logging

import paddle
import paddle.nn as nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .mhc_metrics import amax_gain, branch_residual_share, gate_stats, h_post_structure_stats

logger = logging.getLogger(__name__)


# mHC classes are optional: this monitor must be a no-op when they are absent
# (non-mHC model, or a PaddleFleet build without hyper-connections). Bind to
# None on import failure and gate every code path on that.
try:
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule
    from paddlefleet.transformer.transformer_layer import HyperConnectionTransformerLayer
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
)

# Only the worst-token branch ratio keeps extremum semantics across
# microbatches / layers / ranks; everything else is a mean. The name ends in
# `_max`, so training_logs' suffix classifier agrees on the full key too.
_MAX_METRICS = frozenset({"branch_residual_share_max"})

# amax_gain_fwd is 1 by construction (Sinkhorn), so it is a gatekeeper for that
# invariant rather than a per-layer diagnostic: one global curve per component
# carries the same signal as 13 layers x 2 components of flat lines. Any layer
# drifting still moves the global mean.
_GLOBAL_ONLY_METRICS = frozenset({"amax_gain_fwd"})

# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)


def _iter_chunks(model):
    """Yield ``(chunk_id, chunk)`` pairs for VPP-aware discovery.

    PaddleFleet exposes VPP model chunks via ``_model_chunks`` on the pipeline
    layer; each chunk owns its own ``run_function`` and executes in its own
    forward pass, so discovery has to walk all of them.

    Falls back to a single chunk ``(0, model)`` when no VPP chunking is present.
    """
    candidates = [model]
    if hasattr(model, "_layers"):
        candidates.append(model._layers)
    if hasattr(model, "module"):
        candidates.append(model.module)
    for cand in candidates:
        if cand is None:
            continue
        chunks = getattr(cand, "_model_chunks", None)
        if chunks:
            yield from enumerate(chunks)
            return
    yield 0, model


class PaddleMHCHealthMonitor(PaddleProbe):
    """Monitor the h_pre / h_post / h_res mappings of mHC hyper-connection layers."""

    METRIC_PREFIX = "mhc_health"
    # All metrics but one are means over tokens/batch (and over microbatches /
    # ranks at flush); `branch_residual_share_max` is the single extremum series.
    # Component-prefixed names are registered in __init__, because
    # record_layer_metric classifies on the name it is handed.
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
    ):
        self.MAX_AGGREGATED = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MAX_METRICS}
        self.GLOBAL_ONLY = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _GLOBAL_ONLY_METRICS}
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        # (module, attribute name, original bound method) triples, for remove_hooks().
        self._wrapped: list[tuple[nn.Layer, str, object]] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_hc_modules(self, chunk):
        """Return ``[(global_idx, comp, hc_module)]`` for one chunk.

        Empty (auto-skip) when the mHC classes are unavailable or the chunk has
        no ``HyperConnectionTransformerLayer``. Uses ``isinstance`` against the
        real classes rather than duck-typing, so a plain ``TransformerLayer``
        (or an ``IdentityOp`` placeholder) is never matched.
        """
        if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
            return []
        layers = get_decoder_layers(chunk)
        if not layers:
            return []

        def matches(layer):
            return isinstance(layer, HyperConnectionTransformerLayer)

        monitor_layers = iter_monitor_layers(layers, matches, pp_rank=self.pp_rank)
        entries: list[tuple[int, str, nn.Layer]] = []
        # is_mtp so we can mark them for `_mtp` suffix in metric keys.
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)
        for item in monitor_layers:
            for comp, attr in _COMPONENTS:
                mod = getattr(item.layer, attr, None)
                if not isinstance(mod, HyperConnectionModule):
                    continue
                entries.append((item.idx, comp, mod))
        return entries

    # ------------------------------------------------------------------
    # Setup: prepare (declare) -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self):
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

    def _prepare(self, model):
        """Discover all hc modules across chunks and declare metric schema.

        Returns a list of chunks' entries; buffers are still unallocated so
        every ``declare_layer_metric`` call remains legal.
        """
        self._init_parallel_state()
        all_targets: list[list[tuple[int, str, nn.Layer]]] = []
        for _chunk_id, chunk in _iter_chunks(model):
            entries = self._find_hc_modules(chunk)
            if not entries:
                continue
            for global_idx, comp, _ in entries:
                for name in _METRIC_NAMES:
                    self.declare_layer_metric(global_idx, f"{comp}_{name}")
            all_targets.append(entries)
        return all_targets

    def _attach(self, all_targets):
        for entries in all_targets:
            for layer_idx, comp, mod in entries:
                orig = mod.compute_mappings
                mod.compute_mappings = self._make_capture(orig, layer_idx, comp)
                self._wrapped.append((mod, "compute_mappings", orig))
                # The branch/residual ratio needs the sublayer output, which only
                # exists at the point where the two update terms are combined.
                orig_bda = getattr(mod, "fused_h_res_h_post_bda", None)
                if orig_bda is None:
                    continue
                mod.fused_h_res_h_post_bda = self._make_bda_capture(orig_bda, layer_idx, comp)
                self._wrapped.append((mod, "fused_h_res_h_post_bda", orig_bda))
        logger.info(f"[PaddleMHCMonitor] Wrapped {len(self._wrapped)} hyper-connection methods.")

    def register_hooks(self, model):
        """Discover, declare, allocate, and attach — the whole three-phase setup."""
        all_targets = self._prepare(model)
        if not any(all_targets):
            logger.info("[PaddleMHCMonitor] No hyper-connection layers found; skipping.")
            return
        self.allocate_buffers()
        self._attach(all_targets)

    def remove_hooks(self):
        # Restore the original bound methods so the monitor holds no module
        # references after teardown. ``del mod.<attr>`` removes the instance
        # attribute and falls back to the class method; if that fails (e.g.
        # slots), we re-bind the captured original.
        for mod, attr, orig in self._wrapped:
            try:
                delattr(mod, attr)
            except AttributeError:
                setattr(mod, attr, orig)
        self._wrapped = []
        super().remove_hooks()

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd
        graph; we ``.detach()`` them and do all metric math under ``no_grad`` so
        no stored tensor pins the graph through backward. The wrapper keeps no
        state across calls — a deliberate constraint, since a monitored module
        may be entered more than once per step (recompute replay, and mHC's own
        fp32 / bf16 paths), and any cross-call accumulator would then depend on
        execution order.
        """

        def wrapped(x):
            out = orig(x)  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                h_pre, h_post, h_res = out
                with paddle.no_grad():
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

                    # Row sums of a Sinkhorn-projected h_res: 1 by construction,
                    # so this is a NaN / divergence sentinel, not a diagnostic.
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_fwd", amax_gain(h_res, axis=-1))
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_bda_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``fused_h_res_h_post_bda`` to record the branch/residual ratio.

        This is the only place where both terms of the mHC update are available:
        the call receives ``h_res`` + ``original_residual`` (the residual term's
        inputs) and ``layer_output_with_bias`` (the branch term's input). Metrics
        are derived from the **arguments**, so both the fused fast path and the
        sequential dropout path are covered identically, and the call itself is
        forwarded untouched.

        Arguments are read by keyword with a positional fallback: PaddleFleet's
        transformer layer calls this with keywords today, but relying on that
        alone would silently stop recording if it ever switched.
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

                with paddle.no_grad():
                    stats = branch_residual_share(h_res, original_residual, h_post, layer_output, bias)
                    for name, value in stats.items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error bda layer {layer_idx}/{component}: {e}")
            return out

        return wrapped


def setup_mhc_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    monitor_dict: dict | None = None,
):
    """Enable the mHC health monitor. No-op on any non-mHC model.

    Multi-chunk (VPP / interleaved 1F1B) safe: declares the schema across all
    chunks before ``allocate_buffers`` locks it, then attaches.
    """
    # No-op guarantee #1: mHC classes unavailable -> touch nothing.
    if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
        logger.info("[PaddleMHCMonitor] Hyper-connection classes unavailable; skipping.")
        return model

    monitor = PaddleMHCHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    if monitor._wrapped and monitor_dict is not None:
        monitor_dict["mhc_health"] = monitor
    logger.info(f"[PaddleMHCMonitor] Setup complete. Monitoring {len(monitor._wrapped)} hc modules.")
    return model
