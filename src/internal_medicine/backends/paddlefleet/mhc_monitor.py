"""mHC Health Monitor for PaddleFleet.

Paddle port of ``backends/megatron/mhc_monitor.py``. Monitors the three per-token
mappings of every ``HyperConnectionModule`` in an mHC (Manifold-Constrained
Hyper-Connections) model, plus the relative magnitude of the two update terms of
``x_{l+1} = H_resᵀ x_l + H_postᵀ F(H_pre x_l)`` (i.e. how much this layer still
writes into the residual streams).

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 14
series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_h_post_stream_concentration  {attn,mlp}_h_post_token_std
    {attn,mlp}_branch_residual_share  {attn,mlp}_branch_residual_share_max
    {attn,mlp}_amax_gain_fwd  {attn,mlp}_amax_gain_bwd
    {attn,mlp}_h_res_softmax_min  {attn,mlp}_h_res_softmax_max
    {attn,mlp}_composite_amax_gain_fwd_max  {attn,mlp}_composite_amax_gain_bwd_max

Forward hooks are not enough, so three bound methods are wrapped instead (see
``mhc_metrics`` for what each metric means):

- ``compute_mappings`` — ``h_pre`` is not in ``forward``'s return value.
- ``_compute_h``       — the mixing logits only exist before the Sinkhorn
  projection, and wrapping here also sidesteps the ``use_fused_mhc`` fork.
- ``fused_h_res_h_post_bda`` — the only point where both update terms are
  visible.

Everything is detached and computed under ``no_grad`` (see the VRAM-safety notes
on the hook).

Which axis carries signal: both the single-layer and the composite gains are
computed on the operator that really acts on the stream vector, ``h_res^T``
(``apply_h_res`` and the fused cuTile kernel both contract over ``h_res``'s first
index). ``_fwd`` is therefore a column sum of ``h_res`` and ``_bwd`` a row sum.
Our ``_sinkhorn_normalize`` ends on a **column** normalization, so ``_fwd`` is
pinned to 1 by construction and ``_bwd`` carries the truncated-iteration
residual. The paper's Eq. (9) ends on a row normalization, so its Fig. 7 plots
the mirror image — the curve that moves there is its forward gain, ours is
``_bwd``. Both directions are emitted anyway so the pinned side acts as a guard.

``composite_amax_gain_*`` and ``amax_gain_bwd`` were all retired once and are
back. The composite objections were the cost of keeping a per-token ``h_res``
resident for the whole step and order-dependence under recompute: averaging over
tokens first shrinks the snapshot to one ``n x n`` matrix per layer, and keying
snapshots by ``layer_idx`` instead of accumulating in call order removes the
order-dependence (see ``_record_composite`` for the remaining scope caveat).
``amax_gain_bwd`` was dropped for reading equal to ``amax_gain_fwd``, which is
expected given the pinned axis above; it is kept now as the explicit invariant
guard.

Residual-stream amplification itself is still unmonitored; the candidates are the
Sinkhorn convergence residual (row sums vs 1) and the spectrum of ``h_res``.

The monitor is a hard no-op unless the model actually uses the mHC layer: if the
mHC classes cannot be imported, or no ``HyperConnectionTransformerLayer`` is
found, ``setup_mhc_monitor`` attaches nothing and registers no metrics.

Hot-path discipline (no D2H sync, no hook-time collectives, schema fixed at
registration): see ``.claude/skills/monitor-hook-perf-rules``.
"""

import logging

import paddle
import paddle.nn as nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .mhc_metrics import (
    amax_gain,
    branch_residual_share,
    gate_stats,
    h_post_structure_stats,
    h_res_softmax_extrema,
)

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
    "amax_gain_bwd",
    "h_res_softmax_min",
    "h_res_softmax_max",
    "composite_amax_gain_fwd_max",
    "composite_amax_gain_bwd_max",
)

# Extrema, not means: a worst case that a mean over 43 layers would bury. The
# `_max` / `_min` suffixes keep training_logs' classifier in agreement on the
# full key too.
_MAX_METRICS = frozenset(
    {
        "branch_residual_share_max",
        "h_res_softmax_max",
        "composite_amax_gain_fwd_max",
        "composite_amax_gain_bwd_max",
    }
)
_MIN_METRICS = frozenset({"h_res_softmax_min"})

# Nothing is global-only: the single-layer amax gains used to be, on the grounds
# that they are 1 by construction, but the per-layer profile is wanted alongside
# the composite one.
_GLOBAL_ONLY_METRICS: frozenset[str] = frozenset()

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
    ):
        self.MAX_AGGREGATED = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MAX_METRICS}
        self.MIN_AGGREGATED = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MIN_METRICS}
        self.GLOBAL_ONLY = {f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _GLOBAL_ONLY_METRICS}
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        # (module, attribute name, original bound method) triples, for remove_hooks().
        self._wrapped: list[tuple[nn.Layer, str, object]] = []
        # (layer_idx, component) -> token-mean of the *operator* h_res^T [n, n],
        # for the composite product. Keyed rather than appended so the product is
        # built in layer order, not call order (see _record_composite).
        self._h_res_snapshot: dict[tuple[int, str], paddle.Tensor] = {}
        # MTP layers are off the main trunk and must not enter the product.
        self._mtp_layer_ids: set[int] = set()

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
            self._mtp_layer_ids.update(mtp_layer_ids)
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
                # Wrapping `_compute_h` also sidesteps the `use_fused_mhc` fork:
                # the fused path swaps `_sinkhorn_op` for a cuTile kernel whose
                # intermediates are unreachable, but `_compute_h` is plain paddle
                # either way.
                orig_h = getattr(mod, "_compute_h", None)
                if orig_h is not None:
                    mod._compute_h = self._make_logits_capture(orig_h, layer_idx, comp)
                    self._wrapped.append((mod, "_compute_h", orig_h))
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

        Built from snapshots sorted by ``layer_idx``, so the result does not
        depend on the order the hooks fired in — under recompute that order is
        the reverse-layer backward replay, which is what retired the previous
        composite metric. MTP layers are skipped: they are off the main trunk, so
        multiplying them into the chain has no physical meaning.

        Scope is the layers this rank owns: the whole model under sharding (every
        rank holds all layers), but one stage under real pipeline parallelism,
        which we can neither detect nor correct while
        ``get_pipeline_model_parallel_{rank,world_size}`` are stubs in PaddleFleet
        (they return 0 / 1 unconditionally).
        """
        for comp, _ in _COMPONENTS:
            chain = sorted(
                (
                    (idx, mat)
                    for (idx, c), mat in self._h_res_snapshot.items()
                    if c == comp and idx not in self._mtp_layer_ids
                ),
                key=lambda item: item[0],
            )
            composite = None
            for layer_idx, mat in chain:
                composite = mat if composite is None else paddle.matmul(mat, composite)
                self.record_layer_metric(
                    layer_idx, f"{comp}_composite_amax_gain_fwd_max", amax_gain(composite, axis=-1)
                )
                self.record_layer_metric(
                    layer_idx, f"{comp}_composite_amax_gain_bwd_max", amax_gain(composite, axis=-2)
                )

    def _flush_buffers(self) -> None:
        if self._h_res_snapshot:
            try:
                with paddle.no_grad():
                    self._record_composite()
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error composite: {e}")
            self._h_res_snapshot.clear()
        super()._flush_buffers()

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd graph,
        so they are detached and all metric math runs under ``no_grad`` — nothing
        stored may pin the graph through backward. The wrapper also keeps no state
        across calls, since a monitored module can be entered more than once per
        step (recompute replay, and mHC's own fp32 / bf16 paths) and any
        cross-call accumulator would depend on execution order.
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

                    # Gains of the *operator*, not of `h_res` as written: streams
                    # mix as `out_i = Σ_j h_res[j, i] · x_j`, so stream i's
                    # forward gain is a **column** sum of `h_res` and its
                    # backward gain a row sum. Columns are pinned to 1 by
                    # construction (the Sinkhorn iteration ends on a column
                    # normalization), so `_fwd` is a flat invariant guard while
                    # `_bwd` carries the convergence residual.
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_fwd", amax_gain(h_res, axis=-2))
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_bwd", amax_gain(h_res, axis=-1))

                    # Token-mean snapshot of the *operator* for the composite
                    # product. `apply_h_res` mixes streams as
                    # `mixed = h_res^T @ residual`, so the matrix that actually
                    # acts on the stream vector is the transpose — composing
                    # `h_res` itself would give neither the forward nor the
                    # backward chain. Paper Fig 7b averages over tokens too, so
                    # this holds one n x n matrix per layer/component. Last write
                    # wins if the module is entered more than once per step.
                    n = int(h_res.shape[-1])
                    self._h_res_snapshot[(layer_idx, component)] = (
                        h_res.astype("float32").reshape([-1, n, n]).mean(axis=0).transpose([1, 0])
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_logits_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``_compute_h`` to record the saturation sentinel.

        ``_compute_h`` returns ``(h_pre, h_post, h_res_logits)``; ``h_res_logits``
        is the raw ``[..., n*n]`` mixing logits, which ``compute_mappings``
        consumes immediately (``h_res = self._sinkhorn_op(...)``), so a wrapper
        one level up cannot see it. ``n`` is read from ``h_pre``'s last axis
        rather than config, in case ``num_residual_streams`` ever varies by layer.
        Same VRAM discipline as ``_make_capture``.
        """

        def wrapped(proj, r):
            out = orig(proj, r)  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                h_pre, _h_post, h_res_logits = out
                n = int(h_pre.shape[-1])
                with paddle.no_grad():
                    for name, value in h_res_softmax_extrema(h_res_logits, n).items():
                        self.record_layer_metric(layer_idx, f"{component}_{name}", value)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error logits layer {layer_idx}/{component}: {e}")
            return out

        return wrapped

    def _make_bda_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``fused_h_res_h_post_bda`` to record the branch/residual ratio.

        The only place both mHC update terms are available: the call receives
        ``h_res`` + ``original_residual`` and ``layer_output_with_bias``. Metrics
        come from the **arguments**, so the fused fast path and the sequential
        dropout path are covered identically. Arguments are read by keyword with a
        positional fallback — PaddleFleet passes keywords today, but relying on
        that alone would silently stop recording if it switched.
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
