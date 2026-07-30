"""mHC Health Monitor for PaddleFleet.

Paddle port of ``backends/megatron/mhc_monitor.py``. Monitors the three
per-token mappings of every ``HyperConnectionModule`` in an mHC
(Manifold-Constrained Hyper-Connections) model:

- ``h_pre`` / ``h_post`` — the aggregation / expansion gates: mean and std.
- ``h_res``              — the doubly-stochastic residual-mixing matrix: the
  paper's ``amax_gain`` forward (max-abs row sum) and backward (max-abs column
  sum), computed both on this layer's ``h_res`` and on the running **composite
  mapping** (cumulative product of ``h_res`` across the layers local to this
  pipeline stage / VPP chunk).

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 8
mean-aggregated series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_amax_gain_fwd            {attn,mlp}_amax_gain_bwd
    {attn,mlp}_composite_amax_gain_fwd  {attn,mlp}_composite_amax_gain_bwd

``h_pre`` is not part of ``HyperConnectionModule.forward``'s return, so we
cannot use a forward hook. Instead we wrap the module's ``compute_mappings``
bound method to capture its real ``(h_pre, h_post, h_res)`` — no recompute.
Everything is detached and computed under ``no_grad`` (see the VRAM-safety
notes on the hook).

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
from .mhc_metrics import amax_gain, gate_stats

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
    "amax_gain_fwd",
    "amax_gain_bwd",
    "composite_amax_gain_fwd",
    "composite_amax_gain_bwd",
)

# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)


def _iter_chunks(model):
    """Yield ``(chunk_id, chunk)`` pairs for VPP-aware discovery.

    PaddleFleet exposes VPP model chunks via ``_model_chunks`` on the pipeline
    layer. When present, each chunk owns its own ``run_function`` and executes
    in its own forward pass — so the running composite mapping must reset at
    each chunk's first hc module, and never carry over.

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
    # Every metric is a mean over tokens/batch (and over microbatches/ranks at
    # flush). No max/min series, so both classification sets stay empty. The
    # metric names end in _mean/_std/_fwd/_bwd — never a `_max`/`_min` boundary —
    # so training_logs' suffix classifier also keeps them as "mean".
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        # chunk_id -> running composite mapping [s*b, n, n], detached (no graph).
        self._composite: dict[int, paddle.Tensor] = {}
        # (module, original_compute_mappings) pairs, for remove_hooks() restoration.
        self._wrapped: list[tuple[nn.Layer, object]] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_hc_modules(self, chunk, chunk_id: int):
        """Return ``[(global_idx, comp, hc_module, chunk_id, is_root)]`` for one chunk.

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
        entries: list[tuple[int, str, nn.Layer, int, bool]] = []
        # is_mtp so we can mark them for `_mtp` suffix in metric keys.
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)
        first_attn_seen = False
        for item in monitor_layers:
            for comp, attr in _COMPONENTS:
                mod = getattr(item.layer, attr, None)
                if not isinstance(mod, HyperConnectionModule):
                    continue
                # Chunk root = the attn hc of the first monitored layer in this
                # chunk's execution order; it resets the composite each forward.
                is_root = comp == "attn" and not first_attn_seen
                if is_root:
                    first_attn_seen = True
                entries.append((item.idx, comp, mod, chunk_id, is_root))
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
        all_targets: list[list[tuple[int, str, nn.Layer, int, bool]]] = []
        for chunk_id, chunk in _iter_chunks(model):
            entries = self._find_hc_modules(chunk, chunk_id=chunk_id)
            if not entries:
                continue
            for global_idx, comp, _, _, _ in entries:
                for name in _METRIC_NAMES:
                    self.declare_layer_metric(global_idx, f"{comp}_{name}")
            all_targets.append(entries)
        return all_targets

    def _attach(self, all_targets):
        for entries in all_targets:
            for layer_idx, comp, mod, chunk_id, is_root in entries:
                orig = mod.compute_mappings
                mod.compute_mappings = self._make_capture(orig, layer_idx, comp, chunk_id, is_root)
                self._wrapped.append((mod, orig))
        logger.info(f"[PaddleMHCMonitor] Wrapped {len(self._wrapped)} hyper-connection modules.")

    def register_hooks(self, model):
        """Discover, declare, allocate, and attach — the whole three-phase setup."""
        all_targets = self._prepare(model)
        if not any(all_targets):
            logger.info("[PaddleMHCMonitor] No hyper-connection layers found; skipping.")
            return
        self.allocate_buffers()
        self._attach(all_targets)
        self._set_compatible_sources(len(self._wrapped))

    def remove_hooks(self):
        # Restore the original bound methods and drop all cross-call state so
        # the monitor holds no module references or composite tensors after
        # teardown. ``del mod.compute_mappings`` removes the instance attribute
        # and falls back to the class method; if that fails (e.g. slots), we
        # re-bind the captured original.
        for mod, orig in self._wrapped:
            try:
                del mod.compute_mappings
            except AttributeError:
                mod.compute_mappings = orig
        self._wrapped = []
        self._composite.clear()
        super().remove_hooks()

    def step(self):
        super().step()
        # Release the running composite between train steps so a [s*b, n, n]
        # buffer never sits idle in VRAM; the root reseeds it next forward, so
        # clearing is correctness-neutral.
        self._composite.clear()

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str, chunk_id: int, is_root: bool):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd
        graph; we ``.detach()`` them and do all metric/composite math under
        ``no_grad`` so no stored tensor pins the graph through backward. The
        composite slot holds one detached ``[s*b, n, n]`` tensor per chunk
        (cloned on seed so it never aliases the model's ``h_res`` storage);
        ``step()`` clears it.
        """

        def wrapped(x):
            out = orig(x)  # the real mappings the model consumes — returned unchanged
            # Gate the whole capture: metrics are only recorded on monitored
            # steps, and the composite is reset at the chunk root on each such
            # step, so gating here keeps the composite self-consistent (root
            # reset -> ordered bmm builds within one monitored forward).
            # _should_monitor() also requires grad enabled, selecting the
            # training (not a no-grad / recompute) forward.
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

                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_fwd", amax_gain(h_res, axis=-1))
                    self.record_layer_metric(layer_idx, f"{component}_amax_gain_bwd", amax_gain(h_res, axis=-2))

                    # Composite mapping M_k = h_res_k @ M_{k-1} (per token).
                    # h_res arrives as [..., n, n]; flatten leading dims to a
                    # single batch axis for bmm.
                    shape = h_res.shape
                    n = shape[-1]
                    hb = h_res.reshape([-1, n, n])
                    prev = self._composite.get(chunk_id)
                    # Reset at the chunk root each forward; also self-heal on
                    # first fire / shape drift (variable s*b across
                    # microbatches). identity @ h_res == h_res, so seed with
                    # h_res; clone() so the slot owns a fresh buffer and never
                    # pins the model's h_res storage.
                    if is_root or prev is None or prev.shape != hb.shape:  # noqa: SIM108
                        M = hb.clone()
                    else:
                        M = paddle.bmm(hb, prev)  # fresh tensor, no graph
                    self._composite[chunk_id] = M

                    self.record_layer_metric(layer_idx, f"{component}_composite_amax_gain_fwd", amax_gain(M, axis=-1))
                    self.record_layer_metric(layer_idx, f"{component}_composite_amax_gain_bwd", amax_gain(M, axis=-2))
                    self._record_observation(layer_idx)
            except Exception as e:
                self._record_error()
                if self.verbose:
                    logger.error(f"[PaddleMHCMonitor] Error layer {layer_idx}/{component}: {e}")
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
    chunks before ``allocate_buffers`` locks it, then attaches. Each model
    chunk gets its own composite slot so a later chunk's layers never
    contaminate an earlier chunk's running product.
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
