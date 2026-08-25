"""Megatron/Torch-specific Probe base class.

Adds the GPU-buffer recording API on top of the backend-agnostic
``Probe``. All ``torch`` imports live here, so the paddle backend can use
``Probe`` (via ``...core.base_monitor``) without dragging torch into the
import graph.

Hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.
``declare_*`` is the schema gate; ``record_*`` only has a disabled-key guard
for metrics suppressed by log_per_layer/log_global.
"""

import logging
import time

import torch

from ...core.base_monitor import Probe
from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)


class TorchProbe(Probe):
    """Probe with a torch-backed GPU-buffer accumulator + recompute guard.

    Two recording APIs coexist:

    1. Legacy ``_record_metrics(layer_idx, dict_of_floats)`` — the caller has
       already paid ``.item()`` per scalar, syncing the host on every call.
       Slow on the hot path; kept for unmigrated probes.

    2. GPU-buffer API — ``declare_mean`` / ``declare_max`` / ``declare_min``
       at hook-registration time, then ``record_mean`` / ``record_max`` /
       ``record_min`` inside hooks with **GPU 0-dim tensors**. No D2H sync
       fires until ``step()`` does a single batched flush.
    """

    def __init__(
        self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False, hook_timing_enabled=False
    ):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self._mean_keys: set[str] = set()
        self._max_keys: set[str] = set()
        self._min_keys: set[str] = set()
        self._gpu_acc: dict[str, torch.Tensor] = {}
        self._gpu_cnt: dict[str, int] = {}
        # Layer-metric grouping: global_key -> (agg_kind, [layer_keys]).
        # Globals are derived at flush time from layer accs to halve hot-path
        # kernel launches (no double-write per record).
        self._layer_metric_groups: dict[str, tuple[str, list[str]]] = {}
        self._layer_metric_keys: set[str] = set()
        self._disabled_keys: set[str] = set()
        # Per-layer vector metrics: key -> size, plus the expanded element keys
        # the flush writes. Mean-aggregated only; they do not derive a global.
        self._vector_keys: dict[str, int] = {}
        self._vector_elem_keys: dict[str, list[str]] = {}
        self._gpu_vec: dict[str, torch.Tensor] = {}
        self._gpu_vec_cnt: dict[str, int] = {}
        self._buffers_allocated = False
        self.hook_timing_enabled = hook_timing_enabled
        self._hook_timing: dict[str, tuple[int, float]] = {}

    def _should_monitor(self, in_forward: bool = True) -> bool:
        # in_forward=False lets off-forward callers (e.g. the expert-bias update
        # at finalize_model_grads time) keep the interval gate without the
        # recompute-pass grad guard, which only makes sense inside a forward.
        if in_forward and not torch.is_grad_enabled():
            return False
        return super()._should_monitor()

    def _flush_buffers(self) -> None:
        if not self._buffers_allocated:
            return
        flushed = self._flush_gpu_buffer()
        if flushed:
            training_logs.update(**flushed)

    def timed_hook(self, hook_name: str, hook_fn):
        """Optionally wrap a forward hook and accumulate monitored-step CPU timing."""
        if not self.hook_timing_enabled:
            return hook_fn

        def wrapped(*args, **kwargs):
            should_time = self._should_monitor()
            if not should_time:
                return hook_fn(*args, **kwargs)

            start = time.perf_counter()
            try:
                return hook_fn(*args, **kwargs)
            finally:
                calls, total = self._hook_timing.get(hook_name, (0, 0.0))
                self._hook_timing[hook_name] = (calls + 1, total + time.perf_counter() - start)

        return wrapped

    def pop_hook_timing(self) -> dict[str, tuple[int, float]]:
        timing = self._hook_timing
        self._hook_timing = {}
        return timing

    # ------------------------------------------------------------------
    # GPU-buffer API: declare → allocate → record → flush
    # ------------------------------------------------------------------

    def declare_mean(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_mean({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys, (
            f"duplicate metric key: {key!r}"
        )
        self._mean_keys.add(key)

    def declare_max(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_max({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys, (
            f"duplicate metric key: {key!r}"
        )
        self._max_keys.add(key)

    def declare_min(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_min({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys, (
            f"duplicate metric key: {key!r}"
        )
        self._min_keys.add(key)

    def allocate_buffers(self, device, dtype: torch.dtype = torch.float32) -> None:
        """Materialize all declared accumulators on ``device``. Idempotent.

        Call from ``register_hooks`` after the model is on GPU. Subsequent
        ``declare_*`` calls will raise.
        """
        if self._buffers_allocated:
            return
        for k in self._mean_keys:
            self._gpu_acc[k] = torch.zeros((), device=device, dtype=dtype)
            self._gpu_cnt[k] = 0
        for k in self._max_keys:
            self._gpu_acc[k] = torch.full((), float("-inf"), device=device, dtype=dtype)
            self._gpu_cnt[k] = 0
        for k in self._min_keys:
            self._gpu_acc[k] = torch.full((), float("inf"), device=device, dtype=dtype)
            self._gpu_cnt[k] = 0
        # vector: one [size] accumulator per layer key, expanded per element at flush.
        for k, size in self._vector_keys.items():
            self._gpu_vec[k] = torch.zeros(size, device=device, dtype=dtype)
            self._gpu_vec_cnt[k] = 0
        self._buffers_allocated = True
        if self.verbose:
            logger.info(
                f"[{self.METRIC_PREFIX}] GPU buffer budget: "
                f"mean={len(self._mean_keys)} max={len(self._max_keys)} "
                f"min={len(self._min_keys)} vector={len(self._vector_keys)} keys"
            )

    def record_mean(self, key: str, val: torch.Tensor) -> None:
        """Hot path. Declaration is the gate; disabled keys return early."""
        if key in self._disabled_keys:
            return
        self._gpu_acc[key].add_(val.detach())
        self._gpu_cnt[key] += 1

    def record_max(self, key: str, val: torch.Tensor) -> None:
        """Hot path. Declaration is the gate; disabled keys return early."""
        if key in self._disabled_keys:
            return
        torch.maximum(self._gpu_acc[key], val.detach(), out=self._gpu_acc[key])
        self._gpu_cnt[key] += 1

    def record_min(self, key: str, val: torch.Tensor) -> None:
        """Hot path. Declaration is the gate; disabled keys return early."""
        if key in self._disabled_keys:
            return
        torch.minimum(self._gpu_acc[key], val.detach(), out=self._gpu_acc[key])
        self._gpu_cnt[key] += 1

    # ------------------------------------------------------------------
    # Convenience: declare/record a metric using class-level aggregation
    # rules (MAX_AGGREGATED / MIN_AGGREGATED).
    # ------------------------------------------------------------------

    def _layer_key(self, layer_idx: int, metric_name: str) -> str:
        return f"{self.METRIC_PREFIX}/layer_{layer_idx}/{metric_name}"

    def _global_key(self, metric_name: str) -> str:
        return f"{self.METRIC_PREFIX}/global_{metric_name}"

    def _should_disable_explicit_key(self, key: str) -> bool:
        return key.startswith(f"{self.METRIC_PREFIX}/global_") and not self.log_global

    def declare_layer_metric(self, layer_idx: int, metric_name: str) -> None:
        """Declare a per-layer key. The matching global key is derived from
        the per-layer accumulators at flush time (no per-record double-write).
        """
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name)
        global_key = self._global_key(metric_name)
        all_declared = self._mean_keys | self._max_keys | self._min_keys
        assert global_key not in all_declared, (
            f"{global_key!r} was declared explicitly via declare_*; "
            f"layer-metric grouping would overwrite it at flush time"
        )
        if self._is_max_aggregated(metric_name):
            agg = "max"
            if layer_key not in all_declared:
                self.declare_max(layer_key)
        elif metric_name in self.MIN_AGGREGATED:
            agg = "min"
            if layer_key not in all_declared:
                self.declare_min(layer_key)
        else:
            agg = "mean"
            if layer_key not in all_declared:
                self.declare_mean(layer_key)
        self._layer_metric_keys.add(layer_key)
        if not self.log_global:
            return
        existing = self._layer_metric_groups.get(global_key)
        if existing is None:
            self._layer_metric_groups[global_key] = (agg, [layer_key])
        else:
            assert existing[0] == agg, f"aggregation mismatch for {global_key!r}: {existing[0]} vs {agg}"
            existing[1].append(layer_key)

    def record_layer_metric(self, layer_idx: int, metric_name: str, val: torch.Tensor) -> None:
        """Hot path. Writes only to the per-layer accumulator; globals are
        derived at flush time from those accumulators.
        """
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name)
        if self._is_max_aggregated(metric_name):
            self.record_max(layer_key, val)
        elif metric_name in self.MIN_AGGREGATED:
            self.record_min(layer_key, val)
        else:
            self.record_mean(layer_key, val)

    # ------------------------------------------------------------------
    # Per-layer vector metrics (one curve per element, e.g. per mapping cell)
    # ------------------------------------------------------------------

    def declare_layer_vector(self, layer_idx: int, metric_name: str, size: int, elem_tag: str = "e") -> None:
        """Declare a per-layer vector metric, mean-aggregated, expanded at flush.

        For "one curve per element per layer" series (a mapping cell, an expert
        share): the hot path pays a single ``add_`` for the whole vector instead
        of one kernel per element. Vector metrics derive **no** global key — a
        mean of one cell over all layers has no diagnostic meaning.
        """
        assert not self._buffers_allocated, f"declare_layer_vector({metric_name!r}) after allocate_buffers"
        if not self.log_per_layer:
            return
        key = self._layer_key(layer_idx, metric_name)
        assert key not in self._vector_keys, f"declare_layer_vector({key!r}) declared twice"
        size = int(size)
        assert size > 0
        self._vector_keys[key] = size
        self._vector_elem_keys[key] = [f"{key}_{elem_tag}{i}" for i in range(size)]

    def record_layer_vector(self, layer_idx: int, metric_name: str, vec: torch.Tensor) -> None:
        """Hot path: one in-place accumulate for the whole ``[size]`` vector."""
        key = self._layer_key(layer_idx, metric_name)
        buf = self._gpu_vec.get(key)
        if buf is None:  # log_per_layer=False, or this layer was never declared
            return
        buf.add_(vec.detach().to(buf.dtype))
        self._gpu_vec_cnt[key] += 1

    def _flush_gpu_buffer(self) -> dict[str, float]:
        """Single batched D2H of all declared metrics, then reset.

        Per-layer keys flush their own acc. Globals are derived here from
        the per-layer accs (no per-record double-write) by reducing across
        the layer keys registered for each global in declare_layer_metric.
        """
        keys: list[str] = []
        tensors: list[torch.Tensor] = []

        for k in self._mean_keys:
            cnt = self._gpu_cnt[k]
            if cnt == 0:
                continue
            if self.log_per_layer or k not in self._layer_metric_keys:
                keys.append(k)
                tensors.append(self._gpu_acc[k] / cnt)

        for k in self._max_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self.log_per_layer or k not in self._layer_metric_keys:
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        for k in self._min_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self.log_per_layer or k not in self._layer_metric_keys:
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        if self.log_global:
            for global_key, (agg, layer_keys) in self._layer_metric_groups.items():
                active = [lk for lk in layer_keys if self._gpu_cnt.get(lk, 0) > 0]
                if not active:
                    continue
                if agg == "mean":
                    # Count-weighted: equivalent to mean-of-per-layer-means when each
                    # layer fires the same number of times per step (typical case).
                    total_sum = torch.stack([self._gpu_acc[lk] for lk in active]).sum()
                    total_cnt = sum(self._gpu_cnt[lk] for lk in active)
                    tensors.append(total_sum / total_cnt)
                elif agg == "max":
                    tensors.append(torch.stack([self._gpu_acc[lk] for lk in active]).max())
                else:  # min
                    tensors.append(torch.stack([self._gpu_acc[lk] for lk in active]).min())
                keys.append(global_key)

        out: dict[str, float] = {}
        # Single D2H for scalars and vectors alike: flatten everything into one
        # concat so the vector series cost no extra sync point.
        vec_key_groups: list[list[str]] = []
        vec_tensors: list[torch.Tensor] = []
        for k, elem_keys in self._vector_elem_keys.items():
            cnt = self._gpu_vec_cnt[k]
            if cnt == 0:
                continue
            vec_key_groups.append(elem_keys)
            vec_tensors.append(self._gpu_vec[k] / cnt)

        if tensors or vec_tensors:
            flat = torch.cat([t.reshape(-1) for t in tensors] + vec_tensors)
            vals = flat.cpu().tolist()
            out = dict(zip(keys, vals, strict=False))
            offset = len(keys)
            for elem_keys in vec_key_groups:
                out.update(zip(elem_keys, vals[offset : offset + len(elem_keys)], strict=False))
                offset += len(elem_keys)

        for k in self._mean_keys:
            self._gpu_acc[k].zero_()
            self._gpu_cnt[k] = 0
        for k in self._max_keys:
            self._gpu_acc[k].fill_(float("-inf"))
            self._gpu_cnt[k] = 0
        for k in self._min_keys:
            self._gpu_acc[k].fill_(float("inf"))
            self._gpu_cnt[k] = 0
        for k in self._vector_keys:
            self._gpu_vec[k].zero_()
            self._gpu_vec_cnt[k] = 0
        return out
