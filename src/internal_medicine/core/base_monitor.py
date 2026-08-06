"""Base class for all monitors (probes) — backend-agnostic.

This module deliberately does not import ``torch`` or ``paddle``. The GPU-buffer
recording API (declare_*/record_*/allocate_buffers/_flush_gpu_buffer) lives in
backend-specific subclasses (e.g. ``backends/megatron/base.py:TorchProbe``).
The legacy CPU-float ``_record_metrics`` API lives here for backends that have
not migrated.
"""

from abc import ABC, abstractmethod

from .training_logs import training_logs


class Probe(ABC):
    """Backend-agnostic probe lifecycle and legacy CPU-float metrics.

    Subclasses add backend-specific recording APIs. The Megatron backend's
    ``TorchProbe`` adds the GPU-buffer API (zero-D tensor accumulators flushed
    once per step). Paddle backends currently only use the legacy
    ``_record_metrics`` path defined here.

    Class-level ``MAX_AGGREGATED`` / ``MIN_AGGREGATED`` sets classify metric
    *names* (without the layer/global prefix) so both APIs agree on whether a
    metric reduces by max, min, or mean across layers and ranks.
    """

    METRIC_PREFIX: str = ""
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()
    # Metric names whose per-layer series is accumulated (the global value is
    # derived from it) but never written to the log. For "gatekeeper" metrics
    # that are constant by construction and only speak when an invariant breaks:
    # one global curve carries the same signal as N layers x M components.
    GLOBAL_ONLY: set[str] = set()

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        self.log_per_layer = log_per_layer
        self.log_global = log_global
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        self.hooks = []
        self.step_count = 0
        self.pp_rank = 0
        self._global_accum: dict[str, float] = {}
        self._global_metric_counts: dict[str, int] = {}
        self._global_count: int = 0

    @abstractmethod
    def register_hooks(self, model) -> None: ...

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def step(self):
        self.step_count += 1
        self._flush_buffers()
        if self.log_global and self._global_accum:
            self._flush_global_metrics()

    def _flush_buffers(self) -> None:
        """Hook for backend-specific batched flush. Default: no-op.

        ``TorchProbe`` overrides this to D2H-flush its GPU accumulators.
        """
        return None

    def _should_monitor(self) -> bool:
        if not self.monitor_interval:
            return False
        return self.step_count % self.monitor_interval == 0

    # ------------------------------------------------------------------
    # Legacy CPU-float API
    # ------------------------------------------------------------------

    def _record_metrics(self, layer_idx: int, metrics: dict[str, float]):
        """Log per-layer metrics and accumulate for global aggregation.

        Increments _global_count once (one layer observation).
        Skips entirely when metrics is empty.
        """
        if not metrics:
            return
        self._log_per_layer_metrics(layer_idx, metrics)
        if self.log_global:
            self._accumulate_global(metrics)
            self._count_global_observation()

    def _log_per_layer_metrics(self, layer_idx: int, metrics: dict[str, float]):
        """Write per-layer metric keys without touching global aggregation."""
        if self.log_per_layer and metrics:
            training_logs.update(**{f"{self.METRIC_PREFIX}/layer_{layer_idx}/{k}": v for k, v in metrics.items()})

    def _accumulate_global(self, metrics: dict[str, float]):
        """Accumulate into global buffer WITHOUT incrementing count.

        Use when a layer emits metrics from multiple hooks (e.g., MoE
        router + expert). Pair with a single _count_global_observation()
        call when the complete layer observation is finished.
        """
        for name, val in metrics.items():
            if self._is_max_aggregated(name):
                self._global_accum[name] = max(self._global_accum.get(name, float("-inf")), val)
            elif name in self.MIN_AGGREGATED:
                self._global_accum[name] = min(self._global_accum.get(name, float("inf")), val)
            else:
                self._global_accum[name] = self._global_accum.get(name, 0.0) + val

    def _is_max_aggregated(self, name: str) -> bool:
        return name in self.MAX_AGGREGATED

    def _count_global_observation(self, metric_names: set[str] | None = None):
        """Count one complete layer observation for global averages."""
        if self.log_global:
            self._global_count += 1
            if metric_names is None:
                metric_names = set(self._global_accum)
            for name in metric_names:
                self._global_metric_counts[name] = self._global_metric_counts.get(name, 0) + 1

    def _flush_global_metrics(self):
        """Aggregate accumulated metrics into global keys and write to training_logs."""
        if self._global_count == 0:
            self._global_accum.clear()
            self._global_metric_counts.clear()
            return
        log_dict = {}
        for name, val in self._global_accum.items():
            if self._is_max_aggregated(name) or name in self.MIN_AGGREGATED:
                log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val
            else:
                count = self._global_metric_counts.get(name, self._global_count)
                if count > 0:
                    log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val / count
        training_logs.update(**log_dict)
        self._global_accum = {}
        self._global_metric_counts = {}
        self._global_count = 0

    def _resolve_layer_idx(self, layer, local_idx: int, num_local_layers: int, layer_offset: int = 0) -> int:
        """Resolve a stable global layer id when model layers expose one."""
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + layer_offset + local_idx


BaseMonitor = Probe
