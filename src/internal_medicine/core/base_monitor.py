"""Base class for all monitors (probes)."""

from abc import ABC, abstractmethod

from .training_logs import training_logs


class Probe(ABC):
    """Base class shared by all backend-specific monitors.

    Subclasses set METRIC_PREFIX plus aggregation overrides, then call
    _record_metrics() from their hooks. Global aggregation and
    flushing are handled here automatically.
    """

    METRIC_PREFIX: str = ""
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        self.log_per_layer = log_per_layer
        self.log_global = log_global
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        self.hooks = []
        self.step_count = 0
        self.pp_rank = 0
        self._global_accum: dict[str, float] = {}
        self._global_count: int = 0

    @abstractmethod
    def register_hooks(self, model) -> None: ...

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def step(self):
        self.step_count += 1
        if self.log_global and self._global_accum:
            self._flush_global_metrics()

    def _should_monitor(self) -> bool:
        return self.step_count % self.monitor_interval == 0

    def _record_metrics(self, layer_idx: int, metrics: dict[str, float]):
        """Log per-layer metrics and accumulate for global aggregation.

        Increments _global_count once (one layer observation).
        Skips entirely when metrics is empty.
        """
        if not metrics:
            return
        if self.log_per_layer:
            training_logs.update(**{f"{self.METRIC_PREFIX}/layer_{layer_idx}/{k}": v for k, v in metrics.items()})
        if self.log_global:
            self._accumulate_global(metrics)
            self._global_count += 1

    def _accumulate_global(self, metrics: dict[str, float]):
        """Accumulate into global buffer WITHOUT incrementing count.

        Use when a layer emits metrics from multiple hooks (e.g., MoE
        router + expert). Pair with a single _record_metrics call on
        the final hook to get the count increment.
        """
        for name, val in metrics.items():
            if name in self.MAX_AGGREGATED:
                self._global_accum[name] = max(self._global_accum.get(name, float("-inf")), val)
            elif name in self.MIN_AGGREGATED:
                self._global_accum[name] = min(self._global_accum.get(name, float("inf")), val)
            else:
                self._global_accum[name] = self._global_accum.get(name, 0.0) + val

    def _flush_global_metrics(self):
        """Aggregate accumulated metrics into global keys and write to training_logs."""
        if self._global_count == 0:
            self._global_accum.clear()
            return
        log_dict = {}
        for name, val in self._global_accum.items():
            if name in self.MAX_AGGREGATED or name in self.MIN_AGGREGATED:
                log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val
            else:
                log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val / self._global_count
        training_logs.update(**log_dict)
        self._global_accum = {}
        self._global_count = 0


BaseMonitor = Probe
