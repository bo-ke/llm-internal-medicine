"""
Framework-agnostic training metrics store.

All values stored as Python floats. Framework-specific tensor conversion
is the caller's responsibility.

Cross-rank aggregation has two implementations:

1. Preferred — ``_reduce_fn``: a fixed-schema tensor all_reduce. Payload is
   ``O(number of distinct keys)`` and independent of world size.
2. Fallback — ``_gather_fn``: ``all_gather_object``. Payload is
   ``O(keys x world_size)`` pickled dicts, which at a few thousand ranks costs
   tens of seconds of pure CPU per flush and gigabytes of resident memory.
   Kept for single-card runs and as the escape hatch when the schema cannot be
   made world-consistent.
"""

import logging
import zlib
from collections import defaultdict
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ("SmoothedValue", "TrainingLogs", "training_logs")

MAX_AGGREGATED_SUFFIXES = (
    "topk_channel_norm",
    "channel_max_ratio",
    "channel_median",
    "channel_p95",
    "channel_p99",
    "activation_rms",
    "massive_act_channel_count",
)

# Absolute-threshold channel counts: the threshold is the trailing token
# (``channel_count_gt_10``), so this cannot be a fixed suffix.
CHANNEL_COUNT_GT_MARKER = "channel_count_gt_"


class SmoothedValue:
    """Track a series of scalar values and provide smoothed access."""

    def __init__(self, mode="mean"):
        self.total = 0.0
        self.count = 0
        self.mode = mode
        self.latest_value = 0.0
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")

    def update(self, value: float):
        self.latest_value = value
        self.count += 1
        self.total += value
        if self.mode == "max":
            self.max_value = max(self.max_value, value)
        if self.mode == "min":
            self.min_value = min(self.min_value, value)

    @property
    def global_avg(self):
        return self.total / max(self.count, 1e-6)

    @property
    def log(self):
        if self.mode == "max":
            return self.max_value
        if self.mode == "min":
            return self.min_value
        return self.global_avg

    @property
    def latest(self):
        return self.latest_value

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.latest_value = 0.0
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")


class TrainingLogs:
    """Singleton metric store. All monitors write here."""

    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, gather_fn: Callable | None = None):
        if not hasattr(self, "meters"):
            self.meters = {}
        if not hasattr(self, "_pre_aggregated_keys"):
            self._pre_aggregated_keys: set[str] = set()
        if not hasattr(self, "_schema"):
            self._reset_schema()
        if gather_fn is not None:
            self._gather_fn = gather_fn

    def _reset_schema(self) -> None:
        self._schema: list[str] | None = None
        self._schema_set: set[str] = set()
        self._schema_mean: list[str] = []
        self._schema_max: list[str] = []
        self._schema_min: list[str] = []
        self._schema_checksum: float = 0.0
        self._tensor_path_disabled = False

    def set_gather_fn(self, fn: Callable):
        """Set the distributed gather function (backend provides this)."""
        self._gather_fn = fn

    def set_reduce_fn(self, fn: Callable | None):
        """Install the backend's fixed-schema tensor all_reduce.

        ``fn(sum_vec, max_vec, min_vec)`` returns the three lists reduced with
        SUM / MAX / MIN across the whole world, or ``None`` when unavailable.
        Preferred over ``_gather_fn``: payload no longer scales with world size.
        """
        self._reduce_fn = fn
        self._reset_schema()

    def set_schema_sync_fn(self, fn: Callable | None):
        """Install the backend's key-list all_gather, used to build the schema.

        ``fn(local_keys)`` returns one key list per rank of the *pipeline* group.
        Ranks that share a pipeline stage hold the same layers and therefore
        declare the same keys, so the pipeline group's union already covers the
        world — gathering key names over the full world would itself cost
        hundreds of MB at a few thousand ranks.
        """
        self._schema_sync_fn = fn
        self._reset_schema()

    def mark_pre_aggregated(self, keys):
        """Declare keys whose cross-rank reduction already happened on device.

        They are then excluded from both aggregation paths. Used by vector
        metrics, which emit one key per element: at 512 experts x 43 layers they
        would dominate the payload while carrying a value every rank agrees on.
        Schema-level, so ``reset()`` deliberately does not clear it.
        """
        self._pre_aggregated_keys.update(keys)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, k, v):
        val = float(v) if isinstance(v, int | float) else float(v.item() if hasattr(v, "item") else v)
        if k not in self.meters:
            if self._is_max_metric(k):
                mode = "max"
            elif self._is_min_metric(k):
                mode = "min"
            else:
                mode = "mean"
            self.meters[k] = SmoothedValue(mode=mode)
        self.meters[k].update(val)

    @staticmethod
    def _is_max_metric(key: str) -> bool:
        return (
            "/max" in key
            or key.endswith("_max")
            or key.endswith(MAX_AGGREGATED_SUFFIXES)
            or CHANNEL_COUNT_GT_MARKER in key
        )

    @staticmethod
    def _is_min_metric(key: str) -> bool:
        return "/min" in key or key.endswith("_min")

    def __getitem__(self, v):
        return self.meters[v]

    def dict(self, smoothed: bool = False):
        return {k: (v.log if smoothed else v.latest) for k, v in self.meters.items()}

    def get_latest(self, prefix=None, smoothed: bool = False):
        result = {}
        for k, v in self.meters.items():
            if prefix is None or k.startswith(prefix):
                result[k] = v.log if smoothed else v.latest
        return result

    def print_metrics(self, metrics=None, prefix=None, format_fn=None):
        if format_fn is None:
            format_fn = logger.info
        if metrics is None:
            metrics = self.get_latest(prefix)
        elif prefix:
            metrics = {k: v for k, v in metrics.items() if k.startswith(prefix)}
        if not metrics:
            return
        grouped = defaultdict(dict)
        for k, v in metrics.items():
            if "/" in k:
                parts = k.split("/")
                category = parts[0]
                metric_name = "/".join(parts[1:])
                grouped[category][metric_name] = v
            else:
                grouped["other"][k] = v
        for category, items in sorted(grouped.items()):
            format_fn(f"[{category}]")
            for name, value in sorted(items.items()):
                format_fn(f"  {name}: {value:.4f}")

    def reset(self):
        self.meters.clear()

    def _build_schema(self, local_keys) -> None:
        """Freeze the global key order. Identical on every rank by construction."""
        sync_fn = getattr(self, "_schema_sync_fn", None)
        keys = set(local_keys)
        if sync_fn is not None:
            key_lists = sync_fn(sorted(keys))
            for other in key_lists or ():
                keys.update(other)
        self._schema = sorted(keys)
        self._schema_set = set(self._schema)
        # Same max-then-min-then-mean order as __setitem__, so a key's reduction
        # op matches the SmoothedValue mode it was stored under.
        self._schema_max = [k for k in self._schema if self._is_max_metric(k)]
        self._schema_min = [k for k in self._schema if not self._is_max_metric(k) and self._is_min_metric(k)]
        extrema = set(self._schema_max) | set(self._schema_min)
        self._schema_mean = [k for k in self._schema if k not in extrema]
        self._schema_checksum = float(zlib.crc32("\n".join(self._schema).encode()))

    def _schema_agreed(self, reduce_fn):
        """Fixed-size probe of ``(schema length, checksum)``. ``None`` if unavailable.

        This has to run *before* the payload reduce: if two ranks froze different
        schemas their payload vectors have different lengths, and a mismatched
        all_reduce hangs rather than raising. Two elements, so the probe itself can
        never be the thing that hangs.
        """
        probe = [float(len(self._schema)), self._schema_checksum]
        reduced = reduce_fn([], list(probe), list(probe))
        if reduced is None:
            return None
        _, r_max, r_min = reduced
        return r_max[0] == r_min[0] and r_max[1] == r_min[1]

    def _tensor_aggregate(self, metrics: "dict") -> "dict | None":
        """Aggregate over a fixed schema with a handful of all_reduce calls.

        Layout, all float64 (exact for the crc32 checksum and for the counts):

        - SUM: ``[mean values (M)] + [presence mask (M)]``
        - MAX: ``[resync flag] + [max values (X)]``
        - MIN: ``[min values (I)]``

        Missing keys contribute 0 / ``-inf`` / ``+inf`` and a 0 mask, so a key no
        rank reported stays absent from the output — matching the legacy path.
        Dividing by the reduced mask reproduces the legacy mean exactly: an
        unweighted mean over the ranks that hold the key, *not* count-weighted.

        Every branch below is taken on the value of a *reduced* quantity, so all
        ranks make the same decision and issue the same sequence of collectives.

        Returns ``None`` to hand the round back to the legacy gather path.
        """
        reduce_fn = self._reduce_fn
        local_keys = set(metrics)
        if self._schema is None:
            self._build_schema(local_keys)

        for attempt in (0, 1):
            agreed = self._schema_agreed(reduce_fn)
            if agreed is None:
                return None
            if not agreed:
                # Ranks sharing a pipeline stage declared different keys, so the
                # pipeline-group union is not the world union. Widening the sync
                # group is the real fix; until then stay correct via the old path.
                logger.warning(
                    "[training_logs] metric schema differs across ranks; "
                    "falling back to all_gather_object aggregation for the rest of the run"
                )
                self._tensor_path_disabled = True
                return None

            unknown = local_keys - self._schema_set
            mean_keys, max_keys, min_keys = self._schema_mean, self._schema_max, self._schema_min
            sum_vec = [metrics.get(k, 0.0) for k in mean_keys]
            sum_vec += [1.0 if k in metrics else 0.0 for k in mean_keys]
            max_vec = [1.0 if unknown else 0.0]
            max_vec += [metrics[k] if k in metrics else float("-inf") for k in max_keys]
            min_vec = [metrics[k] if k in metrics else float("inf") for k in min_keys]

            reduced = reduce_fn(sum_vec, max_vec, min_vec)
            if reduced is None:
                return None
            r_sum, r_max, r_min = reduced

            if r_max[0] > 0:
                # Some rank holds a key outside the frozen schema. Every rank sees
                # this same flag, so rebuilding is a collectively consistent
                # decision — retry once with the widened schema.
                if attempt == 0:
                    self._build_schema(local_keys)
                    continue
                logger.warning(
                    "[training_logs] schema still incomplete after a rebuild; "
                    "falling back to all_gather_object aggregation for the rest of the run"
                )
                self._tensor_path_disabled = True
                return None
            break

        aggregated: dict[str, float] = {}
        offset = len(mean_keys)
        for i, k in enumerate(mean_keys):
            present = r_sum[offset + i]
            if present > 0:
                aggregated[k] = r_sum[i] / present
        for i, k in enumerate(max_keys):
            value = r_max[1 + i]
            if value != float("-inf"):
                aggregated[k] = value
        for i, k in enumerate(min_keys):
            value = r_min[i]
            if value != float("inf"):
                aggregated[k] = value
        return aggregated

    def gather_and_aggregate(self):
        """Gather metrics from all ranks and aggregate by naming convention."""
        all_metrics = self.get_latest()

        # Keys already reduced on device are excluded from the payload; every rank
        # holds the same value, so shipping one copy per rank buys nothing.
        pre_aggregated = {k: v for k, v in all_metrics.items() if k in self._pre_aggregated_keys}
        to_reduce = {k: v for k, v in all_metrics.items() if k not in pre_aggregated} if pre_aggregated else all_metrics

        reduce_fn = getattr(self, "_reduce_fn", None)
        if reduce_fn is not None and not self._tensor_path_disabled:
            aggregated = self._tensor_aggregate(to_reduce)
            if aggregated is not None:
                aggregated.update(pre_aggregated)
                return aggregated

        gather_fn = getattr(self, "_gather_fn", None)
        if gather_fn is None:
            return all_metrics

        info_list = gather_fn(to_reduce)
        if info_list is None:
            return all_metrics

        aggregated = {}
        all_keys = {key for item in info_list for key in item}
        for k in all_keys:
            values = [v[k] for v in info_list if k in v]
            if not values:
                continue
            if self._is_max_metric(k):
                aggregated[k] = max(values)
            elif self._is_min_metric(k):
                aggregated[k] = min(values)
            else:
                aggregated[k] = sum(values) / len(values)
        aggregated.update(pre_aggregated)
        return aggregated


training_logs = TrainingLogs()
