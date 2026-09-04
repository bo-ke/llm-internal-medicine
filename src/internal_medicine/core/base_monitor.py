"""Base class for all monitors (probes) — backend-agnostic.

This module deliberately does not import ``torch`` or ``paddle``. The GPU-buffer
recording API (declare_*/record_*/allocate_buffers/_flush_gpu_buffer) lives in
backend-specific subclasses (e.g. ``backends/megatron/base.py:TorchProbe``).
The legacy CPU-float ``_record_metrics`` API lives here for backends that have
not migrated.
"""

from abc import ABC, abstractmethod

from .metric_families import parse_exclude
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

    def __init__(
        self,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        exclude_families=None,
    ):
        self.log_per_layer = log_per_layer
        self.log_global = log_global
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        # Name the families to switch off; everything else stays on, so a config
        # that never heard of families behaves exactly as before. Only exclusion
        # is offered: a whitelist would turn a taxonomy hole into a silently
        # dropped metric, while an unlisted family under exclusion is merely a
        # payload that did not shrink. An unknown family name raises here, at
        # setup, rather than quietly changing what is collected.
        self.family_selection = parse_exclude(self.METRIC_PREFIX, exclude_families)
        self.hooks = []
        # Forward hooks are attached only for the steps that actually sample.
        # ``nn.Layer.__call__`` takes a fast path that skips ``_dygraph_call_func``
        # only while the layer has *no* forward hooks, so a permanently attached
        # hook costs every microbatch of every step even when its body returns
        # immediately. Keeping the specs here lets ``step()`` re-attach them just
        # for the sampling step and drop them again afterwards.
        self._hook_specs: list[tuple[object, str, object]] = []
        self._armed_handles: list = []
        self._hooks_armed = False
        self.step_count = 0
        self.sampled_this_step = False
        self._skip_steps_remaining = 0
        self.pp_rank = 0
        self._global_accum: dict[str, float] = {}
        self._global_metric_counts: dict[str, int] = {}
        self._global_count: int = 0

    def family_allows(self, key: str) -> bool:
        """Whether ``key``'s metric family is selected for collection.

        Takes a full key (``moe_health/layer_3/router_entropy``) so it can sit on
        the single ``declare_*`` chokepoint every monitor already goes through —
        no monitor needs to know families exist. Filtering has to happen here at
        declaration and nowhere later: the schema must be complete before
        ``allocate_buffers``, and a metric dropped at record time would still cost
        its buffer and its slot in the cross-rank reduction.
        """
        if not self.family_selection.excluded:
            return True
        return self.family_selection.allows(self._metric_sub(key))

    def _metric_sub(self, key: str) -> str:
        """Full key → metric name, dropping the monitor prefix and layer segment.

        ``global_`` is left in place; ``metric_families.classify`` strips it, so
        a cross-layer aggregate stays in the same family as the per-layer curves.
        """
        prefix = f"{self.METRIC_PREFIX}/"
        sub = key[len(prefix) :] if key.startswith(prefix) else key
        if sub.startswith("layer_"):
            _layer, _, rest = sub.partition("/")
            if rest:
                sub = rest
        return sub

    @abstractmethod
    def register_hooks(self, model) -> None: ...

    # ------------------------------------------------------------------
    # Hook attachment: declared once, armed only on sampling steps
    # ------------------------------------------------------------------

    def attach_post_hook(self, target, fn):
        """Declare a forward *post* hook; attach it now if this step samples.

        Monitors call this instead of ``target.register_forward_post_hook(fn)`` so
        the probe owns the (target, fn) pair and can detach/re-attach across steps.
        The metric schema is unaffected: ``declare_*`` still runs at registration,
        before ``allocate_buffers``, so a disarmed step keeps its buffer slots and
        the cross-rank schema stays identical on every rank.
        """
        return self._declare_hook(target, "post", fn)

    def attach_pre_hook(self, target, fn):
        """Declare a forward *pre* hook; attach it now if this step samples."""
        return self._declare_hook(target, "pre", fn)

    def _declare_hook(self, target, kind: str, fn):
        self._hook_specs.append((target, kind, fn))
        if self._hooks_armed:
            handle = self._attach_one(target, kind, fn)
            if handle is not None:
                self._armed_handles.append(handle)
                self.hooks.append(handle)
            return handle
        # Registration happens before the first step, so honour that step's phase
        # rather than leaving the model hookless until the first ``step()``.
        if self._interval_reached():
            self.arm_hooks()
        return None

    @staticmethod
    def _attach_one(target, kind: str, fn):
        register = target.register_forward_pre_hook if kind == "pre" else target.register_forward_post_hook
        return register(fn)

    def arm_hooks(self) -> None:
        """Attach every declared hook. Idempotent."""
        if self._hooks_armed:
            return
        for target, kind, fn in self._hook_specs:
            handle = self._attach_one(target, kind, fn)
            if handle is not None:
                self._armed_handles.append(handle)
                # Mirrored into ``hooks`` so that list keeps meaning "everything
                # currently installed on the model".
                self.hooks.append(handle)
        self._hooks_armed = True

    def disarm_hooks(self) -> None:
        """Detach every attached hook, keeping the specs for the next sampling step.

        Only handles this class attached are touched. ``self.hooks`` may also hold
        monkey-patch removers that some monitors install once and must keep for the
        whole run; those are torn down by ``remove_hooks`` alone.
        """
        if not self._hooks_armed:
            return
        armed_ids = {id(handle) for handle in self._armed_handles}
        for handle in self._armed_handles:
            handle.remove()
        self.hooks = [handle for handle in self.hooks if id(handle) not in armed_ids]
        self._armed_handles = []
        self._hooks_armed = False

    def sync_hook_arming(self) -> None:
        """Arm or disarm so the *upcoming* step matches its sampling phase."""
        if not self._hook_specs:
            return
        if self._interval_reached():
            self.arm_hooks()
        else:
            self.disarm_hooks()

    def remove_hooks(self):
        """Tear down for good: detach everything and forget the specs."""
        self.disarm_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hook_specs = []

    def step(self):
        # Latch *before* the increment: the hooks that ran during this step's
        # forward saw the pre-increment ``step_count``, so this is the answer to
        # "did this step sample anything?". A trainer callback that wants to skip
        # the cross-rank gather on non-sampling steps must read this instead of
        # re-deriving the phase from its own global step counter — that counter
        # resumes from a checkpoint while ``step_count`` restarts at 0, and the
        # two drift apart whenever the resume step is not a multiple of
        # ``monitor_interval``.
        skip_step = self._skip_steps_remaining > 0
        self.sampled_this_step = self._interval_reached()
        if skip_step:
            self._skip_steps_remaining -= 1
        self.step_count += 1
        if not skip_step:
            self._flush_buffers()
            if self.log_global and self._global_accum:
                self._flush_global_metrics()
        # ``step_count`` now names the step that is about to run, so this decides
        # whether that step keeps its hooks attached.
        self.sync_hook_arming()

    def _flush_buffers(self) -> None:
        """Hook for backend-specific batched flush. Default: no-op.

        ``TorchProbe`` overrides this to D2H-flush its GPU accumulators.
        """
        return None

    def _interval_reached(self) -> bool:
        """Whether ``step_count`` puts this step on the sampling interval.

        The single owner of the sampling phase. ``_should_monitor`` adds
        backend-specific conditions on top; ``step`` latches this one alone so
        the flag stays a pure function of the step counter.
        """
        if self._skip_steps_remaining > 0 or not self.monitor_interval:
            return False
        return self.step_count % self.monitor_interval == 0

    def _should_monitor(self) -> bool:
        return self._interval_reached()

    def skip_next_steps(self, count: int = 1) -> None:
        """Suppress metric collection for the next ``count`` training steps.

        The gate lives on the probe so forward/backward hooks see it before doing
        metric work. ``step()`` still advances the local interval counter while
        consuming the suppression, preserving the existing process-local phase.
        """
        self._skip_steps_remaining = max(self._skip_steps_remaining, max(0, int(count)))
        # A suppressed step must not pay for attached hooks either.
        self.sync_hook_arming()

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
