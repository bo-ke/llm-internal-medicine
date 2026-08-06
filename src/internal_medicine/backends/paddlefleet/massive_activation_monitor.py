"""
Massive Activation Monitor for PaddleFleet.

Monitors massive activations in post-residual hidden states — extreme outlier
values that appear in a few channels and persist across intermediate layers
via the residual connection.

Based on findings from:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    "The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks." arXiv:2603.05498.

This monitor hooks into pre-norm points (residual stream BEFORE RMSNorm)
to capture raw activation magnitudes, and post-norm points to detect the
sparsification that enables attention sinks.

Metrics produced:
    massive_act/layer_{i}/channel_max
    massive_act/layer_{i}/channel_median
    massive_act/layer_{i}/channel_p95
    massive_act/layer_{i}/channel_p99
    massive_act/layer_{i}/channel_max_ratio
    massive_act/layer_{i}/massive_act_channel_count
    massive_act/layer_{i}/channel_count_gt_{x}
    massive_act/layer_{i}/topk_channel_norm
    massive_act/layer_{i}/activation_rms
    massive_act/layer_{i}/post_norm_sparsity
    massive_act/layer_{i}/post_norm_cosine
    massive_act/global_*
"""

import logging

import paddle
import paddle.nn as nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .massive_activation_metrics import (
    DEFAULT_ABSOLUTE_THRESHOLDS,
    _threshold_key,
    compute_activation_scale_stats,
    compute_per_channel_max,
    compute_post_norm_cosine_stability,
    compute_post_norm_sparsity,
    summarize_per_channel_max,
)

logger = logging.getLogger(__name__)


class PaddleMassiveActivationMonitor(PaddleProbe):
    """Monitor massive activations in the residual stream.

    Hooks into Transformer layers via forward pre-hooks to capture the
    hidden_states (residual stream) before normalization.
    """

    METRIC_PREFIX = "massive_act"
    MAX_AGGREGATED = {
        "channel_max",
        "channel_median",
        "channel_p95",
        "channel_p99",
        "channel_max_ratio",
        "topk_channel_norm",
        "activation_rms",
        "massive_act_channel_count",
    }

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        spike_threshold_multiplier: float = 100.0,
        topk_channels: int = 3,
        sparsity_epsilon: float = 0.01,
        cosine_sample_pairs: int = 256,
        sample_layers: list[int] | None = None,
        absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.topk_channels = topk_channels
        self.sparsity_epsilon = sparsity_epsilon
        self.cosine_sample_pairs = cosine_sample_pairs
        self.sample_layers = set(sample_layers) if sample_layers else None
        self.absolute_thresholds = tuple(absolute_thresholds)
        self.MAX_AGGREGATED = self.MAX_AGGREGATED | {
            f"channel_count_gt_{_threshold_key(t)}" for t in self.absolute_thresholds
        }
        self.tp_size = 1
        self.tp_group = None
        self.cp_size = 1
        self.cp_group = None
        self._warned_per_channel_aggregate = False
        self._post_norm_failed_layers: set[int] = set()
        self._hc_aggregate_failed_layers: set[int] = set()

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
            self.tp_group = pg.tp
            self.tp_size = get_pg_size(pg.tp)
        except Exception:
            pass
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"])
            self.cp_group = pg.cp
            self.cp_size = get_pg_size(pg.cp)
        except Exception:
            pass

        def is_transformer_layer(layer):
            return hasattr(layer, "self_attn") or hasattr(layer, "self_attention") or hasattr(layer, "input_layernorm")

        layers = get_decoder_layers(model)
        if layers is None:
            transformer_layers = [
                sublayer for _name, sublayer in model.named_sublayers() if is_transformer_layer(sublayer)
            ]
            layers = transformer_layers if transformer_layers else None
        if not layers:
            logger.warning("[MassiveActMonitor] No transformer layers found!")
            return

        monitor_layers = iter_monitor_layers(layers, is_transformer_layer, pp_rank=self.pp_rank)
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)

        # Declare metric schema
        metric_names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "topk_channel_norm",
            "activation_rms",
            "post_norm_sparsity",
            "post_norm_cosine",
        ] + [f"channel_count_gt_{_threshold_key(t)}" for t in self.absolute_thresholds]

        registered = 0
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            for m in metric_names:
                self.declare_layer_metric(item.idx, m, attn_type=item.attn_type)
            registered += 1

        self.allocate_buffers()

        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            hook = item.layer.register_forward_pre_hook(self._make_residual_hook(item.idx, item.attn_type))
            self.hooks.append(hook)

        logger.info(f"[MassiveActMonitor] Registered {registered} hooks.")

    def _make_residual_hook(self, layer_idx: int, attn_type: str | None = None):
        def hook_fn(module, inputs):
            if not module.training:
                return
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(inputs)
                if hidden_states is None:
                    return

                with paddle.no_grad():
                    self._compute_and_log(layer_idx, hidden_states.detach(), module, attn_type=attn_type)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MassiveActMonitor] Error at layer {layer_idx}: {e}")

        return hook_fn

    @staticmethod
    def _extract_hidden_states(inputs):
        """Extract hidden_states from pre-hook inputs.

        Handles two PaddleFleet conventions:
        - Dict-based: forward(self, dict_args) where dict_args["hidden_states"] is the tensor
        - Positional: forward(self, hidden_states, ...) where inputs[0] is the tensor
        """
        if not inputs:
            return None
        first = inputs[0]
        if isinstance(first, dict):
            return first.get("hidden_states")
        if isinstance(first, paddle.Tensor):
            return first
        return None

    def _compute_and_log(
        self, layer_idx: int, hidden_states: paddle.Tensor, module: nn.Layer, attn_type: str | None = None
    ):
        # HyperConnection (e.g. DSv4): the residual stream is expanded to [..., n*h].
        hc = getattr(module, "self_attention_hyper_connection", None)
        analysis_input = hidden_states
        if hc is not None:
            try:
                hc_out = hc(hidden_states)
                aggregated = hc_out[0] if isinstance(hc_out, (tuple, list)) else hc_out
                if isinstance(aggregated, paddle.Tensor) and aggregated.shape[-1] != hidden_states.shape[-1]:
                    analysis_input = aggregated
            except Exception as e:
                if self.verbose and layer_idx not in self._hc_aggregate_failed_layers:
                    logger.warning(f"[MassiveActMonitor] hyper_connection aggregate failed at layer {layer_idx}: {e}")
                    self._hc_aggregate_failed_layers.add(layer_idx)

        per_channel_max = compute_per_channel_max(analysis_input)
        per_channel_max = self._aggregate_per_channel_max(per_channel_max)
        stats = summarize_per_channel_max(
            per_channel_max,
            threshold_multiplier=self.spike_threshold_multiplier,
            k=self.topk_channels,
            absolute_thresholds=self.absolute_thresholds,
        )
        scale_stats = compute_activation_scale_stats(analysis_input)

        for name, val in stats.items():
            self.record_layer_metric(layer_idx, name, val, attn_type=attn_type)
        for name, val in scale_stats.items():
            self.record_layer_metric(layer_idx, name, val, attn_type=attn_type)

        norm_layer = getattr(module, "input_layernorm", None)
        if norm_layer is not None:
            try:
                normalized = norm_layer(analysis_input)
                if isinstance(normalized, (tuple, list)):
                    normalized = normalized[0]
                self.record_layer_metric(
                    layer_idx,
                    "post_norm_sparsity",
                    compute_post_norm_sparsity(normalized, epsilon=self.sparsity_epsilon),
                    attn_type=attn_type,
                )
                self.record_layer_metric(
                    layer_idx,
                    "post_norm_cosine",
                    compute_post_norm_cosine_stability(normalized, num_sample_pairs=self.cosine_sample_pairs),
                    attn_type=attn_type,
                )
            except Exception as e:
                if self.verbose and layer_idx not in self._post_norm_failed_layers:
                    logger.warning(f"[MassiveActMonitor] Post-norm metrics disabled at layer {layer_idx}: {e}")
                    self._post_norm_failed_layers.add(layer_idx)

    def _aggregate_per_channel_max(self, per_channel_max: paddle.Tensor) -> paddle.Tensor:
        """Aggregate token-sharded per-channel maxima across TP + CP groups when available.

        - TP MAX all_reduce: covers SP-in-TP (residual stream sharded along seq
          within the TP group under PaddleFleet's SP convention).
        - CP MAX all_reduce: covers Context Parallel, which shards the sequence
          dim over an independent CP group. Without this, each CP rank only sees
          its local S/CP tokens and per-channel-max is a strict lower bound of
          the true global max.

        Both collectives are correctness-required (see monitor-hook-perf-rules
        exception clause) and use MAX ops that are safe under repeated calls.
        """
        try:
            import paddle.distributed as dist
        except Exception as e:
            if self.verbose and not self._warned_per_channel_aggregate:
                logger.warning(f"[MassiveActMonitor] paddle.distributed unavailable: {e}")
                self._warned_per_channel_aggregate = True
            return per_channel_max
        if self.tp_size > 1 and self.tp_group is not None:
            try:
                dist.all_reduce(per_channel_max, op=dist.ReduceOp.MAX, group=self.tp_group)
            except Exception as e:
                if self.verbose and not self._warned_per_channel_aggregate:
                    logger.warning(f"[MassiveActMonitor] TP per-channel aggregation failed: {e}")
                    self._warned_per_channel_aggregate = True
        if self.cp_size > 1 and self.cp_group is not None:
            try:
                dist.all_reduce(per_channel_max, op=dist.ReduceOp.MAX, group=self.cp_group)
            except Exception as e:
                if self.verbose and not self._warned_per_channel_aggregate:
                    logger.warning(f"[MassiveActMonitor] CP per-channel aggregation failed: {e}")
                    self._warned_per_channel_aggregate = True
        return per_channel_max


def setup_massive_activation_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    spike_threshold_multiplier: float = 100.0,
    topk_channels: int = 3,
    sparsity_epsilon: float = 0.01,
    cosine_sample_pairs: int = 256,
    sample_layers: list[int] | None = None,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
    monitor_dict: dict | None = None,
):
    monitor = PaddleMassiveActivationMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        spike_threshold_multiplier=spike_threshold_multiplier,
        topk_channels=topk_channels,
        sparsity_epsilon=sparsity_epsilon,
        cosine_sample_pairs=cosine_sample_pairs,
        sample_layers=sample_layers,
        absolute_thresholds=absolute_thresholds,
    )
    monitor.register_hooks(model)
    logger.info(f"[MassiveActMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["massive_act"] = monitor
    return model
