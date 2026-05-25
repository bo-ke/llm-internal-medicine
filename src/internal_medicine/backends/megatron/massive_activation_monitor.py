"""
Massive Activation Monitor for Megatron-Bridge.

Monitors massive activations in post-residual hidden states — extreme outlier
values that appear in a few channels and persist across intermediate layers
via the residual connection.

Based on findings from:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    "The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks." arXiv:2603.05498.

Key insight: Massive activations follow a "rise-plateau-fall" lifecycle:
    1. Step-up blocks (early FFN) inject extreme values via directional
       quadratic amplification (SwiGLU as high-gain amplifier)
    2. Residual accumulation preserves them across intermediate layers
    3. Step-down blocks (late FFN) neutralize them with opposite-sign values

This monitor hooks into pre-norm points (i.e., the residual stream BEFORE
RMSNorm) to capture raw activation magnitudes, and post-norm points to
detect the sparsification that enables attention sinks.

Metrics produced:
    massive_act/layer_{i}/channel_max          — peak channel magnitude
    massive_act/layer_{i}/channel_max_ratio    — outlier severity (max/median)
    massive_act/layer_{i}/massive_act_channel_count — number of massive activation channels
    massive_act/layer_{i}/topk_channel_norm    — L2 norm of top-K channels
    massive_act/layer_{i}/post_norm_sparsity   — near-zero fraction after norm
    massive_act/layer_{i}/post_norm_cosine     — token similarity after norm
    massive_act/global_*                       — layer-aggregated versions
"""

import logging

import torch
import torch.nn as nn

from ...core.base_monitor import Probe
from .massive_activation_metrics import (
    compute_per_channel_max,
    compute_post_norm_cosine_stability,
    compute_post_norm_sparsity,
    summarize_per_channel_max,
)

logger = logging.getLogger(__name__)


class MassiveActivationMonitor(Probe):
    """Monitor massive activations in the residual stream.

    Hooks into Transformer layers at two points:
    1. Pre-norm input (raw residual stream) — measures activation magnitudes
    2. Post-norm output (after RMSNorm) — measures sparsification

    Both are captured via forward pre-hooks on the attention/FFN sub-layers,
    which receive the already-normalized input in Megatron's pre-norm architecture.
    The raw residual is obtained from the layer's input before normalization.
    """

    METRIC_PREFIX = "massive_act"
    MAX_AGGREGATED = {"channel_max", "channel_max_ratio", "topk_channel_norm"}

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
    ):
        """
        Args:
            spike_threshold_multiplier: multiplier on median channel max to define
                "massive activation" threshold for channel count metric.
            topk_channels: number of top channels for top-K norm metric.
            sparsity_epsilon: threshold for post-norm sparsity detection.
            cosine_sample_pairs: number of random pairs for cosine stability.
            sample_layers: if provided, only monitor these layer indices (global).
                Default: monitor all layers.
        """
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
        self.tp_size = 1
        self.tp_group = None
        self._warned_per_channel_aggregate = False
        self._post_norm_failed_layers: set[int] = set()

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        """Register forward hooks on Transformer layer normalization points."""
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

        layers = self._find_transformer_layers(model)
        if not layers:
            logger.warning("[MassiveActMonitor] No transformer layers found!")
            return

        registered = 0
        for local_idx, layer in layers:
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers), layer_offset)

            if self.sample_layers and global_idx not in self.sample_layers:
                continue

            hook = layer.register_forward_pre_hook(self._make_residual_hook(global_idx), with_kwargs=True)
            self.hooks.append(hook)
            registered += 1

        logger.info(f"[MassiveActMonitor] Registered {registered} hooks.")

    def _find_transformer_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        """Find transformer layers in Megatron model hierarchy."""
        if hasattr(model, "module"):
            model = model.module

        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers

        if layers is None:
            return []

        return list(enumerate(layers))

    def _resolve_layer_idx(self, layer: nn.Module, local_idx: int, num_local_layers: int, layer_offset: int = 0) -> int:
        """Resolve a stable global layer id when model layers expose one."""
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + layer_offset + local_idx

    def _make_residual_hook(self, layer_idx: int):
        """Create a forward pre-hook that captures the residual stream."""

        def hook_fn(module, args, kwargs=None):
            if not torch.is_grad_enabled():
                return
            if not self._should_monitor():
                return
            try:
                if args:
                    hidden_states = args[0]
                elif kwargs and "hidden_states" in kwargs:
                    hidden_states = kwargs["hidden_states"]
                else:
                    return
                if hidden_states is None:
                    return

                with torch.no_grad():
                    self._compute_and_log(layer_idx, hidden_states.detach(), module)
            except Exception as e:
                logger.error(f"[MassiveActMonitor] Error at layer {layer_idx}: {e}")

        return hook_fn

    def _compute_and_log(self, layer_idx: int, hidden_states: torch.Tensor, module: nn.Module):
        """Compute all massive activation metrics and log them."""
        per_channel_max = compute_per_channel_max(hidden_states)
        per_channel_max = self._aggregate_per_channel_max(per_channel_max)
        tensor_metrics = summarize_per_channel_max(
            per_channel_max,
            threshold_multiplier=self.spike_threshold_multiplier,
            k=self.topk_channels,
        )
        metrics = {
            "channel_max": tensor_metrics["channel_max"].item(),
            "channel_max_ratio": tensor_metrics["channel_max_ratio"].item(),
            "massive_act_channel_count": tensor_metrics["massive_act_channel_count"].item(),
            "topk_channel_norm": tensor_metrics["topk_channel_norm"].item(),
        }

        norm_layer = getattr(module, "input_layernorm", None)
        if norm_layer is not None:
            try:
                normalized = norm_layer(hidden_states)
                sparsity = compute_post_norm_sparsity(normalized, epsilon=self.sparsity_epsilon)
                metrics["post_norm_sparsity"] = sparsity.item()

                cosine = compute_post_norm_cosine_stability(normalized, num_sample_pairs=self.cosine_sample_pairs)
                metrics["post_norm_cosine"] = cosine.item()
            except Exception as e:
                if self.verbose and layer_idx not in self._post_norm_failed_layers:
                    logger.warning(f"[MassiveActMonitor] Post-norm metrics disabled at layer {layer_idx}: {e}")
                    self._post_norm_failed_layers.add(layer_idx)

        self._record_metrics(layer_idx, metrics)

    def _aggregate_per_channel_max(self, per_channel_max: torch.Tensor) -> torch.Tensor:
        """Aggregate token-sharded per-channel maxima across the TP group when available."""
        if self.tp_size <= 1 or self.tp_group is None:
            return per_channel_max
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(per_channel_max, op=dist.ReduceOp.MAX, group=self.tp_group)
        except Exception as e:
            if self.verbose and not self._warned_per_channel_aggregate:
                logger.warning(f"[MassiveActMonitor] TP per-channel aggregation failed; using local values: {e}")
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
    monitor_dict: dict | None = None,
):
    """Setup the Massive Activation Monitor."""
    monitor = MassiveActivationMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        spike_threshold_multiplier=spike_threshold_multiplier,
        topk_channels=topk_channels,
        sparsity_epsilon=sparsity_epsilon,
        cosine_sample_pairs=cosine_sample_pairs,
        sample_layers=sample_layers,
    )

    models = [model] if not isinstance(model, list) else model
    layer_offset = 0
    for m in models:
        layers = monitor._find_transformer_layers(m)
        monitor.register_hooks(m, layer_offset=layer_offset)
        layer_offset += len(layers)
    logger.info(f"[MassiveActMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")

    if monitor_dict is not None:
        monitor_dict["massive_act"] = monitor

    return model
