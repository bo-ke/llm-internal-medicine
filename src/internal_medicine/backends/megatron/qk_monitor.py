"""
QK Stats Monitor for Megatron-Bridge + Transformer Engine.
Migrated from src/internal_medicine/qk_logits/.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from ...core.base_monitor import Probe
from .sink_head_metrics import compute_sink_head_classification
from .triton_kernels import compute_qk_stats

logger = logging.getLogger(__name__)

# Above this seq_len, skip QK stats to avoid OOM in pytorch fallback path
_MAX_SEQ_LEN_FOR_QK = 8192


class QKStatsMonitor(Probe):
    METRIC_PREFIX = "qk_stats"
    MAX_AGGREGATED = {"max", "entropy_max", "sink_head_max"}
    MIN_AGGREGATED = {"entropy_min"}

    def __init__(
        self,
        causal: bool = True,
        use_triton: bool = True,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        sink_head_threshold: float = 0.3,
    ):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.causal = causal
        self.use_triton = use_triton
        self.sink_head_threshold = sink_head_threshold
        self.compute_count = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.tp_group = None

    def register_hooks(self, model: nn.Module):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
                self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

        attention_layers = self._find_attention_layers(model)
        if len(attention_layers) == 0:
            logger.warning("[QKMonitor] No attention layers found!")
            return

        if self.verbose:
            logger.info(f"[QKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size}")

        for layer_idx, attention_module in attention_layers:
            if hasattr(attention_module, "core_attention"):
                hook = attention_module.core_attention.register_forward_pre_hook(self._make_compute_hook(layer_idx))
                self.hooks.append(hook)

        logger.info(f"[QKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        attention_layers = []
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            return []
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            attn = None
            if hasattr(layer, "self_attention"):
                attn = layer.self_attention
            elif hasattr(layer, "attention"):
                attn = layer.attention
            if attn:
                attention_layers.append((global_idx, attn))
        return attention_layers

    def _resolve_layer_idx(self, layer: nn.Module, local_idx: int, num_local_layers: int) -> int:
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + local_idx

    def _aggregate_tp_stats(self, stats: dict) -> dict:
        data = torch.tensor(
            [stats["max_global"], stats["mean_global"], stats["entropy_global"], stats["sink_global"]],
            device="cuda",
            dtype=torch.float32,
        )
        max_val = data[0:1].clone()
        if self.tp_group is not None:
            dist.all_reduce(max_val, op=dist.ReduceOp.MAX, group=self.tp_group)
        sum_vals = data[1:4].clone()
        if self.tp_group is not None:
            dist.all_reduce(sum_vals, op=dist.ReduceOp.SUM, group=self.tp_group)
        avg_vals = sum_vals / self.tp_size

        local_head_entropy = stats["entropy_per_head"].mean(dim=0)
        full_head_entropy = local_head_entropy
        if self.tp_size > 1 and self.tp_group is not None:
            gathered_list = [torch.zeros_like(local_head_entropy) for _ in range(self.tp_size)]
            dist.all_gather(gathered_list, local_head_entropy, group=self.tp_group)
            full_head_entropy = torch.cat(gathered_list)

        # Aggregate sink_per_head across TP ranks
        local_sink = stats["sink_per_head"].mean(dim=0) if stats["sink_per_head"].dim() > 1 else stats["sink_per_head"]
        full_sink = local_sink
        if self.tp_size > 1 and self.tp_group is not None:
            gathered_sink = [torch.zeros_like(local_sink) for _ in range(self.tp_size)]
            dist.all_gather(gathered_sink, local_sink, group=self.tp_group)
            full_sink = torch.cat(gathered_sink)

        return {
            "max_global": max_val.item(),
            "mean_global": avg_vals[0].item(),
            "entropy_global": avg_vals[1].item(),
            "sink_global": avg_vals[2].item(),
            "entropy_per_head_tensor": full_head_entropy,
            "sink_per_head": full_sink,
        }

    def _make_compute_hook(self, layer_idx: int):
        def hook_fn(module, args):
            if not torch.is_grad_enabled():
                return
            if not self._should_monitor():
                return
            try:
                query, key = args[0].detach(), args[1].detach()
                if query.dim() == 3:
                    query = query.unsqueeze(1)
                    key = key.unsqueeze(1)
                seq_len = query.shape[0]
                if seq_len > _MAX_SEQ_LEN_FOR_QK and not self.use_triton:
                    return
                with torch.no_grad():
                    stats = compute_qk_stats(query, key, causal=self.causal, use_triton=self.use_triton)
                if self.tp_size > 1 and dist.is_initialized():
                    stats = self._aggregate_tp_stats(stats)
                else:
                    stats["entropy_per_head_tensor"] = stats["entropy_per_head"].mean(dim=0)

                all_heads = stats["entropy_per_head_tensor"]
                sink_per_head = stats["sink_per_head"]
                sink_for_classify = sink_per_head.mean(dim=0) if sink_per_head.dim() > 1 else sink_per_head
                sink_class = compute_sink_head_classification(sink_for_classify, threshold=self.sink_head_threshold)

                metrics = {
                    "max": stats["max_global"],
                    "mean": stats["mean_global"],
                    "entropy_avg": stats["entropy_global"],
                    "sink": stats["sink_global"],
                    "entropy_min": all_heads.min().item(),
                    "entropy_max": all_heads.max().item(),
                    "entropy_std": all_heads.std().item(),
                    "sink_head_ratio": sink_class["sink_head_ratio"].item(),
                    "sink_head_max": sink_class["sink_head_max"].item(),
                    "sink_nonsink_gap": sink_class["sink_nonsink_gap"].item(),
                }
                self._record_metrics(layer_idx, metrics)
                self.compute_count += 1
            except Exception as e:
                logger.error(f"[QKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model: nn.Module,
    causal: bool = True,
    use_triton: bool = True,
    verbose: bool = False,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    sink_head_threshold: float = 0.3,
    monitor_dict: dict | None = None,
) -> nn.Module:
    monitor = QKStatsMonitor(
        causal=causal,
        use_triton=use_triton,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sink_head_threshold=sink_head_threshold,
    )
    models = [model] if not isinstance(model, list) else model
    for m in models:
        monitor.register_hooks(m)
    logger.info(f"[QKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    return model
