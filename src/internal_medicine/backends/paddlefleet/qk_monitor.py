"""
QK Stats Monitor for PaddleFleet.

Uses paddle hooks on core_attention to capture Q, K tensors and compute
attention statistics via Triton kernel on GPU.
"""

import logging

import paddle
import paddle.nn as nn

from ...core.base_monitor import Probe

logger = logging.getLogger(__name__)


_triton_driver_patched = False


def _ensure_triton_driver():
    """Patch triton's NVIDIA CudaDriver to use paddle CUDA instead of torch.

    Triton 3.x's CudaDriver delegates to torch.cuda for device/stream management.
    In pfleet venvs, torch's CUDA may fail (driver version mismatch) while paddle's
    CUDA works fine. This patches the driver instance to use paddle equivalents.
    """
    global _triton_driver_patched
    if _triton_driver_patched:
        return
    try:
        from triton.backends.nvidia.driver import CudaDriver
    except ImportError:
        raise ImportError("triton is required for GPU QK stats computation") from None

    CudaDriver.is_active = staticmethod(lambda: True)

    _orig_init = CudaDriver.__init__

    def _patched_init(self):
        _orig_init(self)
        self.get_current_device = lambda: int(paddle.framework._current_expected_place().get_device_id())
        self.set_current_device = lambda dev: paddle.device.set_device(f"gpu:{dev}")
        self.get_device_capability = lambda dev=None: paddle.device.cuda.get_device_capability(dev)
        self.get_current_stream = lambda dev: paddle.device.current_stream(f"gpu:{dev}").stream_base.cuda_stream

    CudaDriver.__init__ = _patched_init
    _triton_driver_patched = True


def _compute_qk_stats_triton(q: paddle.Tensor, k: paddle.Tensor, causal: bool = True) -> dict:
    """Triton kernel path for paddle tensors. q, k: [B, H, S, D] on GPU."""
    _ensure_triton_driver()
    from ...core.triton_qk_kernel import qk_stats_kernel

    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    max_logits = paddle.empty([batch, num_heads], dtype="float32")
    mean_logits = paddle.empty([batch, num_heads], dtype="float32")
    entropy = paddle.empty([batch, num_heads], dtype="float32")
    sink = paddle.empty([batch, num_heads], dtype="float32")
    count = paddle.empty([batch, num_heads], dtype="float32")

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 if head_dim <= 64 else 128
    grid = (batch * num_heads,)

    qk_stats_kernel[grid](
        q,
        k,
        max_logits,
        mean_logits,
        entropy,
        sink,
        count,
        batch,
        num_heads,
        seq_len,
        head_dim,
        q.strides[0],
        q.strides[1],
        q.strides[2],
        q.strides[3],
        k.strides[0],
        k.strides[1],
        k.strides[2],
        k.strides[3],
        max_logits.strides[0],
        max_logits.strides[1],
        scale=scale,
        apply_causal_mask=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": float(max_logits.max()),
        "mean_global": float(mean_logits.mean()),
        "entropy_global": float(entropy.mean()),
        "sink_global": float(sink.mean()),
    }


def compute_qk_stats_paddle(q: paddle.Tensor, k: paddle.Tensor, causal: bool = True) -> dict:
    """Compute QK stats via Triton kernel.

    Args:
        q: [B, S, H, D] — PaddleFleet core_attention input format
        k: same shape as q
    """
    # [B, S, H, D] → [B, H, S, D]
    q = q.transpose([0, 2, 1, 3]).contiguous()
    k = k.transpose([0, 2, 1, 3]).contiguous()

    if not q.place.is_gpu_place():
        raise RuntimeError("[PaddleQKMonitor] QK stats requires GPU (triton kernel)")

    return _compute_qk_stats_triton(q, k, causal)


class PaddleQKStatsMonitor(Probe):
    METRIC_PREFIX = "qk_stats"
    MAX_AGGREGATED = {"max", "entropy_max"}
    MIN_AGGREGATED = {"entropy_min"}

    def __init__(self, causal=True, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.causal = causal
        self.tp_size = 1
        self.pp_rank = 0

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
            self.tp_size = get_pg_size(pg.tp)
        except Exception:
            pass

        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        attention_layers = self._find_attention_layers(model)
        if not attention_layers:
            logger.warning("[PaddleQKMonitor] No attention layers found!")
            return

        if self.verbose:
            logger.info(f"[PaddleQKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size}")

        for layer_idx, attn_module in attention_layers:
            if hasattr(attn_module, "core_attention"):
                hook = attn_module.core_attention.register_forward_pre_hook(self._make_compute_hook(layer_idx))
                self.hooks.append(hook)

        logger.info(f"[PaddleQKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        attention_layers = []
        layers = self._get_decoder_layers(model)
        if layers is None:
            return []
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            if attn is not None:
                attention_layers.append((global_idx, attn))
        return attention_layers

    def _resolve_layer_idx(self, layer: nn.Layer, local_idx: int, num_local_layers: int) -> int:
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + local_idx

    def _get_decoder_layers(self, model):
        if hasattr(model, "_layers") and hasattr(model._layers, "run_function"):
            model = model._layers
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "run_function"):
            return [
                layer for layer in model.run_function if hasattr(layer, "self_attn") or hasattr(layer, "self_attention")
            ]
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        transformer_layers = []
        for _name, sublayer in model.named_sublayers():
            if hasattr(sublayer, "self_attn") or hasattr(sublayer, "self_attention"):
                transformer_layers.append(sublayer)
        return transformer_layers if transformer_layers else None

    def _make_compute_hook(self, layer_idx: int):
        def hook_fn(layer, inputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                query, key = inputs[0], inputs[1]
                with paddle.no_grad():
                    # Handle GQA
                    if query.shape[2] != key.shape[2]:
                        heads_per_group = query.shape[2] // key.shape[2]
                        key = key.repeat_interleave(heads_per_group, axis=2)
                    stats = compute_qk_stats_paddle(query, key, causal=self.causal)

                all_heads = stats["entropy_per_head"]
                metrics = {
                    "max": stats["max_global"],
                    "mean": stats["mean_global"],
                    "entropy_avg": stats["entropy_global"],
                    "sink": stats["sink_global"],
                    "entropy_min": float(all_heads.min()),
                    "entropy_max": float(all_heads.max()),
                    "entropy_std": float(all_heads.std()),
                }

                self._record_metrics(layer_idx, metrics)
            except Exception as e:
                logger.error(f"[PaddleQKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model,
    causal=True,
    verbose=False,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    monitor_dict=None,
):
    monitor = PaddleQKStatsMonitor(
        causal=causal,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    logger.info(f"[PaddleQKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    return model
