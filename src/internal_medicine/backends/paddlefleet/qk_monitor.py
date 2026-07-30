"""
QK Stats Monitor for PaddleFleet.

Uses paddle hooks on core_attention to capture Q, K tensors and compute
attention statistics via Triton kernel on GPU.
"""

import logging

import paddle
import paddle.nn as nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)


def compute_sink_head_classification(sink_per_head: paddle.Tensor, threshold: float = 0.3) -> dict:
    """Classify attention heads as sink vs non-sink.

    Mirrors the megatron-side ``sink_head_metrics.compute_sink_head_classification``
    using paddle ops. ``sink_per_head`` is a 1-D tensor of length num_heads
    holding the mean attention weight on token-0 per head (already averaged
    across batch).

    Returns a dict with three GPU 0-dim tensors:
        sink_head_ratio  — fraction of heads with sink weight > threshold
        sink_head_max    — max sink weight across heads
        sink_nonsink_gap — mean(sink) - mean(non-sink); 0 if no sinks; mean if all sinks
    """
    num_heads = int(sink_per_head.numel())
    if num_heads == 0:
        zero = sink_per_head.sum()
        return {"sink_head_ratio": zero, "sink_head_max": zero, "sink_nonsink_gap": zero}

    is_sink = sink_per_head > threshold
    is_sink_f = is_sink.astype("float32")
    is_nonsink_f = paddle.logical_not(is_sink).astype("float32")
    sink_count = is_sink_f.sum()
    nonsink_count = is_nonsink_f.sum()
    sink_head_ratio = sink_count / float(num_heads)
    sink_head_max = sink_per_head.max()

    zero = sink_per_head.sum() * 0.0
    sink_sum = (sink_per_head * is_sink_f).sum()
    nonsink_sum = (sink_per_head * is_nonsink_f).sum()
    sink_mean = sink_sum / sink_count.clip(min=1.0)
    nonsink_mean = nonsink_sum / nonsink_count.clip(min=1.0)
    gap = paddle.where(
        sink_count == 0,
        zero,
        paddle.where(nonsink_count == 0, sink_per_head.mean(), sink_mean - nonsink_mean),
    )

    return {
        "sink_head_ratio": sink_head_ratio,
        "sink_head_max": sink_head_max,
        "sink_nonsink_gap": gap,
    }


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


def _compute_qk_stats_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    causal: bool = True,
    heads_per_group: int = 1,
    row_stride: int = 1,
    q_row_offset: int = 0,
    sliding_window: tuple[int, int] | list[int] | None = None,
    softmax_scale: float | None = None,
) -> dict:
    """Triton kernel path for paddle tensors.

    q: [B, H, S_q, D] (query heads). k: [B, H_kv, S_k, D] (KV heads, NOT expanded).
    ``heads_per_group = H // H_kv``; the kernel maps each query head to its KV
    head internally so we never materialize a repeat_interleave of k.
    ``row_stride`` subsamples query rows (see kernel docstring).

    Under Context-Parallel (CP > 1), Q may stay local (``S_q = S/CP``) while
    K has been all_gather'd to full seq (``S_k = S``). ``q_row_offset`` is the
    starting global row index of the local Q shard, used only by the kernel's
    causal masking so that local row m compares against key col ``m + offset``
    rather than the ambiguous local ``m``.
    """
    _ensure_triton_driver()
    from ...core.triton_qk_kernel import qk_stats_partial_kernel

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    scale = softmax_scale or 1.0 / (head_dim**0.5)
    window_left, window_right = sliding_window or (0, 0)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 if head_dim <= 64 else 128

    # Stage-1 grid: parallelize over (B*H, num_M_blocks_over_Q) instead of (B*H,) only.
    m_block_span = BLOCK_M * row_stride
    num_m_blocks = (seq_len_q + m_block_span - 1) // m_block_span

    partial_shape = [batch, num_heads, num_m_blocks]
    partial_max = paddle.empty(partial_shape, dtype="float32")
    partial_sum_logit = paddle.empty(partial_shape, dtype="float32")
    partial_sum_row_mean = paddle.empty(partial_shape, dtype="float32")
    partial_count = paddle.empty(partial_shape, dtype="float32")
    partial_sum_entropy = paddle.empty(partial_shape, dtype="float32")
    partial_sum_sink = paddle.empty(partial_shape, dtype="float32")
    partial_valid_rows = paddle.empty(partial_shape, dtype="float32")

    grid = (batch * num_heads, num_m_blocks)

    qk_stats_partial_kernel[grid](
        q,
        k,
        partial_max,
        partial_sum_logit,
        partial_sum_row_mean,
        partial_count,
        partial_sum_entropy,
        partial_sum_sink,
        partial_valid_rows,
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        heads_per_group,
        num_m_blocks,
        q_row_offset,
        q.strides[0],
        q.strides[1],
        q.strides[2],
        q.strides[3],
        k.strides[0],
        k.strides[1],
        k.strides[2],
        k.strides[3],
        partial_max.strides[0],
        partial_max.strides[1],
        partial_max.strides[2],
        scale=scale,
        apply_causal_mask=causal,
        apply_sliding_window=sliding_window is not None,
        window_left=int(window_left),
        window_right=int(window_right),
        ROW_STRIDE=row_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reduce: pure paddle GPU ops, deterministic reduction order, no D2H sync.
    max_logits = partial_max.max(axis=-1)  # [B, H]
    total_rows = partial_valid_rows.sum(axis=-1).clip(min=1.0)
    mean_logits = partial_sum_row_mean.sum(axis=-1) / total_rows
    entropy = partial_sum_entropy.sum(axis=-1) / total_rows
    sink = partial_sum_sink.sum(axis=-1) / total_rows

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": max_logits.max(),
        "mean_global": mean_logits.mean(),
        "entropy_global": entropy.mean(),
        "sink_global": sink.mean(),
    }


def compute_qk_stats_paddle(
    q: paddle.Tensor,
    k: paddle.Tensor,
    causal: bool = True,
    row_stride: int = 1,
    q_row_offset: int = 0,
    sliding_window: tuple[int, int] | list[int] | None = None,
    softmax_scale: float | None = None,
) -> dict:
    """Compute QK stats via Triton kernel.

    Args:
        q: [B, S_q, H, D] — PaddleFleet core_attention input format
        k: [B, S_k, H_kv, D] — KV heads (may be fewer than query heads for GQA)
        row_stride: subsample every ``row_stride``-th query row (1 == exact)
        q_row_offset: global row-index start for this Q shard. Zero unless the
            caller is running with Context-Parallel and Q is kept local. Used
            only by the kernel's causal-mask logic.

    The [B, S, H, D] -> [B, H, S, D] reordering is expressed via strides
    (no contiguous copy); the triton kernel reads strided memory directly.
    GQA grouping is handled inside the kernel, so k is NOT repeat-expanded.
    """
    if not q.place.is_gpu_place():
        raise RuntimeError("[PaddleQKMonitor] QK stats requires GPU (triton kernel)")

    num_q_heads = q.shape[2]
    num_k_heads = k.shape[2]
    heads_per_group = num_q_heads // num_k_heads if num_k_heads > 0 else 1

    # [B, S, H, D] -> [B, H, S, D] without copying (stride permute only).
    q = q.transpose([0, 2, 1, 3])
    k = k.transpose([0, 2, 1, 3])

    return _compute_qk_stats_triton(
        q,
        k,
        causal=causal,
        heads_per_group=heads_per_group,
        row_stride=row_stride,
        q_row_offset=q_row_offset,
        sliding_window=sliding_window,
        softmax_scale=softmax_scale,
    )


class PaddleQKStatsMonitor(PaddleProbe):
    METRIC_PREFIX = "qk_stats"
    MAX_AGGREGATED = {
        "q_norm_max",
        "k_norm_max",
        "v_norm_max",
        "max",
        "entropy_max",
        "sink_head_max",
    }
    MIN_AGGREGATED = {"entropy_min"}

    def __init__(
        self,
        causal=True,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        sink_head_threshold: float = 0.3,
        row_stride: int = 1,
    ):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.causal = causal
        self.tp_size = 1
        self.cp_size = 1
        self.cp_rank = 0
        self.cp_group = None
        self.pp_rank = 0
        self.sink_head_threshold = sink_head_threshold
        if int(row_stride) < 1:
            raise ValueError(f"[PaddleQKMonitor] row_stride must be >= 1, got {row_stride}")
        self.row_stride = int(row_stride)
        # Warn at most once if a runtime seq_len ends up smaller than row_stride.
        self._warned_row_stride_gt_seqlen = False
        # Warn at most once if the CP-gather path fails at runtime (fallback:
        # skip this hook invocation entirely to preserve correctness).
        self._warned_cp_gather_failed = False

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
            self.tp_size = get_pg_size(pg.tp)
        except Exception:
            pass

        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"])
            self.cp_group = pg.cp
            self.cp_size = get_pg_size(pg.cp)
            if self.cp_group is not None and self.cp_size > 1:
                # ``paddle.distributed.get_rank(group=...)`` returns the rank
                # within the given group, which is the seq-shard index we need.
                import paddle.distributed as dist

                self.cp_rank = dist.get_rank(group=self.cp_group)
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
            logger.info(
                f"[PaddleQKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size} CP={self.cp_size}"
            )

        if self.cp_size > 1:
            logger.warning(
                "[PaddleQKMonitor] CP=%d detected. Using K-only all_gather + "
                "per-head CP mean-reduce to preserve exact metric semantics "
                "while keeping Q local. If qk_stats overhead is still too high, raise "
                "internal_medicine_monitor_interval or remove 'qk_stats' from "
                "internal_medicine_monitors.",
                self.cp_size,
            )

        for layer_idx, _attn_module, attn_type in attention_layers:
            for m in (
                "q_norm_mean",
                "q_norm_max",
                "k_norm_mean",
                "k_norm_max",
                "v_norm_mean",
                "v_norm_max",
                "max",
                "mean",
                "entropy_avg",
                "sink",
                "entropy_min",
                "entropy_max",
                "entropy_std",
                "sink_head_ratio",
                "sink_head_max",
                "sink_nonsink_gap",
            ):
                self.declare_layer_metric(layer_idx, m, attn_type=attn_type)

        self.allocate_buffers()

        for layer_idx, attn_module, attn_type in attention_layers:
            if hasattr(attn_module, "core_attention"):
                hook = attn_module.core_attention.register_forward_pre_hook(
                    self._make_compute_hook(layer_idx, attn_type)
                )
                self.hooks.append(hook)

        self._set_compatible_sources(len(attention_layers))
        logger.info(f"[PaddleQKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer, str | None]]:
        def has_attention(layer):
            return hasattr(layer, "self_attn") or hasattr(layer, "self_attention")

        layers = get_decoder_layers(model)
        if layers is None:
            transformer_layers = [sublayer for _name, sublayer in model.named_sublayers() if has_attention(sublayer)]
            layers = transformer_layers if transformer_layers else None
        if layers is None:
            return []

        monitor_layers = iter_monitor_layers(layers, has_attention, pp_rank=self.pp_rank)
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)
        attention_layers = []
        for item in monitor_layers:
            attn = getattr(item.layer, "self_attn", None) or getattr(item.layer, "self_attention", None)
            if attn is not None:
                attention_layers.append((item.idx, attn, item.attn_type))
        return attention_layers

    def _cp_gather_seq(self, tensor: paddle.Tensor) -> paddle.Tensor | None:
        """All-gather ``tensor`` along its seq dim across the CP group."""
        if self.cp_group is None or self.cp_size <= 1:
            return tensor
        try:
            import paddle.distributed as dist

            # all_gather concatenates along axis=0. Move seq to axis 0 first,
            # gather, then reshape back to [B, S_full, H, D].
            # tensor: [B, S_local, H, D] -> [S_local, B, H, D]
            t = tensor.transpose([1, 0, 2, 3]).contiguous()
            gathered = paddle.empty([self.cp_size * t.shape[0], *t.shape[1:]], dtype=t.dtype)
            dist.all_gather(gathered, t, group=self.cp_group)
            # gathered: [S_full, B, H, D] -> [B, S_full, H, D]
            return gathered.transpose([1, 0, 2, 3]).contiguous()
        except Exception as e:
            if not self._warned_cp_gather_failed:
                logger.warning(
                    "[PaddleQKMonitor] CP all_gather failed (%s); skipping qk_stats "
                    "for this step to preserve correctness. Consider removing "
                    "qk_stats from internal_medicine_monitors under CP.",
                    e,
                )
                self._warned_cp_gather_failed = True
            return None

    def _make_compute_hook(self, layer_idx: int, attn_type: str | None = None):
        def hook_fn(layer, inputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                query, key = inputs[0], inputs[1]
                value = inputs[2] if len(inputs) > 2 else None
                with paddle.no_grad():
                    for name, tensor in (("q", query), ("k", key), ("v", value)):
                        if not isinstance(tensor, paddle.Tensor):
                            continue
                        norm = paddle.linalg.norm(tensor.detach().astype("float32"), axis=-1)
                        self.record_layer_metric(layer_idx, f"{name}_norm_mean", norm.mean(), attn_type=attn_type)
                        self.record_layer_metric(layer_idx, f"{name}_norm_max", norm.max(), attn_type=attn_type)
                    if self.cp_size > 1:
                        # CP > 1: gather K only, keep Q local.
                        query = query.detach()
                        q_local_seq = query.shape[1]  # rows on this rank
                        q_row_offset = self.cp_rank * q_local_seq
                        gathered_k = self._cp_gather_seq(key.detach())
                        if gathered_k is None:
                            return
                        key = gathered_k
                    else:
                        q_row_offset = 0

                    # query: [B, S_q, H, D]; seq_len is a static shape int (no D2H sync).
                    seq_len = query.shape[1]
                    effective_stride = self.row_stride
                    if effective_stride > seq_len:
                        # Stride coarser than the sequence would visit a single
                        # (or zero) row -> statistically useless. Clamp to a full
                        # pass and warn once instead of silently degrading.
                        if not self._warned_row_stride_gt_seqlen:
                            logger.warning(
                                "[PaddleQKMonitor] row_stride=%d exceeds seq_len=%d; "
                                "clamping to 1 (full pass) for this run.",
                                self.row_stride,
                                seq_len,
                            )
                            self._warned_row_stride_gt_seqlen = True
                        effective_stride = 1
                    # GQA grouping is handled inside the kernel; do NOT
                    # repeat_interleave the KV tensor on the hot path.
                    stats = compute_qk_stats_paddle(
                        query,
                        key,
                        causal=self.causal,
                        row_stride=effective_stride,
                        q_row_offset=q_row_offset,
                        sliding_window=getattr(layer, "sliding_window", None),
                        softmax_scale=getattr(layer, "softmax_scale", None),
                    )

                    if self.cp_size > 1 and self.cp_group is not None:
                        import paddle.distributed as dist

                        for key_name in ("entropy_per_head", "sink_per_head"):
                            t = stats[key_name].astype("float32")
                            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.cp_group)
                            stats[key_name] = t / float(self.cp_size)

                all_heads = stats["entropy_per_head"]
                self.record_layer_metric(layer_idx, "max", stats["max_global"], attn_type=attn_type)
                self.record_layer_metric(layer_idx, "mean", stats["mean_global"], attn_type=attn_type)
                self.record_layer_metric(layer_idx, "entropy_avg", stats["entropy_global"], attn_type=attn_type)
                self.record_layer_metric(layer_idx, "sink", stats["sink_global"], attn_type=attn_type)
                self.record_layer_metric(layer_idx, "entropy_min", all_heads.min(), attn_type=attn_type)
                self.record_layer_metric(layer_idx, "entropy_max", all_heads.max(), attn_type=attn_type)
                self.record_layer_metric(layer_idx, "entropy_std", all_heads.std(), attn_type=attn_type)
                # sink_per_head: [B, H] — average across batch to get [H]
                sink_per_head = stats["sink_per_head"]
                sink_for_classify = sink_per_head.mean(axis=0) if sink_per_head.ndim > 1 else sink_per_head
                sink_class = compute_sink_head_classification(sink_for_classify, threshold=self.sink_head_threshold)
                for name, val in sink_class.items():
                    self.record_layer_metric(layer_idx, name, val, attn_type=attn_type)
                self._record_observation(layer_idx)
            except Exception as e:
                self._record_error()
                logger.error(f"[PaddleQKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model,
    causal=True,
    verbose=False,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    sink_head_threshold: float = 0.3,
    row_stride: int = 1,
    monitor_dict=None,
):
    monitor = PaddleQKStatsMonitor(
        causal=causal,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sink_head_threshold=sink_head_threshold,
        row_stride=row_stride,
    )
    monitor.register_hooks(model)
    logger.info(f"[PaddleQKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    return model
