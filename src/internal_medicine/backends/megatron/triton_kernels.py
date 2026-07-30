"""
Triton kernels for efficient QK attention statistics computation.
Includes: Max Logits, Mean Logits, Attention Entropy, and Attention Sink Weights.
"""

import logging

import torch

from ...core.triton_qk_kernel import qk_stats_kernel

logger = logging.getLogger(__name__)


def compute_qk_stats_triton(q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> dict:
    """
    Compute QK statistics using optimized Triton kernel.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).
    Returns: Max Logits, Mean Logits, Entropy, Sink Weights.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # Output tensors [batch, num_heads]
    max_logits = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    mean_logits = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    entropy = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    sink = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    count = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)

    grid = (batch * num_heads,)

    # Tuning block sizes for performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    if head_dim > 64:
        BLOCK_K = 128

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
        seq_len,  # seq_len_q
        seq_len,  # seq_len_k: symmetric (no CP sharding in this monitor path)
        head_dim,
        1,  # heads_per_group: k is already repeat_interleave-expanded above
        0,  # q_row_offset: 0 without CP (Q is the full local sequence)
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        max_logits.stride(0),
        max_logits.stride(1),
        scale=scale,
        apply_causal_mask=causal,
        apply_sliding_window=False,
        window_left=0,
        window_right=0,
        ROW_STRIDE=1,  # full-sequence (exact) behavior, unchanged
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

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


def compute_qk_stats_pytorch(q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> dict:
    """
    Reference PyTorch implementation including Entropy and Sink.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # 1. Logits
    # [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        logits.masked_fill_(mask, float("-inf"))

    # 2. Stats: Max & Mean (Raw Logits)
    # Filter out -inf for mean calculation
    valid_mask = logits > -1e9
    max_per_head = logits.max(dim=-1)[0].max(dim=-1)[0]

    logits_zeroed = torch.where(valid_mask, logits, torch.tensor(0.0, device=q.device))
    sum_logits = logits_zeroed.sum(dim=(-2, -1))
    count = valid_mask.sum(dim=(-2, -1))
    mean_per_head = sum_logits / count.clamp(min=1)

    # 3. Softmax & Entropy & Sink
    # Softmax along last dim (keys)
    probs = torch.softmax(logits, dim=-1)  # [B, H, S, S]

    # Entropy: -sum(p * log(p))
    # Note: masked positions have p=0, 0*log0=0 (handled by where)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -(probs * log_probs)
    # Mask out invalid (causal) entries which might be NaN
    entropy_map = torch.where(valid_mask, entropy_map, torch.tensor(0.0, device=q.device))
    row_entropy = entropy_map.sum(dim=-1)  # [B, H, S]
    # Average entropy over valid queries (rows)
    # For causal, first row has 1 valid, second 2...
    avg_entropy = row_entropy.mean(dim=-1)  # [B, H]

    # Sink: Probability of token 0
    sink_probs = probs[..., 0]  # [B, H, S]
    avg_sink = sink_probs.mean(dim=-1)  # [B, H]

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": avg_entropy,
        "sink_per_head": avg_sink,
        "max_global": max_per_head.max(),
        "mean_global": mean_per_head.mean(),
        "entropy_global": avg_entropy.mean(),
        "sink_global": avg_sink.mean(),
    }


def compute_qk_stats(q: torch.Tensor, k: torch.Tensor, causal: bool = True, use_triton: bool = True) -> dict:
    """
    Unified entry point. Input layout: [S, B, H, D] (Megatron core_attention convention).
    Permutes to [B, H, S, D] before dispatching to backend.
    """
    seq_len, batch, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape

    if num_q_heads != num_k_heads:
        heads_per_group = num_q_heads // num_k_heads
        k = k.repeat_interleave(heads_per_group, dim=2)

    # Permute [S, B, H, D] -> [B, H, S, D] for both backends
    q = q.permute(1, 2, 0, 3).contiguous()
    k = k.permute(1, 2, 0, 3).contiguous()

    if use_triton:
        return compute_qk_stats_triton(q, k, causal)

    return compute_qk_stats_pytorch(q, k, causal)
