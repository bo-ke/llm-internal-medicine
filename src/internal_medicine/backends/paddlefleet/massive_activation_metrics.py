"""
Massive Activation Metrics for PaddleFleet.

Monitors massive activations (extreme outlier values in hidden state channels)
as characterized by Sun et al. (2026) arXiv:2603.05498.

Core metrics:
1. Channel Max — absolute maximum activation across all channels
2. Channel Max Ratio — ratio of max channel to median channel (outlier severity)
3. Massive Activation Channel Count — channels exceeding a magnitude threshold
4. Top-K Channel Norm — L2 norm of the top-K largest channels
5. Post-Norm Sparsity — fraction of near-zero entries after RMSNorm
6. Post-Norm Cosine Stability — cosine similarity of normalized representations
"""

import paddle


def compute_per_channel_max(hidden_states: paddle.Tensor) -> paddle.Tensor:
    """Per-channel maximum absolute activation for [..., H] hidden states."""
    h = hidden_states.reshape([-1, hidden_states.shape[-1]]).astype("float32")
    return h.abs().max(axis=0)


def summarize_per_channel_max(
    per_channel_max: paddle.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
) -> dict[str, float]:
    """Derive scalar massive-activation metrics from per-channel maxima."""
    channel_max = float(per_channel_max.max())
    channel_median = float(paddle.median(per_channel_max))
    channel_max_ratio = channel_max / max(channel_median, 1e-8)

    threshold = channel_median * threshold_multiplier
    massive_act_channel_count = float((per_channel_max > threshold).astype("float32").sum())

    topk_vals, _ = paddle.topk(per_channel_max, min(k, per_channel_max.shape[0]))
    topk_channel_norm = float(topk_vals.norm())

    return {
        "channel_max": channel_max,
        "channel_max_ratio": channel_max_ratio,
        "massive_act_channel_count": massive_act_channel_count,
        "topk_channel_norm": topk_channel_norm,
    }


def compute_pre_norm_metrics(
    hidden_states: paddle.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
) -> dict[str, float]:
    """All pre-norm massive activation metrics in one pass.

    Computes per_channel_max once and derives all dependent metrics.

    Args:
        hidden_states: [..., H] post-residual hidden states.
        threshold_multiplier: multiplier on median to define "spike" threshold.
        k: number of top channels for top-K norm.

    Returns:
        Dict with channel_max, channel_max_ratio, massive_act_channel_count, topk_channel_norm.
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states),
        threshold_multiplier=threshold_multiplier,
        k=k,
    )


def compute_post_norm_sparsity(
    normalized_states: paddle.Tensor,
    epsilon: float = 0.01,
) -> float:
    """Fraction of near-zero entries in post-RMSNorm hidden states.

    Args:
        normalized_states: [..., H] hidden states AFTER RMSNorm.
        epsilon: threshold below which a value is considered "near-zero".

    Returns:
        Fraction of entries with |x| < epsilon.
    """
    h = normalized_states.reshape([-1]).astype("float32")
    return float((h.abs() < epsilon).astype("float32").mean())


def compute_post_norm_cosine_stability(
    normalized_states: paddle.Tensor,
    num_sample_pairs: int = 256,
) -> float:
    """Cosine similarity among token representations after normalization.

    Args:
        normalized_states: [..., H] post-RMSNorm hidden states.
        num_sample_pairs: number of random pairs to sample.

    Returns:
        Mean pairwise cosine similarity (sampled).
    """
    h = normalized_states.reshape([-1, normalized_states.shape[-1]]).astype("float32")
    num_tokens = h.shape[0]

    if num_tokens < 2:
        return 1.0

    n_pairs = min(num_sample_pairs, num_tokens * (num_tokens - 1) // 2)
    idx_a = paddle.randint(0, num_tokens, [n_pairs])
    idx_b = paddle.randint(0, num_tokens - 1, [n_pairs])
    idx_b = idx_b + (idx_b >= idx_a).astype("int64")

    vec_a = h[idx_a]
    vec_b = h[idx_b]

    cosine = paddle.nn.functional.cosine_similarity(vec_a, vec_b, axis=-1)
    return float(cosine.mean())
