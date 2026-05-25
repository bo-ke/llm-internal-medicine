"""
Massive Activation Metrics Computation Functions.

Monitors massive activations (extreme outlier values in hidden state channels)
as characterized by Sun et al. (2026) "The Spike, the Sparse and the Sink:
Anatomy of Massive Activations and Attention Sinks" (arXiv:2603.05498).

Core metrics:
1. Channel Max — absolute maximum activation across all channels
2. Channel Max Ratio — ratio of max channel to median channel (outlier severity)
3. Massive Activation Channel Count — number of channels exceeding a magnitude threshold
4. Top-K Channel Norm — L2 norm of the top-K largest channels
5. Post-Norm Sparsity — fraction of near-zero entries after RMSNorm (sparsification)
6. Post-Norm Cosine Stability — cosine similarity of normalized representations
   across tokens (near-constant vector detection)

All metrics compute local values only; cross-rank aggregation is handled
by training_logs.gather_and_aggregate().

Reference:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks. arXiv:2603.05498.
"""

import torch


def compute_per_channel_max(hidden_states: torch.Tensor) -> torch.Tensor:
    """Per-channel maximum absolute activation for [..., H] hidden states."""
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return h.abs().max(dim=0).values


def summarize_per_channel_max(
    per_channel_max: torch.Tensor,
    threshold_multiplier: float = 100.0,
    k: int = 3,
) -> dict[str, torch.Tensor]:
    """Derive scalar massive-activation metrics from per-channel maxima."""
    channel_max = per_channel_max.max()
    channel_median = per_channel_max.median()
    channel_max_ratio = channel_max / channel_median.clamp(min=1e-8)
    threshold = channel_median * threshold_multiplier
    topk_vals = per_channel_max.topk(min(k, per_channel_max.shape[0])).values

    return {
        "channel_max": channel_max,
        "channel_median": channel_median,
        "channel_max_ratio": channel_max_ratio,
        "massive_act_channel_count": (per_channel_max > threshold).sum().float(),
        "topk_channel_norm": topk_vals.norm(),
    }


def compute_channel_max(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-channel maximum absolute activation statistics.

    Tracks the "rise–plateau–fall" lifecycle of massive activations across layers.
    A sudden spike in channel_max indicates a step-up block is injecting outliers.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
            (Flattened to [*, H] internally)

    Returns:
        Dict with:
            channel_max: max absolute value across all positions and channels (scalar)
            channel_median: median of per-channel max absolute values (scalar)
            channel_max_ratio: channel_max / channel_median (outlier severity)
    """
    return summarize_per_channel_max(compute_per_channel_max(hidden_states))


def compute_massive_activation_channel_count(
    hidden_states: torch.Tensor,
    threshold_multiplier: float = 100.0,
) -> torch.Tensor:
    """Count channels with activations exceeding a dynamic threshold.

    The threshold is set relative to the median channel magnitude:
        threshold = median_channel_max × threshold_multiplier

    This captures Property (ii) from Sun et al. (2026): massive activations
    are confined to a small subset of channels.

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        threshold_multiplier: multiplier on median to define "spike" threshold.

    Returns:
        Scalar tensor: number of channels exceeding the threshold.
    """
    return summarize_per_channel_max(
        compute_per_channel_max(hidden_states),
        threshold_multiplier=threshold_multiplier,
    )["massive_act_channel_count"]


def compute_topk_channel_norm(
    hidden_states: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """L2 norm of the top-K largest channel activations.

    Tracks the magnitude of the most extreme channels. In models with massive
    activations, this should show the "rise–plateau–fall" pattern across layers
    (Figure 1 in Sun et al. 2026).

    Args:
        hidden_states: [S, B, H] or [B, S, H] post-residual hidden states.
        k: number of top channels to include.

    Returns:
        Scalar tensor: L2 norm of the top-K per-channel-max values.
    """
    return summarize_per_channel_max(compute_per_channel_max(hidden_states), k=k)["topk_channel_norm"]


def compute_post_norm_sparsity(
    normalized_states: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Fraction of near-zero entries in post-RMSNorm hidden states.

    After normalization, spike tokens become sparse vectors where non-spike
    channels are suppressed to near-zero (Equation 24, Sun et al. 2026).
    High sparsity indicates the model is creating "implicit parameters" via
    the normalization-spike interaction.

    Args:
        normalized_states: [S, B, H] or [B, S, H] hidden states AFTER RMSNorm.
        epsilon: threshold below which a value is considered "near-zero".

    Returns:
        Scalar tensor: fraction of entries with |x| < epsilon.
    """
    h = normalized_states.reshape(-1).float()
    return (h.abs() < epsilon).float().mean()


def compute_post_norm_cosine_stability(
    normalized_states: torch.Tensor,
    num_sample_pairs: int = 256,
) -> torch.Tensor:
    """Cosine similarity among token representations after normalization.

    Near-constant post-norm representations (cosine → 1.0) indicate that
    normalization has collapsed diverse spike tokens into identical vectors
    (Figure 5, Sun et al. 2026). This is a precondition for attention sinks.

    Args:
        normalized_states: [S, B, H] post-RMSNorm hidden states.
        num_sample_pairs: number of random pairs to sample for efficiency.

    Returns:
        Scalar tensor: mean pairwise cosine similarity (sampled).
    """
    # Flatten to [num_tokens, hidden_dim]
    h = normalized_states.reshape(-1, normalized_states.shape[-1]).float()
    num_tokens = h.shape[0]

    if num_tokens < 2:
        return torch.tensor(1.0, device=h.device)

    # Sample random pairs
    n_pairs = min(num_sample_pairs, num_tokens * (num_tokens - 1) // 2)
    idx_a = torch.randint(0, num_tokens, (n_pairs,), device=h.device)
    idx_b = torch.randint(0, num_tokens - 1, (n_pairs,), device=h.device)
    # Avoid same index
    idx_b = idx_b + (idx_b >= idx_a).long()

    vec_a = h[idx_a]
    vec_b = h[idx_b]

    cosine = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=-1)
    return cosine.mean()
