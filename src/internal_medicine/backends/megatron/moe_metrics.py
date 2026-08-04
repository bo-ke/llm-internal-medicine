"""
MoE Health Metrics Computation Functions (Simplified).

Core metrics:
1. Router Entropy - 路由熵
2. Router Score Sum - TopK选择的score求和
3. Bias-Affinity Correlation - 使用Jaccard相似度
4. Expert Norms - 专家权重L2范数
5. Shared/Routed Ratio - Shared Expert与Routed Expert的比例

注意: 所有指标只计算本地值，EP 聚合在 training_logs 层面统一处理
(通过 gather_object_list 收集所有卡的结果，然后按 mean/max/min 聚合)
"""

import torch


def compute_router_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    """
    计算 Router Entropy (路由熵).

    H = -sum(p * log(p))

    Args:
        router_probs: Softmax probabilities [tokens, num_experts] (scores_for_aux_loss)

    Returns:
        entropy_mean: 本地 batch 的平均熵 (Scalar Tensor)
    """
    probs = router_probs.float().clamp(min=1e-10)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [tokens]
    return entropy.mean()


def compute_topk_score_sum(scores: torch.Tensor, topk: int) -> dict[str, torch.Tensor]:
    """
    计算 Router TopK Score Sum.

    Args:
        scores: Router scores [tokens, num_experts]，softmax/sigmoid 后的分数
        topk: TopK 值

    Returns:
        Dict with score_sum statistics
    """
    topk_scores, _ = scores.float().topk(topk, dim=-1)  # [tokens, topk]
    score_sum = topk_scores.sum(dim=-1)  # [tokens]
    return {
        "score_sum_mean": score_sum.mean(),
        "score_sum_min": score_sum.min(),
        "score_sum_max": score_sum.max(),
    }


def compute_bias_affinity_jaccard(
    routing_map_before_bias: torch.Tensor,
    routing_map_after_bias: torch.Tensor,
    num_experts: int | None = None,
) -> torch.Tensor:
    """
    计算 Bias-Affinity Jaccard 相似度.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        routing_map_before_bias: Bias前的routing map [tokens, topk] (index form)
            or [tokens, num_experts] (one-hot/bool form).
        routing_map_after_bias: Bias后的routing map.
        num_experts: required when routing_map is in index form (last dim == topk).
            Pass router.num_experts. Avoids a D2H sync on the hot path.

    Returns:
        jaccard: Jaccard相似度 (0-1)
    """
    if routing_map_before_bias.dim() == 2:
        is_onehot = routing_map_before_bias.shape[-1] != 1 and routing_map_before_bias.dtype in (
            torch.bool,
            torch.uint8,
        )

        if not is_onehot and num_experts is not None and routing_map_before_bias.dtype not in (torch.bool, torch.uint8):
            num_tokens, topk = routing_map_before_bias.shape
            before_onehot = torch.zeros(
                num_tokens, num_experts, device=routing_map_before_bias.device, dtype=torch.bool
            )
            after_onehot = torch.zeros(num_tokens, num_experts, device=routing_map_after_bias.device, dtype=torch.bool)

            for k in range(topk):
                before_onehot.scatter_(1, routing_map_before_bias[:, k : k + 1].long(), True)
                after_onehot.scatter_(1, routing_map_after_bias[:, k : k + 1].long(), True)

            routing_map_before_bias = before_onehot
            routing_map_after_bias = after_onehot

    before = routing_map_before_bias.bool()
    after = routing_map_after_bias.bool()

    intersection = (before & after).float().sum()
    union = (before | after).float().sum()
    return intersection / union.clamp(min=1e-8)


def compute_load_balance_ratios(tokens_per_expert: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """专家负载极值比值, 基于已 all-reduce 的 tokens-per-expert.

    输入是 mcore 在 get_updated_expert_bias 里跨 TPxCPxDP all-reduce 之后的
    全局 per-expert token 计数 (每张卡上都相同), 因此这里算出的比值本身就是全局
    正确值, 无需 monitor 侧 collective, 也无需 training_logs 跨 rank 聚合.

    对每一层 (行) 计算:
      - load_max_min_ratio    = #tokens(most-routed) / #tokens(least-routed)
      - load_max_median_ratio = #tokens(most-routed) / #tokens(median expert)
      - load_cv               = std(counts) / mean(counts)  (变异系数, 完全均衡=0)

    median 用 torch.median (偶数个专家时取下中位, 即某个真实专家的计数, 不做插值).
    极值比值分母 clamp(min=1.0) 防止死专家 (count=0) 产生 inf/NaN. CV 用 population
    std (unbiased=False) 除以 mean, mean clamp(min=1.0) 防止空批次除零. 全程 GPU
    tensor, 不在 hot path 上运行 (调用点在 finalize_model_grads, forward hook 之外).

    Args:
        tokens_per_expert: reduced counts. [num_layers, num_experts] (stacked)
            or [num_experts] (single layer). Last dim is the expert axis.

    Returns:
        Dict of per-layer 0-dim/1-dim tensors keyed by metric name (shape matches
        the leading layer dim), or None if the expert axis has < 2 experts.
    """
    counts = tokens_per_expert.to(torch.float32)
    if counts.dim() == 1:
        counts = counts.unsqueeze(0)
    if counts.dim() != 2 or counts.shape[-1] < 2:
        return None

    max_count = counts.max(dim=-1).values
    min_count = counts.min(dim=-1).values
    median_count = counts.median(dim=-1).values
    # Population std (unbiased=False): with the global per-expert totals this is
    # the exact CV, not a sample estimate. Balanced load -> std=0 -> CV=0.
    mean_count = counts.mean(dim=-1)
    std_count = counts.std(dim=-1, unbiased=False)

    return {
        "load_max_min_ratio": max_count / min_count.clamp(min=1.0),
        "load_max_median_ratio": max_count / median_count.clamp(min=1.0),
        "load_cv": std_count / mean_count.clamp(min=1.0),
    }


def compute_expert_norms(expert_weights: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    计算 Expert Norms (专家权重L2范数).

    Args:
        expert_weights: 每个本地专家的权重张量列表

    Returns:
        Dict with norm statistics (本地专家)
    """
    if not expert_weights:
        return {
            "expert_norm_mean": torch.tensor(0.0),
            "expert_norm_std": torch.tensor(0.0),
            "expert_norm_min": torch.tensor(0.0),
            "expert_norm_max": torch.tensor(0.0),
        }

    norms = torch.stack([w.float().norm() for w in expert_weights])
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std() if norms.numel() > 1 else torch.tensor(0.0, device=norms.device),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


def compute_shared_expert_norm(shared_expert_weights: list[torch.Tensor]) -> torch.Tensor:
    """
    计算 SharedExpert 的 L2 Norm.

    Args:
        shared_expert_weights: Shared Expert 的权重张量列表

    Returns:
        shared_norm: L2 Norm
    """
    if not shared_expert_weights:
        return torch.tensor(0.0)

    all_params = torch.cat([w.flatten() for w in shared_expert_weights])
    return all_params.float().norm()


def compute_combine_coef_sharpness(probs: torch.Tensor, topk: int) -> torch.Tensor:
    """Per-token sharpness of the k-way combine coefficients, averaged over tokens.

    ``probs`` is the router's output: ``[num_tokens, num_experts]``, sparse, with exactly
    ``topk`` non-zero entries per token — those non-zeros ARE the coefficients the
    dispatcher uses to weight expert outputs during combine. For each token::

        ratio_t = max(coef_t) / median(coef_t)      # over the topk coefficients

    and the returned scalar is ``mean_t(ratio_t)``.

    Read it as how peaked the combine is: ``1.0`` means the k experts contribute equally
    (a true k-way blend), while a large value means one expert dominates and the MoE is
    effectively behaving as top-1 despite paying for topk. Because the reduction over
    tokens is a MEAN, this reports the typical token rather than the worst one — a few
    confidently-routed tokens do not move it, a broad shift toward peakiness does.

    ``topk == 1`` makes max == median by construction, so the ratio is identically 1 and
    the metric carries no information; callers should skip it in that case.

    Note ``torch.median`` returns the LOWER middle element for an even ``k`` (not the
    average of the two middles), so at ``topk == 2`` this is exactly ``max/min``, and at
    ``topk == 8`` the denominator is the 4th-smallest coefficient. That matches the
    convention ``massive_act/channel_max_ratio`` and ``latent_combine_*`` already use.

    Returns a 0-dim GPU tensor — no host sync (perf-rules Rule 1).
    """
    p = probs.detach().float().reshape(-1, probs.shape[-1])
    if p.shape[0] == 0 or topk < 1:
        return torch.zeros((), device=probs.device)
    k = min(int(topk), p.shape[-1])
    coefs = p.topk(k, dim=-1).values  # [num_tokens, k] — the combine weights
    per_token = coefs.amax(dim=-1) / coefs.median(dim=-1).values.clamp(min=1e-8)
    return per_token.mean()


def compute_latent_combine_stats(hidden_states: torch.Tensor, eps: float | None = None) -> dict[str, torch.Tensor]:
    """Magnitude stats for the k-way-combined expert output, in LATENT dim.

    The measured tensor is what ``token_dispatcher.combine_postprocess`` RETURNS inside
    ``MoELayer.postprocess`` — the raw sum of the ``topk`` expert outputs weighted by
    their router probs, before anything downstream touches it. That is deliberately
    upstream of ``fc2_latent_proj``: a model may put an RMSNorm between the combine and
    the latent up-projection, which would pin the RMS to ~1 and make this metric
    constant. Measuring at the combine keeps it a true magnitude signal.

    Args:
        hidden_states: ``combine_postprocess`` output, ``[..., latent_size]``.
        eps: the RMSNorm epsilon the downstream latent norm uses
            (``config.layernorm_epsilon``). When given, ``latent_eps_ratio`` is emitted.
            Pass ``None`` on models with no latent norm.

    Returns:
        - ``latent_combine_rms``: RMS over all elements, the overall scale of the
          combined output.
        - ``latent_combine_channel_max_median_ratio``: ``max_c / median_c`` over the
          per-channel maximum absolute activation. 1.0 means every latent channel
          peaks equally; a large value means a few channels dominate (the
          massive-activation signature). The MEDIAN denominator matches
          ``massive_act/channel_max_ratio``, so the two are directly comparable, and it
          is robust to the outlier channels the metric is meant to detect — a mean
          denominator is itself inflated by the spike, which damps the very signal
          being measured.
        - ``latent_eps_ratio`` (only when ``eps`` is given): ``eps / (mean(h**2) + eps)``,
          i.e. how much of the RMSNorm denominator is the epsilon floor rather than the
          signal. ~0 means the activations dominate and the norm behaves as intended;
          ->1 means ``eps`` dominates, so the norm is dividing by a constant instead of
          the token's own scale — the "epsilon swamps the hidden" failure. Computed
          per-token then averaged, because the norm applies per token: a global
          ``mean(h**2)`` would let large-norm tokens mask the small-norm ones that are
          actually being floored.

    The first two are MAX-aggregated (per-layer -> global, and across ranks): they exist
    to catch magnitude blow-up, so the worst observation is the informative one and a
    mean would average a spike away. ``latent_eps_ratio`` is MEAN-aggregated — it is
    already a per-token average and describes the typical token.

    All are 0-dim GPU tensors — no host sync (perf-rules Rule 1).
    """
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    if h.shape[0] == 0:
        zero = torch.zeros((), device=hidden_states.device)
        out = {"latent_combine_rms": zero, "latent_combine_channel_max_median_ratio": zero}
        if eps is not None:
            out["latent_eps_ratio"] = zero
        return out
    per_channel_max = h.abs().amax(dim=0)
    metrics = {
        "latent_combine_rms": h.square().mean().sqrt(),
        "latent_combine_channel_max_median_ratio": per_channel_max.max() / per_channel_max.median().clamp(min=1e-8),
    }
    if eps is not None:
        mean_sq = h.square().mean(dim=-1)
        metrics["latent_eps_ratio"] = (eps / (mean_sq + eps)).mean()
    return metrics


def compute_shared_routed_ratio(
    shared_norm: torch.Tensor,
    routed_norm_mean: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Shared/Routed Ratio.

    Args:
        shared_norm: SharedExpert 的 L2 Norm
        routed_norm_mean: Routed Experts 的平均 L2 Norm

    Returns:
        ratio: Shared/Routed Ratio
    """
    return shared_norm / routed_norm_mean.clamp(min=1e-8)
