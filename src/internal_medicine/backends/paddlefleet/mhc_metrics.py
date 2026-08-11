"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Paddle port of ``backends/megatron/mhc_metrics.py``. Pure, stateless tensor
helpers for the ``mhc_health`` monitor, operating on the three mappings a
``HyperConnectionModule`` produces per token (``n = num_residual_streams``):

- ``h_pre``  [..., n]     — stream aggregation gate (sigmoid)
- ``h_post`` [..., n]     — stream expansion gate (2 * sigmoid)
- ``h_res``  [..., n, n]  — Sinkhorn doubly-stochastic residual-mixing matrix

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe on a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import paddle

_EPS = 1e-12


def amax_gain(mat: paddle.Tensor, axis: int) -> paddle.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, meaned over tokens.

    The paper's worst-case gain bound. Which axis is "forward" depends on the
    matrix passed in: streams mix as ``out = h_res^T @ x``, so on ``h_res`` as
    stored ``axis=-2`` (column sums) is the forward gain and ``axis=-1`` the
    backward one; on an already-transposed matrix the two swap. Returns a 0-dim
    tensor.
    """
    sums = mat.sum(axis=axis)  # [..., n]
    return sums.abs().max(axis=-1).mean()  # 0-dim


def h_res_softmax_extrema(h_res_logits: paddle.Tensor, n: int) -> dict[str, paddle.Tensor]:
    """Extreme mixing weights of the row-wise softmax, before Sinkhorn.

    Saturation sentinel. ``min`` is the informative half: bounded by ``(0, 1/n]``
    with ``1/n`` = uniform row, falling as the row peaks, and it crosses fp32's
    smallest normal (``1.18e-38``) exactly when the model's own ``softmax``
    underflows and the within-row proportions are lost. ``max`` is capped at 1 by
    the row sum and pins at exactly 1.0 in fp32 from a within-row logit spread of
    ~18 onwards, so it only answers "saturated yes/no", not how far.

    What it diagnoses is **gradient freezing**, not NaN: a saturated row's softmax
    Jacobian ``diag(p) - p pᵀ`` is ~0, so ``mapping_proj`` / ``alpha_res`` stop
    receiving gradient and that layer's ``h_res`` freezes, while the finished
    ``h_res`` still reports row/column sums of ~1 and every other metric
    here reads healthy. (NaN was ruled out separately: both the native and cuTile
    Sinkhorn paths stay finite in fp32 even on ``-inf`` logits, because the
    ``eps`` in the normalization denominators caps amplification at ``1/eps``.)

    The softmax deliberately omits the model's ``+ eps`` (``_sinkhorn_normalize``
    in ``hyper_connection.py``), which would floor every entry at 1e-6 and clamp
    ``min`` there. The trade is that ``min`` reads exactly 0 past a spread of
    ~103 — past the alarm, and the model cannot tell 0 from 1e-44 either.

    Args:
        h_res_logits: ``[..., n*n]`` raw mixing logits, i.e. ``_compute_h``'s
            third return value — before the reshape and Sinkhorn projection.
        n: ``num_residual_streams``.

    Returns:
        ``h_res_softmax_min`` / ``h_res_softmax_max`` over every row of every
        token, as 0-dim tensors; one softmax serves both. No host sync.
    """
    logits = h_res_logits.detach().astype("float32").reshape([-1, n, n])
    weights = paddle.nn.functional.softmax(logits, axis=-1)
    return {
        "h_res_softmax_min": weights.min(),
        "h_res_softmax_max": weights.max(),
    }


def gate_stats(h: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()


def h_post_structure_stats(h_post: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Structure of the expansion gate across streams and across tokens.

    ``h_post_mean`` / ``h_post_std`` pool the token and stream axes together, so
    a shrinking mean cannot tell "all n streams went quiet" from "n-1 died and
    one carries everything". These two split the axes:

    ``h_post_stream_concentration`` — ``mean_t(max_i h_post / mean_i h_post)``,
    taken per token *before* averaging so it stays bounded by ``[1, n]``
    regardless of token-level scale: 1 = streams equally weighted, ``n`` = all
    mass on one stream (the ``num_residual_streams`` budget is unused).

    ``h_post_token_std`` — std over tokens of the per-token stream mean. The gate
    is a function of the input (``h = r · proj · α + bias``), so →0 means it
    stopped discriminating tokens and is effectively a constant scalar.

    Returns 0-dim GPU tensors; no host sync.
    """
    flat = h_post.detach().astype("float32")
    flat = flat.reshape([-1, flat.shape[-1]])  # [tokens, n]

    per_token_mean = flat.mean(axis=-1)  # [tokens]
    per_token_max = flat.max(axis=-1)  # [tokens]
    concentration = (per_token_max / paddle.clip(per_token_mean, min=_EPS)).mean()
    token_std = per_token_mean.std() if flat.shape[0] > 1 else paddle.zeros(())

    return {
        "h_post_stream_concentration": concentration,
        "h_post_token_std": token_std,
    }


def branch_residual_share(
    h_res: paddle.Tensor,
    original_residual: paddle.Tensor,
    h_post: paddle.Tensor,
    layer_output: paddle.Tensor,
    bias: paddle.Tensor | None = None,
) -> dict[str, paddle.Tensor]:
    """How much of the mHC update this layer wrote itself, per token.

    mHC propagates ``x_{l+1} = H_resᵀ x_l + H_postᵀ F(H_pre x_l)``: the first term
    only re-mixes what earlier layers wrote, the second is this layer's own
    contribution. The share

        ``branch_residual_share = b / (b + r)``,
        ``b = ‖H_postᵀ F(·)‖_F``, ``r = ‖H_resᵀ x_l‖_F``

    reads "is this layer still writing anything" — which ``h_post_mean`` cannot,
    since a shrinking gate may be offset by a growing ``F(·)``.

    The bounded form is deliberate: ``r`` really does hit 0 (an all-zero token in
    ``layer_0`` or an MTP layer's shifted slot), and the raw ratio ``b / r`` then
    pinned at the epsilon floor ``b / 1e-6 ≈ 1e6`` and hijacked the token mean
    and the derived ``global_*`` series. Here such a token reads 1.0 and every
    token contributes at most 1.0.

    Both norms are exact but materialise no ``[tokens, n, C]`` intermediate: the
    branch term is an outer product (``‖h_post_t ⊗ xb_t‖_F = ‖h_post_t‖₂·‖xb_t‖₂``)
    and the residual term contracts two ``[tokens, n, n]`` matrices —
    ``‖mixed_t‖²_F = Σ_{j,k} S[t,j,k]·G[t,j,k]`` with ``S`` the ``h_res``
    autocorrelation and ``G`` the Gram matrix of the n streams. Cost is
    ``C/n²`` times cheaper than the layer's own projections.

    Args:
        h_res: ``[..., n, n]`` Sinkhorn doubly-stochastic mixing matrix.
        original_residual: ``[..., n*C]`` incoming n-stream hidden states.
        h_post: ``[..., n]`` expansion gate.
        layer_output: ``[..., C]`` the sublayer output ``F(·)``.
        bias: optional ``[C]`` bias added to the sublayer output.

    Returns:
        ``branch_residual_share`` (token mean) and
        ``branch_residual_share_max`` (worst token), both in ``[0, 1]``.

    Note:
        The branch term is measured *before* dropout, so configs with
        ``hidden_dropout_prob > 0`` slightly over-estimate it; the pretraining
        configs here use 0.
    """
    n = int(h_post.shape[-1])
    xb = layer_output.detach().astype("float32")
    if bias is not None:
        xb = xb + bias.detach().astype("float32").reshape([1] * (xb.ndim - 1) + [-1])
    xb = xb.reshape([-1, xb.shape[-1]])  # [tokens, C]

    gate = h_post.detach().astype("float32").reshape([-1, n])  # [tokens, n]
    branch_norm = gate.norm(axis=-1) * xb.norm(axis=-1)  # [tokens]

    streams = original_residual.detach().astype("float32")
    streams = streams.reshape([-1, n, streams.shape[-1] // n])  # [tokens, n, C]
    res = h_res.detach().astype("float32").reshape([-1, n, n])  # [tokens, n, n]

    gram = paddle.einsum("tjc,tkc->tjk", streams, streams)  # [tokens, n, n]
    mix = paddle.einsum("tji,tki->tjk", res, res)  # [tokens, n, n]
    residual_sq = (gram * mix).sum(axis=[-2, -1])  # [tokens]
    residual_norm = paddle.sqrt(paddle.clip(residual_sq, min=0.0))

    # eps only guards the both-terms-zero token (share 0); it does not bound the
    # value, which is why this form is safe where the raw ratio was not.
    share = branch_norm / paddle.clip(branch_norm + residual_norm, min=_EPS)
    return {
        "branch_residual_share": share.mean(),
        "branch_residual_share_max": share.max(),
    }
