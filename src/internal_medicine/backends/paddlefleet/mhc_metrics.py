"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Paddle port of ``backends/megatron/mhc_metrics.py``. Pure, stateless tensor
helpers for the ``mhc_health`` monitor. They operate on the three mappings a
``HyperConnectionModule`` produces per token:

- ``h_pre``  [..., n]     — stream aggregation gate (sigmoid)
- ``h_post`` [..., n]     — stream expansion gate (2 * sigmoid)
- ``h_res``  [..., n, n]  — Sinkhorn doubly-stochastic residual-mixing matrix

``n = num_residual_streams``.

The ``amax_gain`` diagnostic follows the mHC paper: the max-abs **row** sum of a
mixing matrix bounds the worst-case forward-pass expansion, and the max-abs
**column** sum bounds the backward-pass expansion. For a single doubly-stochastic
``h_res`` both sit at ~1.0; on the *composite* mapping (cumulative product of
``h_res`` across layers) they drift away from 1.0 with depth, flagging
residual-stream amplification.

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe to call from a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import paddle

_EPS = 1e-12


def amax_gain(mat: paddle.Tensor, axis: int) -> paddle.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, meaned over tokens.

    ``axis=-1`` sums over columns -> per-row sums (forward gain); ``axis=-2``
    sums over rows -> per-column sums (backward gain). ``sum(axis)`` collapses
    one ``n``-axis to ``[..., n]``; ``abs().max(axis=-1)`` takes the worst
    stream per token; ``mean()`` averages over all tokens.

    Returns a 0-dim tensor on ``mat``'s device/dtype (fp32 from
    ``compute_mappings``).
    """
    sums = mat.sum(axis=axis)  # [..., n]
    return sums.abs().max(axis=-1).mean()  # 0-dim


def gate_stats(h: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()


def h_post_structure_stats(h_post: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Structure of the expansion gate across streams and across tokens.

    ``h_post_mean`` / ``h_post_std`` pool the token axis and the stream axis
    together, so a shrinking mean cannot tell "all n streams went quiet" apart
    from "n-1 streams died and one still carries everything". These two split
    the axes:

    ``h_post_stream_concentration`` — ``mean_t(max_i h_post[t,i] / mean_i h_post[t,i])``.
    Computed per token *before* averaging, so it is bounded by ``[1, n]``
    regardless of token-level scale: 1 = all streams weighted equally,
    ``n`` = all the mass on a single stream (the layer degenerated to a plain
    single-stream residual and the ``num_residual_streams`` budget is unused).

    ``h_post_token_std`` — std over tokens of the per-token stream mean. The
    gate is a function of the input (``h = r · proj · α + bias``), so this is
    its input sensitivity: →0 means the gate stopped discriminating tokens and
    has effectively become a constant scalar.

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

    mHC propagates ``x_{l+1} = H_resᵀ x_l + H_postᵀ F(H_pre x_l)``: the first
    term only re-mixes what the previous layers wrote, the second is this
    layer's own contribution. The share

        ``branch_residual_share = b / (b + r)``,
        ``b = ‖H_postᵀ F(·)‖_F``, ``r = ‖H_resᵀ x_l‖_F``

    is therefore the direct read on "is this layer still writing anything".
    ``h_post_mean`` alone cannot answer that: a shrinking gate may be offset by
    a growing ``F(·)``.

    The bounded form is deliberate. The raw ratio ``b / r`` is unbounded and
    ``r`` really does hit 0 — an all-zero token (padding, or the shifted slot of
    an MTP layer) has all ``n`` streams at zero, and only ``layer_0`` and MTP
    layers can see one, since deeper layers always carry earlier writes. Dividing
    by ``r`` then pinned the value at the epsilon floor (``b / 1e-6 ≈ 1e6``),
    which blew up the token mean and the derived ``global_*`` series along with
    it. Here such a token simply reads 1.0, and every token contributes at most
    1.0 to the mean, so a handful of them can no longer hijack the series.

    Both norms are computed exactly, but without materialising any
    ``[tokens, n, C]`` intermediate:

    - the branch term is an outer product, so
      ``‖h_post_t ⊗ xb_t‖_F = ‖h_post_t‖₂ · ‖xb_t‖₂``;
    - for the residual term, with ``mixed_t[i] = Σ_j h_res[t,j,i] · x_l[t,j]``,
      ``‖mixed_t‖²_F = Σ_{j,k} S[t,j,k] · G[t,j,k]`` where
      ``S = Σ_i h_res[t,j,i] h_res[t,k,i]`` and ``G`` is the Gram matrix of the
      n streams. Both are ``[tokens, n, n]``, i.e. negligible memory, and the
      cost is ``O(tokens · n² · C)`` — for ``n = 4`` that is ``C/n²`` times
      cheaper than the layer's own projections.

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
        The branch term is measured *before* dropout. Configs that run
        ``hidden_dropout_prob > 0`` therefore see a slight over-estimate of the
        branch magnitude at train time; the pretraining configs here use 0.
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
