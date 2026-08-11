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


def h_res_logits_extrema(h_res_logits: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Min/max of the raw residual-mixing logits before Sinkhorn."""
    logits = h_res_logits.detach().astype("float32")
    return {
        "h_res_logits_min": logits.min(),
        "h_res_logits_max": logits.max(),
    }


def gate_stats(h: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements."""
    return h.mean(), h.std()


def h_post_structure_stats(h_post: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """Structure of the expansion gate across streams and across tokens."""
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
    """How much of the mHC update this layer wrote itself, per token?

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
