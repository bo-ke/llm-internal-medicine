"""mHC (Manifold-Constrained Hyper-Connections) metric compute functions.

Pure, stateless tensor -> tensor helpers for the ``mhc_health`` monitor. They
operate on the three mappings a ``HyperConnectionModule`` produces per token
(``n = num_residual_streams``, ``s`` = sequence, ``b`` = micro-batch):

- ``h_pre``  [s, b, n]      — stream aggregation gate (sigmoid)
- ``h_post`` [s, b, n]      — stream expansion gate (2*sigmoid)
- ``h_res``  [s, b, n, n]   — Sinkhorn doubly-stochastic residual-mixing matrix

The ``amax_gain`` diagnostic follows the mHC paper: the max-abs sum along the
operator's contracted axis bounds the worst-case forward expansion, the other
axis the backward one. For a single doubly-stochastic ``h_res`` both sit at
~1.0; on the *composite* mapping (cumulative product of ``h_res`` across layers)
they drift away from 1.0 with depth, flagging residual-stream amplification.

All functions return 0-dim GPU tensors and never sync the host (no ``.item()`` /
``.cpu()``), so they are safe to call from a forward hot path. See
``.claude/skills/monitor-hook-perf-rules``.
"""

import torch

_EPS = 1e-12


def amax_gain(mat: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-token max-abs {row|col} sum of a batched ``[..., n, n]`` matrix, meaned over tokens.

    The paper's worst-case gain bound. Which axis is "forward" depends on the
    matrix passed in: streams mix as ``out = h_res^T @ x`` (see
    ``HyperConnectionModule.apply_h_res``), so on ``h_res`` as stored ``dim=-2``
    (column sums) is the forward gain and ``dim=-1`` the backward one; on an
    already-transposed matrix the two swap.

    ``sum(dim)`` collapses one ``n``-axis to ``[..., n]``; ``abs().amax(-1)``
    takes the worst stream per token; ``mean()`` averages over all tokens.
    Returns a 0-dim tensor on ``mat``'s device/dtype.
    """
    sums = mat.sum(dim=dim)  # [..., n]
    return sums.abs().amax(dim=-1).mean()  # 0-dim


def gate_stats(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and (unbiased) std of a gate tensor ``h`` over all elements.

    Returns two 0-dim tensors ``(mean, std)``; no host sync.
    """
    return h.mean(), h.std()


def h_res_logits_extrema(h_res_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Min/max of the raw residual-mixing logits before Sinkhorn."""
    logits = h_res_logits.detach().float()
    return {
        "h_res_logits_min": logits.min(),
        "h_res_logits_max": logits.max(),
    }


def h_post_structure_stats(h_post: torch.Tensor) -> dict[str, torch.Tensor]:
    """Structure of the expansion gate across streams and across tokens."""
    flat = h_post.detach().float()
    flat = flat.reshape(-1, flat.shape[-1])  # [tokens, n]

    per_token_mean = flat.mean(dim=-1)  # [tokens]
    per_token_max = flat.amax(dim=-1)  # [tokens]
    concentration = (per_token_max / per_token_mean.clamp(min=_EPS)).mean()
    token_std = per_token_mean.std() if flat.shape[0] > 1 else torch.zeros((), device=flat.device, dtype=flat.dtype)

    return {
        "h_post_stream_concentration": concentration,
        "h_post_token_std": token_std,
    }


def branch_residual_share(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    layer_output: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
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
        ``hidden_dropout > 0`` slightly over-estimate it; the pretraining
        configs here use 0.
    """
    n = int(h_post.shape[-1])
    xb = layer_output.detach().float()
    if bias is not None:
        xb = xb + bias.detach().float().reshape(*([1] * (xb.dim() - 1)), -1)
    xb = xb.reshape(-1, xb.shape[-1])  # [tokens, C]

    gate = h_post.detach().float().reshape(-1, n)  # [tokens, n]
    branch_norm = gate.norm(dim=-1) * xb.norm(dim=-1)  # [tokens]

    streams = original_residual.detach().float()
    streams = streams.reshape(-1, n, streams.shape[-1] // n)  # [tokens, n, C]
    res = h_res.detach().float().reshape(-1, n, n)  # [tokens, n, n]

    gram = torch.einsum("tjc,tkc->tjk", streams, streams)  # [tokens, n, n]
    mix = torch.einsum("tji,tki->tjk", res, res)  # [tokens, n, n]
    residual_sq = (gram * mix).sum(dim=(-2, -1))  # [tokens]
    residual_norm = residual_sq.clamp(min=0.0).sqrt()

    # eps only guards the both-terms-zero token (share 0); it does not bound the
    # value, which is why this form is safe where the raw ratio was not.
    share = branch_norm / (branch_norm + residual_norm).clamp(min=_EPS)
    return {
        "branch_residual_share": share.mean(),
        "branch_residual_share_max": share.max(),
    }
