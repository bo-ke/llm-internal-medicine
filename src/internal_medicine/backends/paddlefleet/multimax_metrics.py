"""MultiMax (SegLU lm_head) output-distribution metrics.

Implements the sparsity / multi-modality metrics of

    Zhou, Fritz, Keuper. *MultiMax: Sparse and Multi-Modal Attention Learning*.
    ICML 2024 (arXiv:2406.01189), Definitions 3.2 and 3.3

plus predictive entropy and top-k probability mass, computed on the LM-head
output distribution.

Definitions, with ``phi`` the output distribution (softmax of the modulated
logits) and ``eps`` a relevance threshold on the logits ``x``:

    Multi-modality (Def 3.2), over the N entries with eps < x_n < x_max:
        M(x) = 1 - (1/N) * sum_n (phi(x)_max - phi(x)_n)

    Sparsity (Def 3.3), over the L entries with x_l < eps:
        S(x) = (1/L) * sum_l exp((s - phi(x)_l) / s - 1)

``S`` simplifies exactly: ``(s - p)/s - 1 == -p/s``, and with the paper's
suggested reference ``s = phi(x)_min`` the ratio is shard-safe and
scale-free, ``p_l / s = exp(x_l - x_min)``. That form needs no probability
underflow guard, which matters at LM-head vocab sizes where ``phi_min``
rounds to 0 in fp32.

Hot-path discipline (``.claude/skills/monitor-hook-perf-rules``): every
function here returns 0-dim GPU tensors and never syncs to host.
"""

import math

import paddle

_EXP_CLAMP = 80.0  # exp(80) is finite in fp32; exp(-exp(80)) underflows to 0


def apply_seglu(logits: paddle.Tensor, ranges: paddle.Tensor, ts: paddle.Tensor) -> paddle.Tensor:
    """Mirror of PaddleFleet ``models/gpt/lm_head.py:SegLU``.

    Kept as a local copy on purpose: the fused-CE path never materializes
    logits, so the monitor has to reproduce the modulation on its own sampled
    tile. ``ranges``/``ts`` are the head's own ``[4]`` parameters, so this
    matches the training math exactly (identity while both are zero-init).
    """
    relu = paddle.nn.functional.relu
    x = logits.astype("float32")
    r = ranges.astype("float32")
    t = ts.astype("float32")
    # Every term is a function of the *input* x (not of the accumulator), which
    # is what upstream does; folding them sequentially would change the math.
    out = x + t[0] * relu(r[0] - x)
    out = out + t[1] * relu(x - r[1])
    out = out + t[2] * relu(r[2] - x) ** 2
    return out + t[3] * relu(x - r[3]) ** 2


def _relevance_threshold(
    log_z: paddle.Tensor,
    vocab_size: int,
    prob_eps: float | None,
    logit_eps: float | None,
) -> paddle.Tensor:
    """Per-row logit threshold ``eps`` for "is this entry relevant".

    The paper leaves ``eps`` free ("any reasonable threshold"). A fixed logit
    value is not meaningful for an LM head whose logit scale drifts during
    training, so the default is expressed in probability space and mapped back
    through the row's own log-partition: ``eps = log(prob_eps) + log_z``. The
    map is monotone, so the definition is unchanged. ``prob_eps = 1/V`` means
    "an entry is relevant once it beats the uniform distribution".
    """
    if logit_eps is not None:
        return paddle.full_like(log_z, float(logit_eps))
    p_eps = (1.0 / vocab_size) if prob_eps is None else float(prob_eps)
    return math.log(p_eps) + log_z


def compute_distribution_metrics(
    logits: paddle.Tensor,
    vocab_size: int | None = None,
    prob_eps: float | None = None,
    logit_eps: float | None = None,
    topk: int = 10,
) -> dict[str, paddle.Tensor]:
    """Token-mean distribution metrics for one ``[rows, vocab]`` logits tile.

    ``logits`` must cover the **full** vocab (the caller all-gathers
    vocab-parallel shards first) and must already carry the SegLU modulation.
    ``vocab_size`` defaults to the tile's last dim; pass it explicitly only if
    the tile is a strict subset of the vocab.

    Returns 0-dim fp32 GPU tensors: ``entropy``, ``entropy_norm``,
    ``top1_prob``, ``top{k}_prob``, ``multi_modality``, ``sparsity``,
    ``relevant_count``, ``sparse_count``.
    """
    x = logits.astype("float32")
    if x.ndim > 2:
        x = x.reshape([-1, x.shape[-1]])
    rows, width = x.shape
    vocab_size = width if vocab_size is None else int(vocab_size)

    log_z = paddle.logsumexp(x, axis=-1, keepdim=True)  # [rows, 1]
    p = paddle.exp(x - log_z)
    x_max = x.max(axis=-1, keepdim=True)
    x_min = x.min(axis=-1, keepdim=True)
    p_max = paddle.exp(x_max - log_z)
    eps = _relevance_threshold(log_z, vocab_size, prob_eps, logit_eps)

    # Entropy via H = log_z - E_p[x]: no separate log of the probabilities.
    entropy = log_z.squeeze(-1) - (p * x).sum(axis=-1)

    # Def 3.2 -- relevant entries are eps < x_n < x_max (the max is excluded).
    relevant = paddle.logical_and(x > eps, x < x_max).astype("float32")
    n_relevant = relevant.sum(axis=-1)
    gap_sum = ((p_max - p) * relevant).sum(axis=-1)
    multi_modality = 1.0 - gap_sum / paddle.clip(n_relevant, min=1.0)
    # Rows whose only relevant entry is the max leave M undefined (N == 0);
    # drop them from the mean instead of scoring them as fully multi-modal.
    mm_valid = (n_relevant > 0).astype("float32")

    # Def 3.3 -- exp((s - p_l)/s - 1) == exp(-p_l/s) == exp(-exp(x_l - x_min))
    # for the paper's reference value s = phi(x)_min. The bound is strict, as in
    # the paper: an entry sitting exactly at eps belongs to neither set, so a
    # uniform row (every entry at the threshold) yields L == 0 and is dropped
    # from the mean rather than scoring as maximally sparse.
    sparse = (x < eps).astype("float32")
    n_sparse = sparse.sum(axis=-1)
    ratio = paddle.clip(x - x_min, max=_EXP_CLAMP)
    sparsity_terms = paddle.exp(-paddle.exp(ratio)) * sparse
    sparsity = sparsity_terms.sum(axis=-1) / paddle.clip(n_sparse, min=1.0)
    sp_valid = (n_sparse > 0).astype("float32")

    k = min(int(topk), width)
    topk_mass = paddle.topk(p, k=k, axis=-1)[0].sum(axis=-1)

    def _masked_mean(values: paddle.Tensor, weights: paddle.Tensor) -> paddle.Tensor:
        return (values * weights).sum() / paddle.clip(weights.sum(), min=1.0)

    return {
        "entropy": entropy.mean(),
        "entropy_norm": entropy.mean() / math.log(vocab_size) if vocab_size > 1 else entropy.mean(),
        "top1_prob": p_max.mean(),
        f"top{k}_prob": topk_mass.mean(),
        "multi_modality": _masked_mean(multi_modality, mm_valid),
        "sparsity": _masked_mean(sparsity, sp_valid),
        "relevant_count": n_relevant.mean(),
        "sparse_count": n_sparse.mean(),
        "rows": paddle.full((), float(rows), dtype="float32"),
    }


def compute_param_metrics(ranges: paddle.Tensor, ts: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """The head's learned SegLU coefficients, one metric per component.

    ``ranges`` are the four thresholds (``b``/``d`` of each order) and ``ts``
    the four scales; both are ``[4]``. Zero on both means SegLU is still the
    identity, so these series answer "did multimax actually start learning".
    """
    r = ranges.astype("float32").reshape([-1])
    t = ts.astype("float32").reshape([-1])
    out: dict[str, paddle.Tensor] = {}
    for i in range(min(r.shape[0], 4)):
        out[f"range_{i}"] = r[i]
    for i in range(min(t.shape[0], 4)):
        out[f"t_{i}"] = t[i]
    return out
