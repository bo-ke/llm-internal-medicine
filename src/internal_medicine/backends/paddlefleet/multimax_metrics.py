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

``M`` is emitted with ``eps`` = the k-th largest entry, i.e. the relevant set is
the top-k minus the max, so ``M_topk = 1 - (phi_max - mean(phi_2..phi_k))`` --
"the average gap between top-1 and the other top-k entries". The eps-based form
over the *full* relevant set was dropped (2026-08-22): at the default
``eps = 1/V`` the set holds thousands of entries two to three orders of
magnitude below ``phi_max``, so ``M`` collapses onto ``1 - top1_prob`` (measured
gap constant at +0.023..+0.026 across three checkpoints) and carried no
information ``top1_prob`` did not already have.

``S`` simplifies exactly: ``(s - p)/s - 1 == -p/s``, so each term is
``exp(-p_l / s)``. The paper leaves ``s`` free ("can be any reference value")
and asks only that ``S`` stay a smooth step approximation normalized to
``[0, 1]``, larger meaning sparser. That normalization holds only while ``s`` is
**independent of the distribution being scored**: with ``s = phi(x)_min`` every
``p_l / s >= 1`` by construction, so each term is capped at ``e^-1`` and
``S -> e^-1 / L`` once ``L`` is vocab-sized. The score then tracks ``1 / L``
rather than sparsity, and is not comparable across steps because the reference
moves with the distribution.

``s`` defaults to the geometric mean of the *unmodulated* softmax's own tail
(``sparsity_ref_mode="geomean"``), which sits mid-tail and so leaves ``S``
responsive; ``"uniform"`` falls back to ``s = 1 / V``, where nearly every term
saturates near 1. Both are per-run yardsticks -- **cross-model comparison
requires passing an explicit constant ``sparsity_ref`` shared by every run.**
The ratio is evaluated as ``exp(x_l - log_z - log s)``, so nothing underflows at
LM-head vocab sizes where ``phi_min`` rounds to 0 in fp32.

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


SPARSITY_REF_MODES = ("geomean", "uniform")


def _sparsity_log_ref(
    vocab_size: int,
    sparsity_ref: float | None,
    ref_logits: paddle.Tensor | None,
    prob_eps: float | None = None,
    logit_eps: float | None = None,
    mode: str = "geomean",
) -> paddle.Tensor | float:
    """``log s`` for Def 3.3, in priority order: explicit constant, then mode.

    Def 3.3 lets ``s`` be "any reference value for a non-linear scaling", with
    the smallest SoftMax(t=1) probability given as one example. The choice only
    has to be independent of the row being scored, but it decides where the
    smooth step ``exp(-p_l/s)`` sits:

    * ``uniform`` (``s = 1/V``): a run-independent constant, but so large that
      every irrelevant entry already sits below it -- the terms saturate near 1
      (measured ~0.98) and the score has almost no resolution.
    * ``geomean`` (default): the geometric mean of the *baseline* tail
      probabilities, i.e. the mean of ``log phi_ref`` over the reference's own
      irrelevant set. That is a middle-of-the-tail reference, so a tail pushed
      below the baseline lands in the responsive part of the step.

    **``geomean`` is per-run and not comparable across models.** It is derived
    from that model's own unmodulated logits, so two runs are each scored against
    their own yardstick: useful for watching one run evolve, wrong for "is the
    MultiMax model sparser than the softmax baseline". Cross-model comparison
    needs an explicit ``sparsity_ref`` constant shared by every run.

    The paper's ``min SoftMax(x_raw)`` example was dropped: it is both a per-run
    reference and orders of magnitude below ``1/V``, so nearly every term
    underflows to 0 (measured on a live run: ``S`` p50 = 0.013).

    The baseline's tail is selected by the reference distribution's own ``eps``,
    never by the scored row's, so ``s`` stays independent of the modulation --
    see the module docstring for why a statistic of the scored row collapses the
    score onto ``e^-1/L``.
    """
    if sparsity_ref is not None:
        return math.log(float(sparsity_ref))
    uniform = math.log(1.0 / vocab_size)
    if ref_logits is None or mode == "uniform":
        return uniform
    ref = ref_logits.astype("float32")
    if ref.ndim > 2:
        ref = ref.reshape([-1, ref.shape[-1]])
    ref_log_z = paddle.logsumexp(ref, axis=-1, keepdim=True)
    ref_eps = _relevance_threshold(ref_log_z, vocab_size, prob_eps, logit_eps)
    # bool mask + where, not a materialized fp32 mask: at LM-head scale each
    # full-size fp32 tile is ~200 MB (see _gather_vocab's sizing note).
    tail = ref < ref_eps
    n_tail = tail.sum(axis=-1, keepdim=True).astype("float32")
    log_phi = ref - ref_log_z
    log_geo = paddle.where(tail, log_phi, paddle.zeros_like(log_phi)).sum(axis=-1, keepdim=True) / paddle.clip(
        n_tail, min=1.0
    )
    # A baseline row with no tail at all (uniform reference) has no geometric
    # mean to take; fall back to the uniform probability for those rows only.
    return paddle.where(n_tail > 0, log_geo, paddle.full_like(log_geo, uniform))


QUANTILES: tuple[float, ...] = (0.5, 0.95, 0.98)


def _summarize(
    values: paddle.Tensor, weights: paddle.Tensor | None = None
) -> tuple[paddle.Tensor, dict[str, paddle.Tensor]]:
    """Token mean plus ``QUANTILES`` of a per-row metric, all 0-dim tensors.

    Every metric here is bounded below (entropy, counts) or to ``[0, 1]``
    (probabilities, M, S) and right-skewed, so a symmetric mean +- sigma band
    leaves the metric's own domain: the lower edge of an entropy band goes
    negative even though no token can have negative entropy. Quantiles describe
    the same spread without assuming symmetry -- p50 is the typical token, p95 /
    p98 say where the tail sits.

    ``weights`` is the 0/1 validity mask for metrics that drop rows (M with
    N == 0, S with L == 0). Invalid rows are pushed to ``+inf`` so the sort
    parks them past the end, and the quantile index is derived from the valid
    count as a tensor -- masking must not become a python-level filter, which
    would need a D2H sync on the hot path.

    Quantiles are not linear, so they cannot be pooled across hook calls the way
    a mean can: ``record_mean`` averaging per-call quantiles is exact only while
    there is one call per logged step (``gradient_accumulation_steps=1``) and an
    approximation otherwise. Raise ``sample_tokens`` rather than relying on that
    average if accumulation is enabled.
    """
    if weights is None:
        weights = paddle.ones_like(values)
    n_valid = weights.sum()
    total = paddle.clip(n_valid, min=1.0)
    mean = (values * weights).sum() / total

    ordered = paddle.sort(paddle.where(weights > 0, values, paddle.full_like(values, float("inf"))))
    quantiles: dict[str, paddle.Tensor] = {}
    for q in QUANTILES:
        # Nearest-rank percentile: rank ceil(q*n) among the valid rows. Unlike
        # floor(q*(n-1)) ("lower" interpolation) this does not collapse p95/p98
        # onto p50 when only a couple of rows survive the mask.
        rank = paddle.clip(paddle.ceil(q * total) - 1.0, min=0.0)
        idx = paddle.minimum(rank, total - 1.0).astype("int64").reshape([1])
        picked = paddle.take_along_axis(ordered, idx, axis=0).squeeze()
        # With no valid row at all, `total` is clipped to 1 and the pick lands on
        # the +inf sentinel; report 0 as the mean already does, because an inf
        # entering record_mean poisons that key's running average for the step.
        # paddle.where, not a 0/1 multiply: inf * 0 would be nan.
        quantiles[_quantile_suffix(q)] = paddle.where(n_valid > 0, picked, paddle.zeros_like(picked))
    return mean, quantiles


def _quantile_suffix(q: float) -> str:
    """``0.5 -> 'p50'``, ``0.98 -> 'p98'``; the metric key suffix for quantile q."""
    return f"p{round(q * 100)}"


def compute_distribution_metrics(
    logits: paddle.Tensor,
    vocab_size: int | None = None,
    prob_eps: float | None = None,
    logit_eps: float | None = None,
    topk: int = 10,
    sparsity_ref: float | None = None,
    ref_logits: paddle.Tensor | None = None,
    sparsity_ref_mode: str = "geomean",
) -> dict[str, paddle.Tensor]:
    """Token-mean distribution metrics for one ``[rows, vocab]`` logits tile.

    ``logits`` must cover the **full** vocab (the caller all-gathers
    vocab-parallel shards first) and must already carry the SegLU modulation.
    ``vocab_size`` defaults to the tile's last dim; pass it explicitly only if
    the tile is a strict subset of the vocab.

    ``sparsity_ref`` pins Def 3.3's reference probability ``s`` to a constant --
    the only setting that makes ``sparsity`` comparable across models. Otherwise
    ``sparsity_ref_mode`` picks how it is derived from ``ref_logits``
    (``geomean`` / ``uniform``, see ``_sparsity_log_ref``). ``s`` must never
    depend on the scored distribution (see the module docstring).

    Returns 0-dim fp32 GPU tensors: ``entropy``, ``entropy_norm``,
    ``top1_prob``, ``top{k}_prob``, ``multi_modality_top{k}``, ``sparsity``,
    ``relevant_count``, ``sparse_count`` -- each with ``_p50`` / ``_p95`` /
    ``_p98`` companions -- plus ``rows``.
    """
    x = logits.astype("float32")
    if x.ndim > 2:
        x = x.reshape([-1, x.shape[-1]])
    rows, width = x.shape
    vocab_size = width if vocab_size is None else int(vocab_size)

    log_z = paddle.logsumexp(x, axis=-1, keepdim=True)  # [rows, 1]
    p = paddle.exp(x - log_z)
    x_max = x.max(axis=-1, keepdim=True)
    p_max = paddle.exp(x_max - log_z)
    eps = _relevance_threshold(log_z, vocab_size, prob_eps, logit_eps)

    # Entropy via H = log_z - E_p[x]: no separate log of the probabilities.
    entropy = log_z.squeeze(-1) - (p * x).sum(axis=-1)

    # top-k before the Def 3.3 block so ``p`` (a full [rows, vocab] fp32 tile)
    # can be released before the mask work allocates its own: at the LM-head
    # scale each of these is ~200 MB, so their overlap is what sets the peak.
    k = min(int(topk), width)
    p_top = paddle.topk(p, k=k, axis=-1)[0]  # descending, p_top[:, 0] == p_max
    topk_mass = p_top.sum(axis=-1)
    del p

    # Def 3.2 with eps = the k-th largest entry: the relevant set is the top-k
    # minus the max, so this is literally "the average gap between top-1 and the
    # other top-k entries", and its terms are comparable in magnitude to phi_max.
    if k > 1:
        mm_topk = 1.0 - (p_top[:, :1] - p_top[:, 1:].mean(axis=-1, keepdim=True)).squeeze(-1)
    else:
        mm_topk = paddle.ones_like(topk_mass)

    # Def 3.2's relevant set, kept only as an eps sanity check (see
    # relevant_count): the metric built on it was dropped, because with the
    # default eps = 1/V the set runs to ~1e3 entries whose probabilities are
    # orders of magnitude below phi_max, so (1/N)sum(phi_max - phi_n) collapses
    # onto phi_max and M becomes 1 - top1_prob (measured: the difference is
    # pinned at +0.023..+0.026 across three checkpoints). The top-k form above
    # is the one that tracks the head's shape.
    # Summed as bool (1 byte/entry) rather than a materialized fp32 mask: only
    # the count is used, and the fp32 cast would be a second full-size tile.
    n_relevant = paddle.logical_and(x > eps, x < x_max).sum(axis=-1).astype("float32")

    # Def 3.3 -- exp((s - p_l)/s - 1) == exp(-p_l/s), evaluated in log space as
    # exp(-exp(x_l - log_z - log s)) so phi_min underflow never appears. ``s`` is
    # a fixed reference, never a statistic of this row -- see the module
    # docstring for why phi_min collapses the score onto e^-1/L.
    # The set bound is strict, as in the paper: an entry sitting exactly at eps
    # belongs to neither set, so a uniform row (every entry at the threshold)
    # yields L == 0 and is dropped from the mean rather than scoring as
    # maximally sparse.
    sparse = x < eps
    n_sparse = sparse.sum(axis=-1).astype("float32")
    log_s = _sparsity_log_ref(vocab_size, sparsity_ref, ref_logits, prob_eps, logit_eps, sparsity_ref_mode)
    ratio = paddle.clip(x - log_z - log_s, max=_EXP_CLAMP)
    # where(mask, term, 0) instead of term * mask.astype("float32"): same value,
    # one fewer full-size fp32 tile alive.
    terms = paddle.exp(-paddle.exp(ratio))
    del ratio
    sparsity = paddle.where(sparse, terms, paddle.zeros_like(terms)).sum(axis=-1) / paddle.clip(n_sparse, min=1.0)
    sp_valid = (n_sparse > 0).astype("float32")

    # Every distribution metric is a token mean; each also reports p50/p95/p98
    # over the sampled tokens, which the viewer draws as a band plus a tail line.
    log_v = math.log(vocab_size) if vocab_size > 1 else 1.0
    out: dict[str, paddle.Tensor] = {}
    for name, values, weights in (
        ("entropy", entropy, None),
        ("entropy_norm", entropy / log_v, None),
        ("top1_prob", p_max.squeeze(-1), None),
        (f"top{k}_prob", topk_mass, None),
        (f"multi_modality_top{k}", mm_topk, None),
        ("sparsity", sparsity, sp_valid),
        ("relevant_count", n_relevant, None),
        ("sparse_count", n_sparse, None),
    ):
        mean, quantiles = _summarize(values, weights)
        out[name] = mean
        for suffix, value in quantiles.items():
            out[f"{name}_{suffix}"] = value
    out["rows"] = paddle.full((), float(rows), dtype="float32")
    return out


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
