"""Activation-gradient magnitude metrics for Megatron/Torch.

Pure tensor functions, no monitor state. Everything returns a 0-dim GPU tensor so
the caller hands it straight to ``record_layer_metric`` without a D2H sync --
hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.

The quantity being measured is dL/d(activation) at a module's *output*, i.e. the
gradient that flows back into that module. This is the backward-path counterpart
of ``massive_act``'s forward-path magnitudes, and the positions are named to
match so a gradient curve and its activation curve line up in the viewer.

A port of ``backends/paddlefleet/grad_metrics.py``; the constants are kept
identical so the metric keys are the same on both backends. The reductions differ:
nothing here upcasts the gradient to fp32 as a whole tensor, because on a
``[8192, 1, 4096]`` bf16 gradient that costs 256 MiB of transient allocation per
hook call -- inside backward, where activation memory is already at its peak.
"""

from __future__ import annotations

import torch

# Where on the backward path a gradient is read. ``layer_out`` is the residual
# stream leaving a decoder layer -- the series that answers "does the gradient
# survive depth"; the other two attribute a layer's share to its two branches.
POSITIONS = ("attn_out", "ffn_or_moe_out", "layer_out")

# Per-shard magnitudes, all degree-1 in ``g``. ``rms`` is the one comparable across
# layers and layouts; ``norm`` is shape-dependent, ``abs_max`` is the spike detector.
METRICS = ("norm", "rms", "abs_max")

# Exact cross-rank quantities. They need a collective, so they are produced at
# flush time from a running sum of squares rather than in the hook -- see
# ``grad_monitor.ExactNormReducer``. Names are separate keys, so the three
# approximate series above keep their exact previous values.
#
# - ``norm_mb``     one microbatch's *complete* tensor, i.e. summed back over
#                   whatever shards the same tensor (SP / CP / SP-in-TP), then
#                   averaged over microbatches and data ranks. Equals ``norm``
#                   exactly on a layout that does not shard activations.
# - ``rms_global``  sqrt(sum g^2 / sum N) over every rank and microbatch. The
#                   only one of the six that is invariant to cluster size,
#                   parallel layout and gradient-accumulation depth.
# - ``norm_global`` sqrt(sum g^2) over every rank and microbatch, i.e. the whole
#                   global batch. Closest in spirit to the optimizer's global
#                   grad norm, but it grows like sqrt(world_size * gas), so it is
#                   only comparable within one cluster shape.
GLOBAL_METRICS = ("norm_mb", "rms_global", "norm_global")

# Per-token decomposition. ``abs_max`` finds the largest single *element*, which
# cannot tell "one whole token carries a huge gradient" (a data problem: a rare
# token, an EOS, a dirty sample) from "one cell ran away" (a numerical / channel
# problem). Those two want opposite fixes, so the token axis gets its own series.
#
# The token axis, not the channel axis, is the right decomposition here:
# ``massive_act`` splits activations per channel because massive activations are
# a channel phenomenon, while gradient spikes are usually carried by individual
# tokens.
#
# All four are exact across ranks with no collective: every position is a
# post-projection full-hidden tensor, so each rank holds *whole* tokens and a max
# over ranks is a max over all tokens. ``token_outlier_ratio`` is a fraction
# rather than a count for the same reason -- a count would be per-shard.
TOKEN_METRICS = (
    "token_norm_max",
    "token_norm_p99",
    "token_norm_ratio",
    "token_outlier_ratio",
    "token_zero_ratio",
)

# Degree-0 NaN/Inf share of the activation gradient -- the bf16/fp16 overflow alert.
# Max-aggregated (see MAX_METRICS) so a localized overflow is not averaged away.
HEALTH_METRICS = ("nonfinite_fraction",)

# Max over microbatches / ranks / layers -- a spike/overflow detector wants the
# worst, not the average. The remaining metrics are means.
MAX_METRICS = ("abs_max", "token_norm_max", "token_norm_ratio", "nonfinite_fraction")

# How far above the per-token median a token counts as an outlier. Matches
# ``massive_act``'s 10x convention so the forward and backward "outlier_ratio"
# series are read the same way.
TOKEN_OUTLIER_MULTIPLIER = 10.0

# Degree-0: ratios/fractions invariant to a positive rescale, so AMP de-scale skips
# them (``rms_depth_ratio`` is a ratio of norms -- scale cancels). Rest are degree-1.
SCALE_INVARIANT = (
    "token_norm_ratio",
    "token_outlier_ratio",
    "token_zero_ratio",
    "nonfinite_fraction",
    "rms_depth_ratio",
)

ALL_METRICS = tuple(
    f"{position}_{metric}"
    for position in POSITIONS
    for metric in METRICS + GLOBAL_METRICS + TOKEN_METRICS + HEALTH_METRICS
)

MAX_AGGREGATED = frozenset(f"{position}_{metric}" for position in POSITIONS for metric in MAX_METRICS)


def token_norms(grad: torch.Tensor) -> torch.Tensor:
    """fp32 ``||g_t||`` for every token, flattening all leading axes.

    ``dtype=torch.float32`` upcasts *inside* the reduction, so a bf16 gradient is
    read once and no fp32 copy of it is ever allocated. Works for both ``[B, S, H]``
    and ``[S, H]`` layouts without being told which it got.
    """
    flat = grad.reshape(-1, grad.shape[-1])
    return torch.linalg.vector_norm(flat, dim=-1, dtype=torch.float32)


def grad_token_stats(token_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token gradient-vector statistics from the per-token norms.

    ``token_norm`` is what :func:`token_norms` returns, i.e. one fp32 scalar per
    token; the gradient itself is never materialized in fp32.

    - ``token_norm_max``      the loudest token's ``||g_t||``
    - ``token_norm_p99``      the tail's shape (over gradient-carrying tokens); its
                              distance from the max says whether the spike is one
                              isolated token or many
    - ``token_norm_ratio``    ``max / median``, the token-level peakiness --
                              the counterpart of ``massive_act``'s
                              ``channel_max_ratio``
    - ``token_outlier_ratio`` fraction of gradient-carrying tokens above ``10x`` the
                              median
    - ``token_zero_ratio``    fraction of tokens with no gradient at all

    **The median, p99 and outlier fraction are all taken over gradient-carrying
    tokens only.** A first version used the plain median and measured the wrong
    thing on a real 8k run: loss-masked positions gave whole rows of exact zeros,
    the median sat at ~0, and ``token_norm_ratio`` came out 2000-19000 -- a reading
    of *how much of the batch is masked* rather than of token peakiness. Dropping
    the dead rows restores the intended meaning for all three, and
    ``token_zero_ratio`` exposes the masked fraction as its own series.
    """
    alive = token_norm > 0
    # NaN for the dead rows so the nan-aware reductions ignore them without a
    # host-side boolean index (which would need the count on the CPU, a D2H sync).
    nonzero = torch.where(alive, token_norm, torch.full_like(token_norm, float("nan")))
    median = torch.nanmedian(nonzero)
    # All-masked microbatch -> nan reductions are NaN; fall back to 0 so the ratio
    # stays finite instead of poisoning the step with NaN.
    median = torch.where(torch.isnan(median), torch.zeros_like(median), median).clamp(min=1e-30)
    p99 = torch.nanquantile(nonzero, 0.99)
    p99 = torch.where(torch.isnan(p99), torch.zeros_like(p99), p99)
    peak = token_norm.max()
    alive_count = alive.float().sum().clamp(min=1.0)
    return {
        "token_norm_max": peak,
        "token_norm_p99": p99,
        "token_norm_ratio": peak / median,
        "token_outlier_ratio": (token_norm > TOKEN_OUTLIER_MULTIPLIER * median).float().sum() / alive_count,
        "token_zero_ratio": 1.0 - alive.float().mean(),
    }


def grad_square_and_stats(grad: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """``(display stats, sum of squares, element count)`` with no fp32 copy.

    Every number comes from the per-token norms plus two whole-tensor extremes, so
    the three magnitude metrics, the five per-token series and the exact-reduction
    summand together allocate one ``[tokens]`` vector instead of three full-size
    fp32 intermediates. ``numel`` is Python metadata, so it never forces a D2H
    sync.

    ``norm`` and ``rms`` are both read off ``sum_sq`` rather than recomputed: they
    differ only by ``sqrt(N)``, and ``N`` is metadata.
    """
    value = grad.detach()
    token_norm = token_norms(value)
    sum_sq = token_norm.square().sum()
    numel = max(1, value.numel())
    # A zero-token microbatch makes every reduction below throw (``max`` of empty);
    # keep the schema filled with zeros so the whole record does not abort.
    if token_norm.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=value.device)
        stats = {name: zero for name in ("norm", "rms", "abs_max", *TOKEN_METRICS, *HEALTH_METRICS)}
        return stats, sum_sq, numel
    stats = {
        "norm": torch.sqrt(sum_sq),
        "rms": torch.sqrt(sum_sq / numel),
        # max|g| without an abs() copy of the whole tensor: the two extremes of g
        # bracket it, and both are fused reductions in the gradient's own dtype.
        "abs_max": torch.maximum(value.max(), value.min().neg()).float(),
        # NaN/Inf share. Count finite and subtract so only a bool temp is reduced to a
        # scalar -- no full-size fp32 copy of the gradient (same discipline as above).
        "nonfinite_fraction": 1.0 - torch.isfinite(value).sum().float() / numel,
    }
    stats.update(grad_token_stats(token_norm))
    return stats, sum_sq, numel
