"""Activation-gradient magnitude metrics for PaddleFleet.

Pure tensor functions, no monitor state. Everything returns a 0-dim GPU tensor so
the caller hands it straight to ``record_layer_metric`` without a D2H sync --
hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.

The quantity being measured is dL/d(activation) at a module's *output*, i.e. the
gradient that flows back into that module. This is the backward-path counterpart
of ``massive_act``'s forward-path magnitudes, and the positions are named to
match so a gradient curve and its activation curve line up in the viewer.
"""

from __future__ import annotations

import math

import paddle

# Where on the backward path a gradient is read. ``layer_out`` is the residual
# stream leaving a decoder layer -- the series that answers "does the gradient
# survive depth"; the other two attribute a layer's share to its two branches.
POSITIONS = ("attn_out", "ffn_or_moe_out", "layer_out")

METRICS = ("norm", "rms", "abs_max")

# ``abs_max`` is a max over microbatches / ranks / layers, the rest are means.
MAX_METRICS = ("abs_max",)

ALL_METRICS = tuple(f"{position}_{metric}" for position in POSITIONS for metric in METRICS)

MAX_AGGREGATED = frozenset(f"{position}_{metric}" for position in POSITIONS for metric in MAX_METRICS)


def grad_magnitude_stats(grad: paddle.Tensor) -> dict[str, paddle.Tensor]:
    """``norm`` / ``rms`` / ``abs_max`` of one activation gradient.

    All three are homogeneous of degree 1 in ``grad``. That is what lets a single
    division by the AMP loss scale de-scale the whole set at finalize time, for
    both the mean accumulators (a sum over microbatches) and the max ones.

    - ``norm`` -- the L2 norm the caller asked for, over *this rank's shard*.
      Shape-dependent, so read it across steps, not across layers.
    - ``rms`` -- ``norm / sqrt(N)``. Shard- and shape-invariant under equal
      sharding (each rank's mean square is an unbiased estimate of the global
      one), so this is the series that is comparable across layers and across
      parallel layouts.
    - ``abs_max`` -- ``max|g|``, the spike detector. Max-aggregated, so one
      outlier microbatch or rank still reaches the global key.

    Two reductions, not three: ``N`` comes from ``shape``, which is Python
    metadata in dygraph, so ``norm`` is a scale of ``rms`` rather than a second
    sum over the tensor.
    """
    value = grad.detach().astype("float32")
    rms = paddle.sqrt(value.square().mean())
    sqrt_numel = math.sqrt(max(1, math.prod(value.shape)))
    return {
        "norm": rms * sqrt_numel,
        "rms": rms,
        "abs_max": value.abs().max(),
    }
