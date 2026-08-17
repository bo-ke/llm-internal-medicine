"""LSE / gate head-dispersion metrics for the QK stats monitor.

The learnable softmax offset ``beta_h`` is not an independent knob: with an offset
(equivalently a value-0 sink token whose logit is ``beta_h``), attention output is a pure
rescale of vanilla attention,

    o_tilde_t = sigma(lse_t - beta_h) * o_van_t

so the pair ``(lse_t, beta_h)`` fully determines a per-head logistic gate. ``beta_h`` alone
says nothing -- whether a head passes signal through depends on where ``lse_t`` sits
relative to it. These metrics make that comparison directly observable.

The kernels emit per-head means and per-head stds over query rows; everything here is the
head-dimension reduction on top of that. Head dispersion needs its own summaries because
``beta_h`` is learned per head: a mean over heads cannot distinguish "every head half
open" from "half the heads open, half shut", and the second is the interesting failure.
"""

import torch


def _dispersion(values: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """``max(values) / denom``, guarded against a zero/degenerate denominator."""
    return values.max() / denom.clamp_min(1e-8)


def compute_gate_head_metrics(
    lse_per_head: torch.Tensor,
    gate_per_head: torch.Tensor | None,
    lse_std_per_head: torch.Tensor | None = None,
    gate_std_per_head: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Reduce per-head LSE/gate statistics over the head dimension.

    Args:
        lse_per_head: ``[H]`` mean pure-QK log-sum-exp per head (offset excluded).
        gate_per_head: ``[H]`` mean ``sigma(lse - beta_h)`` per head, or ``None`` when the
            model has no ``softmax_offset``. Gate metrics are then omitted entirely rather
            than reported with an implied ``beta = 0``, which would be a different model.
        lse_std_per_head: ``[H]`` std of ``lse`` over query rows, per head.
        gate_std_per_head: ``[H]`` std of the gate over query rows, per head.

    Returns:
        Whichever of these the inputs support:
            lse                      mean over heads
            lse_std                  mean over heads of the per-head row-std
            lse_std_max_mean_ratio   max/mean over heads of that std
            gate_avg                 mean over heads
            gate_max_median_ratio    max/median over heads
            gate_min_median_ratio    min/median over heads
            gate_std                 mean over heads of the per-head row-std
            gate_std_max_mean_ratio  max/mean over heads of that std

    The two gate ratios normalise by the MEDIAN so they are scale-free: they report how
    far heads have diverged from each other, not how low the gate is in absolute terms,
    which drifts with lr and training stage. They are complementary -- one dead head shows
    up only in ``min/median`` (measured 0.002 with 15 healthy heads), while "most heads
    shut, one dominant" shows up only in ``max/median`` (measured 19-32x). Both near 1.0
    means the heads are moving together.

    ``torch.median`` takes the lower middle element for an even head count (no
    interpolation), matching the convention used elsewhere in this repo.

    All returned values are 0-dim tensors; no host sync.
    """
    out: dict[str, torch.Tensor] = {}
    if lse_per_head is None or lse_per_head.numel() == 0:
        return out

    out["lse"] = lse_per_head.mean()
    if lse_std_per_head is not None and lse_std_per_head.numel() > 0:
        out["lse_std"] = lse_std_per_head.mean()
        out["lse_std_max_mean_ratio"] = _dispersion(lse_std_per_head, lse_std_per_head.mean())

    if gate_per_head is None or gate_per_head.numel() == 0:
        return out

    median = gate_per_head.median()
    out["gate_avg"] = gate_per_head.mean()
    out["gate_max_median_ratio"] = gate_per_head.max() / median.clamp_min(1e-8)
    out["gate_min_median_ratio"] = gate_per_head.min() / median.clamp_min(1e-8)
    if gate_std_per_head is not None and gate_std_per_head.numel() > 0:
        out["gate_std"] = gate_std_per_head.mean()
        out["gate_std_max_mean_ratio"] = _dispersion(gate_std_per_head, gate_std_per_head.mean())
    return out
