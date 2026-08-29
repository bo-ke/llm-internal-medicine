"""Metric families: the sub-monitor unit of selection.

A single monitor owns far too many metrics to be a useful on/off switch —
``mhc_health`` emits 106 keys per layer, ``moe_health`` 56 plus two per expert.
Turning a whole monitor off to save payload throws away the handful of curves
that are actually being watched; keeping it on pays for all of them.

So each monitor's metrics are partitioned into *families*: `gate`, `mix`,
`gain`, `param`, ... — groups that answer the same kind of question. Selection
happens at ``(monitor, family)`` granularity.

``components`` is an orthogonal axis, not a family. ``mhc_health`` reports every
one of its families twice, once for the attention hyper-connection and once for
the MLP one; folding that into the family list would double the family count for
no gain, so it stays a separate dimension.

This taxonomy is shared with the visualisation layer's chip groups
(``tools/internal_medicine/viewer.py``) on purpose: "I turned off
``mhc_health:mix``" and "the 混合矩阵 group of charts is gone" have to be the
same sentence. ``tests/test_metric_families.py`` pins the two against each other
so they cannot drift.
"""

from __future__ import annotations

import re

__all__ = [
    "METRIC_TAXONOMY",
    "FAMILY_OTHER",
    "UnknownFamilyError",
    "classify",
    "components_of",
    "families_of",
    "parse_exclude",
    "parse_exclusions",
    "parse_families",
    "strip_prefixes",
    "FamilySelection",
]

FAMILY_OTHER = "other"

# Attention-kind and branch prefixes are part of the key, not of the metric
# identity: `hca_h_res_cell0` and `mla_h_res_cell0` are the same quantity
# measured on two attention stacks and belong in the same family.
_ATTN_PREFIX_RE = re.compile(r"^(?:swa|full|mla|mqa|hca|csa|dsa|window)_(?P<base>.+)$")
_VHA_BRANCH_RE = re.compile(r"^(?:main|sparse)_(?P<base>.+)$")

# Families are matched in declaration order, first match wins. Order is
# therefore load-bearing wherever one pattern is a prefix of another — see the
# note on moe_health below.
METRIC_TAXONOMY: dict[str, dict] = {
    "mhc_health": {
        "components": ("attn", "mlp"),
        "families": (
            ("gate", "门控强度 h_pre / h_post", r"^h_(pre|post)(_|$)"),
            ("mix", "混合矩阵 h_res", r"^h_res(_|$)"),
            ("gain", "增益放大 amax / composite", r"amax_gain"),
            ("param", "可学习参数 α / bias", r"^(alpha|bias)_"),
            ("share", "残差占比", r"^branch_residual_share"),
        ),
    },
    "qk_stats": {
        "families": (
            ("sink", "sink head", r"^sink"),
            ("logit", "logits 量级", r"^(max|mean|attn_sink_logit)$"),
            ("entropy", "注意力熵", r"^entropy_"),
            ("qkv", "q / k / v 范数", r"^[qkv]_norm_"),
        ),
    },
    "massive_act": {
        "families": (
            ("channel", "通道量级分布", r"^(channel_(?!count_gt)|topk_channel_|massive_act_channel_count)"),
            ("outlier", "超大通道计数", r"^channel_count_gt_"),
            ("module", "各模块输出量级", r"^(attn_out|ffn_or_moe_out|layer_input|post_ffn_residual)_"),
            ("norm", "归一化后表示", r"^post_norm_"),
            ("overall", "整体激活量级", r"^activation_rms$"),
        ),
    },
    "ape_health": {
        "components": ("core", "indexer"),
        "families": (
            ("softmax", "位置 softmax 形状", r"^softmax_"),
            ("coverage", "位置覆盖", r"^(effective_positions|position_range)"),
            ("scale", "量级与数值健康", r"^(rms|centered_rms|non_finite_ratio)$"),
        ),
    },
    "vha_health": {
        "components": ("main", "mtp"),
        "families": (
            ("mix", "postmix 混合矩阵", r"^postmix_(u|v|uv|offdiag)"),
            ("gain", "postmix 增益与扰动", r"^postmix_(amax_gain|delta_rel)"),
            ("head", "头间一致性", r"^(postmix_head_cos|head_out_norm)"),
        ),
    },
    "moe_health": {
        # `spectrum` and `act` must precede `shared`, or `shared_gate_stable_rank`
        # and `shared_act_*` would be swallowed by the `^shared_` catch.
        "families": (
            ("router", "路由打分质量", r"^router_"),
            ("balance", "负载均衡", r"^(assignment_load|gate_mass)_"),
            ("norm", "专家范数与 bias", r"^(expert_norm|expert_bias|score_sum|bias_affinity)"),
            ("spectrum", "gate 谱性质", r"gate_(stable_rank|singular_entropy)"),
            ("act", "激活量级", r"_act_(abs_max|mean|norm)$|^routed_act|^shared_act"),
            ("shared", "共享 vs 路由", r"^shared_"),
            ("expert", "按专家分布", r"^expert_(token|weight)_share"),
        ),
    },
}


def strip_prefixes(sub: str) -> str:
    """Reduce a metric name to the form the family patterns match against.

    Drops the ``global_`` marker (the cross-layer aggregate of a quantity belongs
    in that quantity's family), then the attention kind, then the vha branch.
    """
    if sub.startswith("global_"):
        sub = sub[len("global_") :]
    m = _ATTN_PREFIX_RE.match(sub)
    if m:
        sub = m["base"]
    m = _VHA_BRANCH_RE.match(sub)
    if m:
        sub = m["base"]
    return sub


def classify(monitor: str, sub: str) -> tuple[str, str]:
    """``(family, component)`` for one metric name.

    A monitor with no taxonomy entry, or a name matching none of its patterns,
    yields ``FAMILY_OTHER``. Callers that gate collection must treat that as
    "always collect" — an unclassified metric is a gap in this table, and
    silently dropping it would be the worst possible failure mode.
    """
    spec = METRIC_TAXONOMY.get(monitor)
    if not spec:
        return FAMILY_OTHER, ""
    rest = strip_prefixes(sub)
    component = ""
    for candidate in spec.get("components", ()):
        if rest.startswith(candidate + "_"):
            component, rest = candidate, rest[len(candidate) + 1 :]
            break
    for key, _label, pattern in spec.get("families", ()):
        if re.search(pattern, rest):
            return key, component
    return FAMILY_OTHER, component


def families_of(monitor: str) -> tuple[str, ...]:
    """Declared family keys for one monitor, in declaration order."""
    spec = METRIC_TAXONOMY.get(monitor) or {}
    return tuple(key for key, _label, _pattern in spec.get("families", ()))


def components_of(monitor: str) -> tuple[str, ...]:
    spec = METRIC_TAXONOMY.get(monitor) or {}
    return tuple(spec.get("components", ()))


class FamilySelection:
    """Which families of one monitor to *skip*. Empty set means collect everything.

    Exclusion, not inclusion, is the primary model on purpose. The operational
    question is always "which of these is too expensive to keep on", never "which
    three do I want" — a whitelist of 27 families would have to be rewritten every
    time a monitor gains a metric, and forgetting to extend it loses data silently.

    It also makes the safe default fall out for free: a metric matching no family
    pattern is not in any exclusion set, so a hole in ``METRIC_TAXONOMY`` costs
    payload rather than deleting a curve.
    """

    __slots__ = ("monitor", "excluded")

    def __init__(self, monitor: str, excluded=()):
        self.monitor = monitor
        self.excluded = frozenset(excluded)

    def allows(self, sub: str) -> bool:
        if not self.excluded:
            return True
        family, _component = classify(self.monitor, sub)
        return family not in self.excluded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        dropped = "+".join(sorted(self.excluded)) or "nothing"
        return f"FamilySelection({self.monitor}, excluding {dropped})"


class UnknownFamilyError(ValueError):
    """A config named a family the monitor does not declare."""


def _split_names(value) -> list[str]:
    if isinstance(value, str):
        return [n for n in re.split(r"[+,\s]+", value) if n]
    return [str(n).strip() for n in value if str(n).strip()]


def _validate(monitor: str, names: list[str], action: str) -> None:
    declared = families_of(monitor)
    if not declared:
        raise UnknownFamilyError(
            f"{monitor!r} declares no metric families, so {action} {names!r} is meaningless; "
            f"add it to METRIC_TAXONOMY first"
        )
    unknown = [n for n in names if n not in declared]
    if unknown:
        raise UnknownFamilyError(
            f"unknown {monitor!r} metric families {unknown!r}; declared families are {list(declared)!r}"
        )


def parse_exclude(monitor: str, exclude) -> FamilySelection:
    """Build a selection from the families to switch off.

    ``exclude`` accepts ``None`` (drop nothing), a ``"mix+param"`` string, or any
    iterable of names. Unknown names raise rather than being ignored: a typo here
    means a family stays on that someone believed was off, and the only symptom is
    a payload that did not shrink.
    """
    if exclude is None:
        return FamilySelection(monitor)
    names = _split_names(exclude)
    if not names or names == ["none"]:
        return FamilySelection(monitor)
    _validate(monitor, names, "excluding")
    return FamilySelection(monitor, names)


def parse_families(monitor: str, families) -> FamilySelection:
    """Build a selection from the families to *keep* — the inverse of the above.

    Kept for callers that genuinely want a whitelist (a focused debugging run,
    mostly). Expressed as an exclusion so there is only one code path in
    ``allows``.
    """
    if families is None:
        return FamilySelection(monitor)
    names = _split_names(families)
    if not names or names == ["all"]:
        return FamilySelection(monitor)
    _validate(monitor, names, "narrowing to")
    return FamilySelection(monitor, set(families_of(monitor)) - set(names))


def parse_exclusions(spec) -> dict[str, list[str]]:
    """``"mhc_health:mix, moe_health:expert+act"`` → per-monitor family lists.

    A monitor written without a ``:`` part, or an absent/empty spec, yields no
    entry — i.e. nothing is switched off and behaviour is exactly what it was
    before families existed. Family names are validated later, per monitor, by
    ``parse_exclude``; monitor names are the registry's business.
    """
    if not spec:
        return {}
    entries = spec.split(",") if isinstance(spec, str) else [str(entry) for entry in spec]
    excluded: dict[str, list[str]] = {}
    for entry in entries:
        name, _, family_part = entry.strip().partition(":")
        name = name.strip()
        families = [n for n in re.split(r"[+\s]+", family_part) if n]
        if name and families:
            excluded.setdefault(name, []).extend(families)
    return excluded
