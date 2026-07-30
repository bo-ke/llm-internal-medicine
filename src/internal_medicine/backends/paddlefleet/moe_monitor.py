"""
MoE Monitor for PaddleFleet.

Monitors MoE router health and expert weight norms using paddle hooks.

PaddleFleet MoE structure:
- MoELayer.gate — varies by model:
    StandardMoEGate: (capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss)
    Other gates may return different formats.
- MoELayer.experts (nn.LayerList) or .grouped_gemm_experts (GroupedMLPExpert)
- MoELayer.shared_experts (StandardMLPSharedExpert or None)

TODO: Gate output format is model-specific. When switching to a different model
family, verify that outputs[2] (top_idx) is still at index 2 in the gate's
return tuple. Currently adapted for StandardMoEGate (PaddleFormers).
"""

import logging
import math

import paddle
import paddle.nn as nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)


def _compute_router_entropy(probs):
    """Router entropy from probability distribution. probs: [tokens, experts]."""
    probs = probs.astype("float32").clip(min=1e-10)
    probs = probs / probs.sum(axis=-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(axis=-1)
    return entropy.mean()


def _distribution_metrics(mass: paddle.Tensor) -> dict[str, paddle.Tensor]:
    mass = mass.astype("float32").reshape([-1]).clip(min=0.0)
    fraction = mass / mass.sum().clip(min=1e-30)
    num_experts = int(fraction.numel())
    uniform = 1.0 / max(num_experts, 1)
    safe = fraction.clip(min=1e-12)
    entropy = -(fraction * safe.log()).sum()
    entropy_norm = entropy / math.log(num_experts) if num_experts > 1 else entropy * 0.0 + 1.0
    minimum = fraction.min()
    return {
        "cv": paddle.sqrt(((fraction - uniform) ** 2).mean()) / uniform,
        "entropy_norm": entropy_norm,
        "kl_uniform": (fraction * (safe * num_experts).log()).sum(),
        "max_frac": fraction.max(),
        "min_frac": minimum,
        "max_min_ratio": fraction.max() / minimum.clip(min=1e-12),
    }


def _assignment_mass(probabilities: paddle.Tensor, expert_indices: paddle.Tensor):
    num_experts = int(probabilities.shape[-1])
    flat_probabilities = probabilities.reshape([-1, num_experts])
    expert_indices = expert_indices.reshape([flat_probabilities.shape[0], -1])
    flat_indices = expert_indices.reshape([-1])
    valid = ((flat_indices >= 0) & (flat_indices < num_experts)).astype("float32")
    safe_indices = flat_indices.clip(min=0, max=num_experts - 1)
    assignment = paddle.bincount(safe_indices, weights=valid, minlength=num_experts)
    selected_mass = paddle.take_along_axis(
        flat_probabilities,
        expert_indices.clip(min=0, max=num_experts - 1),
        axis=-1,
    ).reshape([-1])
    gate_mass = paddle.bincount(safe_indices, weights=selected_mass * valid, minlength=num_experts)
    return assignment, gate_mass


def _compute_bias_affinity_jaccard(top_idx_with_bias, gates_no_bias, k, n_group=1, topk_group=1):
    """Compute mean Jaccard similarity between routing with and without correction_bias.

    Paddle port of megatron/moe_metrics.compute_bias_affinity_jaccard, extended with
    group-limited topk support for PaddleFleet's TopKRouter.

    Args:
        top_idx_with_bias: [tokens, k] — actual routing indices (with bias)
        gates_no_bias: [tokens, experts] — original gate scores (no bias)
        k: num_experts_per_tok
        n_group / topk_group: group-limited topk params

    Returns:
        mean Jaccard similarity (1 = identical routing, 0 = completely different)
    """
    num_tokens, num_experts = gates_no_bias.shape

    if n_group > 1 and topk_group >= 1:
        group_size = num_experts // n_group
        gates_reshaped = gates_no_bias.reshape([num_tokens, n_group, group_size])
        group_max = gates_reshaped.max(axis=-1)
        _, top_groups = paddle.topk(group_max, topk_group, axis=-1)
        group_mask = paddle.zeros([num_tokens, n_group], dtype="int32")
        group_mask = group_mask.put_along_axis(top_groups, paddle.to_tensor(1, dtype="int32"), axis=1)
        group_mask = group_mask.unsqueeze(-1).expand([-1, -1, group_size]).reshape([num_tokens, num_experts])
        masked_gates = gates_no_bias.clone()
        masked_gates = paddle.where(group_mask > 0, masked_gates, paddle.full_like(masked_gates, float("-inf")))
        _, top_idx_no_bias = paddle.topk(masked_gates, k, axis=-1)
    else:
        _, top_idx_no_bias = paddle.topk(gates_no_bias, k, axis=-1)

    set_with = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_without = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_with = set_with.put_along_axis(top_idx_with_bias, paddle.to_tensor(1, dtype="int32"), axis=1)
    set_without = set_without.put_along_axis(top_idx_no_bias, paddle.to_tensor(1, dtype="int32"), axis=1)

    intersection = (set_with & set_without).astype("float32").sum()
    union = (set_with | set_without).astype("float32").sum()
    return intersection / union.clip(min=1.0)


def _per_expert_stacked_norms(w1, w2=None):
    """Per-expert L2 norm over stacked expert weights, fully vectorized."""
    num_experts = w1.shape[0]
    sq = (w1.detach().astype("float32").reshape([num_experts, -1]) ** 2).sum(axis=-1)
    if w2 is not None:
        sq = sq + (w2.detach().astype("float32").reshape([num_experts, -1]) ** 2).sum(axis=-1)
    return paddle.sqrt(sq)


def _module_sumsq(module):
    """Sum of squares over all parameters of a module. Returns a 0-dim GPU tensor or None."""
    sq = None
    for p in module.parameters():
        part = (p.detach().astype("float32") ** 2).sum()
        sq = part if sq is None else sq + part
    return sq


def _norm_stats(norms):
    """mean/std/min/max stats from a ``[num_experts]`` per-expert norm tensor (GPU tensors)."""
    if norms is None or norms.numel() == 0:
        return {}
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std() if norms.numel() > 1 else paddle.zeros(()),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


class PaddleMoEMonitor(PaddleProbe):
    METRIC_PREFIX = "moe_health"
    MAX_AGGREGATED = {
        "score_sum_max",
        "expert_norm_max",
        "expert_bias_max",
        "router_input_abs_max",
        "router_input_abs_p99",
        "router_logit_abs_max",
        "assignment_load_max_frac",
        "assignment_load_max_min_ratio",
        "gate_mass_max_frac",
        "gate_mass_max_min_ratio",
    }
    MIN_AGGREGATED = {
        "score_sum_min",
        "expert_norm_min",
        "expert_bias_min",
        "assignment_load_min_frac",
        "gate_mass_min_frac",
        "router_margin_min",
    }

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self._patched_gates = []
        self._expert_norm_layers = []

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        moe_layers = self._find_moe_layers(model)
        if not moe_layers:
            logger.warning("[PaddleMoEMonitor] No MoE layers found!")
            return
        if self.verbose:
            logger.info(f"[PaddleMoEMonitor] Found {len(moe_layers)} MoE layers.")

        # Declare metric schema
        for layer_idx, moe_layer in moe_layers:
            gate_metrics = ["router_entropy", "score_sum_mean", "score_sum_min", "score_sum_max"]
            gate_metrics += [
                "router_input_rms",
                "router_input_abs_max",
                "router_input_abs_p99",
                "router_logit_mean",
                "router_logit_std",
                "router_logit_abs_max",
                "prob_entropy_norm",
                "router_margin_mean",
                "router_margin_min",
                "router_margin_p10",
                "router_margin_p01",
            ]
            for prefix in ("assignment_load", "gate_mass"):
                gate_metrics += [
                    f"{prefix}_{suffix}"
                    for suffix in ("cv", "entropy_norm", "kl_uniform", "max_frac", "min_frac", "max_min_ratio")
                ]
            if hasattr(moe_layer, "gate") and hasattr(moe_layer.gate, "e_score_correction_bias"):
                gate_metrics += [
                    "bias_affinity_jaccard",
                    "expert_bias_mean",
                    "expert_bias_std",
                    "expert_bias_max",
                    "expert_bias_min",
                ]
            expert_metrics = [
                "expert_norm_mean",
                "expert_norm_std",
                "expert_norm_min",
                "expert_norm_max",
                "shared_expert_norm",
                "shared_routed_ratio",
            ]
            for m in gate_metrics + expert_metrics:
                self.declare_layer_metric(layer_idx, m)

        self.allocate_buffers()

        self._expert_norm_layers = []
        for layer_idx, moe_layer in moe_layers:
            if hasattr(moe_layer, "gate"):
                self._patch_gate_cache(moe_layer.gate)
                hook = moe_layer.gate.register_forward_post_hook(self._make_gate_hook(layer_idx, moe_layer))
                self.hooks.append(hook)
            # Expert weight norms are NOT collected from a forward hook: under
            # offline FP8 quant the bf16 expert weights are cleared at step
            # begin. collect_expert_norms() reads them before quant instead.
            self._expert_norm_layers.append((layer_idx, moe_layer))

        logger.info(
            f"[PaddleMoEMonitor] Registered {len(self.hooks)} gate hooks and "
            f"{len(self._expert_norm_layers)} expert-norm layers on {len(moe_layers)} MoE layers."
        )
        self._set_compatible_sources(len(moe_layers))

    def _patch_gate_cache(self, gate):
        """Monkey-patch gate.gate_score_func to cache pre-bias gates."""
        if not hasattr(gate, "gate_score_func"):
            if self.verbose:
                logger.warning("[PaddleMoEMonitor] Gate has no gate_score_func; router metrics may be unavailable")
            return
        if hasattr(gate, "_im_patched"):
            if self.verbose:
                logger.warning("[PaddleMoEMonitor] Gate is already patched; skipping duplicate patch")
            return
        original_fn = gate.gate_score_func
        monitor = self

        def cached_gate_score_func(logits):
            gate._cached_logits = logits.detach()
            result = original_fn(logits)
            if monitor._should_monitor():
                gate._cached_gates = result.detach()
            else:
                gate._cached_gates = None
            return result

        gate._im_original_gate_score_func = original_fn
        gate.gate_score_func = cached_gate_score_func

        # Also patch _hash_routing for hash-routed layers (DeepSeek V4+).
        # Hash layers return early from forward() without calling gate_score_func,
        # so we intercept _hash_routing to capture the scores computed there.
        if hasattr(gate, "_hash_routing"):
            original_hash_routing = gate._hash_routing

            def cached_hash_routing(logits, flat_ids):
                result = original_hash_routing(logits, flat_ids)
                if not monitor._should_monitor():
                    gate._cached_gates = None
                    return result

                # _hash_routing computes scores internally. Recompute the full
                # [N, num_experts] distribution so router metrics can use it.
                import paddle.nn.functional as F

                logits_fp32 = logits.cast("float32")
                scoring_func = getattr(gate, "scoring_func", "softmax")
                if scoring_func == "softmax":
                    scores = F.softmax(logits_fp32, axis=-1)
                elif scoring_func == "sigmoid":
                    scores = F.sigmoid(logits_fp32)
                elif scoring_func == "sqrtsoftplus":
                    scores = paddle.sqrt(F.softplus(logits_fp32) + 1e-20)
                else:
                    gate._cached_gates = None
                    return result
                gate._cached_gates = scores.detach()
                return result

            gate._im_original_hash_routing = original_hash_routing
            gate._hash_routing = cached_hash_routing

        gate._im_patched = True
        self._patched_gates.append(gate)

    def _find_moe_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        def has_moe(layer):
            return (
                (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"))
                or hasattr(layer, "moe")
                or hasattr(layer, "gate")
            )

        layers = get_decoder_layers(model)
        if layers is None:
            for _name, sublayer in model.named_sublayers():
                if sublayer.__class__.__name__ == "MoELayer":
                    layers = [] if layers is None else layers
                    layers.append(sublayer)
            if layers is None:
                return []

        monitor_layers = iter_monitor_layers(layers, has_moe, pp_rank=self.pp_rank)
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)
        moe_layers = []
        for item in monitor_layers:
            layer = item.layer
            moe_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                moe_module = layer.mlp
            elif hasattr(layer, "moe"):
                moe_module = layer.moe
            elif hasattr(layer, "gate"):
                moe_module = layer
            if moe_module is not None:
                moe_layers.append((item.idx, moe_module))
        return moe_layers

    def _make_gate_hook(self, layer_idx: int, moe_layer: nn.Layer):
        def hook_fn(layer, inputs, outputs):
            if not layer.training:
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                if hasattr(layer, "_cached_logits"):
                    layer._cached_logits = None
                return
            if not self._should_monitor():
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                if hasattr(layer, "_cached_logits"):
                    layer._cached_logits = None
                return
            try:
                with paddle.no_grad():
                    self._compute_gate_metrics(layer_idx, layer, inputs, outputs, moe_layer)
            except Exception as e:
                self._record_error()
                if self.verbose:
                    logger.error(f"[PaddleMoEMonitor] Gate hook error layer {layer_idx}: {e}")
            finally:
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                if hasattr(layer, "_cached_logits"):
                    layer._cached_logits = None

        return hook_fn

    def collect_expert_norms(self):
        """Compute per-layer expert weight norms for all monitored MoE layers."""
        if not self._buffers_allocated or not self._should_monitor():
            return
        for layer_idx, moe_layer in self._expert_norm_layers:
            try:
                with paddle.no_grad():
                    self._compute_expert_metrics(layer_idx, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMoEMonitor] expert-norm collect error layer {layer_idx}: {e}")

    def _compute_gate_metrics(self, layer_idx, gate, inputs, outputs, moe_layer):
        """Compute router metrics from gate forward output."""
        cached_gates = getattr(gate, "_cached_gates", None)
        k = getattr(gate, "num_experts_per_tok", None)

        if cached_gates is None:
            if self.verbose:
                logger.warning(f"[PaddleMoEMonitor] layer {layer_idx}: _cached_gates is None, gate patch may not work")
            return

        router_input = inputs[0] if inputs and isinstance(inputs[0], paddle.Tensor) else None
        if router_input is not None:
            router_input = router_input.detach().astype("float32")
            absolute = router_input.abs()
            self.record_layer_metric(layer_idx, "router_input_rms", paddle.sqrt(router_input.square().mean()))
            self.record_layer_metric(layer_idx, "router_input_abs_max", absolute.max())
            self.record_layer_metric(layer_idx, "router_input_abs_p99", paddle.quantile(absolute.reshape([-1]), 0.99))
        logits = getattr(gate, "_cached_logits", None)
        if logits is not None:
            logits = logits.astype("float32")
            self.record_layer_metric(layer_idx, "router_logit_mean", logits.mean())
            self.record_layer_metric(layer_idx, "router_logit_std", logits.std())
            self.record_layer_metric(layer_idx, "router_logit_abs_max", logits.abs().max())

        self.record_layer_metric(layer_idx, "router_entropy", _compute_router_entropy(cached_gates))
        num_experts = int(cached_gates.shape[-1])
        self.record_layer_metric(
            layer_idx,
            "prob_entropy_norm",
            _compute_router_entropy(cached_gates) / math.log(num_experts)
            if num_experts > 1
            else cached_gates.sum() * 0 + 1,
        )
        if k is not None:
            topk_vals, topk_idx = paddle.topk(cached_gates, k, axis=-1)
            score_sum = topk_vals.sum(axis=-1)
            self.record_layer_metric(layer_idx, "score_sum_mean", score_sum.mean())
            self.record_layer_metric(layer_idx, "score_sum_min", score_sum.min())
            self.record_layer_metric(layer_idx, "score_sum_max", score_sum.max())
            if k < num_experts:
                boundary = paddle.topk(cached_gates, k + 1, axis=-1, sorted=True)[0]
                margin = boundary[..., k - 1] - boundary[..., k]
                self.record_layer_metric(layer_idx, "router_margin_mean", margin.mean())
                self.record_layer_metric(layer_idx, "router_margin_min", margin.min())
                self.record_layer_metric(layer_idx, "router_margin_p10", paddle.quantile(margin, 0.10))
                self.record_layer_metric(layer_idx, "router_margin_p01", paddle.quantile(margin, 0.01))

            actual_idx = topk_idx
            if isinstance(outputs, tuple) and len(outputs) >= 3 and isinstance(outputs[2], paddle.Tensor):
                actual_idx = outputs[2].astype("int64").reshape([-1, k])
            assignment, gate_mass = _assignment_mass(cached_gates, actual_idx)
            for prefix, values in (
                ("assignment_load", _distribution_metrics(assignment)),
                ("gate_mass", _distribution_metrics(gate_mass)),
            ):
                for name, value in values.items():
                    self.record_layer_metric(layer_idx, f"{prefix}_{name}", value)

        if hasattr(gate, "e_score_correction_bias"):
            top_idx_with_bias = None
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                top_idx_with_bias = outputs[2]
            if top_idx_with_bias is not None and k is not None:
                n_group = getattr(gate, "n_group", 1)
                topk_group = getattr(gate, "topk_group", 1)
                self.record_layer_metric(
                    layer_idx,
                    "bias_affinity_jaccard",
                    _compute_bias_affinity_jaccard(top_idx_with_bias, cached_gates, k, n_group, topk_group),
                )
            bias = gate.e_score_correction_bias
            self.record_layer_metric(layer_idx, "expert_bias_mean", bias.mean())
            self.record_layer_metric(layer_idx, "expert_bias_std", bias.std())
            self.record_layer_metric(layer_idx, "expert_bias_max", bias.max())
            self.record_layer_metric(layer_idx, "expert_bias_min", bias.min())
        self._record_observation(layer_idx)

    def _compute_expert_metrics(self, layer_idx, moe_layer):
        routed_norm_mean = None

        # grouped-gemm experts: weight1/weight2 are [num_experts, ...] blocks.
        if hasattr(moe_layer, "grouped_gemm_experts") and moe_layer.grouped_gemm_experts is not None:
            ggm = moe_layer.grouped_gemm_experts
            if hasattr(ggm, "weight1") and hasattr(ggm, "weight2"):
                norms = _per_expert_stacked_norms(ggm.weight1, ggm.weight2)
                norm_stats = _norm_stats(norms)
                for name, val in norm_stats.items():
                    self.record_layer_metric(layer_idx, name, val)
                routed_norm_mean = norm_stats.get("expert_norm_mean")

        elif hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            experts = moe_layer.experts
            norms = None
            # Fused-expert layout (moe_expert_fusion=True): self.experts is a
            # single module whose up_gate_proj/down_proj weights carry a leading
            # expert dim [num_experts, ...]. Vectorize over that dim.
            if hasattr(experts, "up_gate_proj") and hasattr(experts, "down_proj"):
                w1 = experts.up_gate_proj.weight
                w2 = experts.down_proj.weight
                norms = _per_expert_stacked_norms(w1, w2)
            elif isinstance(experts, (list, nn.LayerList)) or hasattr(experts, "__iter__"):
                # Non-fused layout: LayerList of per-expert modules. One sum-sq
                # per expert (each is a small handful of params), then stack.
                per_expert = []
                for expert in experts:
                    if expert is None:
                        continue
                    sq = _module_sumsq(expert)
                    if sq is not None:
                        per_expert.append(paddle.sqrt(sq))
                if per_expert:
                    norms = paddle.stack(per_expert)
            if norms is not None:
                norm_stats = _norm_stats(norms)
                for name, val in norm_stats.items():
                    self.record_layer_metric(layer_idx, name, val)
                routed_norm_mean = norm_stats.get("expert_norm_mean")

        if hasattr(moe_layer, "shared_experts") and moe_layer.shared_experts is not None:
            shared_sq = _module_sumsq(moe_layer.shared_experts)
            if shared_sq is not None:
                shared_norm = paddle.sqrt(shared_sq)
                self.record_layer_metric(layer_idx, "shared_expert_norm", shared_norm)
                if routed_norm_mean is not None:
                    # clip 防止除零（对齐 megatron compute_shared_routed_ratio），保持 GPU 张量无 D2H
                    self.record_layer_metric(
                        layer_idx, "shared_routed_ratio", shared_norm / routed_norm_mean.clip(min=1e-8)
                    )

    def remove_hooks(self):
        super().remove_hooks()
        for gate in self._patched_gates:
            original_fn = getattr(gate, "_im_original_gate_score_func", None)
            if original_fn is not None:
                gate.gate_score_func = original_fn
            original_hash_routing = getattr(gate, "_im_original_hash_routing", None)
            if original_hash_routing is not None:
                gate._hash_routing = original_hash_routing
            for attr in (
                "_im_original_gate_score_func",
                "_im_original_hash_routing",
                "_im_patched",
                "_cached_gates",
                "_cached_logits",
            ):
                if hasattr(gate, attr):
                    delattr(gate, attr)
        self._patched_gates = []
        self._expert_norm_layers = []

    def step(self):
        super().step()


def setup_moe_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
):
    monitor = PaddleMoEMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    logger.info(f"[PaddleMoEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["moe_health"] = monitor
    return model
