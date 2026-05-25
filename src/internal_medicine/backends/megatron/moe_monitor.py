"""
MoE Specialist Monitor for Megatron-Bridge.
Migrated from src/internal_medicine/moe_specialist/.
"""

import logging
import weakref
from typing import Any

import torch
import torch.nn as nn

from ...core.base_monitor import Probe
from ...core.training_logs import training_logs
from .moe_metrics import (
    compute_bias_affinity_jaccard,
    compute_expert_norms,
    compute_router_entropy,
    compute_shared_expert_norm,
    compute_shared_routed_ratio,
    compute_topk_score_sum,
)

logger = logging.getLogger(__name__)


class MoESpecialistMonitor(Probe):
    METRIC_PREFIX = "moe_health"
    MAX_AGGREGATED = {"score_sum_max", "expert_norm_max"}
    MIN_AGGREGATED = {"score_sum_min", "expert_norm_min"}

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.moe_layers: list[weakref.ref] = []
        self._patched_routers: list[weakref.ref] = []
        self._pending_router_global_layers: set[int] = set()

    def register_hooks(self, model: nn.Module):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

        moe_layers = self._find_moe_layers(model)
        if len(moe_layers) == 0:
            logger.warning("[MoEMonitor] No MoE layers found!")
            return
        if self.verbose:
            logger.info(f"[MoEMonitor] Found {len(moe_layers)} MoE layers.")

        for layer_idx, moe_layer in moe_layers:
            self.moe_layers.append(weakref.ref(moe_layer))
            if hasattr(moe_layer, "router"):
                self._patch_router_cache(moe_layer.router)
                hook = moe_layer.router.register_forward_hook(self._make_router_hook(layer_idx, moe_layer))
                self.hooks.append(hook)
            hook = moe_layer.register_forward_hook(self._make_moe_layer_hook(layer_idx, moe_layer))
            self.hooks.append(hook)

        logger.info(f"[MoEMonitor] Registered {len(self.hooks)} hooks on {len(moe_layers)} layers.")

    def _patch_router_cache(self, router):
        """Monkey-patch router._apply_aux_loss to intercept scores_for_aux_loss.

        Inside routing(), compute_routing_scores_for_aux_loss is already called
        and the result is passed to _apply_aux_loss. We intercept there to cache
        the data with zero additional compute (just detach).
        """
        if not hasattr(router, "_apply_aux_loss"):
            if self.verbose:
                logger.warning("[MoEMonitor] Router has no _apply_aux_loss; router metrics may be unavailable")
            return
        if getattr(router, "_im_patched", False):
            if self.verbose:
                logger.warning("[MoEMonitor] Router is already patched; skipping duplicate patch")
            return
        original_apply = router._apply_aux_loss
        monitor = self

        def patched_apply(probs, scores_for_aux_loss, routing_map, *args, **kwargs):
            if monitor._should_monitor():
                router._cached_scores_for_aux_loss = scores_for_aux_loss.detach()
                router._cached_routing_map_for_aux_loss = routing_map.detach()
            else:
                router._cached_scores_for_aux_loss = None
                router._cached_routing_map_for_aux_loss = None
            return original_apply(probs, scores_for_aux_loss, routing_map, *args, **kwargs)

        router._im_original_apply_aux_loss = original_apply
        router._apply_aux_loss = patched_apply
        router._im_patched = True
        self._patched_routers.append(weakref.ref(router))

    def _find_moe_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        moe_layers = []
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers
            elif hasattr(lm, "encoder") and hasattr(lm.encoder, "layers"):
                layers = lm.encoder.layers
        if layers is None:
            for _, module in model.named_modules():
                if module.__class__.__name__ in ("MoELayer", "BaseMoELayer"):
                    moe_layers.append((len(moe_layers), module))
            return moe_layers
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            moe_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_module = layer.mlp
            elif hasattr(layer, "moe"):
                moe_module = layer.moe
            elif hasattr(layer, "router"):
                moe_module = layer
            if moe_module is not None:
                moe_layers.append((global_idx, moe_module))
        return moe_layers

    def _resolve_layer_idx(self, layer: nn.Module, local_idx: int, num_local_layers: int) -> int:
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + local_idx

    def _make_router_hook(self, layer_idx: int, moe_layer: nn.Module):
        def hook_fn(module, _inputs, outputs):
            if not torch.is_grad_enabled():
                for attr in ("_cached_scores_for_aux_loss", "_cached_routing_map_for_aux_loss"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)
                return
            if not self._should_monitor():
                for attr in ("_cached_scores_for_aux_loss", "_cached_routing_map_for_aux_loss"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)
                return
            try:
                with torch.no_grad():
                    self._compute_router_metrics(layer_idx, module, outputs, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MoEMonitor] Router hook error layer {layer_idx}: {e}")
            finally:
                for attr in ("_cached_scores_for_aux_loss", "_cached_routing_map_for_aux_loss"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)

        return hook_fn

    def _make_moe_layer_hook(self, layer_idx: int, _moe_layer: nn.Module):
        def hook_fn(module, _inputs, _outputs):
            if not torch.is_grad_enabled():
                return
            if not self._should_monitor():
                return
            try:
                with torch.no_grad():
                    self._compute_expert_metrics(layer_idx, module)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MoEMonitor] MoE layer hook error layer {layer_idx}: {e}")

        return hook_fn

    def _compute_router_metrics(self, layer_idx, router, outputs, _moe_layer):
        metrics = {}
        topk = getattr(router, "topk", None)
        scores_for_aux_loss = getattr(router, "_cached_scores_for_aux_loss", None)

        if scores_for_aux_loss is not None:
            metrics["router_entropy"] = compute_router_entropy(scores_for_aux_loss).item()
            if topk is not None:
                score_sum_stats = compute_topk_score_sum(scores_for_aux_loss, topk)
                metrics["score_sum_mean"] = score_sum_stats["score_sum_mean"].item()
                metrics["score_sum_min"] = score_sum_stats["score_sum_min"].item()
                metrics["score_sum_max"] = score_sum_stats["score_sum_max"].item()

        if hasattr(router, "expert_bias") and router.expert_bias is not None:
            metrics["expert_bias_mean"] = router.expert_bias.mean().item()
            metrics["expert_bias_std"] = router.expert_bias.std().item()

        routing_map_for_aux_loss = getattr(router, "_cached_routing_map_for_aux_loss", None)
        if routing_map_for_aux_loss is not None and isinstance(outputs, tuple) and len(outputs) >= 2:
            routing_after = outputs[1]
            jaccard = compute_bias_affinity_jaccard(routing_map_for_aux_loss, routing_after)
            metrics["bias_affinity_jaccard"] = jaccard.item()

        # Per-layer log + global accumulate (no count increment — expert hook does that)
        if self.log_per_layer and metrics:
            training_logs.update(**{f"{self.METRIC_PREFIX}/layer_{layer_idx}/{k}": v for k, v in metrics.items()})
        if self.log_global and metrics:
            self._accumulate_global(metrics)
            self._pending_router_global_layers.add(layer_idx)

    def _compute_expert_metrics(self, layer_idx, moe_layer):
        metrics = {}
        routed_norm_mean = None

        if hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            experts = moe_layer.experts
            expert_weights = []
            if hasattr(experts, "weight1") and hasattr(experts, "weight2"):
                num_experts = experts.num_local_experts
                hidden_size = experts.config.hidden_size
                w1 = experts.weight1.data.view(num_experts, hidden_size, -1)
                w2 = experts.weight2.data.view(num_experts, -1, hidden_size)
                for i in range(num_experts):
                    combined = torch.cat([w1[i].flatten(), w2[i].flatten()])
                    expert_weights.append(combined)
            elif hasattr(experts, "linear_fc1"):
                num_experts = experts.num_local_experts
                for i in range(num_experts):
                    w1 = getattr(experts.linear_fc1, f"weight{i}", None)
                    w2 = getattr(experts.linear_fc2, f"weight{i}", None)
                    if w1 is not None and w2 is not None:
                        combined = torch.cat([w1.data.flatten(), w2.data.flatten()])
                        expert_weights.append(combined)
            elif hasattr(experts, "local_experts"):
                for expert in experts.local_experts:
                    weights = [p.data.flatten() for p in expert.parameters()]
                    if weights:
                        expert_weights.append(torch.cat(weights))

            if expert_weights:
                norm_stats = compute_expert_norms(expert_weights)
                metrics["expert_norm_mean"] = norm_stats["expert_norm_mean"].item()
                metrics["expert_norm_std"] = norm_stats["expert_norm_std"].item()
                metrics["expert_norm_min"] = norm_stats["expert_norm_min"].item()
                metrics["expert_norm_max"] = norm_stats["expert_norm_max"].item()
                routed_norm_mean = norm_stats["expert_norm_mean"]

        if hasattr(moe_layer, "shared_experts") and moe_layer.shared_experts is not None:
            shared_weights = [p.data for p in moe_layer.shared_experts.parameters()]
            if shared_weights:
                shared_norm = compute_shared_expert_norm(shared_weights)
                metrics["shared_expert_norm"] = shared_norm.item()
                if routed_norm_mean is not None:
                    ratio = compute_shared_routed_ratio(shared_norm, routed_norm_mean.clone().detach())
                    metrics["shared_routed_ratio"] = ratio.item()

        has_pending_router = layer_idx in self._pending_router_global_layers

        # _record_metrics handles per-layer log + global accumulate + count++
        if metrics:
            self._record_metrics(layer_idx, metrics)
        elif self.log_global and has_pending_router:
            self._global_count += 1
        self._pending_router_global_layers.discard(layer_idx)

    def remove_hooks(self):
        super().remove_hooks()
        for router_ref in self._patched_routers:
            router = router_ref()
            if router is None:
                continue
            original_apply = getattr(router, "_im_original_apply_aux_loss", None)
            if original_apply is not None:
                router._apply_aux_loss = original_apply
            for attr in (
                "_im_original_apply_aux_loss",
                "_im_patched",
                "_cached_scores_for_aux_loss",
                "_cached_routing_map_for_aux_loss",
            ):
                if hasattr(router, attr):
                    delattr(router, attr)
        self.moe_layers = []
        self._patched_routers = []
        self._pending_router_global_layers.clear()

    def step(self):
        super().step()
        self._pending_router_global_layers.clear()

    def get_health_summary(self) -> dict[str, Any]:
        metrics = training_logs.get_latest(prefix="moe_health")
        summary = {"num_layers_monitored": len(self.moe_layers), "total_steps": self.step_count}
        for key, val in metrics.items():
            if "bias_affinity_jaccard" in key:
                summary["router_conflict"] = "SEVERE" if val < 0.3 else "WARNING" if val < 0.7 else "OK"
            if "shared_routed_ratio" in key:
                summary["shared_expert"] = "MONOPOLY" if val > 3.0 else "INEFFECTIVE" if val < 0.3 else "OK"
        return summary


def setup_moe_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
):
    monitor = MoESpecialistMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    models = [model] if not isinstance(model, list) else model
    for m in models:
        monitor.register_hooks(m)
    logger.info(f"[MoEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["moe_health"] = monitor
    return model
