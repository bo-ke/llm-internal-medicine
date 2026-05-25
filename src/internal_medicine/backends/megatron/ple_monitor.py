"""
PLE Health Monitor for Megatron-Bridge.
Migrated from src/internal_medicine/ple_health/.
"""

import logging

import torch
import torch.nn as nn

from ...core.base_monitor import Probe
from ...core.training_logs import training_logs
from .ple_metrics import (
    compute_branch_cosine,
    compute_branch_norms,
    compute_gate_stats,
    compute_residual_ratio,
)

logger = logging.getLogger(__name__)


class PLEHealthMonitor(Probe):
    METRIC_PREFIX = "ple_health"

    def __init__(
        self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False, gate_sparsity_threshold=0.01
    ):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.gate_sparsity_threshold = gate_sparsity_threshold
        self._token_ple_buf: torch.Tensor | None = None
        self._proj_ple_buf: torch.Tensor | None = None
        self._gate_out_buf: dict[int, torch.Tensor] = {}
        self._hidden_size: int = 0
        self._num_layers: int = 0
        self._hidden_size_ple: int = 0

    def register_hooks(self, model: nn.Module):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

        ple_model = self._find_ple_model(model)
        if ple_model is not None:
            config = ple_model.config
            self._hidden_size = config.hidden_size
            self._num_layers = config.num_layers
            self._hidden_size_ple = config.hidden_size_per_layer_input
            h1 = ple_model.embed_tokens_per_layer.register_forward_hook(self._make_token_ple_hook())
            h2 = ple_model.per_layer_projection_norm.register_forward_hook(self._make_proj_ple_hook())
            h3 = ple_model.register_forward_hook(self._make_model_post_hook())
            self.hooks.extend([h1, h2, h3])

        ple_layers = self._find_ple_layers(model)
        for layer_idx, ple_submodule in ple_layers:
            h_gate = ple_submodule.gate_proj.register_forward_hook(self._make_gate_hook(layer_idx))
            h_ple = ple_submodule.register_forward_hook(self._make_ple_layer_hook(layer_idx, ple_submodule))
            self.hooks.extend([h_gate, h_ple])

        logger.info(f"[PLEMonitor] Registered {len(self.hooks)} hooks ({len(ple_layers)} PLE layers).")

    def _find_ple_model(self, model: nn.Module) -> nn.Module | None:
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "embed_tokens_per_layer"):
            return model
        return None

    def _find_ple_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        from megatron.core.transformer.identity_op import IdentityOp

        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers
        if layers is None:
            return []
        result = []
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            if hasattr(layer, "ple") and not isinstance(layer.ple, IdentityOp):
                result.append((global_idx, layer.ple))
        return result

    def _resolve_layer_idx(self, layer: nn.Module, local_idx: int, num_local_layers: int) -> int:
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + local_idx

    def _make_token_ple_hook(self):
        num_layers = self._num_layers
        H_ple = self._hidden_size_ple

        def hook_fn(module, inputs, output):
            if not torch.is_grad_enabled() or not self._should_monitor():
                return
            with torch.no_grad():
                B, S, _ = output.shape
                token_ple = output.transpose(0, 1).contiguous()
                self._token_ple_buf = token_ple.view(S, B, num_layers, H_ple)

        return hook_fn

    def _make_proj_ple_hook(self):
        H = self._hidden_size

        def hook_fn(module, inputs, output):
            if not torch.is_grad_enabled() or not self._should_monitor():
                return
            with torch.no_grad():
                self._proj_ple_buf = output * (H**-0.5)

        return hook_fn

    def _make_model_post_hook(self):
        per_layer_input_scale = 2.0**-0.5

        def hook_fn(module, inputs, output):
            if not torch.is_grad_enabled() or not self._should_monitor():
                return
            if self._token_ple_buf is None or self._proj_ple_buf is None:
                return
            try:
                with torch.no_grad():
                    token_ple = self._token_ple_buf
                    proj_ple = self._proj_ple_buf
                    norms = compute_branch_norms(token_ple, proj_ple, per_layer_input_scale)
                    cosine = compute_branch_cosine(token_ple, proj_ple)
                    log_dict = {}
                    for name, val in norms.items():
                        log_dict[f"ple_health/global_{name}"] = val.item()
                    log_dict["ple_health/global_token_proj_cosine"] = cosine.item()
                    training_logs.update(**log_dict)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PLEMonitor] Model post-hook error: {e}")
            finally:
                self._token_ple_buf = None
                self._proj_ple_buf = None

        return hook_fn

    def _make_gate_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not torch.is_grad_enabled() or not self._should_monitor():
                return
            with torch.no_grad():
                gate_out = output[0] if isinstance(output, tuple) else output
                self._gate_out_buf[layer_idx] = gate_out

        return hook_fn

    def _make_ple_layer_hook(self, layer_idx: int, ple_submodule: nn.Module):
        act_fn = ple_submodule.act_fn
        threshold = self.gate_sparsity_threshold

        def hook_fn(module, inputs, output):
            if not torch.is_grad_enabled() or not self._should_monitor():
                return
            try:
                with torch.no_grad():
                    hidden_states = inputs[0]
                    metrics = {}
                    ratio = compute_residual_ratio(hidden_states, output)
                    metrics["residual_ratio"] = ratio.item()
                    gate_out = self._gate_out_buf.pop(layer_idx, None)
                    if gate_out is not None:
                        gate_stats = compute_gate_stats(gate_out, act_fn, threshold)
                        metrics["gate_activation_mean"] = gate_stats["gate_activation_mean"].item()
                        metrics["gate_sparsity"] = gate_stats["gate_sparsity"].item()
                    self._record_metrics(layer_idx, metrics)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PLEMonitor] Layer {layer_idx} hook error: {e}")

        return hook_fn

    def step(self):
        super().step()
        self._token_ple_buf = None
        self._proj_ple_buf = None
        self._gate_out_buf.clear()


def setup_ple_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    gate_sparsity_threshold=0.01,
    monitor_dict=None,
    return_monitor=False,
):
    monitor = PLEHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        gate_sparsity_threshold=gate_sparsity_threshold,
    )
    models = [model] if not isinstance(model, list) else model
    for m in models:
        monitor.register_hooks(m)
    logger.info(f"[PLEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["ple_health"] = monitor
    if return_monitor:
        return model, monitor
    return model
