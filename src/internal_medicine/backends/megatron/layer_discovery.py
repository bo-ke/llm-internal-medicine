"""Where the decoder layers of a Megatron model chunk live.

Shared by the monitors that walk the whole stack (``massive_act``,
``grad_health``) so one model layout only has to be taught once.
"""

from __future__ import annotations

import torch.nn as nn


def find_transformer_layers(model: nn.Module) -> list[tuple[int, nn.Module]]:
    """``[(local_idx, layer)]`` for one model chunk, or ``[]`` if none are found.

    One ``.module`` unwrap is enough: monitors are installed from a pre-wrap hook,
    i.e. before ``Float16Module`` / DDP, so the chunk is the bare model. ``local_idx``
    is the index within the chunk -- callers turn it into a global layer id via
    ``Probe._resolve_layer_idx``.
    """
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
        return []
    return list(enumerate(layers))


def unwrap_layer(layer: nn.Module) -> nn.Module:
    """Peel layer-boundary wrappers off ``layer`` to reach the real sublayers.

    ``HyperConnectionHybridLayer`` (layer-boundary mHC on a ``HybridStack``) nests the
    actual transformer layer under ``inner_layer`` and proxies no attributes, so probes
    that look for ``mlp`` / ``input_layernorm`` must unwrap first. Identity for every
    other layout. Hooks that must fire still belong on the wrapper -- the wrapper's fast
    path never calls ``inner_layer.__call__``.
    """
    seen: set[int] = set()
    while True:
        inner = getattr(layer, "inner_layer", None)
        if not isinstance(inner, nn.Module) or id(layer) in seen:
            return layer
        seen.add(id(layer))
        layer = inner


def is_identity_module(module: nn.Module | None) -> bool:
    """Whether ``module`` is a placeholder standing in for an absent sublayer.

    mcore fills unused slots with an ``IdentityOp`` *instance* rather than ``None``
    (a hybrid layer is attention-only or MLP-only, so one slot is always a
    placeholder). Matched by class name to avoid importing mcore.
    """
    return module is None or type(module).__name__ in ("IdentityOp", "IdentityFuncOp", "Identity")


def resolve_mhc_streams(model: nn.Module) -> tuple[int, int | None]:
    """``(num_residual_streams, hidden_size)`` for layer-boundary mHC, ``(1, H)`` without.

    Cold path -- read once at hook-registration time and cached by the caller.
    """
    config = getattr(model, "config", None)
    if config is None:
        return 1, None
    hidden_size = getattr(config, "hidden_size", None)
    if not getattr(config, "enable_mhc_connections", False):
        return 1, hidden_size
    return getattr(config, "mhc_num_residual_streams", 1) or 1, hidden_size
