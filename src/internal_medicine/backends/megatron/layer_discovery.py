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

    if layers is None:
        return []
    return list(enumerate(layers))
