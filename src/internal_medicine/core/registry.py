"""Backend detection and dispatch."""

import logging

logger = logging.getLogger(__name__)

AVAILABLE_MONITORS = {
    "megatron": ["qk_stats", "moe_health", "ple_health", "massive_act", "mhc_health"],
    "paddlefleet": ["ape_health", "qk_stats", "moe_health", "massive_act", "mhc_health", "vha_health"],
}


def detect_backend() -> str:
    try:
        import megatron.core  # noqa: F401

        return "megatron"
    except ImportError:
        pass
    try:
        import paddle  # noqa: F401

        return "paddlefleet"
    except ImportError:
        pass
    raise RuntimeError("No supported backend found. Install megatron-core (torch) or paddlepaddle.")


def get_backend_setup_fn(backend: str = None):
    """Return the setup_monitors function for the given backend."""
    backend = backend or detect_backend()
    if backend == "megatron":
        from ..backends.megatron import setup_monitors

        return setup_monitors
    elif backend == "paddlefleet":
        from ..backends.paddlefleet import setup_monitors

        return setup_monitors
    else:
        raise ValueError(f"Unknown backend: {backend}")
