"""Megatron-Bridge backend for internal_medicine."""

import logging

from .activation_dump_monitor import ActivationDumpMonitor, setup_activation_dump_monitor
from .base import TorchProbe
from .gather import install_gather_fn
from .lar_monitor import LARMonitor, setup_lar_monitor
from .massive_activation_monitor import MassiveActivationMonitor, setup_massive_activation_monitor
from .mhc_monitor import MHCHealthMonitor, setup_mhc_monitor
from .moe_monitor import MoESpecialistMonitor, setup_moe_monitor
from .optim_update_monitor import OptimUpdateMonitor, setup_optim_update_monitor
from .ple_monitor import PLEHealthMonitor, setup_ple_monitor
from .qk_monitor import QKStatsMonitor, setup_qk_monitor

logger = logging.getLogger(__name__)

_ALL_MONITORS = {
    "qk_stats": setup_qk_monitor,
    "moe_health": setup_moe_monitor,
    "ple_health": setup_ple_monitor,
    "massive_act": setup_massive_activation_monitor,
    "mhc_health": setup_mhc_monitor,
}

_MONITOR_MAP = {
    **_ALL_MONITORS,
    "act_dump": setup_activation_dump_monitor,
    "lar": setup_lar_monitor,
}

# Always installed, regardless of the ``monitors`` spec: three scalars per step read off
# the optimizer's own main params, with no forward hooks and no per-layer keys. It patches
# the mcore optimizer classes rather than an instance, so it can be set up from here even
# though the optimizer does not exist yet.
_ALWAYS_ON_MONITORS = {"optim": setup_optim_update_monitor}


def _expand_monitor_names(monitors) -> list[str]:
    """Resolve a monitor spec to an ordered, de-duplicated list of monitor names.

    ``"all"`` expands to ``_ALL_MONITORS``. Anything the caller names alongside it is an
    explicit opt-in and is kept, so ``["all", "act_dump"]`` does enable the dump.
    """
    if monitors is None:
        monitors = ["all"]
    if isinstance(monitors, str):
        monitors = [monitors]
    if "all" not in monitors:
        return list(dict.fromkeys(monitors))
    expanded = list(_ALL_MONITORS) + [name for name in monitors if name != "all"]
    return list(dict.fromkeys(expanded))


def setup_monitors(model, monitors=None, monitor_dict=None, monitor_interval=1, verbose=False, **kwargs):
    """Setup all requested monitors on a Megatron model."""
    install_gather_fn()
    hook_timing_enabled = bool(kwargs.pop("hook_timing_enabled", False))

    monitors = _expand_monitor_names(monitors)
    if monitor_dict is None:
        monitor_dict = {}

    for name in [*_ALWAYS_ON_MONITORS, *monitors]:
        setup_fn = _ALWAYS_ON_MONITORS.get(name) or _MONITOR_MAP.get(name)
        if setup_fn is None:
            logger.warning(f"[InternalMedicine/megatron] Unknown monitor: {name}, skipping")
            continue
        try:
            setup_fn(
                model,
                monitor_dict=monitor_dict,
                monitor_interval=monitor_interval,
                verbose=verbose,
                hook_timing_enabled=hook_timing_enabled,
                **kwargs.get(name, {}),
            )
            logger.info(f"[InternalMedicine/megatron] Enabled monitor: {name}")
        except Exception as e:
            logger.error(f"[InternalMedicine/megatron] Failed to setup {name}: {e}")

    return model


__all__ = [
    "setup_monitors",
    "TorchProbe",
    "QKStatsMonitor",
    "setup_qk_monitor",
    "MoESpecialistMonitor",
    "setup_moe_monitor",
    "PLEHealthMonitor",
    "setup_ple_monitor",
    "MassiveActivationMonitor",
    "setup_massive_activation_monitor",
    "MHCHealthMonitor",
    "setup_mhc_monitor",
    "ActivationDumpMonitor",
    "setup_activation_dump_monitor",
    "LARMonitor",
    "setup_lar_monitor",
    "OptimUpdateMonitor",
    "setup_optim_update_monitor",
]
