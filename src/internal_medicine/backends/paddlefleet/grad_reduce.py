"""Backward-compat shim: the planner now lives in :mod:`internal_medicine.core.grad_reduce`.

The reduction planner is backend-agnostic (pure Python), so it was moved to
``core`` to be shared with the Megatron ``grad_health`` monitor. This module
re-exports it unchanged so existing paddle imports and tests keep working.
"""

from __future__ import annotations

from ...core.grad_reduce import (
    DATA_CANDIDATES,
    SHARD_CANDIDATES,
    GroupInfo,
    describe,
    is_orthogonal,
    plan_total_groups,
)

__all__ = [
    "DATA_CANDIDATES",
    "SHARD_CANDIDATES",
    "GroupInfo",
    "describe",
    "is_orthogonal",
    "plan_total_groups",
]
