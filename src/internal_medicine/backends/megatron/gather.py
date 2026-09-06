"""Megatron distributed aggregation primitives for training_logs."""

import logging

logger = logging.getLogger(__name__)


def _torch_gather(local_metrics: dict) -> list:
    """Gather metrics from all ranks using torch.distributed.

    Legacy path: ``O(keys x world_size)`` pickled dicts. Kept as the fallback for
    when the fixed-schema reduce below cannot be used.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    info_list = [None] * world_size
    dist.all_gather_object(info_list, local_metrics)
    return info_list


def _torch_reduce(sum_vec: list, max_vec: list, min_vec: list):
    """SUM / MAX / MIN all_reduce over the whole world, float64.

    float64 keeps the crc32 schema checksum and the presence counts exact, and
    the vectors are ``O(distinct keys)`` — tens of KB regardless of world size.
    Called at flush time, never from a hook.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return None

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    reduced = []
    for vec, op in (
        (sum_vec, dist.ReduceOp.SUM),
        (max_vec, dist.ReduceOp.MAX),
        (min_vec, dist.ReduceOp.MIN),
    ):
        if not vec:
            reduced.append([])
            continue
        tensor = torch.tensor(vec, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=op)
        reduced.append(tensor.tolist())
    return tuple(reduced)


def _torch_schema_gather(local_keys: list) -> list:
    """All_gather the key lists over the pipeline group.

    Ranks sharing a pipeline stage own the same layers and declare the same keys,
    so the pipeline group's union is already the world union — and the group is
    ``pp_size`` wide, which keeps this one-off pickle cheap. training_logs
    cross-checks a schema checksum on every reduce and falls back to
    ``_torch_gather`` if the ranks disagree.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return None

    group = None
    try:
        from megatron.core import parallel_state

        group = parallel_state.get_pipeline_model_parallel_group()
    except Exception:
        logger.warning(
            "[InternalMedicine/megatron] pipeline group unavailable; "
            "syncing the metric schema over the full world instead (one-off, but large at scale)"
        )

    try:
        group_size = dist.get_world_size(group=group)
        # With PP=1 every rank already holds every layer, so the local key set
        # *is* the global schema and there is nothing to union.
        if group_size <= 1:
            return [list(local_keys)]
        gathered = [None] * group_size
        dist.all_gather_object(gathered, list(local_keys), group=group)
    except Exception as exc:
        # Degrade to a local-only schema: the fixed-size probe in training_logs
        # will notice if that disagrees across ranks and fall back to the legacy
        # path, so this cannot turn into a hang or a wrong number.
        logger.warning(f"[InternalMedicine/megatron] metric schema sync failed ({exc}); using local keys only")
        return [list(local_keys)]
    return gathered


def install_gather_fn():
    """Install the torch-based aggregation primitives into the global training_logs."""
    from ...core.training_logs import training_logs

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            training_logs.set_gather_fn(_torch_gather)
            training_logs.set_schema_sync_fn(_torch_schema_gather)
            training_logs.set_reduce_fn(_torch_reduce)
    except ImportError:
        pass
