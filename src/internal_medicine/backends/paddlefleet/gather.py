"""PaddleFleet distributed aggregation primitives for training_logs."""

import logging

logger = logging.getLogger(__name__)


def _paddle_gather(local_metrics: dict) -> list:
    """Gather metrics from all ranks using paddle.distributed.

    Legacy path: ``O(keys x world_size)`` pickled dicts. Kept as the fallback for
    when the fixed-schema reduce below cannot be used.
    """
    import paddle.distributed as dist

    if not dist.is_initialized():
        return None
    info_list = []
    dist.all_gather_object(info_list, local_metrics)
    return info_list


def _paddle_reduce(sum_vec: list, max_vec: list, min_vec: list):
    """SUM / MAX / MIN all_reduce over the whole world, float64.

    float64 keeps the crc32 schema checksum and the presence counts exact, and
    the vectors are ``O(distinct keys)`` — tens of KB regardless of world size.
    Called at flush time, never from a hook, so it does not contend with the
    hot-path streams the monitor rules are about.
    """
    import paddle
    import paddle.distributed as dist

    if not dist.is_initialized():
        return None

    reduced = []
    for vec, op in (
        (sum_vec, dist.ReduceOp.SUM),
        (max_vec, dist.ReduceOp.MAX),
        (min_vec, dist.ReduceOp.MIN),
    ):
        if not vec:
            reduced.append([])
            continue
        tensor = paddle.to_tensor(vec, dtype="float64")
        dist.all_reduce(tensor, op=op)
        reduced.append(tensor.tolist())
    return tuple(reduced)


def _paddle_schema_gather(local_keys: list) -> list:
    """All_gather the key lists over the pipeline group.

    Ranks sharing a pipeline stage own the same layers and declare the same keys,
    so the pipeline group's union is already the world union — and the group is
    ``pp_size`` wide, which keeps this one-off pickle cheap. A mismatch is not
    assumed away: training_logs cross-checks a schema checksum on every reduce
    and falls back to ``_paddle_gather`` if the ranks disagree.
    """
    import paddle.distributed as dist

    if not dist.is_initialized():
        return None

    group = None
    try:
        from paddlefleet.parallel_state import get_pipeline_model_parallel_group

        group = get_pipeline_model_parallel_group()
    except Exception:
        logger.warning(
            "[InternalMedicine/paddlefleet] pipeline group unavailable; "
            "syncing the metric schema over the full world instead (one-off, but large at scale)"
        )

    # A single-stage group has nothing to union, and paddle rejects all_gather on
    # a one-rank group outright. With PP=1 every rank already holds every layer,
    # so the local key set *is* the global schema.
    if group is not None and getattr(group, "nranks", 0) <= 1:
        return [list(local_keys)]

    gathered = []
    try:
        dist.all_gather_object(gathered, list(local_keys), group=group)
    except Exception as exc:
        # Degrade to a local-only schema: the fixed-size probe in training_logs
        # will notice if that disagrees across ranks and fall back to the legacy
        # path, so this cannot turn into a hang or a wrong number.
        logger.warning(f"[InternalMedicine/paddlefleet] metric schema sync failed ({exc}); using local keys only")
        return [list(local_keys)]
    return gathered


def install_gather_fn():
    """Install the paddle-based aggregation primitives into the global training_logs."""
    from ...core.training_logs import training_logs

    try:
        import paddle.distributed as dist

        if dist.is_initialized():
            training_logs.set_gather_fn(_paddle_gather)
            training_logs.set_schema_sync_fn(_paddle_schema_gather)
            training_logs.set_reduce_fn(_paddle_reduce)
    except ImportError:
        pass
