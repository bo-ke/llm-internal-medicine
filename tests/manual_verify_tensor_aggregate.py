"""Multi-process check for the fixed-schema tensor aggregation (paddlefleet).

Deliberately not named ``test_*``: the unit suite runs single-process, and the
only thing that can validate the collective layout is real ranks.

    PP_SIZE=2 python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
        tests/manual_verify_tensor_aggregate.py

What it pins down:

1. The tensor path and the legacy ``all_gather_object`` path return the same keys
   and the same values. This change is meant to be purely about cost.
2. Per-layer keys owned by a single pipeline stage survive the union schema
   (the shape-consistency assumption that a wrong guess turns into a hang).
3. ``max`` / ``min`` keys reduce to the global extremum over the ranks that hold
   them, not to the ``+-inf`` padding.
4. A key written after the schema froze triggers exactly one collective rebuild
   instead of hanging or being dropped.
"""

import os
import sys
from pathlib import Path

import paddle.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from internal_medicine.backends.paddlefleet.gather import _paddle_gather, _paddle_reduce  # noqa: E402
from internal_medicine.core.training_logs import training_logs  # noqa: E402

LAYERS = 8
METRICS_PER_LAYER = 8


def build_local_metrics(stage, pp_size, rank):
    """Per-layer keys for this stage's layers only, with rank-dependent values.

    Every key here is *stage*-uniform: two ranks of the same pipeline stage hold
    the same key set. That is the invariant the pipeline-group schema sync relies
    on, so the happy path must not violate it.
    """
    layers_per_stage = LAYERS // pp_size
    metrics = {}
    for layer in range(stage * layers_per_stage, (stage + 1) * layers_per_stage):
        for i in range(METRICS_PER_LAYER):
            metrics[f"probe/layer_{layer}/metric_{i}"] = layer * 100.0 + i + rank * 0.01
        metrics[f"probe/layer_{layer}/peak_max"] = 1000.0 - rank
        metrics[f"probe/layer_{layer}/floor_min"] = 1000.0 + rank
    metrics["probe/global_metric_0"] = 7.0 + rank
    metrics[f"probe/stage_{stage}_only"] = 5.0 + rank
    return metrics


def make_pipeline_group(rank, pp_size, dp_size):
    """One rank per stage, mirroring get_pipeline_model_parallel_group().

    Every rank must create every group, in the same order, or new_group deadlocks.
    ``None`` for PP=1: a one-rank group has nothing to union and paddle rejects
    all_gather on it, which is exactly what the backend helper handles too.
    """
    if pp_size == 1:
        return None
    own = None
    for dp_index in range(dp_size):
        members = [stage * dp_size + dp_index for stage in range(pp_size)]
        group = dist.new_group(members)
        if rank in members:
            own = group
    return own


def check(results, label, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol
    results.append((label, ok, got, expected))
    return ok


def step(rank, msg):
    """Progress markers, flushed: a hang has to be locatable from the worker log."""
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    dist.init_parallel_env()
    rank, world = dist.get_rank(), dist.get_world_size()
    pp_size = int(os.environ.get("PP_SIZE", "2"))
    if world % pp_size:
        raise SystemExit(f"world size {world} is not divisible by PP_SIZE={pp_size}")
    dp_size = world // pp_size
    stage = rank // dp_size
    step(rank, f"init ok, world={world} pp={pp_size} dp={dp_size} stage={stage}")
    pp_group = make_pipeline_group(rank, pp_size, dp_size)
    step(rank, "pipeline group ready")

    def schema_sync_fn(local_keys):
        if pp_group is None:
            return [list(local_keys)]
        gathered = []
        dist.all_gather_object(gathered, list(local_keys), group=pp_group)
        return gathered

    local = build_local_metrics(stage, pp_size, rank)

    # Legacy path first, as the reference.
    training_logs.reset()
    training_logs.update(**local)
    training_logs.set_reduce_fn(None)
    training_logs.set_gather_fn(_paddle_gather)
    legacy = training_logs.gather_and_aggregate()
    step(rank, f"legacy path ok, {len(legacy)} keys")

    # Same inputs through the fixed-schema tensor path.
    training_logs.reset()
    training_logs.update(**local)
    training_logs.set_schema_sync_fn(schema_sync_fn)
    training_logs.set_reduce_fn(_paddle_reduce)
    tensor = training_logs.gather_and_aggregate()
    step(rank, f"tensor path ok, {len(tensor)} keys")

    results = []
    key_diff = set(legacy) ^ set(tensor)
    results.append(("same key set", not key_diff, len(legacy), len(tensor)))
    worst, worst_key = 0.0, None
    for k in legacy.keys() & tensor.keys():
        delta = abs(legacy[k] - tensor[k])
        if delta > worst:
            worst, worst_key = delta, k
    results.append((f"max |tensor-legacy| ({worst_key})", worst <= 1e-9, worst, 0.0))

    # The reduction domain for a per-layer key is exactly the DP ranks of the
    # stage that owns the layer — that is what makes the union schema safe.
    owned_layer = stage * (LAYERS // pp_size)
    stage_ranks = range(stage * dp_size, (stage + 1) * dp_size)
    mean_of_ranks = sum(stage_ranks) / dp_size
    check(
        results,
        f"layer_{owned_layer}/metric_0 mean over its stage",
        tensor[f"probe/layer_{owned_layer}/metric_0"],
        owned_layer * 100.0 + 0.01 * mean_of_ranks,
    )
    check(
        results,
        "peak_max is the stage maximum",
        tensor[f"probe/layer_{owned_layer}/peak_max"],
        1000.0 - stage * dp_size,
    )
    check(
        results,
        "floor_min is the stage minimum",
        tensor[f"probe/layer_{owned_layer}/floor_min"],
        1000.0 + stage * dp_size,
    )
    check(results, "global key means over all ranks", tensor["probe/global_metric_0"], 7.0 + (world - 1) / 2)
    check(
        results,
        "stage-unique key means over its stage",
        tensor[f"probe/stage_{stage}_only"],
        5.0 + mean_of_ranks,
    )

    # A key written after the schema froze: the resync flag must widen the schema
    # rather than drop the key or hang.
    training_logs.update(**{"optim/update_rms": 3.0 + rank})
    late = training_logs.gather_and_aggregate()
    step(rank, "late-key rebuild ok")
    check(results, "late key after one rebuild", late.get("optim/update_rms", float("nan")), 3.0 + (world - 1) / 2)
    results.append(("tensor path still enabled", not training_logs._tensor_path_disabled, 0, 0))

    # A rank-*unique* key breaks the stage-uniformity the pipeline-group sync
    # assumes, so the payload vectors would have different lengths on different
    # ranks. The fixed-size probe must catch that and fall back instead of hanging.
    # With dp_size == 1 the pipeline group *is* the world, so there is nothing to
    # disagree about and the key is simply absorbed by a schema rebuild.
    if rank == 0:
        training_logs.update(**{"probe/rank0_only": 42.0})
    fallback = training_logs.gather_and_aggregate()
    step(rank, "rank-unique key handled without a hang")
    if dp_size > 1:
        results.append(("mismatch latched the tensor path off", training_logs._tensor_path_disabled, 1, 1))
    else:
        results.append(("world-wide schema absorbed the key", not training_logs._tensor_path_disabled, 1, 1))
    check(results, "rank-unique key value is right either way", fallback.get("probe/rank0_only", float("nan")), 42.0)
    check(
        results,
        "shared keys still correct after the mismatch",
        fallback["probe/global_metric_0"],
        7.0 + (world - 1) / 2,
    )

    failures = [r for r in results if not r[1]]
    print(
        f"[rank {rank}/{world} stage {stage}] {len(results) - len(failures)}/{len(results)} checks passed"
        f" | schema {len(training_logs._schema)} keys, local {len(local)}"
    )
    for label, ok, got, expected in results:
        if not ok:
            print(f"[rank {rank}] FAIL {label}: got {got!r} expected {expected!r}")
    dist.barrier()
    if failures:
        raise SystemExit(1)
    if rank == 0:
        print("ALL RANKS PASSED")


if __name__ == "__main__":
    main()
