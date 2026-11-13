"""Optimizer update-magnitude monitor for the Megatron backend.

Always installed (no ``monitors`` opt-in), reported every ``monitor_interval`` steps
alongside grad-norm / param-norm:

    optim/update_rms        sqrt(mean((theta_new - theta_old)**2))
    optim/param_rms         sqrt(mean(theta_new**2))
    optim/update_param_ratio  update_rms / param_rms   (the "trust ratio", ~1e-3 healthy)

With ``retraction=True`` (opt-in, Skewon only) three more are logged — the RMS of the
per-step polar-retraction correction ``polar(X+lrB) - (X+lrB)``, drained from the inner
optimizer's ``_retract_pending`` and pooled like update_rms:

    optim/retract_rms           sqrt(mean(retraction_correction**2))
    optim/retract_param_ratio   retract_rms / param_rms
    optim/retract_update_ratio  retract_rms / update_rms  (fraction of net motion that is snap-back)

With ``radial_cos=True`` (opt-in, SkewonR only) one more is logged — the mean over managed
matrices of ``cos(M, W)``, the momentum's alignment with the weight's radial axis. It is the
sign the SkewonR radial rule signs, so it is the direct read on whether the free radius is
being pushed out or in:

    optim/radial_cos           mean cos(momentum, W)  in [-1, 1]

With ``per_param=True`` (opt-in, default off) the same three ratios (and, with
``retraction``, the retraction triplet) are also logged per orthogonalized matrix —
``optim/update_rms/<name>`` etc. — where ``<name>`` is the dotted parameter name. Only
Skewon/Muon-managed matrices are broken out (the inner optimizer exposes ``orthogonalize``);
names are resolved from the model param in the monitor's triple, and a one-time world
all_gather fixes a global name order so one packed SUM reduces every matrix at once.

No copy of the parameters is kept. ``MixedPrecisionOptimizer.step_with_ready_grads``
runs ``self.optimizer.step()`` and then ``self._copy_main_params_to_model_params()``;
between those two calls the fp32 main param already holds ``theta_new`` while the bf16
model param still holds ``theta_old``. Wrapping the copy method therefore reads the
pre-update value for free.

The bf16 ``theta_old`` is rounded, which inflates the raw difference once the update is
small (measured 11x at lr 3e-6). ``mean_sq`` is debiased by subtracting the bf16
round-trip variance of ``theta_new``, which restores agreement to 1.00x across
lr 3e-4 .. 3e-6.
"""

import contextlib
import logging
from typing import Any

import torch
import torch.distributed as dist

from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)

METRIC_PREFIX = "optim"

_CHUNK = 1 << 20


def _param_is_not_shared(param) -> bool:
    return not getattr(param, "shared", False)


def _strip_module_prefix(name: str) -> str:
    """Drop leading DDP/Float16Module ``module.`` wrappers for a readable TB tag."""
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _param_is_not_tp_duplicate(param, tp_group) -> bool:
    """Whether this rank owns the canonical copy of ``param`` for TP purposes.

    A TP-sharded param is distinct on every TP rank, so every rank counts it. A
    replicated param would otherwise be counted once per TP rank, so only rank 0 does.
    Same rule as ``MegatronOptimizer._filter_grads_for_norm``.
    """
    if getattr(param, "tensor_model_parallel", False):
        return True
    try:
        from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

        return bool(param_is_not_tensor_parallel_duplicate(param, tp_group))
    except Exception:
        if tp_group is not None:
            return tp_group.rank() == 0
        return True


def _pair_sums(main: torch.Tensor, model: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(sum_sq_update, sum_sq_param, sum_sq_bf16_noise, numel)`` for one shard.

    ``main`` is fp32 ``theta_new``; ``model`` is the lower-precision ``theta_old`` view.
    Walked in ``_CHUNK`` slices so the elementwise temporary stays a few MB regardless of
    parameter size, instead of allocating a full fp32 copy of the shard.
    """
    flat_main = main.reshape(-1)
    flat_model = model.reshape(-1)
    n = flat_main.numel()
    ss_u = torch.zeros((), dtype=torch.float32, device=flat_main.device)
    ss_q = torch.zeros((), dtype=torch.float32, device=flat_main.device)
    for start in range(0, n, _CHUNK):
        a = flat_main[start : start + _CHUNK]
        b = flat_model[start : start + _CHUNK]
        ss_u += torch.linalg.vector_norm(a - b, dtype=torch.float32).square()
        ss_q += torch.linalg.vector_norm(a - a.to(model.dtype).to(a.dtype), dtype=torch.float32).square()
    ss_p = torch.linalg.vector_norm(flat_main, dtype=torch.float32).square()
    return ss_u, ss_p, ss_q, n


class OptimUpdateMonitor:
    """Measure the RMS parameter update applied by the optimizer, once per step.

    Not a ``TorchProbe``: there are no forward hooks and no per-layer metrics. It wraps
    one optimizer method, accumulates 0-dim GPU tensors there, and reduces + logs in
    ``step()`` (which runs outside any forward, so collectives are free to happen).
    """

    def __init__(
        self,
        monitor_interval: int = 1,
        verbose: bool = False,
        debias_low_precision: bool = True,
        retraction: bool = False,
        per_param: bool = False,
        radial_cos: bool = False,
        model=None,
    ):
        self.monitor_interval = monitor_interval
        self.step_count = 0
        self.verbose = verbose
        self.debias_low_precision = debias_low_precision
        self.retraction = retraction
        self.per_param = per_param
        self.radial_cos = radial_cos
        self._model = model
        self._patched: list[tuple[Any, str, Any, bool]] = []
        self._pending_dense: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        self._pending_expert: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        # Optional retraction-correction sums (opt-in via ``retraction``), drained from a
        # skewon inner optimizer's ``_retract_pending``; (sum_sq, numel) per shard.
        self._pending_dense_retract: list[tuple[torch.Tensor, int]] = []
        self._pending_expert_retract: list[tuple[torch.Tensor, int]] = []
        # Optional cos(momentum, W) samples (opt-in via ``radial_cos``), drained from a
        # skewonR inner optimizer's ``_radial_cos_pending``; (cos 0-dim, count) per shard.
        self._pending_dense_cos: list[tuple[torch.Tensor, float]] = []
        self._pending_expert_cos: list[tuple[torch.Tensor, float]] = []
        # Per-param buckets (opt-in via ``per_param``), orthogonalized matrices only.
        # update: (name, ss_update, ss_param, ss_bf16_noise, numel); retract: (name, ss, numel).
        self._pending_per_param: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        self._pending_per_param_retract: list[tuple[str, torch.Tensor, int]] = []
        self._pending_per_param_cos: list[tuple[str, torch.Tensor, float]] = []
        self._main_id_to_name: dict[int, str] = {}  # id(fp32 main) -> dotted name
        self._model_param_name_by_id: dict[int, str] = {}  # id(model param) -> dotted name
        self._name_map_built = False
        self._pp_name_order: list[str] | None = None  # world-union name list, built once
        self._pp_name_idx: dict[str, int] = {}
        self._groups_resolved = False
        self._dp_cp_group = None
        self._expt_dp_group = None
        self._mp_group = None
        self._expert_group = None
        self._warned_fp32 = False
        self._warned_no_pairs = False
        self.latest: dict[str, float] = {}

    def _should_monitor(self) -> bool:
        """Same interval gate as ``BaseMonitor``, on the optimizer step counter.

        Gating in the WRAPPER (not just at report time) is what makes the interval
        actually cheap: on a skipped step no sums are computed, so the wrapper costs one
        boolean and the collectives in ``step`` never fire.
        """
        if not self.monitor_interval:
            return False
        return self.step_count % self.monitor_interval == 0

    def attach_optimizer(self, optimizer) -> bool:
        """Wrap this optimizer instance's main-param copy. Returns True if any wrapped."""
        self._resolve_groups()
        for opt in self._iter_optimizers(optimizer):
            for method in ("_copy_main_params_to_model_params", "_copy_main_params_to_param_buffer"):
                original = getattr(opt, method, None)
                if original is None or getattr(original, "_im_update_patched", False):
                    continue
                self._install(opt, method, original, instance_level=True)
        if not self._patched and not self._warned_no_pairs:
            self._warned_no_pairs = True
            logger.warning("[OptimUpdateMonitor] no main-param copy method found; update_rms unavailable")
        return bool(self._patched)

    def attach_optimizer_classes(self) -> bool:
        """Wrap ``_copy_main_params_to_model_params`` on the mcore optimizer CLASSES.

        Patching the class rather than an instance is what lets this monitor be installed
        from the same model-side setup path as every other monitor: the optimizer does not
        exist yet at that point, and threading it in later would mean an extra callback in
        every training script. Every optimizer built afterwards is covered.

        Both concrete subclasses that define the copy are wrapped —
        ``Float16OptimizerWithFloat16Params`` (non-distributed) and
        ``DistributedOptimizer`` — since the copy is not defined on the shared
        ``MixedPrecisionOptimizer`` base. ``self`` is read from the call, so one wrapper
        serves all instances.
        """
        self._resolve_groups()
        classes = []
        try:
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
            from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

            classes = [Float16OptimizerWithFloat16Params, DistributedOptimizer]
        except ImportError:
            logger.warning("[OptimUpdateMonitor] megatron.core optimizer unavailable; update_rms disabled")
            return False
        for cls in classes:
            for method in ("_copy_main_params_to_model_params", "_copy_main_params_to_param_buffer"):
                original = cls.__dict__.get(method)
                if original is None or getattr(original, "_im_update_patched", False):
                    continue
                self._install(cls, method, original, instance_level=False)
        return bool(self._patched)

    def _iter_optimizers(self, optimizer):
        chained = getattr(optimizer, "chained_optimizers", None)
        if chained:
            for opt in chained:
                yield from self._iter_optimizers(opt)
        else:
            yield optimizer

    def _install(self, target, method: str, original, instance_level: bool) -> None:
        """Wrap ``target.method`` so the update is measured just before the copy-back.

        ``instance_level`` selects how ``self`` reaches ``_measure``: an instance patch
        closes over the optimizer, a class patch reads it from the call's first argument.
        """
        monitor = self

        if instance_level:

            def patched(*args, **kwargs):
                monitor._measure_safe(target)
                return original(*args, **kwargs)

            was_instance_attr = method in vars(target)
        else:

            def patched(self, *args, **kwargs):
                monitor._measure_safe(self)
                return original(self, *args, **kwargs)

            was_instance_attr = False

        patched._im_update_patched = True
        # For an instance patch, remember whether the name already lived on the instance:
        # these are normally class methods, so the correct revert is to delete our
        # instance attribute and let class lookup resume, rather than leave a bound
        # method (an instance -> bound-method -> instance reference cycle) behind.
        setattr(target, method, patched)
        self._patched.append((target, method, original, was_instance_attr))

    def _measure_safe(self, opt) -> None:
        if not self._should_monitor():
            return
        try:
            self._measure(opt)
        except Exception as e:
            if self.verbose:
                logger.error(f"[OptimUpdateMonitor] measure error: {e}")

    @torch.no_grad()
    def _measure(self, opt) -> None:
        """Accumulate per-shard sums while ``theta_old`` is still in the model params.

        Dense and expert shards are bucketed separately because they reduce over
        different groups (see ``step``). The dedup predicates are evaluated on the
        ORIGINAL model param, not the shard view: ``allreduce`` (the dense/expert marker)
        is not among the attributes copied onto shard views by
        ``copy_optimizer_param_metadata``.
        """
        if getattr(opt, "is_stub_optimizer", False):
            return
        triples = self._param_triples(opt)
        if not triples:
            return
        if getattr(opt, "shard_fp32_groups", None) and not self._warned_fp32:
            self._warned_fp32 = True
            logger.warning(
                "[OptimUpdateMonitor] fp32 params are updated in place (no pre-update copy exists); "
                "they are excluded from update_rms."
            )
        tp_group = getattr(opt, "tp_group", None)
        inner = getattr(opt, "optimizer", None)
        capture_pp = self.per_param and inner is not None and hasattr(inner, "orthogonalize")
        if capture_pp:
            self._ensure_name_map()
        for main, model_shard, model_param in triples:
            if main is None or model_shard is None or main.numel() != model_shard.numel():
                continue
            owner = model_param if model_param is not None else model_shard
            if not _param_is_not_shared(owner) or not _param_is_not_tp_duplicate(owner, tp_group):
                continue
            sums = _pair_sums(main.detach(), model_shard.detach())
            if getattr(owner, "allreduce", True):
                self._pending_dense.append(sums)
            else:
                self._pending_expert.append(sums)
            if capture_pp and main.dim() == 2:
                name = self._model_param_name_by_id.get(id(owner))
                if name is not None:
                    self._main_id_to_name[id(main)] = name
                    ss_u, ss_p, ss_q, n = sums
                    self._pending_per_param.append((name, ss_u, ss_p, ss_q, n))

        if self.retraction:
            self._drain_retraction(opt)
        if self.radial_cos:
            self._drain_radial_cos(opt)

    def _drain_radial_cos(self, opt) -> None:
        """Pull a skewonR inner optimizer's per-block cos(momentum, W) samples into the cos
        buckets. Each entry is ``(cos 0-dim, count, is_expert[, id(main)])``; bucketed like
        the update sums so it reduces over the same groups in ``step``. Tensors are carried
        as-is (no D2H here); the division by count happens after the reduce."""
        inner = getattr(opt, "optimizer", None)
        pending = getattr(inner, "_radial_cos_pending", None)
        if not pending:
            return
        for entry in pending:
            c, n, is_expert = entry[0], entry[1], entry[2]
            if is_expert:
                self._pending_expert_cos.append((c, n))
            else:
                self._pending_dense_cos.append((c, n))
            if self.per_param and len(entry) > 3:
                name = self._main_id_to_name.get(entry[3])
                if name is not None:
                    self._pending_per_param_cos.append((name, c, n))
        inner._radial_cos_pending = []

    def _drain_retraction(self, opt) -> None:
        """Pull a skewon inner optimizer's per-step retraction sums into the retract
        buckets. ``opt`` is the Float16 wrapper; the emerging optimizer is ``opt.optimizer``.
        Each entry is ``(sum_sq GPU 0-dim, numel, is_expert[, id(main)])``; bucketed like
        update sums so it reduces over the same groups in ``step``. The optional 4th field
        is ``id(p)`` of the fp32 main — used to attribute the correction to a param name for
        the per-param path (older skewon without the tag degrades to global-only)."""
        inner = getattr(opt, "optimizer", None)
        pending = getattr(inner, "_retract_pending", None)
        if not pending:
            return
        for entry in pending:
            ss, n, is_expert = entry[0], entry[1], entry[2]
            if is_expert:
                self._pending_expert_retract.append((ss, n))
            else:
                self._pending_dense_retract.append((ss, n))
            if self.per_param and len(entry) > 3:
                name = self._main_id_to_name.get(entry[3])
                if name is not None:
                    self._pending_per_param_retract.append((name, ss, n))
        inner._retract_pending = []

    @staticmethod
    def _param_triples(opt):
        """``(fp32_main_shard, low_precision_shard, original_model_param)`` per param.

        The three group lists are index-parallel by construction: ``DistOpt``'s own
        ``_copy_main_params_to_model_params`` zips ``shard_fp32_from_float16_groups``
        against ``model_float16_groups`` the same way.
        """
        main_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
        shard_groups = getattr(opt, "shard_float16_groups", None)
        model_groups = getattr(opt, "model_float16_groups", None)
        if main_groups is None or shard_groups is None:
            main_groups = getattr(opt, "fp32_from_float16_groups", None)
            shard_groups = getattr(opt, "float16_groups", None)
            model_groups = shard_groups
        if main_groups is None or shard_groups is None:
            return []
        triples = []
        for gi, (main_group, shard_group) in enumerate(zip(main_groups, shard_groups, strict=False)):
            model_group = model_groups[gi] if model_groups is not None and gi < len(model_groups) else []
            for pi, (main, shard) in enumerate(zip(main_group, shard_group, strict=False)):
                model_param = model_group[pi] if pi < len(model_group) else None
                triples.append((main, shard, model_param))
        return triples

    def _resolve_groups(self) -> None:
        if self._groups_resolved:
            return
        self._groups_resolved = True
        try:
            from megatron.core import parallel_state as ps

            if not ps.model_parallel_is_initialized():
                return
            self._dp_cp_group = ps.get_data_parallel_group(with_context_parallel=True)
            self._mp_group = ps.get_model_parallel_group()
            for getter, attr in (
                (getattr(ps, "get_expert_data_parallel_group", None), "_expt_dp_group"),
                (getattr(ps, "get_expert_tensor_model_pipeline_parallel_group", None), "_expert_group"),
            ):
                if getter is None:
                    continue
                with contextlib.suppress(Exception):
                    setattr(self, attr, getter())
        except ImportError:
            pass

    def _ensure_name_map(self) -> None:
        """Build ``id(model_param) -> dotted name`` once from the model's parameters.

        Keyed by the bf16 MODEL param (the object the optimizer keeps in its float16
        groups), not the fp32 main — the main is a fresh copy and carries no name. The
        monitor's param triple hands back that model param alongside the main, so this map
        turns either side into a name. Lazy: the model exists at setup, but building here
        keeps it off the setup path and tolerates a model that is a VPP chunk list.
        """
        if self._name_map_built:
            return
        self._name_map_built = True
        model = self._model
        if model is None:
            return
        chunks = model if isinstance(model, (list, tuple)) else [model]
        for chunk in chunks:
            named = getattr(chunk, "named_parameters", None)
            if named is None:
                continue
            for name, p in named():
                self._model_param_name_by_id[id(p)] = _strip_module_prefix(name)

    def _all_reduce(self, vec: torch.Tensor, group) -> torch.Tensor:
        if group is None or not dist.is_available() or not dist.is_initialized():
            return vec
        dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=group)
        return vec

    def _pool(self, pending, shard_group, model_group, device) -> torch.Tensor:
        """Reduce one bucket's ``(ss_update, ss_param, ss_bf16_noise, count)`` globally.

        Always issues both collectives, even with nothing pending, so every rank makes
        the same number of calls — a rank can legitimately own no expert shard while a
        peer does, and skipping the reduce there would hang.
        """
        acc = torch.zeros(4, dtype=torch.float64, device=device)
        for ss_u, ss_p, ss_q, n in pending:
            acc[0] += ss_u.to(torch.float64)
            acc[1] += ss_p.to(torch.float64)
            acc[2] += ss_q.to(torch.float64)
            acc[3] += float(n)
        self._all_reduce(acc, shard_group)
        self._all_reduce(acc, model_group)
        return acc

    def _pool_retract(self, pending, shard_group, model_group, device) -> torch.Tensor:
        """Reduce one retract bucket's ``(sum_sq, count)`` globally, mirroring ``_pool``.
        Always issues both collectives so every rank makes the same number of calls."""
        acc = torch.zeros(2, dtype=torch.float64, device=device)
        for ss, n in pending:
            acc[0] += ss.to(torch.float64)
            acc[1] += float(n)
        self._all_reduce(acc, shard_group)
        self._all_reduce(acc, model_group)
        return acc

    def _pool_cos(self, pending, shard_group, model_group, device) -> torch.Tensor:
        """Reduce one cos bucket's ``(sum_cos, count)`` globally, mirroring ``_pool``.
        Always issues both collectives so every rank makes the same number of calls."""
        acc = torch.zeros(2, dtype=torch.float64, device=device)
        for c, n in pending:
            acc[0] += c.to(torch.float64)
            acc[1] += float(n)
        self._all_reduce(acc, shard_group)
        self._all_reduce(acc, model_group)
        return acc

    def _all_reduce_world(self, t: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _collective_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _build_pp_name_index(self, per_param, per_param_retract, per_param_cos=()) -> None:
        """All-gather the per-rank name sets once and freeze a sorted global order.

        Each managed matrix has a globally unique dotted name except expert matrices, which
        share a name across their expert-parallel peers by design — those land in one slot
        so the world SUM pools them. Built on the first monitor step that sees any name;
        every rank joins the all_gather, so the call stays symmetric. Left unbuilt (None)
        while nothing is managed anywhere, to retry on a later monitor step.
        """
        if self._pp_name_order is not None:
            return
        local = {name for name, *_ in per_param}
        local.update(name for name, *_ in per_param_retract)
        local.update(name for name, *_ in per_param_cos)
        gathered: list = [local]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
        names: set[str] = set()
        for s in gathered:
            if s:
                names.update(s)
        if names:
            self._pp_name_order = sorted(names)
            self._pp_name_idx = {name: i for i, name in enumerate(self._pp_name_order)}

    def _publish_per_param(self, per_param, per_param_retract, per_param_cos=()) -> None:
        """World-reduce the per-matrix sums against the cached name index and publish
        ``optim/{update,param}_rms/<name>``, their ratio, and (with retraction) the
        per-matrix retraction triplet, plus (with radial_cos) ``optim/radial_cos/<name>``.
        One packed ``[N,4]`` SUM covers update/param; a second ``[N,2]`` SUM covers the
        retraction correction; a third covers the cos samples. Owner-unique for dense,
        expert-pooled for expert — SUM is exactly the per-matrix aggregate either way.
        """
        device = self._collective_device()
        self._build_pp_name_index(per_param, per_param_retract, per_param_cos)
        order = self._pp_name_order
        if not order:
            return
        n = len(order)
        upd = torch.zeros((n, 4), dtype=torch.float64, device=device)
        for name, ss_u, ss_p, ss_q, numel in per_param:
            i = self._pp_name_idx[name]
            upd[i, 0] += ss_u.to(torch.float64)
            upd[i, 1] += ss_p.to(torch.float64)
            upd[i, 2] += ss_q.to(torch.float64)
            upd[i, 3] += float(numel)
        self._all_reduce_world(upd)
        ret = None
        if self.retraction:
            ret = torch.zeros((n, 2), dtype=torch.float64, device=device)
            for name, ss, numel in per_param_retract:
                i = self._pp_name_idx[name]
                ret[i, 0] += ss.to(torch.float64)
                ret[i, 1] += float(numel)
            self._all_reduce_world(ret)
        cos = None
        if self.radial_cos:
            cos = torch.zeros((n, 2), dtype=torch.float64, device=device)
            for name, c, numel in per_param_cos:
                i = self._pp_name_idx[name]
                cos[i, 0] += c.to(torch.float64)
                cos[i, 1] += float(numel)
            self._all_reduce_world(cos)
        upd_c = upd.cpu()
        ret_c = ret.cpu() if ret is not None else None
        cos_c = cos.cpu() if cos is not None else None
        for i, name in enumerate(order):
            cnt = float(upd_c[i, 3])
            if cnt <= 0.0:
                continue
            update_ms = float(upd_c[i, 0]) / cnt
            param_ms = float(upd_c[i, 1]) / cnt
            if self.debias_low_precision:
                update_ms = max(update_ms - float(upd_c[i, 2]) / cnt, 0.0)
            update_rms = update_ms**0.5
            param_rms = param_ms**0.5
            self.latest[f"{METRIC_PREFIX}/update_rms/{name}"] = update_rms
            self.latest[f"{METRIC_PREFIX}/param_rms/{name}"] = param_rms
            self.latest[f"{METRIC_PREFIX}/update_param_ratio/{name}"] = update_rms / param_rms if param_rms > 0 else 0.0
            if ret_c is not None and float(ret_c[i, 1]) > 0.0:
                retract_rms = (float(ret_c[i, 0]) / float(ret_c[i, 1])) ** 0.5
                self.latest[f"{METRIC_PREFIX}/retract_rms/{name}"] = retract_rms
                self.latest[f"{METRIC_PREFIX}/retract_param_ratio/{name}"] = (
                    retract_rms / param_rms if param_rms > 0 else 0.0
                )
                self.latest[f"{METRIC_PREFIX}/retract_update_ratio/{name}"] = (
                    retract_rms / update_rms if update_rms > 0 else 0.0
                )
            if cos_c is not None and float(cos_c[i, 1]) > 0.0:
                self.latest[f"{METRIC_PREFIX}/radial_cos/{name}"] = float(cos_c[i, 0]) / float(cos_c[i, 1])

    def _publish_global(self, dense, expert, dense_retract, expert_retract, dense_cos=(), expert_cos=()) -> None:
        """Pool the global update/param (and optional retraction) sums into ``self.latest``.

        Mirrors ``calc_params_l2_norm``: dense over (DP+CP, model-parallel), expert over
        (expert-DP, expert group). Returns early without a collective when nothing was
        measured — symmetric across ranks because the measure gate is in lockstep.
        """
        if not dense and not expert:
            return
        device = None
        for bucket in (dense, expert):
            if bucket:
                device = bucket[0][0].device
                break
        if device is None:
            return
        acc = self._pool(dense, self._dp_cp_group, self._mp_group, device)
        acc += self._pool(
            expert,
            self._expt_dp_group or self._dp_cp_group,
            self._expert_group or self._mp_group,
            device,
        )
        if float(acc[3]) <= 0.0:
            return
        count = acc[3].clamp(min=1.0)
        update_ms = acc[0] / count
        param_ms = acc[1] / count
        if self.debias_low_precision:
            update_ms = (update_ms - acc[2] / count).clamp(min=0.0)
        update_rms = float(update_ms.sqrt())
        param_rms = float(param_ms.sqrt())
        self.latest[f"{METRIC_PREFIX}/update_rms"] = update_rms
        self.latest[f"{METRIC_PREFIX}/param_rms"] = param_rms
        self.latest[f"{METRIC_PREFIX}/update_param_ratio"] = update_rms / param_rms if param_rms > 0 else 0.0
        if self.retraction and (dense_retract or expert_retract):
            racc = self._pool_retract(dense_retract, self._dp_cp_group, self._mp_group, device)
            racc += self._pool_retract(
                expert_retract,
                self._expt_dp_group or self._dp_cp_group,
                self._expert_group or self._mp_group,
                device,
            )
            if float(racc[1]) > 0.0:
                retract_rms = float((racc[0] / racc[1].clamp(min=1.0)).sqrt())
                self.latest[f"{METRIC_PREFIX}/retract_rms"] = retract_rms
                self.latest[f"{METRIC_PREFIX}/retract_param_ratio"] = retract_rms / param_rms if param_rms > 0 else 0.0
                self.latest[f"{METRIC_PREFIX}/retract_update_ratio"] = (
                    retract_rms / update_rms if update_rms > 0 else 0.0
                )
        if self.radial_cos:
            cacc = self._pool_cos(dense_cos, self._dp_cp_group, self._mp_group, device)
            cacc += self._pool_cos(
                expert_cos,
                self._expt_dp_group or self._dp_cp_group,
                self._expert_group or self._mp_group,
                device,
            )
            if float(cacc[1]) > 0.0:
                self.latest[f"{METRIC_PREFIX}/radial_cos"] = float(cacc[0] / cacc[1].clamp(min=1.0))

    @torch.no_grad()
    def step(self, global_step: int | None = None) -> dict[str, float]:
        """Reduce the step's sums, publish the scalars, and return them.

        Advances the interval counter (mirroring ``BaseMonitor.step``). The global metrics
        report only when the wrapper collected something, issuing no collective on a skipped
        step. The per-param path, when enabled, runs on the shared monitor-step gate
        (lockstep across ranks) independent of the global early-returns: its world
        collectives must fire on every rank or none.

        Main params are sharded over DP+CP by the distributed optimizer and split again
        over TP/PP, so the global sums pool over the shard group and then the model-parallel
        group — mirroring ``calc_params_l2_norm``. Expert params live on different groups
        (expert-DP, then expert-TP/PP), hence the separate bucket. RMS is nonlinear in the
        sums, so all pooling happens before the division.
        """
        if global_step is not None:
            self.step_count = global_step
        was_monitor_step = self._should_monitor()
        self.step_count += 1
        dense, self._pending_dense = self._pending_dense, []
        expert, self._pending_expert = self._pending_expert, []
        dense_retract, self._pending_dense_retract = self._pending_dense_retract, []
        expert_retract, self._pending_expert_retract = self._pending_expert_retract, []
        per_param, self._pending_per_param = self._pending_per_param, []
        per_param_retract, self._pending_per_param_retract = self._pending_per_param_retract, []
        dense_cos, self._pending_dense_cos = self._pending_dense_cos, []
        expert_cos, self._pending_expert_cos = self._pending_expert_cos, []
        per_param_cos, self._pending_per_param_cos = self._pending_per_param_cos, []
        self.latest = {}

        self._publish_global(dense, expert, dense_retract, expert_retract, dense_cos, expert_cos)
        if self.per_param and was_monitor_step:
            self._publish_per_param(per_param, per_param_retract, per_param_cos)

        if self.latest:
            training_logs.update(**self.latest)
        return self.latest

    def remove_hooks(self) -> None:
        for target, method, original, was_instance_attr in self._patched:
            current = target.__dict__.get(method) if isinstance(target, type) else getattr(target, method, None)
            if not getattr(current, "_im_update_patched", False):
                continue
            if isinstance(target, type) or was_instance_attr:
                setattr(target, method, original)
            else:
                target.__dict__.pop(method, None)
        self._patched = []
        self._pending_dense = []
        self._pending_expert = []
        self._pending_dense_retract = []
        self._pending_expert_retract = []
        self._pending_per_param = []
        self._pending_per_param_retract = []
        self._pending_dense_cos = []
        self._pending_expert_cos = []
        self._pending_per_param_cos = []


def setup_optim_update_monitor(
    model=None,
    optimizer=None,
    monitor_dict: dict | None = None,
    monitor_interval: int = 1,
    verbose: bool = False,
    retraction: bool = False,
    per_param: bool = False,
    radial_cos: bool = False,
    **_ignored,
):
    """Enable ``optim/update_rms`` + ``param_rms``. Always installed; no opt-in needed.

    ``model`` is accepted and ignored for the global metrics so this can sit in the same
    registry as the model-side monitors and be driven by the same ``setup_internal_medicine``
    call. With no ``optimizer`` given (the normal path — it does not exist at model-setup
    time) the mcore optimizer CLASSES are patched instead, which covers whatever is built
    later. ``model`` IS used by ``per_param`` to resolve matrix names.

    ``retraction`` (opt-in, default off) additionally logs ``optim/retract_rms`` +
    ``optim/retract_param_ratio`` + ``optim/retract_update_ratio`` for Skewon: the RMS of
    the per-step polar-retraction correction, drained from the inner optimizer's
    ``_retract_pending`` and pooled like ``update_rms``. Absent on Muon/Adam (no retraction).

    ``radial_cos`` (opt-in, default off) additionally logs ``optim/radial_cos`` — the mean
    over managed matrices of cos(momentum, W), drained from a SkewonR inner optimizer's
    ``_radial_cos_pending``. It is the sign SkewonR's radial rule signs, so it reads directly
    on whether the free weight radius is being pushed out (+) or in (−). Absent on
    Muon/MuonH/Adam (nothing publishes it).

    ``per_param`` (opt-in, default off) additionally logs the per-matrix breakdown
    ``optim/update_rms/<name>``, ``param_rms/<name>``, ``update_param_ratio/<name>`` (and,
    with ``retraction``, the per-matrix retraction triplet) for every orthogonalized matrix
    — the set Skewon/Muon actually act on. Names come from ``model.named_parameters()`` via
    the monitor's param triple; a one-time world all_gather fixes a global name order.

    ``monitor_interval`` is honoured exactly like the model-side monitors: these are
    slow-moving scalars, so reporting them every step would only add log noise and two
    collectives per step.

    Registered under the ``optim`` key, matching ``METRIC_PREFIX``, so a caller that
    prints ``training_logs`` by iterating ``monitor_dict`` keys as prefixes picks these up
    with no extra wiring.
    """
    monitor = OptimUpdateMonitor(
        monitor_interval=monitor_interval,
        verbose=verbose,
        retraction=retraction,
        per_param=per_param,
        radial_cos=radial_cos,
        model=model,
    )
    if optimizer is not None:
        monitor.attach_optimizer(optimizer)
    else:
        monitor.attach_optimizer_classes()
    if monitor_dict is not None:
        monitor_dict[METRIC_PREFIX] = monitor
    return model
