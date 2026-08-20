"""MultiMax lm_head monitor (PaddleFleet backend).

Tracks what the learnable SegLU modulation on the LM head does to the output
distribution: the paper's sparsity / multi-modality metrics, predictive
entropy, top-k probability mass, and the learned SegLU coefficients
themselves.

The head is a single module, so every metric is emitted as a ``global_`` key;
there is no per-layer dimension.

Hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``. The hook
records 0-dim GPU tensors only. The one collective (a vocab all-gather of the
sampled tile under tensor parallelism) is required for correctness and is
bounded by ``sample_tokens``; see ``_gather_vocab``.
"""

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .multimax_metrics import apply_seglu, compute_distribution_metrics, compute_param_metrics

logger = logging.getLogger(__name__)

_PARAM_METRICS = tuple(f"range_{i}" for i in range(4)) + tuple(f"t_{i}" for i in range(4))
_DIST_METRICS = (
    "entropy",
    "entropy_norm",
    "top1_prob",
    "multi_modality",
    "sparsity",
    "relevant_count",
    "sparse_count",
    "rows",
)


class PaddleMultiMaxMonitor(PaddleProbe):
    """Monitor the MultiMax/SegLU lm_head output distribution."""

    METRIC_PREFIX = "multimax"
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        sample_tokens: int = 256,
        topk: int = 10,
        prob_eps: float | None = None,
        logit_eps: float | None = None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        assert sample_tokens > 0, "sample_tokens must be positive"
        self.sample_tokens = int(sample_tokens)
        self.topk = int(topk)
        # Effective k is clamped to the vocab width at registration so the
        # declared key and the computed key can never disagree (Rule 3: no
        # lazy/conditional declares on the hot path).
        self._topk_effective = int(topk)
        self.prob_eps = prob_eps
        self.logit_eps = logit_eps
        self.tp_size = 1
        self.tp_group = None
        self._hook_failed = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _metric_names(self, vocab_width: int | None = None) -> tuple[str, ...]:
        # topk is clamped to the vocab width at compute time; the key must be
        # declared up front, so clamp it here too when the width is known.
        k = self.topk if vocab_width is None else min(self.topk, vocab_width)
        return _DIST_METRICS + (f"top{k}_prob",) + _PARAM_METRICS

    @staticmethod
    def _find_heads(model: nn.Layer) -> list[tuple[str, nn.Layer]]:
        """LM heads carrying the multimax SegLU parameters, tagged by role.

        With MTP enabled the model has two heads (``GPTMainLMHead`` and
        ``GPTMTPLMHead``), each with its *own* ``[4]`` SegLU parameters. They get
        separate key namespaces (``multimax/...`` and ``multimax/mtp_...``) so a
        divergence between them is visible instead of averaged away.
        """
        heads = []
        for _name, sublayer in model.named_sublayers(include_self=True):
            if not getattr(sublayer, "use_multimax_lmhead", False):
                continue
            if hasattr(sublayer, "multimax_ranges") and hasattr(sublayer, "multimax_ts"):
                tag = "mtp_" if "mtp" in type(sublayer).__name__.lower() else ""
                heads.append((tag, sublayer))
        return heads

    def _init_parallel_state(self) -> None:
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
            self.tp_group = pg.tp
            self.tp_size = get_pg_size(pg.tp)
        except Exception:
            pass

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        heads = self._find_heads(model)
        if not heads:
            logger.info("[PaddleMultiMaxMonitor] No multimax lm_head on this rank; skipping.")
            return
        if not self.log_global:
            # Every key here is a global key (the head is a single module), so
            # log_global=False would disable the whole monitor silently.
            logger.warning("[PaddleMultiMaxMonitor] log_global=False disables every multimax metric; skipping.")
            return

        vocab_width = None
        weight = getattr(heads[0][1], "weight", None)
        if isinstance(weight, paddle.Tensor) and len(weight.shape) == 2:
            # ColumnParallelLinear weight is [vocab_per_partition, hidden].
            vocab_width = weight.shape[0] * self.tp_size
        if vocab_width:
            self._topk_effective = min(self.topk, vocab_width)
        for tag in dict.fromkeys(tag for tag, _head in heads):
            for name in self._metric_names(vocab_width):
                self.declare_mean(self._global_key(tag + name))
        self.allocate_buffers()

        for tag, head in heads:
            self.hooks.append(head.register_forward_post_hook(self._make_head_hook(tag)))
        logger.info(
            "[PaddleMultiMaxMonitor] Registered %d lm_head hook(s). TP=%d sample_tokens=%d topk=%d",
            len(heads),
            self.tp_size,
            self.sample_tokens,
            self.topk,
        )

    # ------------------------------------------------------------------
    # hot path
    # ------------------------------------------------------------------

    def _sample_rows(self, tensor: paddle.Tensor) -> paddle.Tensor:
        """Deterministic strided token sample, identical on every TP rank."""
        flat = tensor.reshape([-1, tensor.shape[-1]])
        rows = flat.shape[0]
        if rows <= self.sample_tokens:
            return flat
        stride = max(1, rows // self.sample_tokens)
        return flat[::stride][: self.sample_tokens]

    def _gather_vocab(self, logits: paddle.Tensor) -> paddle.Tensor:
        """All-gather the vocab-parallel shards of the sampled tile.

        Justified hook-time collective (Rule 2): vocab-parallel TP shards the
        last dim, and every metric here needs the *global* softmax normalizer,
        so a local shard cannot produce a correct entropy / top-k / sparsity
        value. The payload is bounded by ``sample_tokens x vocab/tp`` (a few MB
        at defaults), it fires only on monitored steps, and the head's rows are
        identical across TP ranks because the parallel linear gathers the
        sequence-parallel input before projecting.
        """
        if self.tp_size <= 1 or self.tp_group is None:
            return logits
        # The sampled tile is a strided view of the logits; make it contiguous
        # before handing it to NCCL.
        logits = logits.contiguous()
        shards = [paddle.empty_like(logits) for _ in range(self.tp_size)]
        paddle.distributed.all_gather(shards, logits, group=self.tp_group)
        return paddle.concat(shards, axis=-1)

    def _fused_logits(self, module: nn.Layer, hidden: paddle.Tensor) -> paddle.Tensor:
        """Recompute the sampled logits for the fused-CE path.

        With ``fused_linear_ce_loss_chunk > 0`` the head returns
        ``(hidden, weight, bias, ranges, ts)`` and the full logits are never
        materialized, so the tile is projected here and modulated with the
        head's own parameters.
        """
        rows = self._sample_rows(hidden)
        weight = module.weight
        # Project in the weight's dtype: casting the [vocab/tp, hidden] weight to
        # fp32 would allocate a full-size copy (GBs at LM-head scale). Only the
        # small [sample_tokens, vocab/tp] tile is promoted.
        logits = paddle.matmul(rows.astype(weight.dtype), weight, transpose_y=True).astype("float32")
        bias = getattr(module, "bias", None)
        if isinstance(bias, paddle.Tensor):
            logits = logits + bias.astype("float32")
        return apply_seglu(logits, module.multimax_ranges, module.multimax_ts)

    def _resolve_logits(self, module: nn.Layer, outputs, mtp: bool = False) -> paddle.Tensor | None:
        """Sampled, SegLU-modulated logits tile for this head.

        Handles every shape the head can return: a bare logits tensor, the
        ``{"logits": ...}`` dict of ``GPTMainLMHead``, the ``[main, mtp...]``
        list when MTP is enabled, and the fused-CE tuple -- including a list of
        fused tuples, which is what MTP + fused-CE produces.

        ``GPTMTPLMHead`` returns the pipeline ``dict_args`` with its predictions
        under ``mtp_logits``; that dict may also still carry the *main* head's
        ``logits``, so the MTP head reads ``mtp_logits`` only and never falls
        back, otherwise the main logits would be counted twice.
        """
        candidate = outputs
        for _ in range(3):
            if isinstance(candidate, dict):
                candidate = candidate.get("mtp_logits") if mtp else candidate.get("logits")
                continue
            if isinstance(candidate, (list, tuple)):
                if not candidate:
                    return None
                if len(candidate) in (3, 5) and candidate[1] is getattr(module, "weight", None):
                    return self._fused_logits(module, candidate[0])
                candidate = candidate[0]  # main prediction comes first
                continue
            break
        if not isinstance(candidate, paddle.Tensor):
            return None
        return self._sample_rows(candidate)

    def _make_head_hook(self, tag: str = ""):
        def hook_fn(module, _inputs, outputs):
            if not module.training or not self._should_monitor():
                return
            try:
                with paddle.no_grad():
                    for name, value in compute_param_metrics(module.multimax_ranges, module.multimax_ts).items():
                        self.record_mean(self._global_key(tag + name), value)
                    tile = self._resolve_logits(module, outputs, mtp=bool(tag))
                    if tile is None:
                        return
                    tile = self._gather_vocab(tile.detach())
                    metrics = compute_distribution_metrics(
                        tile,
                        prob_eps=self.prob_eps,
                        logit_eps=self.logit_eps,
                        topk=self._topk_effective,
                    )
                    for name, value in metrics.items():
                        self.record_mean(self._global_key(tag + name), value)
            except Exception:
                if self.verbose and not self._hook_failed:
                    logger.exception("[PaddleMultiMaxMonitor] lm_head hook raised; this step contributes no metrics")
                self._hook_failed = True

        return hook_fn


def setup_multimax_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_tokens: int = 256,
    topk: int = 10,
    prob_eps: float | None = None,
    logit_eps: float | None = None,
    monitor_dict: dict | None = None,
):
    monitor = PaddleMultiMaxMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_tokens=sample_tokens,
        topk=topk,
        prob_eps=prob_eps,
        logit_eps=logit_eps,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["multimax"] = monitor
    return model
