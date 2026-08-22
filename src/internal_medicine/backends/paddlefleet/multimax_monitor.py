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
from .multimax_metrics import (
    QUANTILES,
    SPARSITY_REF_MODES,
    _quantile_suffix,
    apply_seglu,
    compute_distribution_metrics,
    compute_param_metrics,
)

logger = logging.getLogger(__name__)

_PARAM_METRICS = tuple(f"range_{i}" for i in range(4)) + tuple(f"t_{i}" for i in range(4))
_DIST_METRICS = (
    "entropy",
    "entropy_norm",
    "top1_prob",
    "sparsity",
    "relevant_count",
    "sparse_count",
)
# Token-mean metrics also report p50/p95/p98 over the sampled tokens. Declared
# here, not derived at compute time, so the schema is complete at registration
# (Rule 3: no lazy declares on the hot path).
_DIST_QUANTILE_METRICS = tuple(f"{name}_{_quantile_suffix(q)}" for name in _DIST_METRICS for q in QUANTILES)


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
        sparsity_ref: float | None = None,
        sparsity_ref_mode: str = "geomean",
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
        self.sparsity_ref = sparsity_ref
        assert sparsity_ref_mode in SPARSITY_REF_MODES, (
            f"sparsity_ref_mode must be one of {SPARSITY_REF_MODES}, got {sparsity_ref_mode!r}"
        )
        self.sparsity_ref_mode = sparsity_ref_mode
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
        topk_bases = (f"top{k}_prob", f"multi_modality_top{k}")
        topk_keys = topk_bases + tuple(f"{base}_{_quantile_suffix(q)}" for base in topk_bases for q in QUANTILES)
        return _DIST_METRICS + _DIST_QUANTILE_METRICS + topk_keys + ("rows",) + _PARAM_METRICS

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

    def _project(self, module: nn.Layer, rows: paddle.Tensor) -> paddle.Tensor:
        """Unmodulated logits for the sampled rows (the SegLU input)."""
        weight = module.weight
        # Project in the weight's dtype: casting the [vocab/tp, hidden] weight to
        # fp32 would allocate a full-size copy (GBs at LM-head scale). Only the
        # small [sample_tokens, vocab/tp] tile is promoted.
        logits = paddle.matmul(rows.astype(weight.dtype), weight, transpose_y=True).astype("float32")
        bias = getattr(module, "bias", None)
        if isinstance(bias, paddle.Tensor):
            logits = logits + bias.astype("float32")
        return logits

    def _hidden_reference(self, module: nn.Layer, inputs) -> paddle.Tensor | None:
        """Unmodulated tile recovered from the head's input, for Def 3.3's ``s``.

        The unfused path only hands back modulated logits, and SegLU is not
        cheaply invertible, so the baseline tile is re-projected from the hidden
        states the hook already receives. Same strided sample as the modulated
        tile, so the rows line up.
        """
        hidden = inputs[0] if isinstance(inputs, (list, tuple)) and inputs else inputs
        weight = getattr(module, "weight", None)
        if not isinstance(hidden, paddle.Tensor) or not isinstance(weight, paddle.Tensor):
            return None
        if hidden.shape[-1] != weight.shape[-1]:
            return None
        return self._project(module, self._sample_rows(hidden))

    def _fused_logits(self, module: nn.Layer, hidden: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Recompute the sampled logits for the fused-CE path.

        With ``fused_linear_ce_loss_chunk > 0`` the head returns
        ``(hidden, weight, bias, ranges, ts)`` and the full logits are never
        materialized, so the tile is projected here and modulated with the
        head's own parameters. Returns ``(modulated, unmodulated)``; the
        baseline is free on this path since the projection happens here.
        """
        raw = self._project(module, self._sample_rows(hidden))
        return apply_seglu(raw, module.multimax_ranges, module.multimax_ts), raw

    def _resolve_logits(
        self, module: nn.Layer, outputs, mtp: bool = False
    ) -> tuple[paddle.Tensor, paddle.Tensor | None] | None:
        """Sampled logits tile for this head, as ``(modulated, unmodulated)``.

        The second element is the SegLU input, needed for Def 3.3's reference
        ``s``; it is ``None`` on the unfused path, where the head hands back only
        the modulated tensor (the hook then re-projects it from the hidden
        states).

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
        return self._sample_rows(candidate), None

    def _make_head_hook(self, tag: str = ""):
        def hook_fn(module, inputs, outputs):
            if not module.training or not self._should_monitor():
                return
            try:
                with paddle.no_grad():
                    for name, value in compute_param_metrics(module.multimax_ranges, module.multimax_ts).items():
                        self.record_mean(self._global_key(tag + name), value)
                    resolved = self._resolve_logits(module, outputs, mtp=bool(tag))
                    if resolved is None:
                        return
                    tile, raw = resolved
                    tile = self._gather_vocab(tile.detach())
                    # Def 3.3's reference is the baseline softmax of the SegLU
                    # input; skipped rather than faked when unavailable, in which
                    # case the metrics fall back to the uniform reference.
                    if raw is None and self.sparsity_ref is None:
                        raw = self._hidden_reference(module, inputs)
                    if raw is not None:
                        raw = self._gather_vocab(raw.detach())
                    metrics = compute_distribution_metrics(
                        tile,
                        prob_eps=self.prob_eps,
                        logit_eps=self.logit_eps,
                        topk=self._topk_effective,
                        sparsity_ref=self.sparsity_ref,
                        ref_logits=raw,
                        sparsity_ref_mode=self.sparsity_ref_mode,
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
    sparsity_ref: float | None = None,
    sparsity_ref_mode: str = "geomean",
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
        sparsity_ref=sparsity_ref,
        sparsity_ref_mode=sparsity_ref_mode,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["multimax"] = monitor
    return model
