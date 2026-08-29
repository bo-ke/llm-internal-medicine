"""PaddleFleet-specific Probe base class with GPU-buffer recording API.

Hot-path discipline: see ``.claude/skills/monitor-hook-perf-rules``.
``declare_*`` is the schema gate; ``record_*`` only has a disabled-key guard.
"""

import logging

import paddle

from ...core.base_monitor import Probe
from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)


class PaddleProbe(Probe):
    """Probe with paddle-backed GPU-buffer accumulator + recompute guard.

    GPU-buffer API — ``declare_mean`` / ``declare_max`` / ``declare_min``
    at hook-registration time, then ``record_mean`` / ``record_max`` /
    ``record_min`` inside hooks with **GPU 0-dim tensors**. No D2H sync
    fires until ``step()`` does a single batched flush.
    """

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self._mean_keys: set[str] = set()  # 需要求平均的 metric keys
        self._max_keys: set[str] = set()  # 需要取最大值的 metric keys
        self._min_keys: set[str] = set()  # 需要取最小值的 metric keys
        self._gpu_acc: dict[str, paddle.Tensor] = {}  # GPU 0-dim 累加器
        self._gpu_cnt: dict[str, int] = {}  # 每个 key 被 record 的次数
        # global_key → (聚合方式, [对应的 layer keys])，flush 时从 layer 推导 global
        self._layer_metric_groups: dict[str, tuple[str, list[str]]] = {}
        self._layer_metric_keys: set[str] = set()  # 所有 per-layer key，用于 flush 时判断是否输出
        self._disabled_keys: set[str] = set()  # log_global=False 时被禁用的 global keys
        self._vector_keys: dict[str, int] = {}  # per-layer 向量 key → 长度
        self._gpu_vec: dict[str, paddle.Tensor] = {}  # per-layer 向量累加器 [size]
        self._gpu_vec_cnt: dict[str, int] = {}
        self._vector_elem_keys: dict[str, list[str]] = {}  # 向量 key → 展开后的逐元素 key
        # 向量指标的跨 rank 归约组，见 _vector_reduce_group()。None = 不归约（单卡/组不可用）。
        self._vec_group = None
        self._vec_group_resolved = False
        self._mtp_layer_ids: set[int] = set()
        self._buffers_allocated = False

    def _should_monitor(self) -> bool:
        if not paddle.is_grad_enabled():
            return False
        return super()._should_monitor()

    def _flush_buffers(self) -> None:
        """由 step() 调用，执行唯一的 D2H 传输并写入 training_logs。"""
        if not self._buffers_allocated:
            return
        flushed = self._flush_gpu_buffer()
        if flushed:
            training_logs.update(**flushed)

    # ------------------------------------------------------------------
    # GPU-buffer API: declare → allocate → record → flush
    # ------------------------------------------------------------------

    def declare_mean(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_mean({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys
        self._mean_keys.add(key)

    def declare_max(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_max({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys
        self._max_keys.add(key)

    def declare_min(self, key: str) -> None:
        assert not self._buffers_allocated, f"declare_min({key!r}) after allocate_buffers"
        if self._should_disable_explicit_key(key):
            self._disabled_keys.add(key)
            return
        assert key not in self._mean_keys and key not in self._max_keys and key not in self._min_keys
        self._min_keys.add(key)

    def allocate_buffers(self, dtype=None) -> None:
        """物化所有已声明的累加器为 GPU 0-dim tensor。幂等，调用后 schema 冻结。"""
        if self._buffers_allocated:
            return
        if dtype is None:
            dtype = "float32"
        # mean: 初始化为 0，record 时累加，flush 时除以 count
        for k in self._mean_keys:
            self._gpu_acc[k] = paddle.zeros((), dtype=dtype)
            self._gpu_cnt[k] = 0
        # max: 初始化为 -inf，record 时取 maximum
        for k in self._max_keys:
            self._gpu_acc[k] = paddle.full((), float("-inf"), dtype=dtype)
            self._gpu_cnt[k] = 0
        # min: 初始化为 +inf，record 时取 minimum
        for k in self._min_keys:
            self._gpu_acc[k] = paddle.full((), float("inf"), dtype=dtype)
            self._gpu_cnt[k] = 0
        # vector: 每层一条 [size] 累加器，flush 时按元素展开成独立 key
        for k, size in self._vector_keys.items():
            self._gpu_vec[k] = paddle.zeros([size], dtype=dtype)
            self._gpu_vec_cnt[k] = 0
        # 逐元素 key 的跨 rank 归约在 flush 时于 tensor 侧完成（见 _flush_gpu_buffer），
        # 因此把它们从 gather_and_aggregate 的 all_gather_object payload 里摘掉
        if self._vector_elem_keys:
            training_logs.mark_pre_aggregated(key for elem_keys in self._vector_elem_keys.values() for key in elem_keys)
        self._buffers_allocated = True
        if self.verbose:
            logger.info(
                f"[{self.METRIC_PREFIX}] GPU buffer: "
                f"mean={len(self._mean_keys)} max={len(self._max_keys)} min={len(self._min_keys)} "
                f"vector={len(self._vector_keys)}"
            )

    def record_mean(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上就地累加，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        self._gpu_acc[key].add_(val.detach())
        self._gpu_cnt[key] += 1

    def record_max(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上取 maximum，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        # paddle 不支持 maximum 的 out= 参数，用 assign 实现就地写入
        paddle.assign(paddle.maximum(self._gpu_acc[key], val.detach()), self._gpu_acc[key])
        self._gpu_cnt[key] += 1

    def record_min(self, key: str, val: paddle.Tensor) -> None:
        """热路径：GPU 上取 minimum，不触发 D2H 同步。"""
        if key in self._disabled_keys:
            return
        paddle.assign(paddle.minimum(self._gpu_acc[key], val.detach()), self._gpu_acc[key])
        self._gpu_cnt[key] += 1

    # ------------------------------------------------------------------
    # Convenience: declare/record using class-level aggregation rules
    # ------------------------------------------------------------------

    def mark_mtp_layers(self, layer_ids) -> None:
        """Mark MTP layers before declaring the metric schema."""
        assert not self._buffers_allocated, "mark_mtp_layers after allocate_buffers"
        self._mtp_layer_ids.update(int(layer_idx) for layer_idx in layer_ids)

    def _layer_key(self, layer_idx: int, metric_name: str, attn_type: str | None = None) -> str:
        # Attention type stays on the metric name so window/full charts split
        # naturally. MTP identity stays on the layer token and follows the
        # existing metric gather/JSONL path without separate metadata.
        layer_token = f"layer_{layer_idx}"
        if layer_idx in self._mtp_layer_ids:
            layer_token += "_mtp"
        if attn_type is not None:
            return f"{self.METRIC_PREFIX}/{layer_token}/{attn_type}_{metric_name}"
        return f"{self.METRIC_PREFIX}/{layer_token}/{metric_name}"

    def _global_key(self, metric_name: str, attn_type: str | None = None) -> str:
        if attn_type is not None:
            return f"{self.METRIC_PREFIX}/global_{attn_type}_{metric_name}"
        return f"{self.METRIC_PREFIX}/global_{metric_name}"

    def _should_disable_explicit_key(self, key: str) -> bool:
        return key.startswith(f"{self.METRIC_PREFIX}/global_") and not self.log_global

    def declare_layer_metric(self, layer_idx: int, metric_name: str, attn_type: str | None = None) -> None:
        """声明一个 per-layer 指标。

        1. 根据 MAX_AGGREGATED/MIN_AGGREGATED 选择聚合方式，注册 layer key
        2. 建立 layer_key → global_key 的分组映射，flush 时自动推导 global

        ``attn_type`` (optional): when set (``"mla"`` / ``"hca"`` / ``"csa"`` /
        ``"window"`` / ``"mqa"`` on ``csa_compress_ratios`` stacks, ``"swa"`` /
        ``"full"`` on ``sliding_window`` stacks), the tag is prepended to
        ``metric_name`` in both layer and global keys so the viewer renders each
        attention kind in a separate chart. When ``None``, legacy key layout is
        preserved.
        """
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name, attn_type=attn_type)
        global_key = self._global_key(metric_name, attn_type=attn_type)
        all_declared = self._mean_keys | self._max_keys | self._min_keys
        assert global_key not in all_declared
        # 根据类级别聚合规则选择 declare 方式 (基于原始 metric_name，attn_type 不改变聚合语义)
        if self._is_max_aggregated(metric_name):
            agg = "max"
            if layer_key not in all_declared:
                self.declare_max(layer_key)
        elif metric_name in self.MIN_AGGREGATED:
            agg = "min"
            if layer_key not in all_declared:
                self.declare_min(layer_key)
        else:
            agg = "mean"
            if layer_key not in all_declared:
                self.declare_mean(layer_key)
        self._layer_metric_keys.add(layer_key)
        if not self.log_global:
            return
        # 注册 global 分组: flush 时从这些 layer keys 聚合出 global 值
        existing = self._layer_metric_groups.get(global_key)
        if existing is None:
            self._layer_metric_groups[global_key] = (agg, [layer_key])
        else:
            assert existing[0] == agg
            existing[1].append(layer_key)

    def record_layer_metric(
        self, layer_idx: int, metric_name: str, val: paddle.Tensor, attn_type: str | None = None
    ) -> None:
        """热路径：只写 per-layer 累加器，global 在 flush 时从各层推导。"""
        if not (self.log_per_layer or self.log_global):
            return
        layer_key = self._layer_key(layer_idx, metric_name, attn_type=attn_type)
        if self._is_max_aggregated(metric_name):
            self.record_max(layer_key, val)
        elif metric_name in self.MIN_AGGREGATED:
            self.record_min(layer_key, val)
        else:
            self.record_mean(layer_key, val)

    # ------------------------------------------------------------------
    # Per-layer vector metrics (one curve per element, e.g. per expert)
    # ------------------------------------------------------------------

    def declare_layer_vector(self, layer_idx: int, metric_name: str, size: int, elem_tag: str = "e") -> None:
        """声明一个 per-layer 向量指标，mean 聚合，flush 时展开为 ``{key}_{elem_tag}{i}``。

        用于「每层每个元素一条曲线」的指标（例如每个专家的占比）：热路径上整条
        向量只有一次 ``add_``，而不是每个元素一个 kernel。向量指标不参与 global
        派生 —— 跨层视图由 viewer 自己从各层曲线汇总。
        """
        assert not self._buffers_allocated, f"declare_layer_vector({metric_name!r}) after allocate_buffers"
        if not self.log_per_layer:
            return
        key = self._layer_key(layer_idx, metric_name)
        assert key not in self._vector_keys, f"declare_layer_vector({key!r}) declared twice"
        size = int(size)
        assert size > 0
        self._vector_keys[key] = size
        self._vector_elem_keys[key] = [f"{key}_{elem_tag}{i}" for i in range(size)]

    def record_layer_vector(self, layer_idx: int, metric_name: str, vec: paddle.Tensor) -> None:
        """热路径：整条 ``[size]`` 向量一次就地累加，不触发 D2H 同步。"""
        key = self._layer_key(layer_idx, metric_name)
        buf = self._gpu_vec.get(key)
        if buf is None:  # log_per_layer=False 或该层未声明
            return
        buf.add_(vec.detach().astype(buf.dtype))
        self._gpu_vec_cnt[key] += 1

    def _emits_layer_key(self, key: str) -> bool:
        """per-layer key 是否写进日志（global 派生不受此影响）。"""
        if key not in self._layer_metric_keys:
            return True  # 显式 declare 的非逐层 key
        return self.log_per_layer

    def _vector_reduce_group(self):
        """向量指标的跨 rank 归约组：DP(+CP)，即持有相同层但看到不同 token 的那些 rank。

        - TP rank 复算同一份路由，值本就相同，归约进来只是浪费；
        - PP rank 持有不同的层，定长 collective 在它们之间形状都对不上，必然挂死；

        所以 DP x CP 既是语义上正确的归约域，也是形状安全的那个。解析失败时返回
        None，退化为「只信本 rank 的值」——向量指标本身在单 rank 上已是无偏估计
        （整条归一化占比向量都在本 rank 上算全，跨 rank 平均只降噪）。
        """
        if self._vec_group_resolved:
            return self._vec_group
        self._vec_group_resolved = True
        try:
            import paddle.distributed as dist

            if not dist.is_initialized():
                return None
            from paddlefleet.parallel_state import get_data_parallel_group

            try:
                self._vec_group = get_data_parallel_group(with_context_parallel=True)
            except TypeError:  # 老版本签名里没有 with_context_parallel
                self._vec_group = get_data_parallel_group()
        except Exception as exc:
            logger.warning(
                f"[{self.METRIC_PREFIX}] 无法解析向量指标的 DP 归约组 ({exc}); "
                f"逐元素曲线将只反映本 rank 的观测（噪声更大，但不影响标量指标）"
            )
            self._vec_group = None
        return self._vec_group

    def _vector_means_for_flush(self) -> dict[str, list[float]]:
        """向量 key → 逐元素的全局均值（Python float）。

        一次 collective 覆盖所有向量 key。形状只由 schema 决定（同一 PP stage 的各
        rank 声明相同的层），刻意不按 ``cnt == 0`` 过滤 —— 否则各 rank 会带着不同
        形状进同一个 collective 而挂死（同 tests/test_core_monitoring.py 里记录的
        PP=4 死锁）。同组内各 rank 的 record 次数相同，故 SUM / nranks 等价于旧
        路径「各 rank 均值再等权平均」的语义。

        归约在 **float64** 上做，并单独走一次 D2H 而不与标量共用那次 concat：
        旧路径是把 float32 的 per-rank 均值取回 Python 再用 float64 求平均，若这里
        用 float32 归约，误差会随卡数累积（几千卡下相对误差可达 1e-6 量级），
        单独用 float64 能把差异压回 float64 舍入水平。这是 step 边界的一次额外
        D2H，不在 hook 热路径上。
        """
        order = list(self._vector_keys)
        if not order:
            return {}

        flat = paddle.concat([(self._gpu_vec[k] / max(self._gpu_vec_cnt[k], 1)).astype("float64") for k in order])
        group = self._vector_reduce_group()
        if group is not None:
            import paddle.distributed as dist

            dist.all_reduce(flat, group=group)
            flat = flat / float(group.nranks)

        values = flat.cpu().tolist()
        means: dict[str, list[float]] = {}
        offset = 0
        for k in order:
            size = self._vector_keys[k]
            if self._gpu_vec_cnt[k] > 0:
                means[k] = values[offset : offset + size]
            offset += size
        return means

    def _flush_gpu_buffer(self) -> dict[str, float]:
        """单次批量 D2H：收集所有累加器 → 推导 global → concat→cpu→tolist → 重置。"""
        keys: list[str] = []
        tensors: list[paddle.Tensor] = []

        # 1) 收集 per-layer mean 指标: acc / count
        for k in self._mean_keys:
            cnt = self._gpu_cnt[k]
            if cnt == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k] / cnt)

        # 2) 收集 per-layer max 指标: 直接取累加器值
        for k in self._max_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        # 3) 收集 per-layer min 指标
        for k in self._min_keys:
            if self._gpu_cnt[k] == 0:
                continue
            if self._emits_layer_key(k):
                keys.append(k)
                tensors.append(self._gpu_acc[k].clone())

        # 4) 从各层累加器推导 global 值（不需要 hook 时双写）
        if self.log_global:
            for global_key, (agg, layer_keys) in self._layer_metric_groups.items():
                active = [lk for lk in layer_keys if self._gpu_cnt.get(lk, 0) > 0]
                if not active:
                    continue
                if agg == "mean":
                    total_sum = paddle.stack([self._gpu_acc[lk] for lk in active]).sum()
                    total_cnt = sum(self._gpu_cnt[lk] for lk in active)
                    tensors.append(total_sum / total_cnt)
                elif agg == "max":
                    tensors.append(paddle.stack([self._gpu_acc[lk] for lk in active]).max())
                else:
                    tensors.append(paddle.stack([self._gpu_acc[lk] for lk in active]).min())
                keys.append(global_key)

        # 5) 标量的唯一 D2H 同步点：一次 concat → cpu → tolist
        out: dict[str, float] = {}
        if tensors:
            vals = paddle.concat([t.reshape([-1]) for t in tensors]).cpu().tolist()
            out = dict(zip(keys, vals, strict=False))

        # 6) 向量指标：float64 归约 + 自己的一次 D2H（见 _vector_means_for_flush）。
        #    展开出来的逐元素 key 已是跨 rank 结果，不再进 all_gather_object
        #    （allocate_buffers 里已 mark_pre_aggregated）。
        for k, elements in self._vector_means_for_flush().items():
            out.update(zip(self._vector_elem_keys[k], elements, strict=False))

        # 7) 重置所有累加器，为下一个 step 准备
        for k in self._mean_keys:
            self._gpu_acc[k].zero_()
            self._gpu_cnt[k] = 0
        for k in self._max_keys:
            paddle.assign(paddle.full((), float("-inf"), dtype=self._gpu_acc[k].dtype), self._gpu_acc[k])
            self._gpu_cnt[k] = 0
        for k in self._min_keys:
            paddle.assign(paddle.full((), float("inf"), dtype=self._gpu_acc[k].dtype), self._gpu_acc[k])
            self._gpu_cnt[k] = 0
        for k in self._vector_keys:
            self._gpu_vec[k].zero_()
            self._gpu_vec_cnt[k] = 0
        return out
