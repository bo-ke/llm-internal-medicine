# Internal Medicine — 模型健康监控系统

训练时模型健康的实时监控框架，通过 forward hook 零侵入式采集指标，不影响训练梯度。

包含五大监控模块：
- **[MoE Health](./docs/moe_specialist.md)** — MoE 专家系统健康监控 (18 指标)
- **[QK Stats](./docs/qk_logits.md)** — 注意力 QK 统计监控 (9 指标)
- **[Massive Activation Health](./docs/massive_activation.md)** — Residual Stream Massive Activation 健康监控 (21 指标)
- **[PLE Health](./docs/ple_health.md)** — Per-Layer Embedding 健康监控 (7 指标)
- **[mHC Health](./docs/mhc_health.md)** — Manifold-Constrained Hyper-Connections 映射监控 (每 hc 模块 8 指标；仅在开启 mHC 层时生效)
- **[LAR (Log-Alignment Ratio)](./docs/lar.md)** — output_layer + 每个 MoE router 的 LAR，泛化/过拟合诊断信号（无 SVD、每步 O(1) 通信）

外加一个非指标类工具：
- **[Activation Dump](./docs/activation_dump.md)** — 按 monitor 间隔把残差流 hidden states（默认全量）连同产生它们的输入 batch（`input_ids` / `labels` / `PackedSeqParams`）落盘 (safetensors)，供离线结构分析 (`spec_entropy_explorer.py`)，不上报任何 training_logs 指标

---

## 快速开始

### 统一 API

```python
from internal_medicine import setup_internal_medicine

# 创建 monitor_dict 用于存储 monitor 实例
monitor_dict = {}

# 启用全部指标类监控 (默认)
model = setup_internal_medicine(
    model,
    monitors=['all'],              # 或指定 ['moe_health', 'qk_stats', 'massive_act', 'ple_health']
    monitor_dict=monitor_dict,
    monitor_interval=1,
    verbose=False,
)
```

> **`all` 不包含 `act_dump`**。`all` 的语义是"打开全部**指标**监控"，而 `act_dump` 不上报
> 任何指标、只往磁盘写 hidden-state 张量（当前默认是全量 `[s*b, h]` × 全部层，单个被监控
> step 就是几十 GB，且 `dump_dir` 必须指向真实大容量卷）。让 `all` 顺带把它打开，会让只想
> 看指标的人写爆磁盘。需要落盘时显式点名：`act_dump: {...}`（dict 形式）或
> `monitors=['all', 'act_dump']` —— 与 `all` 并列点名属于显式 opt-in，不会被排除掉。

```python
# 训练循环
for step in range(num_steps):
    loss = model(inputs)
    loss.backward()
    optimizer.step()

    # 每步更新所有 monitor 的计步器
    for monitor in monitor_dict.values():
        monitor.step()
```

### 配合 NeMo Trainer 使用

```python
from functools import partial

cfg.model.register_pre_wrap_hook(partial(
    setup_internal_medicine,
    monitors=['moe_health', 'qk_stats', 'massive_act', 'ple_health'],
    monitor_dict=monitor_dict,
))
```

NeMo YAML 中推荐使用 dict 形式，便于按 monitor 传独立参数：

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_hook_timing: false
internal_medicine_monitors:
    qk_stats: true
    moe_health: true
    massive_act:
        log_activation_rms: false
        log_post_norm_metrics: false
```

`true` 表示启用该 monitor 并使用默认参数；嵌套 dict 会作为该 monitor 的 kwargs 透传，例如上面的 `massive_act` 等价于
`setup_internal_medicine(..., massive_act={"log_activation_rms": False, "log_post_norm_metrics": False})`。
`log_activation_rms` 统一控制残差尺度/增益一组指标（`activation_rms` + `spectral_norm_max/min`）——`activation_rms` 由 spectral hook 的 per-token pre-RMS 免费导出，故合用一个开关。
`log_lipschitz`（默认 `true`）控制每层 Lipschitz/梯度增益指标（`lipschitz_max/min`）——由 forward hook 在输入/输出 hidden 上注册 tensor grad hook，在反向传播采集 `‖∂L/∂x‖/‖∂L/∂y‖`。
`log_logit_lens_entropy`（默认 **`false`**）控制 logit-lens 熵指标（`logit_lens_entropy_mean` + `logit_lens_logsumexp_mean`，均只报告 token 均值）；`log_logit_lens_cross_entropy`（默认 **`false`**）额外开启逐层交叉熵（`logit_lens_cross_entropy_mean`，末层≈LM loss，与熵共用同一次投影，只多一次 gather）——将每层残差经 LM head 投影求 softmax 熵/对数配分/对真实 token 的交叉熵，**开销较大**故默认关闭。配套参数：`logit_lens_chunk_size`（默认 `1024`，按 token 分块的 tile 大小）、`logit_lens_apply_final_norm`（默认 `true`，投影前是否过 `decoder.final_layernorm`）、`logit_lens_layers`（默认 `None`=全部持有 head 的层；传 list 只监控指定 global 层索引）。交叉熵的 label 由 head-owning chunk 上的顶层 pre-hook 从 `forward(labels=...)` 捕获，loss_mask 在模型外施加故报告未加权 token 均值。PP 下仅持有 LM head 的 stage 生效，其余 stage 为 no-op。
`log_hidden_spectral_entropy`（默认 **`false`**）控制 post-RMSNorm 隐状态谱熵指标（`hidden_spectral_entropy`）——对归一化后 hidden 的 Gram 矩阵做 `eigvalsh` 求归一化特征值分布的熵（有效秩，衡量表征多样性/秩坍缩），无 full SVD、无 host sync。set-level 非线性量，跨 shard 均值为近似。
list 形式仍兼容，但需要按 monitor 传参时不要再额外加 `internal_medicine_monitor_kwargs` 字段。

### 读取指标

```python
from internal_medicine import training_logs

# 获取全部最新指标
all_metrics = training_logs.get_latest()

# 按前缀过滤
moe_metrics = training_logs.get_latest(prefix='moe_health')
qk_metrics  = training_logs.get_latest(prefix='qk_stats')
massive_metrics = training_logs.get_latest(prefix='massive_act')
ple_metrics = training_logs.get_latest(prefix='ple_health')

# 跨卡聚合后获取
aggregated = training_logs.gather_and_aggregate()

# 格式化打印
training_logs.print_metrics(prefix='massive_act')

# 重置
training_logs.reset()
```

---

## 架构概览

```
setup_internal_medicine()
    ├── setup_moe_monitor()   → MoESpecialistMonitor → forward hooks on MoE layers
    ├── setup_qk_monitor()    → QKStatsMonitor     → forward pre-hooks on core_attention
    ├── setup_massive_activation_monitor() → MassiveActivationMonitor → forward pre-hooks on transformer layers
    └── setup_ple_monitor()   → PLEHealthMonitor   → forward hooks on PLE modules
                                        │
                                        ▼
                              GPU 0-dim accumulators
                                        │
                                        ▼
                              training_logs (singleton)
                              ├── SmoothedValue (mean/max/min)
                              └── gather_and_aggregate() → 跨卡聚合
```

### 日志键命名规则

所有指标遵循统一的命名格式：

```
{monitor_name}/layer_{global_idx}/{metric_name}        # 普通层逐层指标
{monitor_name}/layer_{global_idx}_mtp/{metric_name}    # MTP 层逐层指标
{monitor_name}/global_{metric_name}                     # 全局聚合指标
```

- `monitor_name`: `moe_health` | `qk_stats` | `massive_act` | `ple_health` | `mhc_health`
- `global_idx`: 考虑 PP (Pipeline Parallelism) 的全局层索引 = `pp_rank × local_layers + local_idx`
- `_mtp`: 仅 MTP layer 带有的层类型标记，随指标走现有聚合和日志链路

---

## 一、MoE Specialist Monitor (moe_health)

> 详细文档: [moe_specialist.md](./docs/moe_specialist.md)

监控 MoE (Mixture of Experts) 路由、专家权重和负载均衡健康状况。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `router_entropy` | `moe_health/.../router_entropy` | `-Σ(p × log(p))` | 每层+全局 | 路由分布均匀度 |
| 2 | `score_sum_mean` | `moe_health/.../score_sum_mean` | `mean(topk_scores.sum())` | 每层+全局 | TopK 分数和均值 |
| 3 | `score_sum_min` | `moe_health/.../score_sum_min` | `min(topk_scores.sum())` | 每层+全局 | TopK 分数和最小值 |
| 4 | `score_sum_max` | `moe_health/.../score_sum_max` | `max(topk_scores.sum())` | 每层+全局 | TopK 分数和最大值 |
| 5 | `expert_bias_mean` | `moe_health/.../expert_bias_mean` | `expert_bias.mean()` | 每层+全局 | 专家偏置均值 |
| 6 | `expert_bias_std` | `moe_health/.../expert_bias_std` | `expert_bias.std()` | 每层+全局 | 专家偏置标准差 |
| 7 | `bias_affinity_jaccard` | `moe_health/.../bias_affinity_jaccard` | `\|A∩B\| / \|A∪B\|` | 每层+全局 | Bias 前后路由一致性 |
| 8 | `expert_norm_mean` | `moe_health/.../expert_norm_mean` | `mean(expert_L2_norms)` | 每层+全局 | 专家权重范数均值 |
| 9 | `expert_norm_std` | `moe_health/.../expert_norm_std` | `std(expert_L2_norms)` | 每层+全局 | 专家权重范数标准差 |
| 10 | `expert_norm_min` | `moe_health/.../expert_norm_min` | `min(expert_L2_norms)` | 每层+全局 | 最小专家范数 |
| 11 | `expert_norm_max` | `moe_health/.../expert_norm_max` | `max(expert_L2_norms)` | 每层+全局 | 最大专家范数 |
| 12 | `shared_expert_norm` | `moe_health/.../shared_expert_norm` | `\|\|shared_params\|\|₂` | 每层+全局 | 共享专家权重范数 |
| 13 | `shared_routed_ratio` | `moe_health/.../shared_routed_ratio` | `shared_norm / routed_mean` | 每层+全局 | 共享/路由专家比例 |
| 14 | `load_max_min_ratio` | `moe_health/.../load_max_min_ratio` | `max(tokens) / min(tokens)` | 每层+全局 | 最忙/最闲专家 token 数比值 |
| 15 | `load_max_median_ratio` | `moe_health/.../load_max_median_ratio` | `max(tokens) / median(tokens)` | 每层+全局 | 最忙/中位专家 token 数比值 |
| 16 | `load_cv` | `moe_health/.../load_cv` | `std(tokens) / mean(tokens)` | 每层+全局 | 专家负载变异系数 (均衡=0) |
| 17 | `latent_combine_rms` | `moe_health/.../latent_combine_rms` | `rms(combine 后的 latent 张量)` | 每层+全局 (max 聚合) | k-way combine 后、latent 上投影前的整体幅度 |
| 18 | `latent_combine_channel_max_median_ratio` | `moe_health/.../latent_combine_channel_max_median_ratio` | `max_c / median_c` (per-channel \|max\|) | 每层+全局 (max 聚合) | 同一张量的 latent 通道集中度 (massive-activation 前兆) |

> **注**: 指标 14-16 (`load_*`) 在满足以下**任一**条件时输出, 优先使用前者:
> 1. `global_aux_loss` 已开启 (`get_aux_loss_coeff("global_aux_loss") > 0`): 复用
>    router 的 `global_tokens_per_expert` 缓冲区 (已跨 TPxDPxCP all-reduce 并在
>    global batch 内累加)。该缓冲区在 `finalize_model_grads` →
>    `reset_model_temporary_tensors` 中被清零, 因此在清零**之前**读取。
> 2. `moe_router_enable_expert_bias=true`: 复用 mcore 在 `get_updated_expert_bias`
>    中已做的 TPxCPxDP all-reduce (softmax 模型无法开启 expert-bias, 属此路径回退项)。
>
> 两种来源都在 global batch 级读取 (不在 forward hook 热路径上), 且计数已跨 rank
> reduce, 因此无需 monitor 侧 collective, 统计值已是全局正确值。load 比值本身
> scale-invariant, 直接用累加计数即可。
>
> **注**: 指标 17-18 (`latent_combine_*`) 仅在 latent-MoE 模型 (`moe_latent_size`
> 已设置, mcore `MoELayer` 因此构造 `fc1/fc2_latent_proj`) 上输出, 其它模型不声明该
> 指标。测点是 `fc2_latent_proj` 的 forward **pre-hook**, 其输入正是
> `combine_postprocess` 产出、尚未做 latent 上投影的张量 —— 即 topk 个专家输出按
> router 权重求和之后的结果。`fc2_latent_proj` 是 `parallel_mode="duplicated"`,
> latent 维不被 TP 切分, 故 per-channel 归约在本 rank 内已完整; token 维按 DP/CP 切分,
> 由 flush 时的 `gather_and_aggregate` 组合 —— 两个指标都取 **max**（不是 mean）: 它们
> 的用途是抓幅度异常, 跨 microbatch / 跨层 / 跨 rank 取平均会把尖峰摊平在正常样本里。
> 因此两者都显式列进 `MAX_AGGREGATED` 与 `MAX_AGGREGATED_SUFFIXES`（名字都不以 `_max`
> 结尾, 不会被自动识别）。分母用 **median** 与 `massive_act/channel_max_ratio` 保持
> 一致, 两者可直接横向对比; 且 median 对本指标要抓的离群通道稳健 —— 用 mean 会被尖峰
> 自身抬高分母, 反而压掉信号。

### 健康阈值

| 指标 | 值 | 状态 | 说明 |
|------|-----|------|------|
| `bias_affinity_jaccard` | > 0.7 | OK | Bias 对路由影响较小 |
| | 0.3 ~ 0.7 | WARNING | Bias 显著改变了路由 |
| | < 0.3 | SEVERE | Bias 强行扭转了大部分路由决策 |
| `shared_routed_ratio` | 0.3 ~ 3.0 | OK | 共享专家与路由专家贡献均衡 |
| | < 0.3 | INEFFECTIVE | 共享专家作用不大 |
| | > 3.0 | MONOPOLY | 共享专家主导，MoE 退化为 Dense |
| `latent_combine_channel_max_median_ratio` | ~1 | OK | latent 通道峰值分布均匀 |
| | 持续上升 | WARNING | 少数 latent 通道开始主导 combine 输出 (massive activation 前兆) |
| `latent_combine_rms` | 平稳 | OK | combine 输出幅度稳定 |
| | 随 step 单调上升 | WARNING | combine 侧幅度累积, 关注下游 `fc2_latent_proj` 与残差流放大 |

---

## 二、QK Stats Monitor (qk_stats)

> 详细文档: [qk_logits.md](./docs/qk_logits.md)

监控注意力 QK logit 的数值稳定性、集中度和 sink 现象。基于 Triton Online Softmax 内核高效计算。
新增 Sink Head 分类指标，基于 Sun et al. (2026) arXiv:2603.05498 的发现。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `max` | `qk_stats/.../max` | `max(Q·K^T/√d)` | 每层+全局 | Logit 最大值，数值稳定性 |
| 2 | `mean` | `qk_stats/.../mean` | `mean(valid_logits)` | 每层+全局 | Logit 基准量级 |
| 3 | `entropy_avg` | `qk_stats/.../entropy_avg` | `-Σ(p·log(p))` 均值 | 每层+全局 | 注意力集中度 |
| 4 | `sink` | `qk_stats/.../sink` | `mean(softmax[..., 0])` | 每层+全局 | Token-0 注意力权重 |
| 5 | `entropy_min` | `qk_stats/.../entropy_min` | `min(per_head_entropy)` | 每层+全局 | 最尖锐 head 的熵 |
| 6 | `entropy_max` | `qk_stats/.../entropy_max` | `max(per_head_entropy)` | 每层+全局 | 最分散 head 的熵 |
| 7 | `sink_head_ratio` | `qk_stats/.../sink_head_ratio` | `count(sink>θ)/N_heads` | 每层+全局 | Sink head 占比 |
| 8 | `sink_head_max` | `qk_stats/.../sink_head_max` | `max(sink_per_head)` | 每层+全局 | 最强 sink head 权重 |
| 9 | `sink_nonsink_gap` | `qk_stats/.../sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | 每层+全局 | Sink/非Sink 分化度 |

---

## 三、Massive Activation Monitor (massive_act)

> 详细文档: [massive_activation.md](./docs/massive_activation.md)

监控 Residual Stream 中的 Massive Activations（极端异常激活值）。
基于 Sun et al. (2026) "The Spike, the Sparse and the Sink" (arXiv:2603.05498) 的发现。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `channel_max` | `massive_act/.../channel_max` | `max(abs(H_i))` | 每层+全局 | 通道峰值，追踪 spike 生命周期 |
| 2 | `channel_median` | `massive_act/.../channel_median` | `median(max(abs(H_i), tokens))` | 每层+全局 | 通道峰值分布的基准量级 |
| 3 | `channel_p95` | `massive_act/.../channel_p95` | `p95(per_channel_max)` | 每层+全局 | 高分位通道幅度 |
| 4 | `channel_p99` | `massive_act/.../channel_p99` | `p99(per_channel_max)` | 每层+全局 | 极高分位通道幅度 |
| 5 | `channel_max_ratio` | `massive_act/.../channel_max_ratio` | `max / median` | 每层+全局 | 少数通道 outlier 严重度 |
| 6 | `massive_act_channel_count` | `massive_act/.../massive_act_channel_count` | `mean_token(count(&#124;h&#124; > median*√H))` | 每层+全局 | 每 token median-relative 异常通道数（对 seqlen 取均值；megatron）|
| 7 | `channel_count_gt_10` | `massive_act/.../channel_count_gt_10` | `count(per_channel_max > 10)` | 每层+全局 | 广泛高于基准量级的通道数量 |
| 8 | `channel_count_gt_20` | `massive_act/.../channel_count_gt_20` | `count(per_channel_max > 20)` | 每层+全局 | 高幅度通道数量 |
| 9 | `channel_count_gt_30` | `massive_act/.../channel_count_gt_30` | `count(per_channel_max > 30)` | 每层+全局 | 接近当前 1.5B FP4 峰值区间的通道数量 |
| 10 | `topk_channel_norm` | `massive_act/.../topk_channel_norm` | `norm(topk(3))` | 每层+全局 | 对应论文 Figure 1 |
| 11 | `activation_rms` | `massive_act/.../activation_rms` | `sqrt(mean(H_i^2))` | 每层+全局 | residual stream 整体 scale |
| 12 | `post_norm_sparsity` | `massive_act/.../post_norm_sparsity` | `mean(abs(x) < eps)` | 每层+全局 | 归一化后稀疏度 |
| 13 | `post_norm_cosine` | `massive_act/.../post_norm_cosine` | `cos_sim(tokens)` | 每层+全局 | 近常量向量检测 |
| 14 | `spectral_norm_max` | `massive_act/.../spectral_norm_max` | `max(post_rms / pre_rms)` | 每层+全局 | 层增益比上界，谱范数(σ_max)下界 |
| 15 | `spectral_norm_min` | `massive_act/.../spectral_norm_min` | `min(post_rms / pre_rms)` | 每层+全局 | 层增益比下界，最小奇异值(σ_min)上界 |
| 16 | `lipschitz_max` | `massive_act/.../lipschitz_max` | `max(‖∂L/∂x‖ / ‖∂L/∂y‖)` | 每层+全局 | 反向梯度增益上界，σ_max(J)下界 = Lipschitz 常数 |
| 17 | `lipschitz_min` | `massive_act/.../lipschitz_min` | `min(‖∂L/∂x‖ / ‖∂L/∂y‖)` | 每层+全局 | 反向梯度增益下界，σ_min(J)上界 |
| 18 | `logit_lens_entropy_mean` | `massive_act/.../logit_lens_entropy_mean` | `mean_t(H(softmax(final_norm(h)·Wᵀ)))` | 每层+全局 | logit-lens 预测熵均值，随深度下降=表征逐层"定型" |
| 19 | `logit_lens_logsumexp_mean` | `massive_act/.../logit_lens_logsumexp_mean` | `mean_t(logsumexp(final_norm(h)·Wᵀ))` | 每层+全局 | logit-lens 对数配分均值，追踪 logit 原始尺度 |
| 20 | `logit_lens_cross_entropy_mean` | `massive_act/.../logit_lens_cross_entropy_mean` | `mean_t(log_z − l[y])` | 每层+全局 | logit-lens 逐层交叉熵，末层≈LM loss |
| 21 | `hidden_spectral_entropy` | `massive_act/.../hidden_spectral_entropy` | `-Σ p_i log p_i, p_i=σ_i²/‖h‖_F²` | 每层+全局 | post-norm 隐状态谱熵(有效秩)，表征多样性/秩坍缩 |

> **注**: 指标 18-20 (`logit_lens_*`) 为**默认关闭**的可选项（熵/logsumexp 由 `log_logit_lens_entropy=True`
> 开启，交叉熵由 `log_logit_lens_cross_entropy=True` 开启）。
> 将每层残差经 LM head 投影到 vocab logits，同一次投影求三组量（**仅报告 token 均值**）：softmax 熵
> `H(p) = -Σ p·log p ∈ [0, log(vocab)]`（随深度下降=逐层定型）、对数配分 `log Z = logsumexp(l)`
> （softmax 归一化项，追踪 logit 原始尺度，直接复用熵里的 `log_z`，零额外开销），以及对真实下一个 token 的
> 交叉熵 `CE = log Z − l[y]`（同样复用 `log_z`，只多一次 gather；**末层 CE ≈ LM loss**）。相当于每监控层每监控步
> 做一次 LM-head 前向，开销较大，建议配合 `monitor_interval` 与 `logit_lens_layers`（只监控指定层）使用。
> 投影**按 token 分块**（`logit_lens_chunk_size`，默认 1024），任一时刻只物化一个 `[chunk, vocab]` tile，
> 绝不 materialize 完整 `[tokens, vocab]` logits。默认先过 `decoder.final_layernorm`
> （`logit_lens_apply_final_norm=True`，LM head 是在 final-norm 后的表征上训练的）。熵用 `H = log_z − E_p[l]`
> 写法（`log_z = torch.logsumexp(l)`、`E_p[l] = Σ softmax(l)·l`），直接用 `torch.logsumexp` / `torch.softmax`，
> 无手写 exp/log。**交叉熵 loss-mask 说明**：label 由 head-owning chunk 上的顶层 forward pre-hook 从
> `model.forward(labels=...)` 捕获并按 seq-major 对齐；loss_mask 在模型 forward 之外施加，monitor 拿不到，故报告
> 的是**未加权 token 均值 CE**（= LM loss up to loss-mask 加权）。label 缺失/对齐失败时当步不产出 CE（不报错）。
> **PP 覆盖**: 只有持有 LM head 权重的 PP stage（最后 stage / tied-embedding stage）会计算这组指标；
> 其余 stage `_resolve_lm_head` 返回 `(None, None)`，不 attach hook、不声明 key、彻底 no-op（不做权重广播）。
> **暂不支持 vocab-parallel TP**：`compute_logit_lens_entropy` 断言 `tp_size <= 1`（TP 切 vocab 时归一化需跨 rank
> reduce，无法与 `torch.logsumexp` 融合，后续再加）。
>
> **注**: 指标 21 (`hidden_spectral_entropy`) 为**默认关闭**的可选项（`log_hidden_spectral_entropy=True` 开启）。
> 对 post-RMSNorm 隐状态 `h∈R^{n,d}` 的谱（矩阵/von Neumann）熵：`p_i=σ_i²/‖h‖_F²` 是较小 Gram 矩阵
> （`hᵀh` 若 n≥d 否则 `h hᵀ`）的归一化特征值，`H=-Σ p_i log p_i∈[0, log(min(n,d))]`，`exp(H)` 即有效秩——
> 衡量 token 集张成多少个有效方向（低=秩坍缩，高=方向丰富）。用 `torch.linalg.eigvalsh` 一次 GPU 调用（无
> full SVD、无 host sync）。这是 **set-level 非线性量**，故跨 rank/microbatch 的均值只是近似（mean-of-shard-
> entropies ≠ 全 token 熵），按 per-shard 值在 flush 时平均，作为坍缩趋势信号可接受。

### 核心洞察

Massive activations 是 pre-norm Transformer 的**架构副产品**，独立于模型性能：
- PPL 不变但 spike 暴涨 → 部署时量化精度严重退化
- Spike token 经 RMSNorm 后变成稀疏近常量向量 → 为 attention sink 创造条件
- 监控 spike lifecycle (rise-plateau-fall) 可定位 step-up/step-down blocks
- `channel_max_ratio` 用于识别少数通道 outlier；`channel_p95/p99`、`activation_rms`、`channel_count_gt_10/20/30` 用于识别 residual stream 整体 scale growth。

---

## 四、PLE Health Monitor (ple_health)

> 详细文档: [ple_health.md](./docs/ple_health.md)

监控 Per-Layer Embedding 双分支架构的健康状况。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `token_ple_norm` | `ple_health/global_token_ple_norm` | `mean(\|\|token_ple\|\|₂, dim=-1)` | 全局 | Token 分支信号强度 |
| 2 | `proj_ple_norm` | `ple_health/global_proj_ple_norm` | `mean(\|\|proj_ple × H^{-0.5}\|\|₂, dim=-1)` | 全局 | 投影分支信号强度 |
| 3 | `per_layer_inputs_norm` | `ple_health/global_per_layer_inputs_norm` | `mean(\|\|(token+proj) × 2^{-0.5}\|\|₂, dim=-1)` | 全局 | 合并信号量级 |
| 4 | `token_proj_cosine` | `ple_health/global_token_proj_cosine` | `mean(cosine_sim(token, proj))` | 全局 | 双分支冗余度 (→1 冗余) |
| 5 | `residual_ratio` | `ple_health/layer_{i}/residual_ratio` | `\|\|output - input\|\| / \|\|input\|\|` | 每层+全局 | PLE 贡献幅度 |
| 6 | `gate_activation_mean` | `ple_health/layer_{i}/gate_activation_mean` | `mean(\|act_fn(gate_out)\|)` | 每层+全局 | 门控激活强度 |
| 7 | `gate_sparsity` | `ple_health/layer_{i}/gate_sparsity` | `(\|act\| < 0.01).mean()` | 每层+全局 | 死门控单元占比 |

---

## 五、mHC Health Monitor (mhc_health)

> 详细文档: [mhc_health.md](./docs/mhc_health.md)

监控 mHC (Manifold-Constrained Hyper-Connections) 层的三个 per-token 映射 `h_pre` / `h_post` / `h_res`。
只在模型开启 mHC 层时生效——mHC 类无法 import 或模型不含 `HyperConnectionTransformerLayer` 时该 monitor 为
彻底 no-op（不 wrap、不产生指标）。每层含两个 hyper-connection 模块（`attn` / `mlp`），各产出以下 8 个指标，
指标名以 `attn_` / `mlp_` 前缀区分；全部按 token/batch 求均值。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `{c}_h_pre_mean` | `mhc_health/layer_{i}/{c}_h_pre_mean` | `mean(h_pre)` | 每层+全局 | 聚合门均值 |
| 2 | `{c}_h_pre_std` | `mhc_health/layer_{i}/{c}_h_pre_std` | `std(h_pre)` | 每层+全局 | 聚合门离散度 |
| 3 | `{c}_h_post_mean` | `mhc_health/layer_{i}/{c}_h_post_mean` | `mean(h_post)` | 每层+全局 | 扩展门均值 |
| 4 | `{c}_h_post_std` | `mhc_health/layer_{i}/{c}_h_post_std` | `std(h_post)` | 每层+全局 | 扩展门离散度 |
| 5 | `{c}_amax_gain_fwd` | `mhc_health/layer_{i}/{c}_amax_gain_fwd` | `mean_t(max_i \|Σ_j h_res_ij\|)` | 每层+全局 | 单层前向最坏放大 (≈1) |
| 6 | `{c}_amax_gain_bwd` | `mhc_health/layer_{i}/{c}_amax_gain_bwd` | `mean_t(max_j \|Σ_i h_res_ij\|)` | 每层+全局 | 单层反向最坏放大 (≈1) |
| 7 | `{c}_composite_amax_gain_fwd` | `mhc_health/layer_{i}/{c}_composite_amax_gain_fwd` | 复合映射 `∏ h_res` 的行和 | 每层+全局 | 跨层累积前向放大 |
| 8 | `{c}_composite_amax_gain_bwd` | `mhc_health/layer_{i}/{c}_composite_amax_gain_bwd` | 复合映射 `∏ h_res` 的列和 | 每层+全局 | 跨层累积反向放大 |

`{c}` ∈ `{attn, mlp}`。复合映射为本 pipeline stage / VPP chunk 内 `h_res` 的累乘（每次 forward 在本 stage 首个
hc 模块处重置）——PP=1 时精确，PP>1 时为 stage 局部近似。

---

## 基础设施

### TrainingLogs

全局单例的指标存储，所有 Monitor 将指标写入此处。

```python
from internal_medicine import training_logs
```

**SmoothedValue 聚合模式** — 由指标键名自动推断：

| 键名模式 | 推断模式 | 输出值 |
|----------|----------|--------|
| 包含 `/max` 或以 `_max` 结尾 | `max` | 历史最大值 |
| 以 `topk_channel_norm`、`channel_max_ratio`、`channel_median`、`channel_p95`、`channel_p99`、`activation_rms` 结尾 | `max` | 历史最大值 |
| `channel_count_gt_*` | `max` | 历史最大值 |
| 包含 `/min` 或以 `_min` 结尾 | `min` | 历史最小值 |
| 其他 | `mean` | 累积均值 |

### 跨卡聚合

`training_logs.gather_and_aggregate()` 通过 `dist.all_gather_object` 收集所有 rank 的指标，然后按键名规则聚合：

| 键名模式 | 聚合方式 |
|----------|----------|
| 包含 `_max` 或以 `/max` 结尾 | `np.max(all_ranks)` |
| `channel_max_ratio`、`channel_median`、`channel_p95`、`channel_p99`、`topk_channel_norm`、`activation_rms` | `np.max(all_ranks)` |
| `channel_count_gt_*` | `np.max(all_ranks)` |
| `massive_act_channel_count`（megatron 每 token 均值）| `np.mean(all_ranks)` |
| 包含 `_min` 或以 `/min` 结尾 | `np.min(all_ranks)` |
| 其他 | `np.mean(all_ranks)` |

注意: QK Stats Monitor 不在 hook 内做 TP `all_reduce`/`all_gather`，避免在 attention 热路径插入通信；跨 rank 聚合统一交给 `gather_and_aggregate()`。MassiveAct 会先对 TP 内 per-channel maxima 做 `MAX all_reduce`（通道维被 TP 切分时这是正确性所需），再依赖 `gather_and_aggregate()` 做跨 rank 聚合；MoE/PLE 同样依赖 `gather_and_aggregate()` 统一处理。

### 通用配置参数

所有 Monitor 共享以下配置参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_per_layer` | `True` | 记录每层指标 |
| `log_global` | `True` | 记录全局聚合指标 |
| `monitor_interval` | `1` | 监控间隔 (每 N 步采集一次) |
| `verbose` | `False` | 打印调试信息 |
| `hook_timing_enabled` | `False` | 记录每个 monitor hook 的 CPU wall time；用于定位监控开销，默认关闭 |

NeMo Trainer 对应字段为 `internal_medicine_hook_timing`。开启后 trainer 会按日志间隔输出各 hook 的累计耗时和平均耗时；该开关只做 Python 侧计时，不会额外插入 CUDA 同步。

---

## 附录: 完整指标速查表

共 50 个指标键 (13 MoE + 9 QK + 21 MassiveAct + 7 PLE)。

| Monitor | 指标 | 公式 | SmoothedValue 模式 | 健康信号 |
|---------|------|------|--------------------|----------|
| **MoE** | `router_entropy` | `-Σ(p log p)` | mean | 高 = 均匀路由 |
| **MoE** | `score_sum_mean` | `mean(topk_sum)` | mean | 适中 |
| **MoE** | `score_sum_min` | `min(topk_sum)` | min | 不应过低 |
| **MoE** | `score_sum_max` | `max(topk_sum)` | max | 不应过高 |
| **MoE** | `expert_bias_mean` | `bias.mean()` | mean | 接近零 |
| **MoE** | `expert_bias_std` | `bias.std()` | mean | 适度 |
| **MoE** | `bias_affinity_jaccard` | `\|A∩B\|/\|A∪B\|` | mean | > 0.7 OK |
| **MoE** | `expert_norm_mean` | `mean(L2)` | mean | 稳定 |
| **MoE** | `expert_norm_std` | `std(L2)` | mean | 不应过大 |
| **MoE** | `expert_norm_min` | `min(L2)` | min | 不应萎缩 |
| **MoE** | `expert_norm_max` | `max(L2)` | max | 不应过载 |
| **MoE** | `shared_expert_norm` | `\|\|shared\|\|₂` | mean | 稳定 |
| **MoE** | `shared_routed_ratio` | `shared/routed` | mean | 0.3 ~ 3.0 OK |
| **MoE** | `load_max_min_ratio` | `max/min(tokens)` | mean | 越接近 1 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_max_median_ratio` | `max/median(tokens)` | mean | 越接近 1 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `load_cv` | `std/mean(tokens)` | mean | 越接近 0 越均衡 (global_aux_loss 或 expert_bias) |
| **MoE** | `latent_combine_rms` | `rms(combine 后 latent)` | max | 平稳 (仅 latent-MoE) |
| **MoE** | `latent_combine_channel_max_median_ratio` | `max_c/median_c(per-channel \|max\|)` | max | 越接近 1 越均匀 (仅 latent-MoE) |
| **QK** | `max` | `max(QK^T/√d)` | max | 不应暴增 |
| **QK** | `mean` | `mean(logits)` | mean | 稳定 |
| **QK** | `entropy_avg` | `-Σ(p log p)` avg | mean | 适中 |
| **QK** | `sink` | `p(token_0)` avg | mean | 不应过高 |
| **QK** | `entropy_min` | `min(head_H)` | min | 不应过低 |
| **QK** | `entropy_max` | `max(head_H)` | max | 合理范围 |
| **QK** | `sink_head_ratio` | `count(sink>threshold)/N_heads` | mean | Sink head 占比 |
| **QK** | `sink_head_max` | `max(sink_per_head)` | max | 最强 sink head |
| **QK** | `sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | mean | Sink vs 非Sink gap |
| **MassiveAct** | `channel_max` | `max(abs(H_i))` | max | 通道峰值激活 |
| **MassiveAct** | `channel_median` | `median(per_channel_max)` | max | 通道峰值基准量级 |
| **MassiveAct** | `channel_p95` | `p95(per_channel_max)` | max | 高分位通道幅度 |
| **MassiveAct** | `channel_p99` | `p99(per_channel_max)` | max | 极高分位通道幅度 |
| **MassiveAct** | `channel_max_ratio` | `max / median` | max | 异常值严重度 |
| **MassiveAct** | `massive_act_channel_count` | `mean_token(count(&#124;h&#124; > median*√H))` | mean | 每 token median-relative 异常通道数（megatron）|
| **MassiveAct** | `channel_count_gt_10` | `count(ch > 10)` | max | 广泛高于基准量级 |
| **MassiveAct** | `channel_count_gt_20` | `count(ch > 20)` | max | 高幅度通道数 |
| **MassiveAct** | `channel_count_gt_30` | `count(ch > 30)` | max | 接近当前 FP4 峰值区间 |
| **MassiveAct** | `topk_channel_norm` | `norm(topk(3))` | max | Top-K 通道范数 |
| **MassiveAct** | `activation_rms` | `sqrt(mean(H_i^2))` | max | residual stream 整体 scale |
| **MassiveAct** | `post_norm_sparsity` | `mean(abs(x) < eps)` | mean | 归一化后稀疏度 |
| **MassiveAct** | `post_norm_cosine` | `cos_sim(tokens)` | mean | 近常量向量检测 |
| **MassiveAct** | `spectral_norm_max` | `max(post_rms / pre_rms)` | max | 谱范数(σ_max)下界 |
| **MassiveAct** | `spectral_norm_min` | `min(post_rms / pre_rms)` | min | 最小奇异值(σ_min)上界 |
| **MassiveAct** | `lipschitz_max` | `max(‖∂L/∂x‖ / ‖∂L/∂y‖)` | max | σ_max(J)下界 = Lipschitz 常数 |
| **MassiveAct** | `lipschitz_min` | `min(‖∂L/∂x‖ / ‖∂L/∂y‖)` | min | σ_min(J)上界 |
| **MassiveAct** | `logit_lens_entropy_mean` | `mean_t H(softmax(final_norm(h)·Wᵀ))` | mean | 预测熵均值 (可选, 默认关) |
| **MassiveAct** | `logit_lens_logsumexp_mean` | `mean_t logsumexp(final_norm(h)·Wᵀ)` | mean | 对数配分均值 (可选, 默认关) |
| **MassiveAct** | `logit_lens_cross_entropy_mean` | `mean_t(log_z − l[y])` | mean | 逐层交叉熵，末层≈LM loss (可选, 默认关) |
| **MassiveAct** | `hidden_spectral_entropy` | `-Σ p_i log p_i, p_i=σ_i²/‖h‖_F²` | mean | post-norm 谱熵/有效秩 (可选, 默认关) |
| **PLE** | `token_ple_norm` | `mean(\|\|token_ple\|\|₂)` | mean | 量级稳定 |
| **PLE** | `proj_ple_norm` | `mean(\|\|proj × H^{-0.5}\|\|₂)` | mean | 与 token 分支匹配 |
| **PLE** | `per_layer_inputs_norm` | `mean(\|\|(t+p)×2^{-0.5}\|\|₂)` | mean | 量级稳定 |
| **PLE** | `token_proj_cosine` | `mean(cos_sim)` | mean | 显著 < 1.0 |
| **PLE** | `residual_ratio` | `\|\|Δ\|\|/\|\|h\|\|` | mean | 适中 |
| **PLE** | `gate_activation_mean` | `mean(\|act(gate)\|)` | mean | 非零 |
| **PLE** | `gate_sparsity` | `dead_ratio` | mean | 不应持续上升 |
