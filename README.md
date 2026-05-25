# Internal Medicine — 模型健康监控系统

训练时模型健康的实时监控框架，通过 forward hook 零侵入式采集指标，不影响训练梯度。

包含四大监控模块：
- **[PLE Health](./docs/ple_health.md)** — Per-Layer Embedding 健康监控 (7 指标)
- **[MoE Health](./docs/moe_specialist.md)** — MoE 专家系统健康监控 (13 指标)
- **[QK Stats](./docs/qk_logits.md)** — 注意力 QK 统计监控 (10 指标)
- **[Massive Activation Health](./docs/massive_activation.md)** — Residual Stream Massive Activation 健康监控 (6 指标)

---

## 快速开始

### 统一 API

```python
from internal_medicine import setup_internal_medicine

# 创建 monitor_dict 用于存储 monitor 实例
monitor_dict = {}

# 启用全部监控 (默认)
model = setup_internal_medicine(
    model,
    monitors=['all'],              # 或指定 ['qk_stats', 'moe_health', 'ple_health']
    monitor_dict=monitor_dict,
    monitor_interval=1,
    verbose=False,
)

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
    monitors=['qk_stats', 'moe_health', 'ple_health'],
    monitor_dict=monitor_dict,
))
```

### 读取指标

```python
from internal_medicine import training_logs

# 获取全部最新指标
all_metrics = training_logs.get_latest()

# 按前缀过滤
ple_metrics = training_logs.get_latest(prefix='ple_health')
moe_metrics = training_logs.get_latest(prefix='moe_health')
qk_metrics  = training_logs.get_latest(prefix='qk_stats')

# 跨卡聚合后获取
aggregated = training_logs.gather_and_aggregate()

# 格式化打印
training_logs.print_metrics(prefix='ple_health')

# 重置
training_logs.reset()
```

---

## 架构概览

```
setup_internal_medicine()
    ├── setup_ple_monitor()   → PLEHealthMonitor   → forward hooks on PLE modules
    ├── setup_moe_monitor()   → MoESpecialistMonitor → forward hooks on MoE layers
    ├── setup_qk_monitor()    → QKStatsMonitor     → forward pre-hooks on core_attention
    └── setup_massive_activation_monitor() → MassiveActivationMonitor → forward pre-hooks on transformer layers
                                        │
                                        ▼
                              training_logs (singleton)
                              ├── SmoothedValue (mean/max/min)
                              └── gather_and_aggregate() → 跨卡聚合
```

### 日志键命名规则

所有指标遵循统一的命名格式：

```
{monitor_name}/layer_{global_idx}/{metric_name}    # 逐层指标
{monitor_name}/global_{metric_name}                 # 全局聚合指标
```

- `monitor_name`: `ple_health` | `moe_health` | `qk_stats` | `massive_act`
- `global_idx`: 考虑 PP (Pipeline Parallelism) 的全局层索引 = `pp_rank × local_layers + local_idx`

---

## 一、PLE Health Monitor (ple_health)

> 详细文档: [ple_health/readme.md](./ple_health/readme.md)

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

## 二、MoE Specialist Monitor (moe_health)

> 详细文档: [moe_specialist/readme.md](./moe_specialist/readme.md)

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

### 健康阈值

| 指标 | 值 | 状态 | 说明 |
|------|-----|------|------|
| `bias_affinity_jaccard` | > 0.7 | OK | Bias 对路由影响较小 |
| | 0.3 ~ 0.7 | WARNING | Bias 显著改变了路由 |
| | < 0.3 | SEVERE | Bias 强行扭转了大部分路由决策 |
| `shared_routed_ratio` | 0.3 ~ 3.0 | OK | 共享专家与路由专家贡献均衡 |
| | < 0.3 | INEFFECTIVE | 共享专家作用不大 |
| | > 3.0 | MONOPOLY | 共享专家主导，MoE 退化为 Dense |

---

## 三、QK Stats Monitor (qk_stats)

> 详细文档: [qk_logits/readme.md](./qk_logits/readme.md)

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
| 7 | `entropy_std` | `qk_stats/.../entropy_std` | `std(per_head_entropy)` | 每层+全局 | Head 行为分化度 |
| 8 | `sink_head_ratio` | `qk_stats/.../sink_head_ratio` | `count(sink>θ)/N_heads` | 每层+全局 | Sink head 占比 |
| 9 | `sink_head_max` | `qk_stats/.../sink_head_max` | `max(sink_per_head)` | 每层+全局 | 最强 sink head 权重 |
| 10 | `sink_nonsink_gap` | `qk_stats/.../sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | 每层+全局 | Sink/非Sink 分化度 |

---


---

## 四、Massive Activation Monitor (massive_act)

> 详细文档: [massive_activation.md](./docs/massive_activation.md)

监控 Residual Stream 中的 Massive Activations（极端异常激活值）。
基于 Sun et al. (2026) "The Spike, the Sparse and the Sink" (arXiv:2603.05498) 的发现。

| # | 指标 | 日志键 | 公式 | 级别 | 诊断意义 |
|---|------|--------|------|------|----------|
| 1 | `channel_max` | `massive_act/.../channel_max` | `max(abs(H_i))` | 每层+全局 | 通道峰值，追踪 spike 生命周期 |
| 2 | `channel_max_ratio` | `massive_act/.../channel_max_ratio` | `max / median` | 每层+全局 | 异常值严重度 |
| 3 | `massive_act_channel_count` | `massive_act/.../massive_act_channel_count` | `count(ch > 100*med)` | 每层+全局 | 异常通道数量 |
| 4 | `topk_channel_norm` | `massive_act/.../topk_channel_norm` | `norm(topk(3))` | 每层+全局 | 对应论文 Figure 1 |
| 5 | `post_norm_sparsity` | `massive_act/.../post_norm_sparsity` | `mean(abs(x) < eps)` | 每层+全局 | 归一化后稀疏度 |
| 6 | `post_norm_cosine` | `massive_act/.../post_norm_cosine` | `cos_sim(tokens)` | 每层+全局 | 近常量向量检测 |

### 核心洞察

Massive activations 是 pre-norm Transformer 的**架构副产品**，独立于模型性能：
- PPL 不变但 spike 暴涨 → 部署时量化精度严重退化
- Spike token 经 RMSNorm 后变成稀疏近常量向量 → 为 attention sink 创造条件
- 监控 spike lifecycle (rise-plateau-fall) 可定位 step-up/step-down blocks

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
| 包含 `/min` 或以 `_min` 结尾 | `min` | 历史最小值 |
| 其他 | `mean` | 累积均值 |

### 跨卡聚合

`training_logs.gather_and_aggregate()` 通过 `dist.all_gather_object` 收集所有 rank 的指标，然后按键名规则聚合：

| 键名模式 | 聚合方式 |
|----------|----------|
| 包含 `_max` 或以 `/max` 结尾 | `np.max(all_ranks)` |
| 包含 `_min` 或以 `/min` 结尾 | `np.min(all_ranks)` |
| 其他 | `np.mean(all_ranks)` |

注意: QK Stats Monitor 额外在 hook 内部实现了 TP 级别的 `all_reduce`/`all_gather` 聚合（因为需要跨 TP ranks 的 per-head 信息），而 PLE/MoE 依赖 `gather_and_aggregate` 统一处理。

### 通用配置参数

所有 Monitor 共享以下配置参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_per_layer` | `True` | 记录每层指标 |
| `log_global` | `True` | 记录全局聚合指标 |
| `monitor_interval` | `1` | 监控间隔 (每 N 步采集一次) |
| `verbose` | `False` | 打印调试信息 |

---

## 附录: 完整指标速查表

共 36 个指标键 (7 PLE + 13 MoE + 10 QK + 6 MassiveAct)。

| Monitor | 指标 | 公式 | SmoothedValue 模式 | 健康信号 |
|---------|------|------|--------------------|----------|
| **PLE** | `token_ple_norm` | `mean(\|\|token_ple\|\|₂)` | mean | 量级稳定 |
| **PLE** | `proj_ple_norm` | `mean(\|\|proj × H^{-0.5}\|\|₂)` | mean | 与 token 分支匹配 |
| **PLE** | `per_layer_inputs_norm` | `mean(\|\|(t+p)×2^{-0.5}\|\|₂)` | mean | 量级稳定 |
| **PLE** | `token_proj_cosine` | `mean(cos_sim)` | mean | 显著 < 1.0 |
| **PLE** | `residual_ratio` | `\|\|Δ\|\|/\|\|h\|\|` | mean | 适中 |
| **PLE** | `gate_activation_mean` | `mean(\|act(gate)\|)` | mean | 非零 |
| **PLE** | `gate_sparsity` | `dead_ratio` | mean | 不应持续上升 |
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
| **QK** | `max` | `max(QK^T/√d)` | mean | 不应暴增 |
| **QK** | `mean` | `mean(logits)` | mean | 稳定 |
| **QK** | `entropy_avg` | `-Σ(p log p)` avg | mean | 适中 |
| **QK** | `sink` | `p(token_0)` avg | mean | 不应过高 |
| **QK** | `entropy_min` | `min(head_H)` | min | 不应过低 |
| **QK** | `entropy_max` | `max(head_H)` | max | 合理范围 |
| **QK** | `entropy_std` | `std(head_H)` | mean | 适度分化 |
| **QK** | `sink_head_ratio` | `count(sink>threshold)/N_heads` | mean | Sink head 占比 |
| **QK** | `sink_head_max` | `max(sink_per_head)` | max | 最强 sink head |
| **QK** | `sink_nonsink_gap` | `mean(sink) - mean(nonsink)` | mean | Sink vs 非Sink gap |
| **MassiveAct** | `channel_max` | `max(abs(H_i))` | max | 通道峰值激活 |
| **MassiveAct** | `channel_max_ratio` | `max / median` | max | 异常值严重度 |
| **MassiveAct** | `massive_act_channel_count` | `count(ch > 100*median)` | mean | 异常通道数 |
| **MassiveAct** | `topk_channel_norm` | `norm(topk(3))` | max | Top-K 通道范数 |
| **MassiveAct** | `post_norm_sparsity` | `mean(abs(x) < eps)` | mean | 归一化后稀疏度 |
| **MassiveAct** | `post_norm_cosine` | `cos_sim(tokens)` | mean | 近常量向量检测 |
