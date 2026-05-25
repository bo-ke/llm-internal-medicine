# Massive Activation Monitor

**Residual Stream Massive Activation 健康监控模块**，监控 6 个核心指标。

基于论文发现实现：

> Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
> *The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks.*
> arXiv:2603.05498.

---

## 背景：什么是 Massive Activations？

在 pre-norm Transformer（如 Llama、Qwen、ERNIE）中，少数 token 在少数 hidden state channel 上出现**极端异常值**——比正常激活大 2-3 个数量级。这些异常值：

1. 由早期 FFN block（step-up block）通过 SwiGLU 的**方向性二次放大**注入
2. 通过残差连接在中间层**持续累积**
3. 在末尾 FFN block（step-down block）被**反向中和**

```
Layer:    1  2  3  4  ...  30  31  32
Spike:    -  -  -  ↑↑↑  ===  ===  ↓↓↓  -
                   step-up    step-down
```

### 为什么要监控？

- **Massive activations 独立于 PPL 变化**：loss 看不出问题，但 spike 暴涨可能导致量化精度严重退化（论文 Table 3）
- **是 attention sink 的上游信号**：spike token 经过 RMSNorm 后变成稀疏近常量向量，为 sink 创造条件
- **Weight decay 关联**：关闭 weight decay 后 spike 从 ~3800 涨到 ~12000，但 PPL 仅变 0.3
- **可诊断训练异常**：spike magnitude 突变通常意味着权重矩阵出现了异常高增益结构

---

## 监控指标

### 1. Channel Max (通道最大激活值)

**数学公式：**
```
channel_max = max(|H_i|)
```

对 post-residual hidden states 所有 position 和 channel 取绝对值最大值。

**诊断意义：**
- 正常 pre-norm Transformer 中间层：数百到数千量级
- 首尾层应显著低于中间层（生命周期特征）
- 训练中突增可能预示数值不稳定

---

### 2. Channel Max Ratio (通道异常比)

**数学公式：**
```
channel_max_ratio = max(per_channel_max) / median(per_channel_max)
```

最大 channel 激活与中位 channel 激活的比值。

**诊断意义：**
- 典型 spike 层：ratio > 100x（少数 channel 远超其他）
- ratio 持续攀升说明模型在强化 spike channel 的放大路径
- 非 spike 层（首尾）应接近 1-10x

---

### 3. Massive Activation Channel Count (异常通道数)

**数学公式：**
```
massive_act_channel_count = |{c : max_pos(|H_i[:, c]|) > median × threshold_multiplier}|
```

超过阈值（默认 100× 中位数）的 channel 数量。

**诊断意义：**
- 论文发现 spike channel 通常只有 **2-5 个**（Property ii）
- 数量突增说明模型正在更多 channel 上制造极端值——可能是训练不稳定的信号
- 配合 topk_channel_norm 看趋势

---

### 4. Top-3 Channel Norm (Top-K 通道范数)

**数学公式：**
```
topk_channel_norm = ||topk(per_channel_max, k=3)||₂
```

前 3 个最大 channel 的 L2 范数。直接对应论文 Figure 1 的 "top-3 channel magnitudes"。

**诊断意义：**
- 应呈现 "rise–plateau–fall" 模式（上升→平台→下降）
- 中间层平台值是模型的 "spike fingerprint"
- 跨步骤对比可检测 spike 强度的趋势变化

---

### 5. Post-Norm Sparsity (归一化后稀疏度)

**数学公式：**
```
post_norm_sparsity = mean(|RMSNorm(H_i)| < ε)
```

RMSNorm 后接近零的 entry 占比（默认 ε=0.01）。

**理论基础（论文 Eq. 24）：**
RMSNorm 将 spike token 的非 spike channel 压制为接近零，产生稀疏向量：
```
RMSNorm(h^(s)) ≈ Σ_{i∈C} h̃_i^(s) e_i
```
其中 C 是 spike channel 集合，结果是一个近似 multi-hot 的稀疏表示。

**诊断意义：**
- 高稀疏度 = 模型正在通过 spike+norm 创造 "implicit parameters"
- 结合 sink 指标看：高 sparsity + 高 sink ratio = 经典的 spike→sink 路径激活

---

### 6. Post-Norm Cosine Stability (归一化后余弦稳定性)

**数学公式：**
```
post_norm_cosine = mean(cosine_sim(RMSNorm(H[a]), RMSNorm(H[b])))
```

随机采样 token 对的归一化表示之间的余弦相似度。

**理论基础（论文 Eq. 25, Figure 5）：**
不同 spike token 归一化后坍缩为**近常量向量**：
```
RMSNorm(h^(a)) ≈ RMSNorm(h^(b))
```
这使得 sink token 的 key 向量几乎不变，创造稳定的 attention sink 位置。

**诊断意义：**
- cosine → 1.0 说明 step-up block 已完成 spike 注入（Figure 5）
- 在 step-up block 之前应该较低，之后应跳升到接近 1.0
- 用于定位 step-up block 的确切位置

---

## 健康阈值参考

| 指标 | 值 | 状态 | 说明 |
|------|-----|------|------|
| `channel_max` | < 100 | NORMAL | 非 spike 层 |
| | 100 ~ 5000 | SPIKE | 典型 massive activation |
| | > 10000 | SEVERE | 极端放大，检查 weight decay |
| `channel_max_ratio` | < 10 | NORMAL | 各 channel 量级接近 |
| | 10 ~ 1000 | SPIKE | 存在少数异常 channel |
| | > 1000 | SEVERE | 极端通道不平衡 |
| `massive_act_channel_count` | 0 ~ 5 | NORMAL | 典型 spike pattern |
| | > 10 | WARNING | 异常 channel 过多 |
| `post_norm_sparsity` | < 0.5 | NORMAL | 归一化后信息丰富 |
| | > 0.8 | HIGH | 高度稀疏，implicit parameter 效应 |
| `post_norm_cosine` | < 0.5 | DIVERSE | token 表示多样 |
| | > 0.9 | COLLAPSED | 近常量向量，sink 前提条件满足 |

---

## 与其他 Monitor 的交叉诊断

| 组合信号 | 诊断 |
|----------|------|
| `channel_max` 暴涨 + `qk_stats/sink` 不变 | Spike 还没传导到 attention（可能在 step-up 前） |
| `channel_max` 暴涨 + `qk_stats/sink` 上升 | 经典 spike→sink 路径激活 |
| `post_norm_sparsity` 高 + `qk_stats/entropy_min` 低 | 存在 dormant sink heads |
| `channel_max` 高 + `moe_health/router_entropy` 低 | 模型在用 spike 走捷径 |
| `topk_channel_norm` 平稳 + `qk_stats/sink_head_ratio` 上升 | Sink 在没有更多 spike 的情况下增长（替代策略） |

---

## 性能说明

### 计算开销
- **Pre-norm 指标**（channel_max 等）：一次 `abs().max(dim=0)`，O(S×H)，几乎零开销
- **Post-norm 指标**：需要额外一次 RMSNorm forward（无梯度），开销约等于一个 norm 层
- **Cosine stability**：采样 256 对，O(256×H)，可忽略

### 内存开销
- 无额外 tensor 持久化（每层 hook 内计算后立即写入 training_logs）
- 不保存激活值，不影响梯度计算

### 推荐配置

```python
# 全量监控（< 32 层的模型）
setup_massive_activation_monitor(model, monitor_interval=10)

# 采样监控（大模型，如 64+ 层）— 只看首、中、尾层
setup_massive_activation_monitor(
    model,
    sample_layers=[0, 1, 2, 3, 16, 30, 31],  # step-up 区 + 中间 + step-down 区
    monitor_interval=10,
)
```

---

## 使用方式

### 基本用法

```python
from internal_medicine import setup_internal_medicine

monitor_dict = {}
model = setup_internal_medicine(
    model,
    monitors=['massive_act'],      # 或 'all' 启用全部
    monitor_dict=monitor_dict,
    monitor_interval=10,
)
```

### 读取指标

```python
from internal_medicine import training_logs

# 获取所有 spike 指标
spike_metrics = training_logs.get_latest(prefix='massive_act')

# 查看特定层
layer_4_max = training_logs.get_latest(prefix='massive_act/layer_4')

# 格式化打印
training_logs.print_metrics(prefix='massive_act')
```

### 定位 Step-Up/Step-Down Blocks

```python
# 在全部层上运行一次，检查 channel_max 的 layer profile
spike_metrics = training_logs.get_latest(prefix='massive_act')

# 找到 channel_max 突增的层 = step-up block
# 找到 channel_max 突降的层 = step-down block
for key, val in sorted(spike_metrics.items()):
    if 'channel_max' in key and 'ratio' not in key and 'global' not in key:
        print(f"{key}: {val:.1f}")
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `log_per_layer` | `True` | 记录每层指标 |
| `log_global` | `True` | 记录全局聚合指标 |
| `monitor_interval` | `1` | 监控间隔 (每 N 步) |
| `verbose` | `False` | 打印调试信息 |
| `spike_threshold_multiplier` | `100.0` | spike channel 判定阈值 = median × 此值 |
| `topk_channels` | `3` | Top-K 通道数（对应论文 Figure 1） |
| `sparsity_epsilon` | `0.01` | post-norm sparsity 判定阈值 |
| `cosine_sample_pairs` | `256` | cosine stability 的采样对数 |
| `sample_layers` | `None` | 要监控的层索引列表，None=全部 |
