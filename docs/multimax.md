# MultiMax Monitor

**LM head SegLU（MultiMax）输出分布监控**，回答两件事：调制有没有真的在学，以及它把输出分布推向了「稀疏」还是「多峰」。

指标定义来自：

> Zhou, Y., Fritz, M., & Keuper, M. (2024).
> *MultiMax: Sparse and Multi-Modal Attention Learning.* ICML 2024, arXiv:2406.01189.
> Definition 3.2（Multi-Modality）与 Definition 3.3（Sparsity）。

---

## 背景

PaddleFleet 在 `multimax_modules: [lm_head]` 时给 LM head 加两个 `[4]` 可学参数
`multimax_ranges`（阈值 b/d）与 `multimax_ts`（温度 t_b/t_d），在 softmax 前对 logits 做分段调制：

```
SegLU(x) = x + t₀·relu(b₀−x) + t₁·relu(x−d₁) + t₂·relu(b₂−x)² + t₃·relu(x−d₃)²
```

两者都零初始化，所以第 0 步 SegLU 是恒等映射。论文的结论是 SoftMax 的温度调参在
稀疏性与多峰性之间存在权衡（Proposition 1），而 MultiMax 能同时改善两者
（Proposition 2）——所以这两个指标必须**一起看**，单看一个没有意义。

---

## 指标

设 `x` 为调制后的 logits，`φ(x) = softmax(x)`，`ε` 为「这一项是否相关」的阈值。

### 1. Multi-Modality（Def 3.2）

```
M(x) = 1 − (1/N) · Σ_{ε < xₙ < x_max} (φ(x)_max − φ(x)ₙ)
```

N 是满足 `ε < xₙ < x_max` 的项数（不含最大项）。相关项的概率越接近最大项，M 越接近 1，
即分布越多峰。**上升 = 更多峰**。

只有 N = 0 的行（除最大项外没有任何相关项）被排除在均值外，而不是记成 1.0——
否则一个完全单峰的分布会被打成「最多峰」。

### 2. Sparsity（Def 3.3）

```
S(x) = (1/L) · Σ_{x_l < ε} exp((s − φ(x)_l)/s − 1)
```

L 是低于阈值的项数，`s ∈ [0,1]` 是**参考概率**。指数项里 `(s−p)/s − 1 = −p/s`，所以

```
S(x) = (1/L) · Σ_{x_l < ε} exp(−p_l / s)
```

论文只要求 `s` 是「任意合理的参考值」，并明确 `S` 是归一化到 `[0,1]` 的平滑阶跃近似、
**越大越稀疏**。这一点只有在 `s` **与被打分的分布无关**时才成立。

> **不要取 `s = φ(x)_min`。** 那样 `p_l / s ≥ 1` 恒成立，每一项被钉在 `e⁻¹` 以下，
> 而且 `L` 一旦是词表量级，`S → e⁻¹/L`（LM head 上就是 1e-6 量级）——此时它衡量的是
> 「有几项贴在最小值上」，几乎只反映 `1/L`，而且因为参考值随分布漂移，跨 step 不可比。

默认 `s = 1/V`（均匀分布概率），与默认 ε 的语义一致（「概率超过均匀分布才算相关」）：
无关项必然满足 `p_l < s`，各项落在 `(e⁻¹, 1)`，序列跨 step 可比。`sparsity_ref` 可覆盖，
例如取论文举例的「**未调制** temperature-1 softmax 的最小概率」。

比值按 `exp(x_l − log_z − log s)` 在对数空间求值，因此在 LM head 的词表规模下
（fp32 里 `φ_min` 会下溢成 0）不会有下溢问题。

阈值是**严格**不等号（`x_l < ε`），和论文一致：正好落在 ε 上的项既不算相关也不算无关。
因此均匀分布那种「每一项都恰好等于阈值」的行会得到 L = 0，被排除在均值外，而不是被判成
「最稀疏」。`sparse_count` 长期为 0 说明 ε 选得太低，这一列的读数没有意义。

### 3. ε 的取法（唯一的实现判断）

论文把 ε 留作「任何合理阈值」。LM head 的 logit 尺度在训练中会漂移，固定 logit 阈值没有
可比性，所以默认在**概率空间**给阈值再映射回 logit：

```
ε = log(prob_eps) + logsumexp(x),     prob_eps 默认 1/V
```

softmax 单调，所以定义不变；`prob_eps = 1/V` 的含义是「概率超过均匀分布才算相关」。
要复现论文的固定阈值口径，传 `logit_eps=<value>` 覆盖。

### 4. 熵与 top-k

| 指标 | 公式 | 含义 |
|------|------|------|
| `entropy` | `H = logsumexp(x) − E_p[x]` | 预测熵（nats），下降 = 越确定 |
| `entropy_norm` | `H / log V` | 归一化到 `[0,1]`，跨词表可比 |
| `top1_prob` | `φ(x)_max` | 最大概率 |
| `top10_prob` | `Σ top-10 φ(x)` | 前 10 项的概率质量；与 `top1_prob` 的差 = 第 2~10 名分掉多少 |
| `relevant_count` | `N` | Def 3.2 的样本量，用来核对 ε 是否选得合理 |
| `sparse_count` | `L` | Def 3.3 的样本量，同上 |
| `range_0..3` / `t_0..3` | `multimax_ranges` / `multimax_ts` 的四个分量 | 全为 0 = SegLU 还是恒等，即 multimax 没开始学 |

全部键都落在 `multimax/global_{metric}`：LM head 不是逐层结构，没有 layer 维度。

开 MTP 时上游有两个 head（`GPTMainLMHead` 和 `GPTMTPLMHead`），各自持有**独立**的
`multimax_ranges` / `multimax_ts`，所以 MTP head 走单独的命名空间
`multimax/global_mtp_{metric}`——两个 head 的 SegLU 参数如果学歪了/分叉了能直接看出来，
而不是被平均掉。只有一个 head 时不会声明任何 `mtp_` 键。

---

## 实现要点

- **hook 点**：带 `use_multimax_lmhead=True` 的 LM head 的 forward post hook。没有这个属性的
  rank（非 head stage、或没开 multimax）直接 no-op，不声明任何键。
- **两条路径都支持**。`fused_linear_ce_loss_chunk = 0` 时 head 直接返回调制后的 logits；
  `> 0` 时它返回 `(hidden, weight, bias, ranges, ts)` 且完整 logits 从不物化，此时 monitor 用
  head 自己的参数把采样到的那一小块 logits 重算一遍（`multimax_metrics.apply_seglu` 是
  `lm_head.py:SegLU` 的镜像，测试里逐元素对齐）。MTP 的 `[main, mtp...]` 列表和
  `GPTMainLMHead` 的 `{"logits": ...}` 字典都会被拆开取 main head；`GPTMTPLMHead` 返回的是
  透传的 `dict_args`，预测在 `mtp_logits` 里，这个 head 只读 `mtp_logits`，不回退到同一个
  字典里可能残留的 main head `logits`（否则 main 会被算两次）。
- **采样**：`sample_tokens`（默认 256）按定长步幅取 token，各 TP rank 取到的是同一批 token。
- **唯一的 hook 内集合通信**：词表切分（vocab-parallel TP）下每个 rank 只有 `V/tp` 列，而所有
  指标都需要全局 softmax 归一化，所以对采样后的 `[sample_tokens, V/tp]` 做一次
  `all_gather` 拼回全词表。代价被 `sample_tokens` 限住，且只在被监控的 step 触发。
  这符合 `monitor-hook-perf-rules` 里「正确性必需时保留最小规模的集合通信」那条例外。

## 配置

```python
setup_monitors(model, monitors=["multimax"], monitor_interval=100,
               multimax={"sample_tokens": 256, "topk": 10, "prob_eps": None})
```

- `sample_tokens`（默认 256）：参与统计的 token 数。TP all_gather 的大小与之成正比。
- `topk`（默认 10）：概率质量的 k，会被词表宽度截断，键名跟着变（`top{k}_prob`）。
- `prob_eps`（默认 `None` → `1/V`）/ `logit_eps`（默认 `None`）：ε 的两种给法，后者优先。

## 判读

- `range_*` / `t_*` 恒为 0：SegLU 没在学，先查 `[MULTIMAX-LMHEAD-APPLIED]` 日志有没有出现。
- M 与 S **同时**上升：这是论文期望的 Pareto 改善方向。
- 一个升一个降：退化成了温度调参的权衡，说明调制没带来额外收益。
- `entropy` 与 `top1_prob` 反向变化属正常；两者同时下降说明概率质量从 top-1 流向了 2~10 名，
  配合 `top10_prob` 一起看。
