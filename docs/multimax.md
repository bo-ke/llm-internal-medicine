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

#### 1b. `multi_modality_top{k}`：把 ε 取成第 k 大项

Def 3.2 里的 ε 是「任何合理的相关性阈值」。取 `ε = 第 k 大的 logit` 时，相关集正好是
**top-k 去掉最大项**，于是式子退化成一句话：

```
M_topk(x) = 1 − (φ_max − mean(φ_2..φ_k))
```

即「**top-1 与其余 top-k 项的平均差距**」，`= 1` 表示前 k 项完全等概率。

**为什么必须同时报这一条。** 默认 `ε` 对应「概率超过均匀分布 1/V」，在 LM head 上相关集有
上千项（实测 `relevant_count` 均值 ≈ 275、p95 ≈ 1000），这些 `φₙ` 比 `φ_max` 小两三个数量级，
于是 `(1/N)Σ(φ_max − φₙ) ≈ φ_max`，`M ≈ 1 − top1_prob`——实测三个 checkpoint 上
`M − (1 − top1_prob)` 恒定在 `+0.023 ~ +0.026`（最大偏离 0.027），也就是说**原式那条曲线
不含 `top1_prob` 之外的任何信息**。`top{k}` 变体的相关集只有 k−1 项、量级与 `φ_max` 可比，
才真正反映头部形状。两条都保留：一条对齐论文原式，一条可读。

复用 `topk_prob` 已经算出的 `paddle.topk` 结果，没有额外的排序开销。

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

**参考概率 `s` 的取法**（`sparsity_ref` > `sparsity_ref_mode`，优先级由高到低）。`s` 只要求与被
打分的行无关，但它决定平滑阶跃 `exp(−p_l/s)` 落在哪一段，两个极端都不可用：

| 取法 | `s` 的量级 | 实测 `S`（44800，V≈201k） | 问题 |
|------|-----------|--------------------------|------|
| `uniform`（`s = 1/V`） | 5e-6 | ≈ 0.98 | 无关项本来就都低于 `s`，整段饱和到 1，没有分辨力 |
| `min`（论文举例：baseline softmax 最小概率） | ~1e-11 | 均值 0.051 / **p50 0.013** / p95 0.265 | 比 `1/V` 小几个数量级，几乎所有项下溢到 0，信号只剩最上面几个百分点 |
| **`geomean`（默认）** | 尾部中段 | 合成数据上 ≈ 0.585 | —— |

`geomean` = **未经 SegLU 调制**的同一行 logits 过 temperature-1 softmax 后，**其自身无关项**
概率的几何平均，即 `log s = mean_{l ∈ ref 的无关集}(x_ref,l − logsumexp(x_ref))`。
落在尾部中段，所以尾部被压到 baseline 以下时 `S` 会明确上升，这才是「越大越稀疏」的可读区间。

注意无关集是按**参考分布自己的 ε** 选的，不是被打分那一行的 ε，否则 `s` 又会重新依赖调制结果。
baseline 本身没有尾部（`n_tail = 0`）的行退化为 `1/V`。

`ref_logits` 由 monitor 在 hook 里用 head 的权重把 baseline 重算一遍得到；拿不到（上游只交出
调制后的结果）时退化为 `uniform`。`sparsity_ref=<常数>` 优先于所有 mode，用来固定一个跨实验
可比的参考。

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

### 5. `*_p50` / `*_p95` / `*_p98`：同一 step 内 token 的分布

上面每个按 token 求均值的指标（`entropy`、`entropy_norm`、`top1_prob`、`top10_prob`、
`multi_modality`、`sparsity`、`relevant_count`、`sparse_count`）都额外报三个分位数，
取的是这一批采样 token 的**最近秩分位**（nearest-rank，秩 `ceil(q·n)`）：

```
p50 = 典型 token,   p95 / p98 = 尾部落在哪
```

**为什么是分位数而不是 ±1σ。** 这些量要么有下界（熵 ≥ 0、计数 ≥ 0），要么落在 `[0,1]`，
而且都右偏：多数 token 近乎确定、少数 token 很不确定。对称的 `mean ± σ` 在这种分布上
下沿会跑出定义域（熵的下沿变成负数），而且 σ 会被尾部拉大到超过均值本身，读起来像是
「指标不稳」，其实只是分布偏斜。分位数不假设对称，直接回答「典型值在哪、尾巴在哪」。

掩码与均值完全一致：无效行（M 的 N = 0、S 的 L = 0）在排序前被推到 `+inf`，
分位下标由有效行数（张量）算出，所以 `multi_modality_p50` 只看 N > 0 的行、
`sparsity_p50` 只看 L > 0 的行。整个过程没有 python 层过滤，因此不会在热路径上触发 D2H 同步。

viewer 把 `p50~p95` 画成均值曲线周围的阴影，`p98` 只在 hover 里给，避免图上线条过多。

**一个聚合上的注意点：分位数不是线性的**，不能像均值那样跨 hook 调用平均。
`record_mean` 对每次调用的分位数求平均，只有在「每个记录 step 只有一次调用」
（`gradient_accumulation_steps=1`）时才等于真分位数；开了梯度累积就是近似。
真要在累积下保持精确，应该调大 `sample_tokens`，而不是依赖这个平均。

`rows` 是常数，没有分位数。

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
- `topk`（默认 10）：概率质量的 k，会被词表宽度截断，键名跟着变（`top{k}_prob`、
  `multi_modality_top{k}`）。
- `prob_eps`（默认 `None` → `1/V`）/ `logit_eps`（默认 `None`）：ε 的两种给法，后者优先。
- `sparsity_ref`（默认 `None`）：Def 3.3 的参考概率 `s` 取常数，优先于 mode。
- `sparsity_ref_mode`（默认 `"geomean"`，可选 `"min"` / `"uniform"`）：不给常数时 `s` 怎么从
  未调制 logits 推出来（见「Sparsity」一节的对比表）。

## 判读

- `range_*` / `t_*` 恒为 0：SegLU 没在学，先查 `[MULTIMAX-LMHEAD-APPLIED]` 日志有没有出现。
- M 与 S **同时**上升：这是论文期望的 Pareto 改善方向。
- 一个升一个降：退化成了温度调参的权衡，说明调制没带来额外收益。
- `entropy` 与 `top1_prob` 反向变化属正常；两者同时下降说明概率质量从 top-1 流向了 2~10 名，
  配合 `top10_prob` 一起看。
