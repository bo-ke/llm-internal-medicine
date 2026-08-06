# mHC Health Monitor

监控 mHC (Manifold-Constrained Hyper-Connections) 层的健康状况。mHC 用如下传播替换普通残差：

```
x_{l+1} = H_res @ x_l + H_post^T · F(H_pre @ x_l)
```

每个 token、每个 hyper-connection 模块学习三个映射（`n = num_residual_streams`）：

- `h_pre`  `[s, b, n]`     — 多流聚合门（sigmoid）
- `h_post` `[s, b, n]`     — 多流扩展门（2·sigmoid）
- `h_res`  `[s, b, n, n]`  — Sinkhorn 双随机（doubly-stochastic）残差混合矩阵

每个 `HyperConnectionTransformerLayer` 含两个 hyper-connection 模块：`self_attention_hyper_connection`
（`attn`）与 `mlp_hyper_connection`（`mlp`），forward 中 attn 先于 mlp 执行。

---

## 开关与 no-op 保证

通过在 `internal_medicine_monitors` 中加入 `mhc_health`（或 `all`）启用。启用它在**任何**模型上都是安全的——
以下任一情况下它彻底 no-op（不 wrap 任何模块、不声明/产生任何指标、不抛异常）：

1. **mHC 类无法 import**：`setup_mhc_monitor` 在 import 失败时将类名绑定为 `None`，直接返回 `model`。
2. **模型未使用 mHC 层**：discovery 用 `isinstance(layer, HyperConnectionTransformerLayer)` 与
   `isinstance(mod, HyperConnectionModule)` 精确匹配（不做 duck-typing），普通 `TransformerLayer` 或
   `IdentityOp` 占位符都不会被匹配。

```yaml
internal_medicine_monitors:
    mhc_health: true
```

---

## 采集方式：wrap `compute_mappings`（非重算）

`h_pre` 不在 `HyperConnectionModule.forward` 的返回中，普通 forward hook 看不到它。因此本 monitor **包裹
（wrap）每个 mHC 模块的 `compute_mappings` 绑定方法**，直接捕获其真实返回的 `(h_pre, h_post, h_res)` —— 不重算。

- `compute_mappings` 是普通 Python 方法（仅 `@nvtx_decorator`，非 `@torch.compile`），在 `_forward_normal` 中以
  `self.compute_mappings(...)` 调用，且**不被 checkpoint**，因此在 grad-enabled 的正向中恰好执行一次；实例属性
  wrapper 干净地遮蔽类方法。
- 整个捕获逻辑受 `_should_monitor()` 门控（含 grad 门），非监控步只是 `orig(x)` + 一次布尔判断。
- `remove_hooks()` 恢复原始方法并清空所有状态。

### VRAM 安全（无泄漏）

wrapper 不保留任何跨调用状态，只有固定的 0 维累加器。规则：

- 捕获后立即对 `h_pre/h_post/h_res` `.detach()`，并在 `no_grad()` 下做全部指标计算——否则一个仍带梯度的
  张量会通过反向把整层 autograd graph 钉住（大泄漏）。
- wrapper 原样返回 `out`，除 0 维标量外不保留任何对它/其视图的引用。
- `remove_hooks()` 恢复原始方法，不残留模块引用。


热路径纪律见 `.claude/skills/monitor-hook-perf-rules`：hook 内无 D2H 同步、无集合通信，schema 在 `allocate_buffers`
前声明。TP 不沿 `n` 切分映射，故无需 hook 内通信；跨 rank 归约在 flush 时由 `gather_and_aggregate` 完成（mean）。

---

## 监控指标

每个 hc 模块产出 9 个指标（本节均为 paddlefleet 后端；megatron 后端未随本次改动调整，仍是 8 个），指标名以
`attn_` / `mlp_` 前缀区分，除 `branch_residual_share_max` 取极值外全部按 token/batch 求均值
（并在 flush 时对 microbatch/rank 求均值）。日志键形如 `mhc_health/layer_{i}/{c}_{name}`，`{c}` ∈ `{attn, mlp}`；
对应的 `mhc_health/global_{c}_{name}` 由逐层累加器在 flush 时自动派生。

### 门控统计（h_pre / h_post）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_pre_mean`  | `mean(h_pre)`  | 聚合门均值 |
| `{c}_h_pre_std`   | `std(h_pre)`   | 聚合门离散度 |
| `{c}_h_post_mean` | `mean(h_post)` | 扩展门均值 |
| `{c}_h_post_std`  | `std(h_post)`  | 扩展门离散度 |

### 结构指标（paddlefleet 独有）

`h_post_mean` / `h_post_std` 把 token 轴和 stream 轴一起池化，因此「n 个流一起变弱」与「n-1 个流死掉、
剩一个扛全部」在均值上不可分；`branch_residual_share` 则回答 `h_post_mean` 回答不了的问题：门变小可能被
`F(·)` 变大抵消，只有两项的相对大小能说明该层是否还在写入。

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_post_stream_concentration` | `mean_t( max_i h_post[t,i] / mean_i h_post[t,i] )` | 流间集中度，区间 `[1, n]`。1 = 各流等权；趋 `n` = 质量压到单流，该层退化成普通单流残差，`num_residual_streams` 预算未被使用 |
| `{c}_h_post_token_std` | `std_t( mean_i h_post[t,i] )` | 门对输入的敏感度（`h = r · proj · α + bias`）。→0 = 门不再区分 token，等价于常数标量 |
| `{c}_branch_residual_share` | `mean_t( b / (b + r) )`，`b = ‖H_postᵀ F(·)‖_F`、`r = ‖H_resᵀ x_l‖_F` | 本层写入量在「写入 + 残差重混」中的占比，区间 `[0, 1]`。0.5 = 两项等量；趋 0 = 该层退化为纯残差搬运。|
| `{c}_branch_residual_share_max` | `max_t( b / (b + r) )` | 最坏 token 的写入占比；贴 1 表示存在残差近零的 token。唯一按极值跨 microbatch/层/rank 归约的指标 |

两个范数都是精确值，但不物化 `[tokens, n, C]` 中间量：分支项是外积（`‖h_post_t ⊗ xb_t‖_F = ‖h_post_t‖₂·‖xb_t‖₂`），
残差项用 `[tokens, n, n]` 的 Gram/mix 收缩得到，`n = 4` 时开销约为本层投影的 `C/n²` 分之一。
分支项在 dropout **之前**测量，`hidden_dropout_prob > 0` 的配置会略微高估分支幅度（此处预训练配置为 0）。

### amax-gain（Sinkhorn 收敛哨兵）

paper 定义的最坏情况增益：矩阵的**最大绝对行和**界定前向传播的最坏放大。对每个 token 的 `n×n` 矩阵计算，
再对 token 求均值：

```
amax_gain_fwd = mean_t( max_i | Σ_j  h_res_ij | )      # 行和（forward）
```

| 指标 | 诊断意义 |
|------|----------|
| `{c}_amax_gain_fwd` | Sinkhorn 是否收敛 / 有无 NaN。双随机 → 恒 ≈1.0。**只输出 `global_*`** |

单层 `h_res` 经 Sinkhorn 投影为双随机矩阵，所以这个值**按构造恒为 1**——实测 0.99999905，动态范围 0.0001%，
偏离量全部来自 Sinkhorn 的 `eps = 1e-6`。它只能当哨兵，不要拿来做诊断。也正因如此它走 `Probe.GLOBAL_ONLY`：
逐层值照常累加、`global_*` 从中派生，只是不写进日志——任一层漂移仍会把全局曲线带走，但看板上只有 2 条线
而不是 13 层 × 2 组件。

### 已下线：amax_gain_bwd 与复合映射

- `{c}_amax_gain_bwd`（最大绝对列和）：Sinkhorn 的最后一步是列归一（`hyper_connection.py` 的迭代循环），
  列和被钉在 1，实测四条 run、每一层都与 `amax_gain_fwd` **逐位相同**。
- `{c}_composite_amax_gain_fwd` / `_bwd`（`M_k = h_res_k @ M_{k-1}` 的行/列和）：双随机矩阵之积仍是双随机矩阵，
  复合增益同样恒为 1（实测 0.99999~0.999975）。更糟的是它依赖执行顺序：开 `recompute_granularity: full` 时
  采集实际发生在反向重放，累乘变成逆序——实测每层累乘因子数为 `1, 25.6, 23.6, …, 2.9`，而正序应为
  `1, 3, 5, …, 25.8`（关掉 recompute 的对照组给出了后者）。让它与顺序无关需要把每层 `h_res` 在整个 step 内
  驻留（43 层约 45MB 常驻显存），为一个常数付这个代价不值得。

要真正观测残差流放大，应监控 **Sinkhorn 收敛残差**（行和与 1 的偏差）或 `h_res` 的谱性质，两者都尚未实现。

---

## 与 microbatch / 激活重算的交互

- 每个梯度累积 microbatch 都会按序重新调用所有 hc 模块，每次调用独立记录一条样本，flush 时对 microbatch 求均值。
- 整个捕获受 `_should_monitor()` 门控；非监控步只是 `orig(x)` + 一次布尔判断。
- `_should_monitor()` 要求 grad enabled。这个判据的名字暗示"跳过重算"，**实际语义相反**：recompute 的真前向跑在
  `no_grad` 下，反向重放才开 grad，所以采集实际发生在反向重放。数值不受影响（重算确定性），但依赖执行顺序的指标
  会出错——这是复合映射被下线的原因之一。
- 这个判据留着是因为它顺带保证了"每个模块每 step 只取一条样本"：mHC 的 hyper-connection 在一次前向里会被进入
  两次（`high_precision_mhc` 打开时 AMP 关闭的 fp32 路径 + bf16 路径），两条都记会稀释所有均值（实测单层
  `amax_gain` 偏离量减半、门控 std 偏移 1~3%）。详见 README「已知限制：采集口径依赖 grad 判据」。
