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
另外包 `_compute_h`（拿 Sinkhorn 之前的 `h_res` logits）和 `fused_h_res_h_post_bda`（拿子层输出，算分支/残差占比）。

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

每个 hc 模块产出 14 个指标（本节均为 paddlefleet 后端；megatron 后端未随本次改动调整，仍是 8 个），指标名以
`attn_` / `mlp_` 前缀区分，除 `branch_residual_share_max`、`h_res_softmax_max`、
`composite_amax_gain_{fwd,bwd}_max` 取极大值与 `h_res_softmax_min` 取极小值外全部按 token/batch 求均值
（并在 flush 时对 microbatch/rank 求均值）。日志键形如 `mhc_health/layer_{i}/{c}_{name}`，
`{c}` ∈ `{attn, mlp}`；对应的 `mhc_health/global_{c}_{name}` 由逐层累加器在 flush 时自动派生。

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

paper 定义的最坏情况增益：算子的**最大绝对行和**界定前向传播的最坏放大。这里的算子不是 `h_res` 本身——
`apply_h_res` 与 fused cuTile kernel 都按 `out_i = Σ_j h_res_ji · x_j` 混流，即真正作用在流向量上的是 `h_resᵀ`
（见 `hyper_connection.py:466-471`、`fused_mhc_kernels.py:367-373`）。所以前向增益是 `h_res` 的**列**和、
反向增益是**行**和。对每个 token 的 `n×n` 矩阵计算，再对 token 求均值：

```
amax_gain_fwd = mean_t( max_i | Σ_j  h_res_ji | )      # h_res 的列和 = h_resᵀ 的行和（forward）
amax_gain_bwd = mean_t( max_j | Σ_i  h_res_ji | )      # h_res 的行和（backward）
```

| 指标 | 诊断意义 |
|------|----------|
| `{c}_amax_gain_fwd` | 单层前向最坏放大。列和被 Sinkhorn 最后一步归一钉死，是不变量守卫 |
| `{c}_amax_gain_bwd` | 单层反向最坏放大。承载 Sinkhorn 截断迭代的收敛残差 |

**哪个轴带信息**：我们的 `_sinkhorn_normalize` 最后一步是**列**归一，所以列和按构造恒为 1，行和才承载截断迭代的
残差（偏离量来自 `eps = 1e-6`，量级很小）。所以 `_bwd` 只能当哨兵、不要做诊断；`_fwd` 一旦偏离 1 反而更严重，
说明归一化本身出了问题（数值异常或实现改动）。

注意论文 Eq. (9) 最后一步是**行**归一，所以它 Fig 7 里会动的是 forward 那条，而我们会动的是 `_bwd`——
**两边曲线不可直接对比。**

### logits 饱和哨兵（迭代前的行内跨度）

`softmax` 在行内**平移不变**，所以决定 `h_res` 有多尖的唯一量是行内跨度 `max_j z_j − min_j z_j`。跨度直接决定
softmax 的最小输出：`min ≈ exp(−跨度)`，于是阈值由 dtype 给出，不需要拍：

- `exp(-87) ≈ 1.2e-38` —— fp32 最小正规数
- `exp(-103) ≈ 1.4e-45` —— fp32 最小 denormal，再往下就是 0

跨度超过 ~87，模型自己的 `softmax` 就开始下溢：行内相对比例被摧毁，紧随其后的 Sinkhorn 把残存比例按最多
`1/eps` 放大。

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_softmax_min` | `min_t min_{i,j} softmax(z)_{t,i,j}` | 最小混合权重。值域 `(0, 1/n]`，`1/n` = 行内均匀，趋 0 = 该行退化成 one-hot。按 **min** 归约 |
| `{c}_h_res_softmax_max` | `max_t max_{i,j} softmax(z)_{t,i,j}` | 单个最大混合权重。值域 `[1/n, 1]`，跨度到约 18 之后在 fp32 里精确等于 1.0 并保持不变，所以只能当"是否已经饱和"的开关看，不反映饱和深度。按 **max** 归约 |

### 复合映射（composite_amax_gain_{fwd,bwd}_max）

`apply_h_res` 的实际运算是 `mixed = h_resᵀ @ residual`，所以
**真正作用在流向量上的算子是 `h_resᵀ`**。从首层到第 k 层的复合算子是

```
A_k = h_res_kᵀ @ ⋯ @ h_res_0ᵀ
```

逐层记录它的最大绝对行和（`_fwd`，前向信号增益）与最大绝对列和（`_bwd`，反向梯度增益）。这条曲线对应论文
Figure 7(b)：单层指标看不出的拱形（中间层高、两端低）只有复合能显示。

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_composite_amax_gain_fwd_max` | `max_i \|Σ_j A_{k,i,j}\|` | 从首层到本层累计的前向最坏放大。逐层输出形成剖面曲线；global 按 **max** 归约给出峰值 |
| `{c}_composite_amax_gain_bwd_max` | `max_j \|Σ_i A_{k,i,j}\|` | 同上取列和 |

四个实现要点：

1. **累乘的是转置。** 直接对 `h_res` 累乘既不是前向链也不是反向链——`h_res_k ⋯ h_res_0` 与真实算子
   `A_k = (h_res_0 ⋯ h_res_k)ᵀ` 是不同的矩阵。快照里存的就是 `h_resᵀ`。单层的 `amax_gain_{fwd,bwd}` 同样按
   `h_resᵀ` 口径计算（只是没有显式转置，直接换了求和轴），所以两组指标的 `_fwd`/`_bwd` 语义一致、可以对着读。
2. **先对 token 求平均再累乘**（论文 Fig 7/8 同口径），所以每层每组件只存一个 `n×n` fp32 快照，而不是把 per-token
   `h_res` 在整个 step 内驻留。这是它当初被下线的第一条理由，现在不成立。
3. **累乘发生在 flush 时、按 `layer_idx` 排序**，不在 hook 里增量累乘。所以结果与 hook 触发顺序无关——而触发顺序
   在 recompute 下是反向重放的逆序，那正是它当初被下线的第二条理由。`_record_composite` 有对应的回归测试
   （故意逆序触发 hook，断言结果仍等于正序累乘）。
4. **MTP 层被排除。** MTP 层不在主干传播路径上，乘进链条没有物理意义。

作用域限制：复合只覆盖**本 rank 持有的层**。当前配置（`sharding: stage1`，无 PP）下每张卡都持有全部 43 层，所以
就是全网语义。真开 PP 后它会退化成 stage 内语义，而我们既检测不到也修不了——`get_pipeline_model_parallel_rank`
和 `get_pipeline_model_parallel_world_size` 在 PaddleFleet 里目前是 stub（无条件返回 0 / 1，
`parallel_state.py:229`、`:246`）。要支持 PP 需要先等这两个函数实现，再补一次 flush 时的 all-gather；届时
**collective 必须放在所有 per-layer `try/except` 之外**，否则单个 rank 吞异常会让整个 job 挂死而不是报错。

### 曾经下线、现已恢复

- `{c}_amax_gain_bwd`：曾因读数与 `amax_gain_fwd` 一致而下线。这个结果本身是预期的（`_fwd` 读的列和被 Sinkhorn
  最后一步钉死为 1），两条现在都保留：`_fwd` 当不变量守卫，`_bwd` 才是承载收敛残差的那条。
- `{c}_composite_amax_gain_{fwd,bwd}_max`：曾因两条理由下线，现在都不成立——per-token `h_res` 全 step 驻留的开销
  （改成先对 token 求平均后只剩一个 `n×n` 快照），以及依赖执行顺序（改成按 `layer_idx` 排序、flush 时累乘后无关）。



---

## 与 microbatch / 激活重算的交互

- 每个梯度累积 microbatch 都会按序重新调用所有 hc 模块，每次调用独立记录一条样本，flush 时对 microbatch 求均值。
- 整个捕获受 `_should_monitor()` 门控；非监控步只是 `orig(x)` + 一次布尔判断。
- `_should_monitor()` 要求 grad enabled。这个判据的名字暗示"跳过重算"，**实际语义相反**：recompute 的真前向跑在
  `no_grad` 下，反向重放才开 grad，所以采集实际发生在反向重放。数值不受影响（重算确定性），但依赖执行顺序的指标
  会出错——这曾经让复合映射下线，现在复合改成按 `layer_idx` 排序、在 flush 时累乘，不再受影响。
- 这个判据留着是因为它顺带保证了"每个模块每 step 只取一条样本"：mHC 的 hyper-connection 在一次前向里会被进入
  两次（`high_precision_mhc` 打开时 AMP 关闭的 fp32 路径 + bf16 路径），两条都记会稀释所有均值。详见 README
  「已知限制：采集口径依赖 grad 判据」。
