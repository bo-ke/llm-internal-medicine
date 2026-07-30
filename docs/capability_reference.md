# Internal Medicine 能力参考

本文档描述当前代码能够采集的指标、计算口径、启用条件和训练生命周期。指标用于定位训练内部机制变化，不应脱离 loss、稳定性和最终质量单独解释。

## 后端支持

| Monitor | Megatron | PaddleFleet | 适用结构 |
| --- | :---: | :---: | --- |
| `qk_stats` | Yes | Yes | Attention |
| `moe_health` | Yes | Yes | MoE |
| `massive_act` | Yes | Yes | Transformer |
| `ple_health` | Yes | No | Per-Layer Embedding |
| `mhc_health` | Yes | Yes | mHC；无 mHC 时自动 no-op |

## 通用键与状态

| 指标模板 | 计算 | 含义 |
| --- | --- | --- |
| `{monitor}/layer_{i}/{metric}` | 第 `i` 层指标 | 逐层定位 |
| `{monitor}/layer_{i}_mtp/{metric}` | MTP 层指标 | 区分主干与 MTP |
| `{monitor}/global_{metric}` | 按指标语义做 mean/max/min | 跨层摘要 |
| `monitor_status/{monitor}/compatible_sources_{min,max}` | 各 rank 发现的兼容源数量 | 安装覆盖 |
| `monitor_status/{monitor}/observations_{min,max}` | 采样 step 成功记录次数 | 实际运行覆盖 |
| `monitor_status/{monitor}/observed_layers_{min,max}` | 采样 step 覆盖层数 | 层覆盖 |
| `monitor_status/{monitor}/runtime_errors_max` | monitor 捕获的异常数 | 静默退化检测 |

PaddleFleet hot hook 只写 GPU 0 维累加器，`monitor.step()` 批量回传 CPU。QK 支持 PP/VPP、MTP、GQA、CP K-only gather、full/SWA 标签与 query-row subsampling。

## QK Stats

设单个 token/head 的向量为 $q,k,v$，attention logit 为

$$
l_{ij} = s q_i^\top k_j,\qquad p_{ij}=\operatorname{softmax}_j(l_{ij}).
$$

| 指标模板 | 计算 | 诊断含义 |
| --- | --- | --- |
| `{q,k,v}_norm_mean` | $\operatorname{mean}\lVert q/k/v\rVert_2$ | Q/K/V 平均尺度 |
| `{q,k,v}_norm_max` | $\max\lVert q/k/v\rVert_2$ | 局部尺度峰值 |
| `max`, `mean` | 有效 attention 区域内 logit 最大值、逐 query 均值 | logit 尺度 |
| `entropy_avg/min/max/std` | $-\sum_jp_{ij}\log p_{ij}$ 的均值、head 极值与离散度 | attention sharpness |
| `sink` | $\operatorname{mean}(p_{i0})$ | token-0 sink 强度 |
| `sink_head_ratio` | sink weight 超阈值的 head 比例 | sink 是否普遍 |
| `sink_head_max` | 最大 head sink weight | 最严重 sink |
| `sink_nonsink_gap` | sink-head 与其他 head 的均值差 | head 分化 |

SWA 指标只在真实窗口

$$
i-w_{\mathrm{left}}\le j\le i+w_{\mathrm{right}}
$$

内统计，并与 causal mask 共同生效。`softmax_scale` 使用模型实际值。

## Massive Activation

| 指标模板 | 计算 | 诊断含义 |
| --- | --- | --- |
| `channel_max/median/p95/p99` | 各 hidden channel 跨 token 绝对峰值的分布 | outlier 尺度与背景 |
| `channel_max_ratio` | 最大 channel peak / 中位数 | 少数通道异常程度 |
| `massive_act_channel_count` | channel peak 超相对阈值的数量 | massive channel 数 |
| `channel_count_gt_{threshold}` | channel peak 超绝对阈值的数量 | 固定尺度告警 |
| `topk_channel_norm` | 最大 K 个 channel peak 的 L2 norm | outlier 总强度 |
| `activation_rms` | $\sqrt{\operatorname{mean}(h^2)}$ | residual 整体尺度 |
| `post_norm_sparsity` | $\operatorname{mean}(|\hat h|<\epsilon)$ | norm 后稀疏性 |
| `post_norm_cosine` | 确定性 token pair cosine 均值 | 表示同质化 |

PaddleFleet 额外在五个机制位置输出 `{position}_{rms,abs_max,abs_p99,outlier_ratio}`：

`layer_input`, `attn_out`, `post_attn_residual`, `ffn_or_moe_out`, `post_ffn_residual`。

其中

$$
\text{outlier ratio}=\operatorname{mean}(|x|>10\operatorname{RMS}(x)).
$$

`attn_update_rms_ratio` 和 `ffn_update_rms_ratio` 分别衡量 attention、FFN/MoE 输出相对其 residual 输入的注入尺度。

Megatron 后端另提供 spectral gain、Lipschitz、logit-lens 和 hidden spectral entropy 等可选指标，详见 [Massive Activation](./massive_activation.md)。

## MoE Health

令 router probability 为 $p_{te}$，实际 top-k assignment 为 $a_{te}\in\{0,1\}$。

| 指标模板 | 计算 | 诊断含义 |
| --- | --- | --- |
| `router_input_rms/abs_max/abs_p99` | router 输入尺度与 tail | 区分输入异常与 router 放大 |
| `router_logit_mean/std/abs_max` | raw router logits 分布 | router 决策尺度 |
| `router_entropy` | token-wise $-\sum_ep_{te}\log p_{te}$ 的均值 | 单 token 路由不确定性 |
| `prob_entropy_norm` | `router_entropy / log(E)` | 跨 expert 数可比熵 |
| `score_sum_mean/min/max` | 每 token top-k probability 之和 | 选中质量 |
| `router_margin_mean/min/p10/p01` | 第 k 与第 k+1 分数之差 | 路由边界脆弱性 |
| `bias_affinity_jaccard` | bias 前后 expert 集合 Jaccard | correction bias 改写程度 |
| `expert_bias_mean/std/min/max` | correction bias 分布 | balance 控制状态 |

Hard assignment 与 gate mass 分开统计：

$$
c_e=\sum_t a_{te},\qquad m_e=\sum_ta_{te}p_{te}.
$$

`assignment_load_{cv,entropy_norm,kl_uniform,max_frac,min_frac,max_min_ratio}` 描述 $c_e$；`gate_mass_{...}` 描述 $m_e$。二者分别回答“专家接收多少 token”和“专家接收多少概率质量”。padding/无效 expert id 不计入。

专家权重指标为 `expert_norm_mean/std/min/max`, `shared_expert_norm`, `shared_routed_ratio`。这些指标需要宿主在 FP8 trainable 权重释放前调用 `collect_expert_norms()`；未接入该生命周期的宿主仍可采集 router、assignment 与 gate-mass 指标。

## PLE Health

仅 Megatron PLE 模型适用：

| 指标 | 计算 |
| --- | --- |
| `token_ple_norm`, `proj_ple_norm`, `per_layer_inputs_norm` | token/projection PLE 分支及合并输入 norm |
| `token_proj_cosine` | 两分支 cosine |
| `residual_ratio` | $\lVert output-input\rVert/\lVert input\rVert$ |
| `gate_activation_mean`, `gate_sparsity` | gate 激活强度与低激活比例 |

## mHC Health

对 attention 与 MLP 两个 hyper-connection 组件分别输出：

| 指标模板 | 计算 |
| --- | --- |
| `{attn,mlp}_h_pre_mean/std` | 聚合门统计 |
| `{attn,mlp}_h_post_mean/std` | 扩展门统计 |
| `{attn,mlp}_amax_gain_fwd/bwd` | 单层 residual 映射绝对行和/列和上界 |
| `{attn,mlp}_composite_amax_gain_fwd/bwd` | PP/VPP chunk 内复合映射增益 |

无 mHC 层时 monitor 自动 no-op；VPP chunk 分别维护复合映射，step 结束释放缓存。

## PaddleFleet 生命周期

```text
on_train_begin      -> setup monitors and hooks
forward             -> GPU-buffer metrics
on_step_end         -> flush GPU buffers and runtime status
on_log              -> distributed aggregation
```

未知 monitor 会告警并跳过；已知 monitor 初始化失败会终止 setup，避免训练在诊断能力缺失时静默继续。
