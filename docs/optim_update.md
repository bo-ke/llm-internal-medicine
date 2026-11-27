# Optimizer Update Monitor (`optim/*`)

每 `monitor_interval` 步上报三个标量，与 grad norm / param norm 并排读：

| 指标 | 公式 | 含义 |
|---|---|---|
| `optim/update_rms` | `sqrt(mean((θ_new − θ_old)²))` | 本 step 实际施加的参数更新幅度 |
| `optim/param_rms` | `sqrt(mean(θ_new²))` | 更新后的参数尺度 |
| `optim/update_param_ratio` | `update_rms / param_rms` | trust ratio，健康量级 ~1e-3 |

**始终装上，没有开关，也不需要调用方多写一行**：它在 `_ALWAYS_ON_MONITORS` 里而不在
`_MONITOR_MAP` 里，因此 `setup_internal_medicine` / `setup_monitors` 无论 `monitors` 传什么
都会装上它，不需要在 `internal_medicine_monitors` 里点名。

**上报频率受 `internal_medicine_monitor_interval` 门控**，与模型侧 monitor 一致：这三个都是
慢变量（实测 50 步内 `param_rms` 一位不变），每 step 报一次只会刷屏，还要多付两次 collective。
门开在 **wrapper 里**而不只是上报时，所以未被监控的 step 上连和都不算 —— wrapper 只多一次布尔
判断，`step()` 里的 collective 也完全不发起。

指标 key 前缀是 `optim/`，而 `monitor_dict` 的 key 也是 `optim` —— 两者刻意一致，这样调用方
"遍历 `monitor_dict` key 当 prefix 打印 `training_logs`" 的既有循环就会自动带上这三个值，
TB 也走既有的 `aggregated_metrics` 写入。**训练脚本侧零改动。**

## 为什么补丁打在类上而不是实例上

优化器在模型侧 monitor 安装时（pre-wrap hook）**还不存在**。若要拿实例，就得在训练脚本里多注册
一个能看到 `ctx.optimizer` 的回调——那等于把这套逻辑摊到第二个仓库里去。因此改为直接包
`Float16OptimizerWithFloat16Params` 与 `DistributedOptimizer` 两个**类**上的
`_copy_main_params_to_model_params`（这两个是真正定义该方法的具体子类，共同基类
`MixedPrecisionOptimizer` 上没有），`self` 从调用里取，之后构造的任何优化器实例都被覆盖。
`remove_hooks` 把类属性还原。

## 不额外保存一份参数

`MixedPrecisionOptimizer.step_with_ready_grads`（`megatron/core/optimizer/optimizer.py:712`）
的顺序是：

```python
self.optimizer.step()                     # fp32 master: θ_old -> θ_new
self._copy_main_params_to_model_params()  # 把 θ_new 写回 bf16 model param
```

**这两行之间**，fp32 master 已经是 θ_new，而 bf16 model param 仍然是 θ_old —— 更新前的值
本来就还活着。因此本 monitor 只把 `_copy_main_params_to_model_params` 包一层，在里面读这一
对张量，**不需要在 step 前 clone 任何参数**（那会让显存翻倍）。

`token_dispatcher` 那套按实例打补丁的手法在这里复用；`remove_hooks` 时还原（类方法的情况
删实例属性，避免留下 bound-method 引用环）。

逐 shard 的和按 1 Mi 元素**分块**累加，因此 elementwise 临时张量恒为几 MB，与参数规模无关。

## bf16 去偏（必需，不是可选优化）

θ_old 是 bf16，本身带舍入误差；当更新量降到 bf16 分辨率附近时，原始差值会被这个误差主导。
实测（200k 元素，θ ~ N(0, 0.02²)）：

| lr | 真实 \|Δθ\| | 原始差值 | 偏差 | 去偏后 |
|---|---|---|---|---|
| 3e-4 | 3.00e-4 | 3.01e-4 | 1.01× | 1.00× |
| 1e-4 | 1.00e-4 | 1.05e-4 | 1.05× | 1.00× |
| 3e-5 | 3.00e-5 | 4.48e-5 | 1.49× | 1.00× |
| 1e-5 | 9.98e-6 | 3.46e-5 | 3.46× | 1.00× |
| 3e-6 | 3.00e-6 | 3.33e-5 | **11.10×** | 1.00× |

cosine decay 尾段 lr 正好落在 1e-5~1e-6，原始形式在那里完全不可用。

去偏方式：用 `θ_new − bf16(θ_new)` 估计 bf16 舍入方差（同一个张量，免费），在 mean-square
空间里减掉。全 lr 段回到 1.00×。`debias_low_precision=False` 可关掉（只用于测试对照）。

## 跨 rank 归约

主参数被 distributed optimizer 按 **DP+CP** 切分，再按 TP/PP 切分，与
`calc_params_l2_norm`（`megatron/bridge/training/utils/train_utils.py:285`）同一套语义：

| bucket | shard 维归约 | model 维归约 |
|---|---|---|
| dense | `dp_cp` | `mp` |
| expert (`allreduce=False`) | expert-DP | expert-TP/PP |

`(ss_update, ss_param, ss_bf16_noise, count)` 打成一个 fp64 4-vector，每个 bucket 两次
collective。RMS 对和是非线性的，所以**先归约再相除**。

count 用 fp64：总参数量 × shard 数远超 fp32 的 `2**24` 精确整数上限。

去重与 grad-norm 一致，且判据取在**原始 model param** 上（`allreduce` 这个 dense/expert
标记不在 `copy_optimizer_param_metadata` 拷到 shard view 的属性里）：

- `shared`（tied embedding，跨 PP stage 出现两次）→ 跳过
- TP **replicated** 参数只在 tp rank 0 计入；TP **sharded** 参数每个 rank 都计入

即使某个 bucket 本 rank 没有 shard，也照样发起两次 collective —— 一个 rank 可能没有 expert
shard 而 peer 有，跳过就会挂。

## 逐参数打点（`per_param`，默认关）

三个聚合标量是**按元素加权**的，所以它们回答不了"某个张量的范数动了没有"。实测的反例：
一个 l12 MoE 上 LN 权重只有 31.9K 个元素但 RMS ~2.5，而 managed 矩阵有 3.73B 个元素、
RMS ~0.017 —— 平方和里前者和后者量级相当，于是 managed 矩阵涨 3 倍聚合值几乎不动，
而 LN 动一下聚合值就跟着走。拿聚合值反推单个矩阵，会错一个量级。

`per_param=True` 额外给每个参数出一份同样的三件套（开了 `retraction` 再多一组）：

```
optim/update_rms/<name>          optim/param_rms/<name>          optim/update_param_ratio/<name>
```

`<name>` 是 `model.named_parameters()` 里的点分名。覆盖范围**与优化器无关**：只要是被监控
优化器持有、且模型能给出名字的参数就出曲线 —— Muon/MuonH/Skewon 管的矩阵、Adam 管的
embedding、LN 与 bias，一视同仁。叫不出名字的参数会被计数并 warning 一次，然后只进聚合值。

两点实现约束值得知道：

- **聚合桶要去重，逐参数不能去重。** 复制的参数（`shared` 的 word embedding、TP 复制的参数）
  被每个 rank 各算一次会歪掉世界均值，所以聚合桶排掉它们；而逐参数是按名字槽位把 `ss` 和
  `numel` 一起 SUM，复制 N 份时两者同倍放大，`sqrt(ss/numel)` 不变 —— 这正是 embedding
  能出曲线却不污染聚合值的原因。TP 分片的矩阵也靠这条把各分片汇成整矩阵范数，expert 同名
  跨 EP 汇聚同理。
- **retraction / radial_cos 的逐参数支路仍然只有 Skewon/MuonH 有**：它们排空的是内层优化器
  自己的 `_retract_pending` / `_radial_cos_pending`，别的优化器不产生这些。

开销：多出来的只是原本被聚合桶跳过的那批参数各算一次平方和（l12 上是 embedding 的 205M
元素加 67 个范数向量），且不新增 D2H、不新增 hook 期 collective —— 名字序的 all_gather 与
打包 SUM 都在 flush 时，规模随"有名参数数"增长。

## 未覆盖的部分

- **纯 fp32 参数**（`shard_fp32_groups`）就地更新，不存在"更新前的副本"，因此不计入
  `update_rms`。首次遇到时打一条 warning。bf16-mixed 训练下这类参数通常为空。
- **Megatron-FSDP** 路径的 main weights 由 `param_and_grad_buffer` 管理，未走本 monitor
  依赖的 group 结构；此时不上报（不会报错）。

## 怎么读

- `update_param_ratio` ~1e-3 是常见的健康量级；持续 ≫1e-2 说明单步动得太狠，
  配合 grad norm 与 lr 一起看。
- `update_rms` 应大致随 lr schedule 走。lr 在降但 update_rms 不降，通常意味着
  Adam 的二阶矩塌了（`v` 变小把 `1/√v` 放大）。
- `update_rms` 突然掉到 ~0 而 grad norm 正常 → 检查是否被 loss-scale 跳步
  （skipped iter）或 clip 压死。
- `param_rms` 单调上升 → weight decay 没有压住增长。
