# Activation Dump (`act_dump`)

按 monitor 间隔把**残差流 hidden states** `[s, b, h]` 落盘，并**同时落盘产生这些激活的
输入 batch**（`input_ids` / `labels` / `position_ids` / `PackedSeqParams`），供离线结构
分析（有效秩 / 各向异性 / 聚类 / massive-outlier 检测——见 `spec_entropy_explorer.py`）。

与其它 monitor 不同，`act_dump` **不上报任何 `training_logs` 标量指标**：它在 forward
hook 里把 hidden 从 GPU 拷到 pinned CPU，在 step 末尾（cold path）写成 safetensors 文件。

也正因如此，它**不被 `monitors=['all']` / `all: true` 包含**：`all` 的语义是"打开全部
指标监控"，而本工具只写盘、体量又大（默认全量 hidden × 全部层，单 step 几十 GB）。要启用
必须显式点名 —— dict 形式写 `act_dump: {...}`，或 `monitors=['all', 'act_dump']`（与
`all` 并列点名视为显式 opt-in，不会被排除）。

## 为什么不复用 Megatron 的 activation offload

Megatron 的 `fine_grained_activation_offloading`（以及 TE 的 CPU offload）只在
**GPU ↔ pinned CPU-RAM** 之间搬运 saved-for-backward 张量，与 autograd 耦合，同一个
step 的反向里 reload + free（pool 复用），**没有落盘路径**，也没有"周期性快照"的概念。
复用它的 manager 会与其 pool-reuse / once-per-step 契约冲突。因此本工具只借用它的
**传输技巧**（side CUDA stream 上的 non-blocking D2H 拷入 pinned buffer，用 CUDA event
定序），把阻塞的 sync + 落盘推迟到 flush（cold path）——这遵守
`monitor-hook-perf-rules`：hot path 上无 `.item()/.cpu()` sync，无 collectives。

## 默认落盘完整 hidden（不采样）

`n_sample_tokens` 默认 `None`，即**落盘全部 `[s*b, h]` 行**。原因：token 子采样后
残差行无法再与 batch 里的 `input_ids` / `labels` / pack 边界对齐，"这一行是哪个 token"
这类分析直接做不了。

需要压磁盘时可以设 `n_sample_tokens=K` 退回采样，此时采的是 **K 个随机 token 位置**而
不是前 K 个：前导 token 受 BOS / attention-sink 的 massive activation 影响，量级与统计
特性异常，前 K 切片**不能代表**残差流。索引在 CPU 上按 `token_sample_seed + step_count`
生成（因此可复现，且同一 step 内所有采样层用**相同**位置，跨层可比），再搬到 device；
CPU 索引同时作为文件元数据，无需 hot-path 的 D2H sync。全量落盘时跳过 `index_select`
（gather 全部行没有意义），`token_index` 退化为恒等排列，metadata 里 `full_dump=True`。

## 输入 batch 落盘

`dump_input_batch`（默认 `True`）在**模型级 forward pre-hook** 上抓取喂进这次 forward
的 batch，每 step 单独写一个 `batch_*.safetensors`。有了它才能把某一行残差反查回它的
token id、它在 pack 里属于第几条序列、以及它是否被 loss mask 掉。

抓取的张量：`input_ids`、`labels`、`position_ids`、`loss_mask`（存在即抓），外加
`PackedSeqParams` 的 `cu_seqlens_q/kv`、`cu_seqlens_q/kv_padded`、`seq_idx`。

刻意**不抓** `attention_mask`：fused / causal 路径下它经常是 `None`，否则是 `[b,1,s,s]`
的巨大 bool 张量，落盘代价与信息量不成比例。`PackedSeqParams.cp_group` 是
`ProcessGroup`，不可序列化，也不抓。

几个约束：

- 只读 **kwargs**。Megatron-Bridge 是 `model(**forward_args)` 全关键字调用
  （`gpt_step.py`），所以真实路径一定在 kwargs 里。刻意不做 `args[0] -> input_ids` 的
  位置回退：那样在 forward 首参不是 input_ids 的模型上会把别的张量误标成 `input_ids`，
  写出一个看起来权威、实际错误的 batch 文件——**抓不到比抓错好**。
- 只保留**第一个 microbatch**，与 `first_microbatch_only` 的激活口径一致，保证 batch
  文件和它旁边的 hidden 文件来自同一次 forward。
- `min_channel_max_ratio` 把该 step 所有 hidden 都过滤掉时，**batch 也不写**，避免留下
  没有激活可对齐的孤儿文件。
- `max_seqlen_q/kv` 在本仓库的 `get_packed_seq_params`（`src/trainers/gpt_step_fix_cp.py`）
  里是 `batch["max_seqlen"].squeeze()` 产生的 **0-dim GPU tensor**，不是 int。对它做
  `str()` 会在 pre-hook 里触发 D2H sync（违反 perf-rules Rule 1），因此 tensor 一律走
  异步拷贝存成张量，只有真正的 python 标量才进 metadata。

## 激进采样默认值（控制磁盘占用）

默认只落盘：被监控的 step × 采样层 × 第一个 microbatch × 一个 DP/TP rank。所有开关可覆盖。

此外 `max_dump_steps`（默认 `20`）做**轮转（rotation）**：每次写盘后，只保留最近
`max_dump_steps` 个 `step_*` 目录，删除更旧的，使磁盘占用有上界，跑多久都不会撑爆。
设为 `null` 关闭轮转（无上限）。

`min_channel_max_ratio`（默认 `None`）在 flush 时按 massive-activation 门限过滤：
只有 `channel_max / channel_median >= min_channel_max_ratio` 的样本会落盘，其他丢弃。
ratio 无论如何都写进 metadata（`channel_max_ratio` 字段），便于事后核对。触发所在的
`step / layer_idx / global_rank / pp / tp / dp` 也已经记进 metadata。适合"只关心
ill-conditioned 状态"的场景，配合 `max_dump_steps` 后每步一层最多写一份、只在异常
step 才写，磁盘占用天然稀疏。

粗略磁盘占用（每 step）≈ `落盘 token 数 × hidden_size × dtype_size × 采样层数`，全量
落盘时"落盘 token 数"就是 `s*b`（**注意这比旧的 512 采样默认值大得多**，长序列 + 多层
下建议同时收紧 `sample_layers` / `max_dump_steps`，或显式设 `n_sample_tokens`）；配合
`max_dump_steps` 后总占用 ≈ 上式 × `max_dump_steps`。开启 `min_channel_max_ratio`
后按实际"异常步数"下压。

## dump_dir 路径校验（构造时就报错）

`dump_dir` 在 `__init__` 里经 `resolve_dump_dir()` 解析成**绝对路径**后校验并保存
（保存绝对路径也顺带免疫运行中途的 `os.chdir()`）。

**相对路径是推荐用法**（默认就是 `./outputs/act_dumps`）：它落在作业启动时所在的运行
目录下，通常就是仓库 checkout。被拒绝的只是"逃逸到根文件系统"的情况：

| 情况 | 结果 |
|---|---|
| `./outputs/act_dumps`（从仓库目录启动） | OK → `<repo>/outputs/act_dumps` |
| `/root/paddlejob/workspace/.../act_dumps` | OK |
| `/` | 拒绝：文件系统根 |
| `/tmp/...`、`/dev/shm/...`、`/var/tmp/...` 等 | 拒绝：小容量共享卷，写满会把整个节点带崩 |
| `/outputs/act_dumps`（= 从 `/` 启动时的 `./outputs/...`） | 拒绝：会在 `/` 下新建顶级目录，**不允许直接写根目录** |

最后一条的判据是**顶级祖先目录必须已存在**：`/outputs/act_dumps` 需要新建 `/outputs`
所以拒绝；`/root/paddlejob/...` 因为 `/root` 已存在所以通过。这条正好拦住"从 `/` 启动
作业 + 相对 dump_dir"这个真实事故路径。

另外可选设 `$INTERNAL_MEDICINE_DUMP_ROOT`，强制 dump 必须落在该前缀之下，适合把大容量
scratch 卷钉死。

选择在构造时抛异常而不是 warning：dump 路径错了通常要等磁盘写满、跑了几小时之后才会
被发现。

## 配置（必须用 dict 形式传 per-monitor kwargs）

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_monitors:
    act_dump:
        dump_dir: "./outputs/act_dumps"   # 相对=运行目录下；禁止解析到 / 或 /tmp
        which: "output"          # "output"=层输出残差 | "input"=层输入残差
        sample_layers: [0, 6, 11] # global 层索引；null=全部层
        n_sample_tokens: null     # null=全量落盘（默认）；设 K 则随机采 K 个位置
        dump_input_batch: true    # 同时落盘 input_ids / labels / PackedSeqParams
```

### 全部 kwargs

| 参数 | 默认 | 说明 |
|---|---|---|
| `dump_dir` | `"./outputs/act_dumps"` | 落盘根目录；构造时解析为绝对路径并校验（禁止 `/`、`/tmp` 等，禁止在 `/` 下新建顶级目录） |
| `which` | `"output"` | `"output"` 取层输出残差，`"input"` 取层输入残差 |
| `sample_layers` | `None` | global 层索引列表；`None`=全部层 |
| `n_sample_tokens` | `None` | `None`=全量落盘；设 K 则随机采 K 个 token 位置 |
| `token_sample_seed` | `0` | 基础种子；每 step 位置用 `seed + step_count`（仅采样时生效） |
| `first_microbatch_only` | `True` | 每 step 每层只落盘第一个 microbatch |
| `dump_input_batch` | `True` | 落盘输入 batch（ids / labels / position_ids / pack params） |
| `dump_dp_ranks` | `[0]` | 哪些 DP rank 落盘 |
| `dump_tp_ranks` | `[0]` | 哪些 TP rank 落盘 |
| `max_dumps_per_step` | `None` | 每 step 文件数安全上限 |
| `max_dump_steps` | `20` | 轮转：只保留最近 N 个 `step_*` 目录；`None`=不轮转 |
| `min_channel_max_ratio` | `None` | flush 门限：`channel_max / channel_median` 低于此值则不落盘（None=不过滤） |

## 文件布局与元数据

```
{dump_dir}/step_{step:07d}/rank{grank}_pp{pp}_tp{tp}_dp{dp}_layer{L}_{which}.safetensors
{dump_dir}/step_{step:07d}/rank{grank}_pp{pp}_tp{tp}_dp{dp}_batch.safetensors
```

激活文件含两个张量：
- `hidden`：`[s*b, hidden_size]`（全量落盘）或 `[n_sample_tokens, hidden_size]`（采样）
- `token_index`：int64，落盘行对应的扁平 token 位置（`s*b` 行主序）；全量时为恒等排列

safetensors metadata（全部为字符串）：`step`、`layer_idx`、`which`、`global_rank`、
`pp_rank`、`tp_rank`、`dp_rank`、`seq`、`batch`、`hidden_size`、`n_tokens`、
`n_sample_tokens`、`full_dump`、`token_sample_seed`、`src_dtype`、`sequence_parallel`、
`token_layout`（`seq_major_flattened_s_times_b`）、`channel_max_ratio`（本次采样
per-channel `max/median`，用于事后按 massive-activation 严重度筛选）。

batch 文件（每 step 一个）含 `input_ids` / `labels` / `position_ids` / `loss_mask`
（存在者）与 `packed_seq_params.*` 张量；metadata 有 `kind="input_batch"`、`step`、
四个 rank 字段、每个张量的 `*_shape` / `*_dtype`、`packed_seq_params_present`，以及
`PackedSeqParams` 里属于 python 标量的字段（`qkv_format` 等）。

## 并行语义

- **TP / Sequence Parallel**：每个 TP rank 只持有序列的一个分片；默认只 dump `tp_rank 0`
  的分片，`sequence_parallel=True` 时 metadata 记录该状态（跨 rank 拼全序列不在范围内）。
  注意 SP 下 hidden 是序列分片的，而 batch 文件里的 `input_ids` 是全长的——对齐时要自己
  按 TP 分片换算。
- **PP / VPP**：每个 PP stage dump 自己的层；三段式 setup（prepare 跨所有 chunk →
  allocate 一次 → 逐 chunk attach）处理 VPP 多 chunk。非首 stage 的 forward 收到的是
  上游 hidden 而不是 `input_ids`，batch 文件因此只在真正拿到这些 kwarg 的 stage 上有内容。
- **step 编号 off-by-one**：文件名里的 `step` 是 hook 触发时 `step()` **自增前**的
  `step_count`，与标量指标的归属方式一致。

## 离线加载

```python
from safetensors import safe_open

with safe_open(path, framework="pt") as f:
    meta = f.metadata()
    hidden = f.get_tensor("hidden")          # [T, hidden_size]
    token_index = f.get_tensor("token_index")

# 同目录的 batch 文件：把残差行反查回 token
with safe_open(batch_path, framework="pt") as f:
    input_ids = f.get_tensor("input_ids")    # [b, s]
    labels = f.get_tensor("labels")          # [b, s]
# hidden 行 i 对应扁平位置 token_index[i]，seq-major：seq = pos // b, batch = pos % b
```

或直接喂给探针工具：

```bash
python spec_entropy_explorer.py <file>.safetensors --key hidden
```
