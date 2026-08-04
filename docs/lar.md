# LAR Online Monitor (`lar`)

Log-Alignment Ratio (LAR) of a linear map, computed online with no SVD, as a
training-time generalization / overfitting diagnostic. See the outer spec at
`env_run/docs/lar_online_monitor.md` and arXiv:2605.28975 for theory.

## What it measures

For a `y = W x` linear on hidden inputs:

```
LAR = log_n( ||W X||_rms / (||W||_rms * ||X||_rms) ),   n = W.shape[1]  (input dim)
```

Hooked at two families of sites:

- **`lm_head`** — the output projection on the last PP stage. Hidden and logits
  are already in the forward pass; nothing recomputed. Works with tied
  input/output embeddings: under `share_embeddings_and_output_weights` Megatron
  builds `output_layer` with `skip_weight_param_allocation=True`, so
  `module.weight is None` and the tensor is only reachable via
  `shared_embedding_or_output_weight()`. The monitor resolves the weight through
  that accessor at attach time and closes over it, since reading `module.weight`
  inside the hook would yield `None` and silently drop the site.
- **`router_{L}`** — every MoE router. The router forward returns
  `(probs, routing_map)` (not raw gating logits), so the monitor recomputes
  `logits = F.linear(hidden.float(), weight.float())` locally in the hook.

Both sites use the SAME log base (`n = H`), so `lar/lm_head/lar` and
`lar/router_{L}/lar` are directly comparable.

## Loss-mask handling

A model-level `forward_pre_hook` captures `labels` (the standard Megatron GPT
kwarg). Tokens where `labels != label_ignore_index` (default `-100`) are used
for the `X` / logits sum-of-squares. The weight sum-of-squares is
token-independent and unaffected. When `labels` is not present on a forward
(eval / inference), the monitor falls back to using all tokens; `lar` still
emits.

There is **no `valid_frac` metric** reporting the mask's keep-rate. Computing it
would need a pre-mask token count, but the hooks index `x_flat[mask]` before
accumulating, so only post-mask counts reach flush time. Whether masking is live
is therefore not observable from the emitted metrics.

**Under `sequence_parallel=True`**, hidden entering routers is seq-sharded
across TP while `labels` are full-length. Router LAR falls back to unmasked
tokens on such runs (documented gotcha). `lm_head` is unaffected — it operates
post-SP-gather.

## Distributed reductions (`_flush_buffers`)

All-reduces are on `(sum_of_squares, count)` pairs — 2 fp64 scalars each,
one collective per stat. LAR is nonlinear in the sums, so per-rank averaging
of LARs is *incorrect* — pooling the sums globally is the point.

| site | ssW, nW | ssX, nX | ssZ, nZ |
|---|---|---|---|
| `lm_head` | TP-sum | DP-sum | TP-sum then DP-sum |
| `router_{L}` | (replicated) | DP-sum | DP-sum |

With TP=1/DP=1 no reduction fires. Cost is negligible either way.

## Metrics emitted per monitored step

Per site (`lm_head`, `router_0`, `router_1`, ...): `lar/{site}/lar` — one metric,
nothing else.

The three RMS norms (`rms_w`, `rms_x`, `rms_z`) are computed at flush time but
**not logged** — only their combination `lar` carries the diagnostic signal, and
raw activation/weight scale is already covered by `massive_act`
(`activation_rms`, `spectral_norm_max/min`).

`k = H ** (2*(1-lar))` (effective dimension) is **not logged** either: it is a
strictly monotone reparametrisation of `lar`, so it carries no information `lar`
does not already have, and its literal "effective rank" reading only holds for a
uniform weight spectrum — which does not hold at 200k-vocab scale (see the outer
spec's §10 measurements: full rank, one dominant direction, heavy tail). Derive
it offline from `lar` if you want the alternate scale.

Globals:
- `lar/global_lm_head_lar` (equals the single lm_head site)
- `lar/global_router_lar` (mean over routers, if any)

## Config

**Not part of `monitors=["all"]`** — LAR reads the `[T, vocab]` logits on every
monitored step and adds a hook plus a recomputed gating matmul per MoE router, so
it is opt-in rather than swept in with the cheap probes. Name it explicitly (this
also works alongside `all`, e.g. `monitors=['all', 'lar']`).

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_monitors:
  lar: true
  # or explicit kwargs:
  # lar:
  #   hook_lm_head: true
  #   hook_moe_router: true
  #   apply_loss_mask: true
  #   label_ignore_index: -100
```

### Kwargs

| 参数 | 默认 | 说明 |
|---|---|---|
| `hook_lm_head` | `True` | 挂 output_layer（末 PP stage 有效） |
| `hook_moe_router` | `True` | 挂每个 MoE router；非 MoE 模型自动 no-op |
| `apply_loss_mask` | `True` | 用 `labels != ignore_index` 过滤 X/Z tokens |
| `label_ignore_index` | `-100` | Megatron 默认 |

## Reading the signal

- `lar ≈ 0.5` at random init.
- Healthy training: `lar` drifts in `(0.5, 0.75)` and stabilizes.
- **Declining `lar`, decline accelerating ⇒ overfitting onset** (paper's core
  signal — the *slope* is the strongest indicator). Consider logging a smoothed
  `d(lar)/dstep` downstream.

## Perf notes

### Compute

- **`lm_head`: nothing is recomputed.** Both `hidden` and `logits` are already
  materialised by the forward pass; the hook only reduces them. Cost is three
  sum-of-squares reductions, and the weight one runs once per step (see below).
- **Routers: one small extra matmul each.** `TopKRouter.forward` returns
  `(probs, routing_map)`, not raw gating logits, so the hook recomputes
  `F.linear(hidden, weight)` — `T x H x E` FLOPs per MoE layer, negligible next
  to the expert GEMMs.
- Weight sum-of-squares runs once per step per site (constant within a step;
  guarded by the `w_done` flag), not once per microbatch — on a `[200019, 1024]`
  tied head that matters.
- All hook outputs are 0-dim GPU tensors: no `.item()/.cpu()` and no collectives
  on the hot path. Reductions happen at flush on `(sum_sq, count)` scalar pairs.

### Memory: why the sums avoid the obvious form

The lm_head `Z` is the **full logits tensor** — `[4096, 200019]` bf16 is 1.53 GiB
at seq 4096 / mbs 1, and it scales with `vocab x seq`. Two naive idioms each cost
multiples of that, so `_sum_of_squares` avoids both (measured on a
`[512, 200019]` bf16 slice = 195 MiB):

| form | extra peak | note |
|---|---|---|
| `Z.float().pow(2).sum()` | **+781 MiB** (~4x input) | fp32 cast **and** squares |
| `Z[mask].float().pow(2).sum()` | **+604 MiB** | plus the masked copy |
| `vector_norm(Z, dtype=fp32).square()` | **+0 MiB** | fp32 accumulate, reads in place |
| `vector_norm(Z, dim=-1, dtype=fp32).square()[mask].sum()` | **+0 MiB** | per-row norms are `[T]` |

Extrapolated to a real L12 microbatch the naive form would have peaked ~6.1 GiB
of transients per monitored step. Values agree to fp32 round-off (~1e-7 relative).

The masked path is why the loss mask is passed **into** `_accumulate` rather than
applied by the caller: `z_flat[mask]` would copy a vocab-wide tensor. The router
hook masks eagerly instead — its matmul is ours to pay for, so dropping invalid
rows first makes it cheaper, and the copy is `[T_valid, H]`, not vocab-wide.

Token **counts** are fp64: `valid_tokens x vocab` runs to ~1e8-1e9, past fp32's
`2**24` exact-integer limit, and the counts are all-reduced as fp64 anyway.

### Not done online

- SVD-based spectral metrics (spec §5, Appendix) are NOT computed online. Run
  offline at checkpoint boundaries if wanted.
