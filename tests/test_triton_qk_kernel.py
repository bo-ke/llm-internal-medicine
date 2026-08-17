"""Correctness tests for the packed (THD) QK-stats Triton kernel.

Compares ``qk_stats_packed_kernel`` (via ``compute_qk_stats_triton_packed``)
against a row-first PyTorch reference, and unit-tests the sync-free
``_cu_seqlens_to_token_arrays`` helper.

Mean uses row-first semantics (mean over valid rows of each row's mean logit)
on BOTH the Triton and PyTorch packed paths, so ``mean_global`` is asserted
directly — no need to skip it.
"""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    triton = importlib.import_module("triton")
except Exception as exc:
    raise unittest.SkipTest(f"torch or triton unavailable: {exc}") from exc

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required for Triton kernel tests")

_kernels = importlib.import_module("internal_medicine.backends.megatron.triton_kernels")
compute_qk_stats_triton_packed = _kernels.compute_qk_stats_triton_packed
compute_qk_stats_pytorch_packed = _kernels.compute_qk_stats_pytorch_packed
compute_qk_stats_packed = _kernels.compute_qk_stats_packed
compute_qk_stats_triton = _kernels.compute_qk_stats_triton
compute_qk_stats_pytorch = _kernels.compute_qk_stats_pytorch
_cu_seqlens_to_token_arrays = _kernels._cu_seqlens_to_token_arrays


def _make_packed_qk(seq_lengths, num_heads, head_dim, dtype=torch.bfloat16, device="cuda", seed=1):
    """Build packed [T, H, D] q/k and int32 cu_seqlens from a list of seq lengths."""
    torch.manual_seed(seed)
    total = sum(seq_lengths)
    q = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device)
    offsets = [0] + list(torch.tensor(seq_lengths).cumsum(0).tolist())
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q, k, cu_seqlens


def _to_htd(q, k):
    return q.permute(1, 0, 2).contiguous(), k.permute(1, 0, 2).contiguous()


class TritonPackedKernelCorrectnessTest(unittest.TestCase):
    ATOL = 1e-2
    RTOL = 1e-2

    def _assert_close(self, triton_out, ref_out, keys, tag=""):
        for key in keys:
            t = triton_out[key].float().item()
            r = ref_out[key].float().item()
            self.assertAlmostEqual(
                t,
                r,
                delta=self.ATOL + self.RTOL * abs(r),
                msg=f"{tag} {key}: triton={t:.6f} ref={r:.6f}",
            )

    _ALL = ("max_global", "mean_global", "entropy_global", "sink_global")

    def test_uniform_sequences_causal(self):
        q, k, cu = _make_packed_qk([64, 64, 64], num_heads=4, head_dim=64)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True),
            self._ALL,
            tag="3x64_causal",
        )

    def test_variable_length_sequences(self):
        q, k, cu = _make_packed_qk([32, 80, 48], num_heads=4, head_dim=64)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True),
            self._ALL,
            tag="var_len_causal",
        )

    def test_head_dim_128(self):
        q, k, cu = _make_packed_qk([48, 80], num_heads=4, head_dim=128)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True),
            self._ALL,
            tag="head_dim_128",
        )

    def test_single_sequence_causal(self):
        # A single packed sequence: packed-triton must match its row-first
        # PyTorch reference (this also exercises the no-boundary fast path).
        seq_len, num_heads, head_dim = 128, 4, 64
        torch.manual_seed(2)
        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True),
            self._ALL,
            tag="single_seq_causal",
        )

    def test_block_boundary_straddle(self):
        # A sequence boundary that falls inside a BLOCK_M=64 block: seq 1 ends
        # at token 50, seq 2 starts at 50 — both live in block [0:64].
        q, k, cu = _make_packed_qk([50, 78], num_heads=4, head_dim=64)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True),
            self._ALL,
            tag="boundary_straddle",
        )

    def test_non_causal(self):
        q, k, cu = _make_packed_qk([64, 64], num_heads=4, head_dim=64)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=False),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=False),
            self._ALL,
            tag="non_causal",
        )

    def test_non_causal_unequal_lengths(self):
        # Unequal sequence lengths, non-causal: row counts differ across rows so
        # row-first mean is a non-trivial average — a good stress for the reduce.
        q, k, cu = _make_packed_qk([32, 96], num_heads=4, head_dim=64)
        qh, kh = _to_htd(q, k)
        self._assert_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=False),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=False),
            self._ALL,
            tag="non_causal_unequal",
        )

    def test_padded_total_tokens(self):
        # total_tokens (tensor T dim) > cu_seqlens[-1]: padding tokens must not
        # contribute to any accumulator and must not cause OOB reads.
        seq_lengths = [48, 64]
        real_T = sum(seq_lengths)  # 112
        padded_T = 128  # next multiple of BLOCK_M=64
        torch.manual_seed(5)
        q = torch.randn(padded_T, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(padded_T, 4, 64, dtype=torch.bfloat16, device="cuda")
        cu = torch.tensor([0, 48, 112], dtype=torch.int32, device="cuda")

        # Reference uses only real tokens; triton sees the padded tensor.
        ref_q, ref_k = _to_htd(q[:real_T], k[:real_T])
        ref_out = compute_qk_stats_pytorch_packed(ref_q, ref_k, cu, causal=True)

        qh, kh = _to_htd(q, k)
        triton_out = compute_qk_stats_triton_packed(qh, kh, cu, causal=True)
        self._assert_close(triton_out, ref_out, self._ALL, tag="padded_T")

    def test_gqa_packed_entry_point(self):
        # GQA via the entry point: num_q_heads=8, num_k_heads=2 (repeat_interleave).
        q_thd, _, cu = _make_packed_qk([48, 80], num_heads=8, head_dim=64)
        torch.manual_seed(8)
        k_gqa = torch.randn(q_thd.shape[0], 2, 64, dtype=torch.bfloat16, device="cuda")
        out_triton = compute_qk_stats_packed(q_thd, k_gqa, cu, causal=True, use_triton=True)
        out_ref = compute_qk_stats_packed(q_thd, k_gqa, cu, causal=True, use_triton=False)
        self._assert_close(out_triton, out_ref, self._ALL, tag="gqa_packed")


class CuSeqlensTokenArraysTest(unittest.TestCase):
    """Unit tests for _cu_seqlens_to_token_arrays (GPU-only, no kernel launch)."""

    def test_no_padding_single_sequence(self):
        cu = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
        s, e = _cu_seqlens_to_token_arrays(cu, 64)
        self.assertEqual(s.shape[0], 64)
        self.assertTrue((s == 0).all(), "all seq_start should be 0")
        self.assertTrue((e == 64).all(), "all seq_end should be 64")

    def test_no_padding_multiple_sequences(self):
        cu = torch.tensor([0, 32, 80, 128], dtype=torch.int32, device="cuda")
        s, e = _cu_seqlens_to_token_arrays(cu, 128)
        self.assertEqual(s.shape[0], 128)
        self.assertTrue((s[:32] == 0).all())
        self.assertTrue((e[:32] == 32).all())
        self.assertTrue((s[32:80] == 32).all())
        self.assertTrue((e[32:80] == 80).all())
        self.assertTrue((s[80:] == 80).all())
        self.assertTrue((e[80:] == 128).all())
        pad_mask = (s == 0) & (e == 0)
        self.assertFalse(pad_mask.any(), "no tokens should be zeroed when total_tokens==cu[-1]")

    def test_padded_tokens_are_zeroed(self):
        # real_tokens=112, total_tokens=128 — 16 padding slots.
        cu = torch.tensor([0, 48, 112], dtype=torch.int32, device="cuda")
        s, e = _cu_seqlens_to_token_arrays(cu, 128)
        self.assertEqual(s.shape[0], 128)
        self.assertTrue((s[:48] == 0).all())
        self.assertTrue((e[:48] == 48).all())
        self.assertTrue((s[48:112] == 48).all())
        self.assertTrue((e[48:112] == 112).all())
        self.assertTrue((s[112:] == 0).all(), f"padding seq_start not zero: {s[112:].tolist()}")
        self.assertTrue((e[112:] == 0).all(), f"padding seq_end not zero: {e[112:].tolist()}")

    def test_zero_length_sequence_skipped(self):
        # cu = [0, 0, 64] — zero-length first sequence must not pollute seq 1.
        cu = torch.tensor([0, 0, 64], dtype=torch.int32, device="cuda")
        s, e = _cu_seqlens_to_token_arrays(cu, 64)
        self.assertEqual(s.shape[0], 64)
        self.assertTrue((s == 0).all(), f"unexpected seq_start: {s[:8].tolist()}")
        self.assertTrue((e == 64).all(), f"unexpected seq_end: {e[:8].tolist()}")


class DenseSinkFoldTest(unittest.TestCase):
    """Fold the per-head sink logit into the dense QK stats vs a materialized ref."""

    ATOL = 2e-2
    RTOL = 2e-2

    def _bhsd(self, seq_len=96, num_heads=4, head_dim=64, seed=3):
        torch.manual_seed(seed)
        q = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        return q, k

    def _reference(self, q, k, sink):
        _, num_heads, seq_len, head_dim = q.shape
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / head_dim**0.5
        logits = logits.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), 1), float("-inf")
        )
        sink_col = sink.reshape(1, num_heads, 1, 1).expand(1, num_heads, seq_len, 1).float()
        ext = torch.cat([logits, sink_col], dim=-1)
        probs = torch.softmax(ext, dim=-1)
        ent = -(probs * torch.log_softmax(ext, dim=-1))
        ent = torch.where(torch.isfinite(ext), ent, torch.zeros_like(ent))
        return ent.sum(-1).mean(-1).flatten(), probs[..., 0].mean(-1).flatten()

    def test_fold_matches_materialized_reference(self):
        q, k = self._bhsd()
        sink = torch.tensor([0.5, -1.0, 2.0, 0.0], device="cuda")
        out = compute_qk_stats_triton(q.contiguous(), k.contiguous(), causal=True, attn_sink=sink)
        ref_ent, ref_sink = self._reference(q, k, sink)
        for h in range(ref_ent.numel()):
            self.assertAlmostEqual(
                out["entropy_per_head"].float().flatten()[h].item(),
                ref_ent[h].item(),
                delta=self.ATOL + self.RTOL * abs(ref_ent[h].item()),
            )
            self.assertAlmostEqual(
                out["sink_per_head"].float().flatten()[h].item(),
                ref_sink[h].item(),
                delta=self.ATOL + self.RTOL * abs(ref_sink[h].item()),
            )

    def test_none_sink_matches_baseline(self):
        q, k = self._bhsd()
        out = compute_qk_stats_triton(q.contiguous(), k.contiguous(), causal=True, attn_sink=None)
        ref = compute_qk_stats_pytorch(q.contiguous(), k.contiguous(), causal=True, attn_sink=None)
        self.assertAlmostEqual(out["entropy_global"].item(), ref["entropy_global"].item(), delta=self.ATOL)

    def test_large_sink_collapses_entropy(self):
        q, k = self._bhsd()
        big = torch.full((4,), 50.0, device="cuda")
        out = compute_qk_stats_triton(q.contiguous(), k.contiguous(), causal=True, attn_sink=big)
        self.assertLess(out["entropy_global"].item(), 1e-2)
        self.assertLess(out["sink_global"].item(), 1e-2)


class PackedSinkFoldTest(unittest.TestCase):
    """Fold the sink logit into the packed (THD) QK stats vs the pytorch reference."""

    ATOL = 2e-2
    RTOL = 2e-2

    def _qk(self, seed=5):
        q, k, cu = _make_packed_qk([48, 80, 64], num_heads=4, head_dim=64, seed=seed)
        return (*_to_htd(q, k), cu)

    def test_fold_matches_reference(self):
        qh, kh, cu = self._qk()
        for sink in (torch.tensor([0.5, -1.0, 2.0, 0.0], device="cuda"), None):
            tri = compute_qk_stats_triton_packed(qh, kh, cu, causal=True, attn_sink=sink)
            ref = compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True, attn_sink=sink)
            for key in ("entropy_global", "sink_global"):
                t, r = tri[key].float().item(), ref[key].float().item()
                self.assertAlmostEqual(t, r, delta=self.ATOL + self.RTOL * abs(r), msg=f"{key}: {t} vs {r}")

    def test_large_sink_collapses_entropy(self):
        qh, kh, cu = self._qk()
        tri = compute_qk_stats_triton_packed(qh, kh, cu, causal=True, attn_sink=torch.full((4,), 50.0, device="cuda"))
        self.assertLess(tri["entropy_global"].item(), 1e-2)
        self.assertLess(tri["sink_global"].item(), 1e-2)


class LseGateKernelTest(unittest.TestCase):
    """LSE / gate outputs on both the dense and packed kernels.

    Parity is asserted in fp32. In bf16 the PyTorch reference is the LESS accurate
    side — it materializes bf16 logits then reduces, while the kernel accumulates in
    fp32 — so a bf16 triton-vs-pytorch delta measures the reference, not the kernel.
    ``test_bf16_kernel_matches_fp64_reference`` pins the kernel itself instead.
    """

    ATOL = 5e-3
    KEYS = ("lse_per_head", "gate_per_head", "lse_std_per_head", "gate_std_per_head")

    def _beta(self, num_heads=4):
        return torch.tensor([0.5, -1.0, 2.0, 0.0], device="cuda")[:num_heads]

    def _dense_fp32(self, batch=2, num_heads=4, seq_len=128, head_dim=64, seed=11):
        torch.manual_seed(seed)
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        return q, k

    def _dense_bf16(self, batch=1, num_heads=4, seq_len=96, head_dim=64, seed=3):
        torch.manual_seed(seed)
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        return q, k

    def _assert_heads_close(self, triton_out, ref_out, keys, tag=""):
        for key in keys:
            t, r = triton_out[key].float(), ref_out[key].float()
            self.assertEqual(t.shape, r.shape, f"{tag} {key}: shape {t.shape} vs {r.shape}")
            diff = (t - r).abs().max().item()
            self.assertLess(diff, self.ATOL, f"{tag} {key}: maxdiff={diff:.3e}\ntriton={t}\nref={r}")

    def test_dense_matches_pytorch_fp32(self):
        q, k = self._dense_fp32()
        beta = self._beta()
        self._assert_heads_close(
            compute_qk_stats_triton(q, k, causal=True, attn_sink=beta),
            compute_qk_stats_pytorch(q, k, causal=True, attn_sink=beta),
            self.KEYS,
            tag="dense_fp32",
        )

    def test_packed_matches_pytorch_fp32(self):
        torch.manual_seed(12)
        # 50/70/72: the first boundary lands inside BLOCK_M=64, so the M-block
        # partial reduce is exercised alongside a mid-block sequence switch.
        q, k, cu = _make_packed_qk([50, 70, 72], num_heads=4, head_dim=64, dtype=torch.float32, seed=12)
        qh, kh = _to_htd(q, k)
        beta = self._beta()
        self._assert_heads_close(
            compute_qk_stats_triton_packed(qh, kh, cu, causal=True, attn_sink=beta),
            compute_qk_stats_pytorch_packed(qh, kh, cu, causal=True, attn_sink=beta),
            self.KEYS,
            tag="packed_fp32",
        )

    def test_gate_equals_one_minus_offset_column_weight(self):
        """``gate`` IS the offset column's complement — the defining identity.

        ``1 - gate`` is the softmax weight of the key-less offset column, which is a
        DIFFERENT column from the one the ``sink`` metric measures (that one is the
        real first token). Asserting against a materialized softmax keeps the two
        from being conflated again.

        Inputs are bf16 on purpose: ``tl.dot`` runs fp32 inputs through TF32 (10-bit
        mantissa, ~1e-5 logit error) whereas bf16 products are exact in the fp32
        accumulator, so bf16 is the tighter test of the identity.
        """
        num_heads, head_dim = 4, 64
        q, k = self._dense_bf16(num_heads=num_heads, head_dim=head_dim)
        batch, _, seq_len, _ = q.shape
        beta = self._beta(num_heads)

        out = compute_qk_stats_triton(q, k, causal=True, attn_sink=beta)

        logits = (q.double() @ k.double().transpose(-2, -1)) / head_dim**0.5
        logits = logits.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), 1), float("-inf")
        )
        offset_col = beta.double().reshape(1, num_heads, 1, 1).expand(batch, num_heads, seq_len, 1)
        probs = torch.softmax(torch.cat([logits, offset_col], dim=-1), dim=-1)
        alpha_offset = probs[..., -1].mean(-1).flatten()

        gate = out["gate_per_head"].double().flatten()
        self.assertLess((gate - (1.0 - alpha_offset)).abs().max().item(), 1e-6)

    def test_bf16_kernel_matches_fp64_reference(self):
        """The kernel accumulates in fp32, so bf16 inputs still land on fp64 truth."""
        torch.manual_seed(0)
        num_heads, head_dim = 4, 64
        seq_lengths = [50, 70, 72]
        q, k, cu = _make_packed_qk(seq_lengths, num_heads, head_dim, dtype=torch.bfloat16, seed=0)
        qh, kh = _to_htd(q, k)
        beta = self._beta(num_heads)

        out = compute_qk_stats_triton_packed(qh, kh, cu, causal=True, attn_sink=beta)

        # fp64 ground truth from the SAME bf16 inputs, one sequence at a time.
        rows_lse = []
        bounds = [0] + list(torch.tensor(seq_lengths).cumsum(0).tolist())
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            length = e - s
            logits = (qh[:, s:e, :].double() @ kh[:, s:e, :].double().transpose(-2, -1)) / head_dim**0.5
            logits = logits.masked_fill(
                torch.triu(torch.ones(length, length, device="cuda", dtype=torch.bool), 1), float("-inf")
            )
            rows_lse.append(torch.logsumexp(logits, dim=-1))
        row_lse = torch.cat(rows_lse, dim=-1)
        row_gate = torch.sigmoid(row_lse - beta.double().reshape(num_heads, 1))

        expected = {
            "lse_per_head": row_lse.mean(-1),
            "gate_per_head": row_gate.mean(-1),
            "lse_std_per_head": row_lse.std(-1, unbiased=False),
            "gate_std_per_head": row_gate.std(-1, unbiased=False),
        }
        for key, ref in expected.items():
            got = out[key].double().flatten()
            diff = (got - ref).abs().max().item()
            self.assertLess(diff, 1e-4, f"{key}: maxdiff={diff:.3e}\ntriton={got}\nfp64={ref}")

    def test_gate_keys_absent_without_offset(self):
        q, k = self._dense_fp32()
        for out in (
            compute_qk_stats_triton(q, k, causal=True, attn_sink=None),
            compute_qk_stats_pytorch(q, k, causal=True, attn_sink=None),
        ):
            self.assertIsNone(out["gate_per_head"])
            self.assertIsNone(out["gate_std_per_head"])
            # lse needs no offset to be meaningful and must survive.
            self.assertIsNotNone(out["lse_per_head"])
            self.assertIsNotNone(out["lse_std_per_head"])

    def test_packed_lse_respects_sequence_boundaries(self):
        """Packed LSE must cover only each row's own sequence.

        Three 64-token sequences vs one 192-token sequence over identical tokens:
        if the kernel leaked across boundaries the packed LSE would inherit the
        longer row sums and come out strictly larger.
        """
        torch.manual_seed(21)
        q, k, _ = _make_packed_qk([192], num_heads=4, head_dim=64, dtype=torch.float32, seed=21)
        qh, kh = _to_htd(q, k)
        split_cu = torch.tensor([0, 64, 128, 192], dtype=torch.int32, device="cuda")
        whole_cu = torch.tensor([0, 192], dtype=torch.int32, device="cuda")

        split = compute_qk_stats_triton_packed(qh, kh, split_cu, causal=True)
        whole = compute_qk_stats_triton_packed(qh, kh, whole_cu, causal=True)

        self.assertTrue(
            (split["lse_per_head"].float() < whole["lse_per_head"].float()).all(),
            f"split={split['lse_per_head']} whole={whole['lse_per_head']}",
        )
        # And the split result must equal the per-sequence reference exactly.
        ref = compute_qk_stats_pytorch_packed(qh, kh, split_cu, causal=True)
        self._assert_heads_close(split, ref, ("lse_per_head", "lse_std_per_head"), tag="split_vs_ref")

    def test_large_offset_shuts_the_gate(self):
        """A huge ``beta`` sends the offset column to weight ~1, so gate -> 0."""
        q, k = self._dense_fp32()
        num_heads = q.shape[1]
        out = compute_qk_stats_triton(q, k, causal=True, attn_sink=torch.full((num_heads,), 50.0, device="cuda"))
        self.assertLess(out["gate_per_head"].float().max().item(), 1e-6)
        # A shut gate is shut on every row, so its spread over rows collapses too.
        self.assertLess(out["gate_std_per_head"].float().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
