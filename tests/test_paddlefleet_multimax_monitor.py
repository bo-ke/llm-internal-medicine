import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    paddle = importlib.import_module("paddle")
except Exception as exc:  # pragma: no cover - optional backend
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

mm_metrics = importlib.import_module("internal_medicine.backends.paddlefleet.multimax_metrics")
mm_monitor = importlib.import_module("internal_medicine.backends.paddlefleet.multimax_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


def _seglu_reference(x, ranges, ts):
    relu = paddle.nn.functional.relu
    out = x.clone()
    out += ts[0] * relu(ranges[0] - x)
    out += ts[1] * relu(x - ranges[1])
    out += ts[2] * relu(ranges[2] - x) ** 2
    out += ts[3] * relu(x - ranges[3]) ** 2
    return out


class SegLUMirrorTest(unittest.TestCase):
    def test_zero_init_is_identity(self):
        x = paddle.to_tensor([[-2.0, 0.0, 3.5]], dtype="float32")
        zeros = paddle.zeros([4], dtype="float32")
        out = mm_metrics.apply_seglu(x, zeros, zeros)
        self.assertTrue(bool(paddle.allclose(out, x)))

    def test_matches_upstream_reference_for_trained_params(self):
        x = paddle.to_tensor([[-4.0, -0.5, 0.0, 1.25, 6.0]], dtype="float32")
        ranges = paddle.to_tensor([-1.0, 2.0, -3.0, 4.0], dtype="float32")
        ts = paddle.to_tensor([0.3, -0.2, 0.05, 0.1], dtype="float32")
        out = mm_metrics.apply_seglu(x, ranges, ts)
        self.assertTrue(bool(paddle.allclose(out, _seglu_reference(x, ranges, ts), atol=1e-6)))


class DistributionMetricsTest(unittest.TestCase):
    def test_uniform_logits_hit_entropy_and_topk_bounds(self):
        logits = paddle.zeros([2, 8], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, topk=2)

        self.assertAlmostEqual(float(m["entropy"]), math.log(8), places=5)
        self.assertAlmostEqual(float(m["entropy_norm"]), 1.0, places=5)
        self.assertAlmostEqual(float(m["top1_prob"]), 1.0 / 8, places=6)
        self.assertAlmostEqual(float(m["top2_prob"]), 2.0 / 8, places=6)
        # Uniform rows have no entry strictly between eps and the max, so the
        # multi-modality mean has no valid row to average.
        self.assertAlmostEqual(float(m["relevant_count"]), 0.0, places=6)
        # ... and none strictly below eps either, so sparsity has no valid row
        # either: a uniform row must not be scored as maximally sparse.
        self.assertAlmostEqual(float(m["sparse_count"]), 0.0, places=6)
        self.assertAlmostEqual(float(m["sparsity"]), 0.0, places=6)

    def test_threshold_boundary_belongs_to_neither_set(self):
        # prob_eps == 1/V puts every entry of a uniform row exactly on eps.
        logits = paddle.zeros([1, 4], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=0.25, topk=1)

        self.assertAlmostEqual(float(m["relevant_count"]), 0.0, places=6)
        self.assertAlmostEqual(float(m["sparse_count"]), 0.0, places=6)

    def test_logit_eps_overrides_the_probability_threshold(self):
        logits = paddle.to_tensor([[3.0, 1.0, -1.0, -5.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, logit_eps=0.0, topk=1)

        # eps == 0: relevant is {1.0} (max excluded), sparse is {-1.0, -5.0}.
        self.assertAlmostEqual(float(m["relevant_count"]), 1.0, places=6)
        self.assertAlmostEqual(float(m["sparse_count"]), 2.0, places=6)

    def test_multi_modality_matches_hand_computed_definition(self):
        # Two relevant modes plus one clearly irrelevant entry.
        logits = paddle.to_tensor([[2.0, 1.0, -30.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=1e-6, topk=1)

        p = paddle.nn.functional.softmax(logits, axis=-1)[0]
        expected = 1.0 - float(p[0] - p[1])  # N == 1: only the second entry
        self.assertAlmostEqual(float(m["multi_modality"]), expected, places=5)
        self.assertAlmostEqual(float(m["relevant_count"]), 1.0, places=6)

    def test_multi_modality_is_higher_for_a_flatter_head(self):
        peaked = paddle.to_tensor([[8.0, 1.0, -30.0]], dtype="float32")
        flat = paddle.to_tensor([[1.2, 1.0, -30.0]], dtype="float32")
        peaked_m = mm_metrics.compute_distribution_metrics(peaked, prob_eps=1e-6, topk=1)
        flat_m = mm_metrics.compute_distribution_metrics(flat, prob_eps=1e-6, topk=1)

        self.assertLess(float(peaked_m["multi_modality"]), float(flat_m["multi_modality"]))

    def test_sparsity_matches_hand_computed_definition(self):
        # Def 3.3 with an explicit reference s: mean of exp(-p_l / s) over the
        # entries below eps. s is chosen so the terms land mid-range, where a
        # wrong reference cannot hide behind saturation at 0 or 1.
        logits = paddle.to_tensor([[6.0, -1.0, -2.0]], dtype="float32")
        ref = 0.05
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=0.01, topk=1, sparsity_ref=ref)

        p = paddle.nn.functional.softmax(logits, axis=-1)[0]
        sparse_p = [float(p[1]), float(p[2])]
        expected = sum(math.exp(-v / ref) for v in sparse_p) / len(sparse_p)
        self.assertAlmostEqual(float(m["sparsity"]), expected, places=5)
        self.assertAlmostEqual(float(m["sparse_count"]), 2.0, places=6)

    def test_sparsity_rises_as_irrelevant_mass_shrinks(self):
        near = paddle.to_tensor([[6.0, 2.5, 2.4]], dtype="float32")
        far = paddle.to_tensor([[6.0, -4.0, -4.5]], dtype="float32")
        kwargs = {"prob_eps": 0.2, "topk": 1, "sparsity_ref": 0.05}
        near_s = float(mm_metrics.compute_distribution_metrics(near, **kwargs)["sparsity"])
        far_s = float(mm_metrics.compute_distribution_metrics(far, **kwargs)["sparsity"])
        self.assertLess(near_s, far_s)

    def test_sparsity_reference_is_independent_of_the_scored_row(self):
        # Regression guard: referencing phi_min (a statistic of the same row)
        # caps every term at e^-1, so a genuinely sparse row could never score
        # above that, and the score decays as 1/L. With a fixed reference it
        # approaches 1, which is the [0, 1] normalization Def 3.3 asks for.
        very_sparse = paddle.to_tensor([[20.0, -20.0, -20.0, -20.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(very_sparse, prob_eps=1e-6, topk=1)
        self.assertGreater(float(m["sparsity"]), math.exp(-1.0))
        self.assertAlmostEqual(float(m["sparsity"]), 1.0, places=5)
        self.assertAlmostEqual(float(m["sparse_count"]), 3.0, places=6)

        # Pushing this row's own minimum further down must not move the score,
        # since the mass below eps is already negligible either way.
        deeper = paddle.to_tensor([[20.0, -40.0, -20.0, -20.0]], dtype="float32")
        deeper_s = float(mm_metrics.compute_distribution_metrics(deeper, prob_eps=1e-6, topk=1)["sparsity"])
        self.assertAlmostEqual(deeper_s, float(m["sparsity"]), places=5)

    def test_ref_logits_supply_the_baseline_softmax_minimum(self):
        # sparsity_ref_mode="min" is the paper's SoftMax_{t=1} example: the
        # smallest probability of the *unmodulated* distribution, far below 1/V.
        # It is no longer the default (it underflows most terms to 0), but it must
        # stay available and exact.
        logits = paddle.to_tensor([[6.0, -1.0, -2.0]], dtype="float32")
        ref_logits = paddle.to_tensor([[5.0, 0.0, -3.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(
            logits, prob_eps=0.01, topk=1, ref_logits=ref_logits, sparsity_ref_mode="min"
        )

        p = paddle.nn.functional.softmax(logits, axis=-1)[0]
        ref_p = paddle.nn.functional.softmax(ref_logits, axis=-1)[0]
        s = float(ref_p.min())
        expected = sum(math.exp(-float(p[i]) / s) for i in (1, 2)) / 2
        self.assertAlmostEqual(float(m["sparsity"]), expected, places=5)
        self.assertLess(s, 1.0 / 3)  # strictly below the uniform fallback

    def test_sparsity_ref_wins_over_ref_logits(self):
        logits = paddle.to_tensor([[6.0, -1.0, -2.0]], dtype="float32")
        ref_logits = paddle.to_tensor([[5.0, 0.0, -3.0]], dtype="float32")
        kwargs = {"prob_eps": 0.01, "topk": 1}
        pinned = mm_metrics.compute_distribution_metrics(logits, sparsity_ref=0.05, ref_logits=ref_logits, **kwargs)
        explicit = mm_metrics.compute_distribution_metrics(logits, sparsity_ref=0.05, **kwargs)
        self.assertAlmostEqual(float(pinned["sparsity"]), float(explicit["sparsity"]), places=6)

    def test_every_token_mean_metric_reports_its_quantiles(self):
        logits = paddle.to_tensor([[6.0, -1.0, -2.0, -8.0], [1.0, 0.9, -3.0, -9.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=0.01, topk=2)

        for name in (
            "entropy",
            "entropy_norm",
            "top1_prob",
            "top2_prob",
            "multi_modality",
            "sparsity",
            "relevant_count",
            "sparse_count",
        ):
            for suffix in ("p50", "p95", "p98"):
                self.assertIn(f"{name}_{suffix}", m, f"{name} must report {suffix}")
        # rows is a constant, so it has no distribution to summarize.
        self.assertNotIn("rows_p50", m)
        # the old symmetric band is gone
        self.assertNotIn("entropy_std", m)

        rows = sorted(
            float(mm_metrics.compute_distribution_metrics(logits[i : i + 1], prob_eps=0.01, topk=2)["entropy"])
            for i in range(2)
        )
        # 2 rows, nearest-rank: ceil(0.5*2)-1 = 0 for p50, ceil(0.95*2)-1 = 1 for
        # the tail quantiles.
        self.assertAlmostEqual(float(m["entropy_p50"]), rows[0], places=5)
        self.assertAlmostEqual(float(m["entropy_p95"]), rows[1], places=5)
        self.assertAlmostEqual(float(m["entropy_p98"]), rows[1], places=5)

    def test_quantiles_match_numpy_on_a_larger_sample(self):
        paddle.seed(0)
        logits = paddle.randn([64, 32], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=1e-6, topk=1)

        per_row = np.sort(
            np.array(
                [
                    float(mm_metrics.compute_distribution_metrics(logits[i : i + 1], prob_eps=1e-6, topk=1)["entropy"])
                    for i in range(64)
                ]
            )
        )
        for q, suffix in ((0.5, "p50"), (0.95, "p95"), (0.98, "p98")):
            expected = per_row[max(0, math.ceil(q * len(per_row)) - 1)]
            self.assertAlmostEqual(float(m[f"entropy_{suffix}"]), expected, places=5)

    def test_quantiles_ignore_rows_the_mean_drops(self):
        # Row 1 has no entry strictly between eps and the max, so it is excluded
        # from multi_modality; the quantiles must be over the surviving row only,
        # not over a +inf sentinel that the mask parked at the end of the sort.
        logits = paddle.to_tensor([[2.0, 1.0, -30.0], [5.0, -30.0, -30.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, prob_eps=1e-6, topk=1)
        single = mm_metrics.compute_distribution_metrics(logits[0:1], prob_eps=1e-6, topk=1)
        self.assertAlmostEqual(float(m["multi_modality"]), float(single["multi_modality"]), places=5)
        for suffix in ("p50", "p95", "p98"):
            self.assertAlmostEqual(
                float(m[f"multi_modality_{suffix}"]),
                float(single["multi_modality"]),
                places=5,
                msg=f"{suffix} must not pick up the masked row",
            )

    def test_multi_modality_topk_is_the_top1_minus_topk_mean_gap(self):
        # The paper's Def 3.2 with eps = the k-th largest entry: "the average
        # difference between the reweighted relevant entries and the maximum",
        # where relevant == top-k minus the max.
        logits = paddle.to_tensor([[6.0, 5.5, 5.0, -2.0, -9.0]], dtype="float32")
        m = mm_metrics.compute_distribution_metrics(logits, topk=3)

        p = paddle.nn.functional.softmax(logits, axis=-1)[0]
        expected = 1.0 - (float(p[0]) - (float(p[1]) + float(p[2])) / 2)
        self.assertAlmostEqual(float(m["multi_modality_top3"]), expected, places=5)

    def test_multi_modality_topk_separates_a_flat_head_from_a_peaked_one(self):
        # The eps-thresholded M cannot: with eps = 1/V the relevant set is a long
        # tail whose probabilities are negligible next to phi_max, so M collapses
        # onto 1 - top1_prob. The top-k variant must stay sensitive to the shape
        # of the head itself.
        peaked = paddle.to_tensor([[9.0, 1.0, 0.9, 0.8]], dtype="float32")
        flat = paddle.to_tensor([[1.1, 1.0, 0.9, 0.8]], dtype="float32")
        peaked_m = mm_metrics.compute_distribution_metrics(peaked, topk=4)
        flat_m = mm_metrics.compute_distribution_metrics(flat, topk=4)
        self.assertLess(float(peaked_m["multi_modality_top4"]), float(flat_m["multi_modality_top4"]))
        self.assertGreater(float(flat_m["multi_modality_top4"]), 0.9)

    def test_sparsity_ref_modes_are_ordered_and_geomean_sits_in_between(self):
        # uniform (s = 1/V) saturates near 1, min (s = the baseline minimum)
        # underflows toward 0; the geomean reference must land between them so the
        # smooth step actually has resolution.
        paddle.seed(0)
        raw = paddle.randn([8, 512], dtype="float32") * 2.0
        mod = raw * 1.3
        scores = {
            mode: float(
                mm_metrics.compute_distribution_metrics(mod, ref_logits=raw, topk=5, sparsity_ref_mode=mode)["sparsity"]
            )
            for mode in mm_metrics.SPARSITY_REF_MODES
        }
        self.assertLess(scores["min"], scores["geomean"])
        self.assertLess(scores["geomean"], scores["uniform"])
        self.assertGreater(scores["geomean"], 0.05)
        self.assertLess(scores["geomean"], 0.95)

    def test_sparsity_reference_never_depends_on_the_scored_row(self):
        # Same baseline, different modulation of the *same* row: the reference is
        # a statistic of ref_logits only, so log s must not move with the tile.
        paddle.seed(1)
        raw = paddle.randn([4, 256], dtype="float32")
        a = mm_metrics._sparsity_log_ref(256, None, raw, None, None, "geomean")
        b = mm_metrics._sparsity_log_ref(256, None, raw, None, None, "geomean")
        self.assertTrue(bool(paddle.all(a == b)))
        # ... and an explicit constant wins over every mode.
        for mode in mm_metrics.SPARSITY_REF_MODES:
            self.assertAlmostEqual(
                float(mm_metrics._sparsity_log_ref(256, 0.01, raw, None, None, mode)),
                math.log(0.01),
                places=6,
            )

    def test_metrics_are_zero_dim_and_finite(self):
        logits = paddle.randn([4, 16], dtype="float32")
        for name, value in mm_metrics.compute_distribution_metrics(logits).items():
            self.assertEqual(value.shape, [], f"{name} must be a 0-dim tensor")
            self.assertTrue(bool(paddle.isfinite(value)), f"{name} must be finite")

    def test_batched_logits_are_flattened_to_rows(self):
        flat = paddle.randn([6, 16], dtype="float32")
        batched = flat.reshape([2, 3, 16])
        flat_m = mm_metrics.compute_distribution_metrics(flat)
        batched_m = mm_metrics.compute_distribution_metrics(batched)

        self.assertEqual(float(batched_m["rows"]), 6.0)
        self.assertAlmostEqual(float(batched_m["entropy"]), float(flat_m["entropy"]), places=5)

    def test_param_metrics_expose_every_component(self):
        out = mm_metrics.compute_param_metrics(
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0], dtype="float32"),
            paddle.to_tensor([4.0, 5.0, 6.0, 7.0], dtype="float32"),
        )
        self.assertEqual(float(out["range_2"]), 2.0)
        self.assertEqual(float(out["t_3"]), 7.0)
        self.assertEqual(len(out), 8)


class _FakeMultimaxHead(paddle.nn.Layer):
    """Minimal stand-in for ``GPTLMHead`` with multimax enabled."""

    def __init__(self, vocab=32, hidden=8, fused=False, mtp=False, as_dict=False):
        super().__init__()
        self.use_multimax_lmhead = True
        self.fused = fused
        self.mtp = mtp
        self.as_dict = as_dict
        self.weight = self.create_parameter(
            shape=[vocab, hidden],
            default_initializer=paddle.nn.initializer.Normal(std=0.05),
        )
        self.bias = None
        self.multimax_ranges = self.create_parameter(shape=[4], default_initializer=paddle.nn.initializer.Constant(0.0))
        self.multimax_ts = self.create_parameter(shape=[4], default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, hidden_states):
        if self.fused:
            out = (hidden_states, self.weight, self.bias, self.multimax_ranges, self.multimax_ts)
        else:
            logits = paddle.matmul(hidden_states, self.weight, transpose_y=True)
            out = mm_metrics.apply_seglu(logits, self.multimax_ranges, self.multimax_ts)
        if self.mtp:
            out = [out, out]
        if self.as_dict:
            out = {"logits": out, "mtp_loss": None}
        return out


class MultiMaxMonitorTest(unittest.TestCase):
    def tearDown(self):
        training_logs.reset()

    def _run(self, head, tokens=12, hidden=8, sample_tokens=8):
        monitor = mm_monitor.PaddleMultiMaxMonitor(sample_tokens=sample_tokens, topk=5, verbose=True)
        monitor.register_hooks(head)
        head.train()
        head(paddle.randn([tokens, hidden], dtype="float32"))
        monitor.step()
        monitor.remove_hooks()
        return training_logs.get_latest(prefix="multimax/")

    def test_unfused_path_emits_every_declared_metric(self):
        logs = self._run(_FakeMultimaxHead())

        for name in ("entropy", "entropy_norm", "top1_prob", "top5_prob", "multi_modality", "sparsity"):
            self.assertIn(f"multimax/global_{name}", logs)
        self.assertIn("multimax/global_range_0", logs)
        self.assertIn("multimax/global_t_3", logs)
        # sample_tokens=8 with 12 tokens -> strided down to 8 rows.
        self.assertLessEqual(logs["multimax/global_rows"], 8.0)

    def test_fused_ce_path_reconstructs_the_logits_tile(self):
        logs = self._run(_FakeMultimaxHead(fused=True))
        self.assertIn("multimax/global_entropy", logs)
        self.assertGreater(logs["multimax/global_entropy"], 0.0)

    def test_mtp_and_dict_outputs_are_unwrapped(self):
        for kwargs in ({"mtp": True}, {"as_dict": True}, {"mtp": True, "fused": True}):
            training_logs.reset()
            logs = self._run(_FakeMultimaxHead(**kwargs))
            self.assertIn("multimax/global_entropy", logs, f"failed for {kwargs}")

    def test_no_head_means_no_hooks_and_no_keys(self):
        plain = paddle.nn.Linear(4, 4)
        monitor = mm_monitor.PaddleMultiMaxMonitor()
        monitor.register_hooks(plain)

        self.assertEqual(monitor.hooks, [])
        self.assertFalse(monitor._buffers_allocated)

    def test_topk_is_clamped_to_the_vocab_width_at_registration(self):
        head = _FakeMultimaxHead(vocab=4)
        monitor = mm_monitor.PaddleMultiMaxMonitor(topk=10)
        monitor.register_hooks(head)

        self.assertEqual(monitor._topk_effective, 4)
        self.assertIn("multimax/global_top4_prob", monitor._mean_keys)
        monitor.remove_hooks()

    def test_token_sampling_caps_the_analyzed_rows(self):
        logs = self._run(_FakeMultimaxHead(), tokens=100, sample_tokens=8)
        self.assertEqual(logs["multimax/global_rows"], 8.0)

    def test_fewer_tokens_than_the_budget_uses_every_row(self):
        logs = self._run(_FakeMultimaxHead(), tokens=5, sample_tokens=8)
        self.assertEqual(logs["multimax/global_rows"], 5.0)

    def test_log_global_false_disables_the_monitor_instead_of_recording_nothing(self):
        head = _FakeMultimaxHead()
        monitor = mm_monitor.PaddleMultiMaxMonitor(log_global=False)
        monitor.register_hooks(head)

        self.assertEqual(monitor.hooks, [])
        self.assertFalse(monitor._buffers_allocated)

    def test_fused_path_does_not_promote_the_head_weight_to_fp32(self):
        # The [vocab, hidden] weight must never be cast (that would allocate a
        # full-size fp32 copy at LM-head scale); only the sampled tile is.
        head = _FakeMultimaxHead(fused=True)
        monitor = mm_monitor.PaddleMultiMaxMonitor(sample_tokens=4, topk=2)
        monitor.register_hooks(head)
        original_astype = paddle.Tensor.astype
        promoted_shapes = []

        def tracking_astype(self, dtype):
            if str(dtype).endswith("float32") and len(self.shape) == 2 and self.shape == head.weight.shape:
                promoted_shapes.append(list(self.shape))
            return original_astype(self, dtype)

        paddle.Tensor.astype = tracking_astype
        try:
            head.train()
            head(paddle.randn([6, 8], dtype="float32"))
        finally:
            paddle.Tensor.astype = original_astype
            monitor.remove_hooks()

        self.assertEqual(promoted_shapes, [])


class _FakeGPTMTPLMHead(_FakeMultimaxHead):
    """Stand-in for ``GPTMTPLMHead``: returns ``dict_args`` with ``mtp_logits``.

    The dict also carries the main head's ``logits`` (the pipeline dict flows
    through both heads), which the monitor must ignore for this head.
    """

    def forward(self, hidden_states):
        peaked = paddle.concat(
            [
                paddle.full([hidden_states.shape[0], 1], 20.0, dtype="float32"),
                paddle.zeros([hidden_states.shape[0], self.weight.shape[0] - 1], dtype="float32"),
            ],
            axis=-1,
        )
        uniform = paddle.zeros([hidden_states.shape[0], self.weight.shape[0]], dtype="float32")
        return {"logits": uniform, "mtp_logits": [peaked], "hidden_states": hidden_states}


class MultiMaxMTPHeadTest(unittest.TestCase):
    def tearDown(self):
        training_logs.reset()

    def test_mtp_head_gets_its_own_namespace_and_reads_mtp_logits(self):
        model = paddle.nn.LayerList([_FakeMultimaxHead(vocab=8), _FakeGPTMTPLMHead(vocab=8)])
        monitor = mm_monitor.PaddleMultiMaxMonitor(sample_tokens=8, topk=2)
        monitor.register_hooks(model)
        self.assertEqual(len(monitor.hooks), 2)

        for head in model:
            head.train()
            head(paddle.randn([6, 8], dtype="float32"))
        monitor.step()
        monitor.remove_hooks()
        logs = training_logs.get_latest(prefix="multimax/")

        self.assertIn("multimax/global_entropy", logs)
        self.assertIn("multimax/global_mtp_entropy", logs)
        self.assertIn("multimax/global_mtp_range_0", logs)
        # mtp_logits is one-hot-peaked; the stale "logits" in the same dict is
        # uniform. Reading the wrong key would give entropy == log(8).
        self.assertLess(logs["multimax/global_mtp_entropy"], 0.01)
        self.assertAlmostEqual(logs["multimax/global_mtp_top1_prob"], 1.0, places=5)

    def test_single_main_head_declares_no_mtp_keys(self):
        monitor = mm_monitor.PaddleMultiMaxMonitor(topk=2)
        monitor.register_hooks(_FakeMultimaxHead(vocab=8))
        self.assertFalse([k for k in monitor._mean_keys if "mtp_" in k])
        monitor.remove_hooks()


if __name__ == "__main__":
    unittest.main()
