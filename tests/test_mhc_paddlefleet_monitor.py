import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

mhc_metrics = importlib.import_module("internal_medicine.backends.paddlefleet.mhc_metrics")
mhc_monitor = importlib.import_module("internal_medicine.backends.paddlefleet.mhc_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs

PaddleMHCHealthMonitor = mhc_monitor.PaddleMHCHealthMonitor


class FakeHC(nn.Layer):
    """Stand-in for HyperConnectionModule exposing compute_mappings."""

    def __init__(self, n, h_pre, h_post, h_res, h_res_logits=None):
        super().__init__()
        self.n = n
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res
        # Raw pre-Sinkhorn mixing logits. Zeros -> uniform rows -> every column
        # sum is exactly 1, i.e. the healthy case.
        if h_res_logits is None:
            h_res_logits = paddle.zeros([*h_pre.shape[:-1], n * n])
        self._h_res_logits = h_res_logits

    def _compute_h(self, proj, r):
        return self._h_pre, self._h_post, self._h_res_logits

    def compute_mappings(self, x):
        # Mirrors the real module: _compute_h yields the raw logits, the Sinkhorn
        # projection then turns them into h_res. Going through self._compute_h is
        # what lets the monitor's instance-attribute shadowing see the logits.
        self._compute_h(None, None)
        return self._h_pre, self._h_post, self._h_res

    def fused_h_res_h_post_bda(
        self,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob=0.0,
        training=True,
        fused=False,
    ):
        x, bias = layer_output_with_bias
        n = self.n
        leading = original_residual.shape[:-1]
        C = original_residual.shape[-1] // n
        mixed = paddle.bmm(
            h_res.reshape([-1, n, n]).transpose([0, 2, 1]),
            original_residual.reshape([-1, n, C]),
        ).reshape([*leading, n, C])
        xb = x if bias is None else x + bias
        return (h_post.unsqueeze(-1) * xb.unsqueeze(-2) + mixed).reshape([*leading, n * C])

    def forward(self, hidden_states):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Layer):
    """Stand-in for HyperConnectionTransformerLayer with the two hc modules."""

    def __init__(self, attn, mlp):
        super().__init__()
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7 (fwd); col sums [4,6] -> 6 (bwd).
        m = paddle.to_tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype="float32")
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(m, axis=-1)), 7.0, places=5)
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(m, axis=-2)), 6.0, places=5)

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = paddle.eye(n).reshape([1, 1, n, n])
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(ident, axis=-1)), 1.0, places=5)
        self.assertAlmostEqual(float(mhc_metrics.amax_gain(ident, axis=-2)), 1.0, places=5)

    def test_gate_stats(self):
        h = paddle.to_tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype="float32")
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(float(mean), 1.5, places=5)
        self.assertAlmostEqual(float(std), float(h.std()), places=5)

    def test_stream_concentration_bounds(self):
        # Uniform across streams -> 1 (lower bound); all mass on one stream -> n.
        uniform = paddle.full([3, 4], 0.7)
        one_hot = paddle.to_tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype="float32")
        self.assertAlmostEqual(
            float(mhc_metrics.h_post_structure_stats(uniform)["h_post_stream_concentration"]),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(mhc_metrics.h_post_structure_stats(one_hot)["h_post_stream_concentration"]),
            4.0,
            places=5,
        )

    def test_token_std_zero_when_gate_is_constant(self):
        const = paddle.full([8, 4], 1.0)
        varying = paddle.to_tensor([[0.2] * 4, [1.8] * 4], dtype="float32")
        self.assertAlmostEqual(float(mhc_metrics.h_post_structure_stats(const)["h_post_token_std"]), 0.0, places=6)
        self.assertGreater(float(mhc_metrics.h_post_structure_stats(varying)["h_post_token_std"]), 0.5)

    def test_branch_residual_share_against_explicit_terms(self):
        # h_res = identity -> residual term is exactly the incoming streams, so
        # the share reduces to b / (b + ‖x_l‖_F) with b = ‖h_post‖₂·‖x‖₂.
        n, c = 2, 3
        h_res = paddle.eye(n).reshape([1, n, n])
        streams = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype="float32")
        h_post = paddle.to_tensor([[3.0, 4.0]], dtype="float32")  # ‖·‖₂ = 5
        x = paddle.to_tensor([[0.0, 0.0, 2.0]], dtype="float32")  # ‖·‖₂ = 2
        stats = mhc_metrics.branch_residual_share(h_res, streams.reshape([1, n * c]), h_post, x)
        branch = 5.0 * 2.0
        residual = float(paddle.sqrt(paddle.to_tensor(5.0)))  # ‖streams‖_F = √5
        expected = branch / (branch + residual)
        self.assertAlmostEqual(float(stats["branch_residual_share"]), expected, places=4)
        self.assertAlmostEqual(float(stats["branch_residual_share_max"]), expected, places=4)

    def test_branch_residual_share_zero_gate_kills_branch(self):
        n, c = 4, 5
        h_res = paddle.eye(n).reshape([1, n, n])
        streams = paddle.randn([1, n * c])
        x = paddle.randn([1, c])
        stats = mhc_metrics.branch_residual_share(h_res, streams, paddle.zeros([1, n]), x)
        self.assertAlmostEqual(float(stats["branch_residual_share"]), 0.0, places=6)

    def test_branch_residual_share_zero_residual_token_stays_bounded(self):
        # An all-zero token (padding, or an MTP layer's shifted slot) drives the
        # residual norm to 0. The raw ratio hit the epsilon floor and returned
        # ~1e6, hijacking the token mean; the bounded share must read 1.0 and
        # leave the mean dominated by the healthy token.
        n, c = 2, 3
        h_res = paddle.eye(n).reshape([1, n, n]).expand([2, n, n])
        streams = paddle.concat([paddle.ones([1, n * c]), paddle.zeros([1, n * c])])
        h_post = paddle.ones([2, n])
        x = paddle.ones([2, c])
        stats = mhc_metrics.branch_residual_share(h_res, streams, h_post, x)
        self.assertAlmostEqual(float(stats["branch_residual_share_max"]), 1.0, places=6)
        self.assertLessEqual(float(stats["branch_residual_share"]), 1.0)
        self.assertGreater(float(stats["branch_residual_share"]), 0.5)

    def test_branch_residual_share_all_zero_token_is_zero(self):
        # Both terms zero must not divide by zero; the eps guard makes it 0.
        n, c = 2, 3
        stats = mhc_metrics.branch_residual_share(
            paddle.zeros([1, n, n]), paddle.zeros([1, n * c]), paddle.zeros([1, n]), paddle.zeros([1, c])
        )
        self.assertAlmostEqual(float(stats["branch_residual_share"]), 0.0, places=6)

    def test_softmax_extrema_are_one_over_n_for_uniform_logits(self):
        stats = mhc_metrics.h_res_softmax_extrema(paddle.zeros([5, 16]), 4)
        self.assertAlmostEqual(float(stats["h_res_softmax_min"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["h_res_softmax_max"]), 0.25, places=6)

    def test_softmax_min_matches_exp_minus_range_over_S(self):
        # p_min = exp(-R) / S with S = 1/p_max, so -log(p_min) = R + log(S).
        # Row 2 has the widest spread (0 - (-4) = 4) and holds the smallest weight.
        logits = paddle.to_tensor(
            [[-2.0, 0.0, -3.0, -2.0], [-3.0, 0.0, -2.0, -1.0], [-3.0, -2.0, -4.0, 0.0], [-2.0, -2.0, -3.0, 0.0]],
            dtype="float32",
        ).reshape([1, 16])
        p = paddle.nn.functional.softmax(logits.reshape([1, 4, 4]), axis=-1)
        s = 1.0 / float(p[0, 2].max())
        stats = mhc_metrics.h_res_softmax_extrema(logits, 4)
        self.assertAlmostEqual(float(stats["h_res_softmax_min"]), math.exp(-4.0) / s, places=6)

    def test_softmax_max_saturates_where_min_still_resolves(self):
        # Row 0 is one winner against three losers at -R; spreads of 18 and 200
        # both pin `max` at exactly 1.0, so it cannot separate them, while `min`
        # and the logit range can.
        near = paddle.zeros([1, 16])
        far = paddle.zeros([1, 16])
        near[0, 1:4] = -18.0
        far[0, 1:4] = -200.0
        s_near = mhc_metrics.h_res_softmax_extrema(near, 4)
        s_far = mhc_metrics.h_res_softmax_extrema(far, 4)
        self.assertEqual(float(s_near["h_res_softmax_max"]), 1.0)
        self.assertEqual(float(s_far["h_res_softmax_max"]), 1.0)
        self.assertGreater(float(s_near["h_res_softmax_min"]), float(s_far["h_res_softmax_min"]))

    def test_softmax_min_underflows_to_zero_past_the_alarm(self):
        # A spread past ~103 drives softmax's smallest entry below fp32's smallest
        # denormal, so the series bottoms out at exactly 0. That is past the alarm
        # (1.18e-38, i.e. a spread of ~87) and the model cannot tell 0 from 1e-44
        # either — `_sinkhorn_normalize` adds the same 1e-6 to both.
        far = paddle.zeros([1, 16])
        far[0, 1:4] = -200.0
        self.assertEqual(float(mhc_metrics.h_res_softmax_extrema(far, 4)["h_res_softmax_min"]), 0.0)


class MHCMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()
        # Discovery uses isinstance against the real classes; point them at the fakes.
        self._orig_layer_cls = mhc_monitor.HyperConnectionTransformerLayer
        self._orig_mod_cls = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = FakeLayer
        mhc_monitor.HyperConnectionModule = FakeHC

    def tearDown(self):
        mhc_monitor.HyperConnectionTransformerLayer = self._orig_layer_cls
        mhc_monitor.HyperConnectionModule = self._orig_mod_cls
        training_logs.reset()

    def _identity_layer(self, n, s, b):
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.assign(ident),  # own storage, not a view of the shared eye
            )

        return FakeLayer(attn=make_hc(), mlp=make_hc())

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings; grad is enabled by default in dygraph.
        for _, _, mod in targets:
            mod.compute_mappings(paddle.randn([4, 2, x_dim]))

    def _prepare_and_attach(self, monitor, model):
        all_targets = monitor._prepare(model)
        monitor.allocate_buffers()
        monitor._attach(all_targets)
        # Return a single flat list of entries for convenience.
        return [entry for chunk_entries in all_targets for entry in chunk_entries]

    def test_gate_and_amax_metrics_are_recorded(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self.assertTrue(targets)

        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ("h_pre_mean", "h_pre_std", "h_post_mean", "h_post_std"):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            # h_pre == 0.5 everywhere, h_post == 1.0 everywhere.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_pre_mean"], 0.5, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_mean"], 1.0, places=5)
            # Both amax gains are emitted per layer and globally.
            for key in ("amax_gain_fwd", "amax_gain_bwd"):
                self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_{key}"], 1.0, places=4)
                self.assertAlmostEqual(latest[f"mhc_health/global_{comp}_{key}"], 1.0, places=4)

    def test_amax_gain_axes_are_not_swapped(self):
        # An asymmetric h_res pins the convention. Streams mix as
        # `out = h_res^T @ x`, so _fwd must read the COLUMN sums of h_res (1.5
        # here) and _bwd the ROW sums (1.0 here) — single layer and composite
        # agree. Identity matrices cannot tell the two apart, which is how the
        # fwd/bwd label mix-up stayed invisible.
        n, s, b = 2, 1, 1
        h_res = paddle.to_tensor([[0.75, 0.25], [0.75, 0.25]]).reshape([1, 1, n, n]).expand([s, b, n, n])
        hc = FakeHC(
            n=n,
            h_pre=paddle.full([s, b, n], 0.5),
            h_post=paddle.ones([s, b, n]),
            h_res=paddle.assign(h_res),
        )
        model = _mhc_model([FakeLayer(attn=hc, mlp=hc)])

        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest["mhc_health/global_attn_amax_gain_fwd"], 1.5, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_amax_gain_bwd"], 1.0, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_composite_amax_gain_fwd_max"], 1.5, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_composite_amax_gain_bwd_max"], 1.0, places=5)

    def test_softmax_extrema_are_recorded_and_reduce_by_extremum_across_layers(self):
        # layer_0 healthy (uniform logits), layer_1 saturated. The global series
        # must reduce by MIN / MAX so the sick layer is not averaged away.
        n, s, b = 4, 2, 3
        saturated = (
            paddle.to_tensor(
                [[-25.7, 0.0, -100.0, -61.0], [-33.4, 0.0, -89.0, -48.0]]
                + [[-34.0, -24.9, -104.0, 0.0], [-79.4, -72.6, -104.0, 0.0]],
                dtype="float32",
            )
            .reshape([1, 1, n * n])
            .expand([s, b, n * n])
        )

        def sick_hc():
            return FakeHC(
                n=n,
                h_pre=paddle.full([s, b, n], 0.5),
                h_post=paddle.ones([s, b, n]),
                h_res=paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n]),
                h_res_logits=paddle.assign(saturated),
            )

        model = _mhc_model([self._identity_layer(n, s, b), FakeLayer(attn=sick_hc(), mlp=sick_hc())])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_softmax_min"], 0.25, places=5)
            self.assertLess(latest[f"mhc_health/layer_1/{comp}_h_res_softmax_min"], 1e-30)
            self.assertLess(latest[f"mhc_health/global_{comp}_h_res_softmax_min"], 1e-30)
            # max: uniform layer also reads 1/n, saturated layer pins at 1.0, and
            # the global series reduces by MAX.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_softmax_max"], 0.25, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_1/{comp}_h_res_softmax_max"], 1.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/global_{comp}_h_res_softmax_max"], 1.0, places=5)

    def test_composite_is_built_in_layer_order_not_call_order(self):
        # Regression test for what retired the previous composite metric: it was
        # accumulated in call order, which under recompute is the reverse-layer
        # backward replay. Here the hooks are fired in reverse on purpose.
        n, s, b = 4, 1, 1
        mats = [
            paddle.to_tensor(
                [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype="float32",
            ),
            paddle.to_tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]],
                dtype="float32",
            ),
            paddle.to_tensor(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                dtype="float32",
            ),
        ]

        def row_gain(m):
            return float(m.sum(axis=-1).abs().max())

        def col_gain(m):
            return float(m.sum(axis=-2).abs().max())

        def chain(seq, gain):
            # apply_h_res mixes with h_res^T, so the composite operator is built
            # from the transposes.
            out, acc = [], None
            for m in seq:
                op = m.transpose([1, 0])
                acc = op if acc is None else paddle.matmul(op, acc)
                out.append(gain(acc))
            return out

        expected = chain(mats, row_gain)
        expected_bwd = chain(mats, col_gain)
        # Guard against a vacuous test: the reversed product must actually differ.
        self.assertNotAlmostEqual(expected[-1], chain(list(reversed(mats)), row_gain)[-1], places=4)

        def layer_for(mat):
            def make_hc():
                return FakeHC(
                    n=n,
                    h_pre=paddle.full([s, b, n], 0.5),
                    h_post=paddle.ones([s, b, n]),
                    h_res=paddle.assign(mat.reshape([1, 1, n, n]).expand([s, b, n, n])),
                )

            return FakeLayer(attn=make_hc(), mlp=make_hc())

        model = _mhc_model([layer_for(m) for m in mats])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(list(reversed(targets)), x_dim=n * 8)  # deliberately out of order
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for idx, (want_fwd, want_bwd) in enumerate(zip(expected, expected_bwd)):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"], want_fwd, places=4
                )
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_bwd_max"], want_bwd, places=4
                )
            # Global reduces by max over layers.
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_fwd_max"], max(expected), places=4
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_bwd_max"], max(expected_bwd), places=4
            )

    def test_composite_is_one_for_identity_chain(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b) for _ in range(3)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for idx in range(3):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"], 1.0, places=5
                )

    def test_composite_skips_mtp_layers(self):
        # MTP layers are off the main trunk; multiplying them into the chain has no
        # physical meaning. Marking layer_1 as MTP must leave the layer_0 value
        # untouched and emit nothing for layer_1.
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b) for _ in range(2)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        monitor._mtp_layer_ids.add(1)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(
                latest[f"mhc_health/layer_0/{comp}_composite_amax_gain_fwd_max"], 1.0, places=5
            )
            self.assertNotIn(f"mhc_health/layer_1/{comp}_composite_amax_gain_fwd_max", latest)

    def test_composite_snapshot_is_cleared_each_step(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        self.assertTrue(monitor._h_res_snapshot)
        monitor.step()
        # Stale matrices must not survive into the next step's product.
        self.assertFalse(monitor._h_res_snapshot)

    def test_branch_residual_share_is_recorded_from_bda(self):
        n, s, b, c = 4, 2, 3, 6
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        streams = paddle.randn([s, b, n * c])
        x = paddle.randn([s, b, c])
        for _, _, mod in targets:
            mod.fused_h_res_h_post_bda(
                h_res=mod._h_res,
                original_residual=streams,
                h_post=mod._h_post,
                layer_output_with_bias=(x, None),
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for key in ("branch_residual_share", "branch_residual_share_max"):
                self.assertIn(f"mhc_health/layer_0/{comp}_{key}", latest)
            # h_post == 1 everywhere so the branch norm is √n·‖x‖ and h_res is the
            # identity, hence the share must sit strictly inside (0, 1].
            share = latest[f"mhc_health/layer_0/{comp}_branch_residual_share"]
            self.assertGreater(share, 0.0)
            self.assertLessEqual(share, 1.0)
            self.assertGreaterEqual(latest[f"mhc_health/layer_0/{comp}_branch_residual_share_max"], share)

    def test_h_post_structure_keys_recorded(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            # h_post is constant 1.0 -> equal across streams and across tokens.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_stream_concentration"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_token_std"], 0.0, places=5)

    def test_remove_hooks_restores_both_wrapped_methods(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor()
        hc = layer.self_attention_hyper_connection
        self._prepare_and_attach(monitor, model)
        self.assertIn("fused_h_res_h_post_bda", vars(hc))
        monitor.remove_hooks()
        self.assertNotIn("fused_h_res_h_post_bda", vars(hc))
        self.assertEqual(hc.fused_h_res_h_post_bda.__func__, FakeHC.fused_h_res_h_post_bda)

    def test_remove_hooks_restores_compute_mappings(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = PaddleMHCHealthMonitor()
        original = layer.self_attention_hyper_connection.compute_mappings
        self._prepare_and_attach(monitor, model)
        self.assertIsNot(layer.self_attention_hyper_connection.compute_mappings, original)
        monitor.remove_hooks()
        # falls back to the (bound) class method after deleting the instance attr
        self.assertEqual(
            layer.self_attention_hyper_connection.compute_mappings.__func__,
            FakeHC.compute_mappings,
        )
        self.assertEqual(monitor._wrapped, [])

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = paddle.eye(n).reshape([1, 1, n, n]).expand([s, b, n, n])
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = paddle.zeros([s, b, n])
        leaf.stop_gradient = False
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = paddle.assign(ident) * (leaf.sum() + 1.0)  # grad-tracked
        hc = FakeHC(n=n, h_pre=h_pre, h_post=h_post, h_res=h_res)
        layer = FakeLayer(attn=hc, mlp=hc)
        model = _mhc_model([layer])

        monitor = PaddleMHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        for _, _, mod in targets:
            mod.compute_mappings(paddle.randn([4, 2, n * 8]))

        # The 0-dim accumulators must not be graph-tracked: if the wrapper forgot
        # to detach, the graph stays pinned through them until backward.
        for key, acc in monitor._gpu_acc.items():
            self.assertTrue(acc.stop_gradient, msg=key)
        monitor.step()


class MHCMonitorNoOpTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_auto_skip_no_hc(self):
        # A plain model (no HyperConnectionTransformerLayer) -> wraps nothing.
        plain_layer = nn.Linear(4, 4)
        model = _mhc_model([plain_layer])
        monitor_dict = {}
        mhc_monitor.setup_mhc_monitor(model, monitor_dict=monitor_dict)
        self.assertEqual(monitor_dict, {})

    def test_no_op_when_unimportable(self):
        # Simulate mHC classes not importable -> setup is a total no-op, no raise.
        orig_layer = mhc_monitor.HyperConnectionTransformerLayer
        orig_mod = mhc_monitor.HyperConnectionModule
        mhc_monitor.HyperConnectionTransformerLayer = None
        mhc_monitor.HyperConnectionModule = None
        try:
            model = _mhc_model([nn.Linear(4, 4)])
            monitor_dict = {}
            returned = mhc_monitor.setup_mhc_monitor(model, monitor_dict=monitor_dict)
            self.assertIs(returned, model)
            self.assertEqual(monitor_dict, {})
        finally:
            mhc_monitor.HyperConnectionTransformerLayer = orig_layer
            mhc_monitor.HyperConnectionModule = orig_mod


if __name__ == "__main__":
    unittest.main()
