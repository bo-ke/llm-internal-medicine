import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

mhc_metrics = importlib.import_module("internal_medicine.backends.megatron.mhc_metrics")
mhc_monitor = importlib.import_module("internal_medicine.backends.megatron.mhc_monitor")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs

MHCHealthMonitor = mhc_monitor.MHCHealthMonitor


class FakeHC(nn.Module):
    """Stand-in for HyperConnectionModule exposing the three wrapped methods."""

    def __init__(self, n, h_pre, h_post, h_res, h_res_logits=None):
        super().__init__()
        self.n = n
        self._h_pre = h_pre
        self._h_post = h_post
        self._h_res = h_res
        # Raw pre-Sinkhorn mixing logits. Zeros -> uniform rows -> every column
        # sum is exactly 1, i.e. the healthy case.
        if h_res_logits is None:
            h_res_logits = torch.zeros(*h_pre.shape[:-1], n * n)
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
        leading = list(original_residual.shape[:-1])
        C = original_residual.shape[-1] // n
        mixed = torch.bmm(
            h_res.reshape(-1, n, n).transpose(1, 2),
            original_residual.reshape(-1, n, C),
        ).reshape(*leading, n, C)
        xb = x if bias is None else x + bias
        return (h_post.unsqueeze(-1) * xb.unsqueeze(-2) + mixed).reshape(*leading, n * C)

    def forward(self, hidden_states):  # pragma: no cover - unused
        return hidden_states, self._h_res, self._h_post


class FakeLayer(nn.Module):
    """Stand-in for HyperConnectionTransformerLayer with the two hc modules."""

    def __init__(self, attn, mlp, layer_number=1):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention_hyper_connection = attn
        self.mlp_hyper_connection = mlp


def _mhc_model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.ModuleList(layers)))


class MHCMetricsTest(unittest.TestCase):
    def test_amax_gain_row_vs_col(self):
        # [[1,2],[3,4]] -> row sums [3,7] -> max abs 7; col sums [4,6] -> 6.
        m = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.assertAlmostEqual(mhc_metrics.amax_gain(m, dim=-1).item(), 7.0, places=5)
        self.assertAlmostEqual(mhc_metrics.amax_gain(m, dim=-2).item(), 6.0, places=5)

    def test_doubly_stochastic_gain_is_one(self):
        n = 4
        ident = torch.eye(n).reshape(1, 1, n, n)
        self.assertAlmostEqual(mhc_metrics.amax_gain(ident, dim=-1).item(), 1.0, places=5)
        self.assertAlmostEqual(mhc_metrics.amax_gain(ident, dim=-2).item(), 1.0, places=5)

    def test_gate_stats(self):
        h = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        mean, std = mhc_metrics.gate_stats(h)
        self.assertAlmostEqual(mean.item(), 1.5, places=5)
        self.assertAlmostEqual(std.item(), h.std().item(), places=6)

    def test_stream_concentration_bounds(self):
        # Uniform across streams -> 1 (lower bound); all mass on one stream -> n.
        uniform = torch.full((3, 4), 0.7)
        one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
        self.assertAlmostEqual(
            mhc_metrics.h_post_structure_stats(uniform)["h_post_stream_concentration"].item(),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            mhc_metrics.h_post_structure_stats(one_hot)["h_post_stream_concentration"].item(),
            4.0,
            places=5,
        )

    def test_token_std_zero_when_gate_is_constant(self):
        const = torch.full((8, 4), 1.0)
        varying = torch.tensor([[0.2] * 4, [1.8] * 4])
        self.assertAlmostEqual(mhc_metrics.h_post_structure_stats(const)["h_post_token_std"].item(), 0.0, places=6)
        self.assertGreater(mhc_metrics.h_post_structure_stats(varying)["h_post_token_std"].item(), 0.5)

    def test_branch_residual_share_against_explicit_terms(self):
        # h_res = identity -> residual term is exactly the incoming streams, so
        # the share reduces to b / (b + ‖x_l‖_F) with b = ‖h_post‖₂·‖x‖₂.
        n, c = 2, 3
        h_res = torch.eye(n).reshape(1, n, n)
        streams = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        h_post = torch.tensor([[3.0, 4.0]])  # ‖·‖₂ = 5
        x = torch.tensor([[0.0, 0.0, 2.0]])  # ‖·‖₂ = 2
        stats = mhc_metrics.branch_residual_share(h_res, streams.reshape(1, n * c), h_post, x)
        branch = 5.0 * 2.0
        residual = 5.0**0.5  # ‖streams‖_F = √5
        expected = branch / (branch + residual)
        self.assertAlmostEqual(stats["branch_residual_share"].item(), expected, places=4)
        self.assertAlmostEqual(stats["branch_residual_share_max"].item(), expected, places=4)

    def test_branch_residual_share_zero_gate_kills_branch(self):
        n, c = 4, 5
        h_res = torch.eye(n).reshape(1, n, n)
        streams = torch.randn(1, n * c)
        x = torch.randn(1, c)
        stats = mhc_metrics.branch_residual_share(h_res, streams, torch.zeros(1, n), x)
        self.assertAlmostEqual(stats["branch_residual_share"].item(), 0.0, places=6)

    def test_branch_residual_share_zero_residual_token_stays_bounded(self):
        # An all-zero token (padding) drives the residual norm to 0. The raw
        # ratio would hit the epsilon floor and return ~1e6, hijacking the token
        # mean; the bounded share must read 1.0 instead.
        n, c = 2, 3
        h_res = torch.eye(n).reshape(1, n, n).expand(2, n, n)
        streams = torch.cat([torch.ones(1, n * c), torch.zeros(1, n * c)])
        stats = mhc_metrics.branch_residual_share(h_res, streams, torch.ones(2, n), torch.ones(2, c))
        self.assertAlmostEqual(stats["branch_residual_share_max"].item(), 1.0, places=6)
        self.assertLessEqual(stats["branch_residual_share"].item(), 1.0)
        self.assertGreater(stats["branch_residual_share"].item(), 0.5)

    def test_branch_residual_share_all_zero_token_is_zero(self):
        # Both terms zero must not divide by zero; the eps guard makes it 0.
        n, c = 2, 3
        stats = mhc_metrics.branch_residual_share(
            torch.zeros(1, n, n), torch.zeros(1, n * c), torch.zeros(1, n), torch.zeros(1, c)
        )
        self.assertAlmostEqual(stats["branch_residual_share"].item(), 0.0, places=6)

    def test_logits_extrema_preserve_signed_raw_range(self):
        logits = torch.tensor([[-200.0, -18.0, 0.0, 7.5]])
        stats = mhc_metrics.h_res_logits_extrema(logits)
        self.assertAlmostEqual(stats["h_res_logits_min"].item(), -200.0, places=6)
        self.assertAlmostEqual(stats["h_res_logits_max"].item(), 7.5, places=6)

    def test_logits_extrema_are_detached_fp32_scalars(self):
        logits = torch.tensor([[-3.0, 2.0]], dtype=torch.float16, requires_grad=True)
        stats = mhc_metrics.h_res_logits_extrema(logits)
        for value in stats.values():
            self.assertEqual(tuple(value.shape), ())
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)


class _MHCFixture:
    """Shared setup for the monitor test cases (mixed into TestCase subclasses)."""

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

    def _identity_layer(self, n, s, b, layer_number=1):
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()

        def make_hc():
            return FakeHC(
                n=n,
                h_pre=torch.full((s, b, n), 0.5),
                h_post=torch.ones(s, b, n),
                h_res=ident.clone(),  # own storage, not a view of the shared eye
            )

        return FakeLayer(attn=make_hc(), mlp=make_hc(), layer_number=layer_number)

    def _drive(self, targets, x_dim):
        # Fire each wrapped compute_mappings (grad enabled -> _should_monitor passes).
        for _, _, mod in targets:
            mod.compute_mappings(torch.randn(4, 2, x_dim))

    def _prepare_and_attach(self, monitor, model):
        targets = monitor._prepare_layers(model)
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(targets)
        return targets


class MHCGateTest(_MHCFixture, unittest.TestCase):
    def test_gate_and_amax_metrics_are_recorded(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
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
        # `out = h_res^T @ x` (see HyperConnectionModule.apply_h_res), so _fwd
        # must read the COLUMN sums of h_res (1.5 here) and _bwd the ROW sums
        # (1.0 here) — single layer and composite agree. Identity matrices cannot
        # tell the two apart, which is how the fwd/bwd label mix-up stayed
        # invisible on this backend.
        n, s, b = 2, 1, 1
        h_res = torch.tensor([[0.75, 0.25], [0.75, 0.25]]).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        hc = FakeHC(n=n, h_pre=torch.full((s, b, n), 0.5), h_post=torch.ones(s, b, n), h_res=h_res)
        model = _mhc_model([FakeLayer(attn=hc, mlp=hc)])

        monitor = MHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest["mhc_health/global_attn_amax_gain_fwd"], 1.5, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_amax_gain_bwd"], 1.0, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_composite_amax_gain_fwd_max"], 1.5, places=5)
        self.assertAlmostEqual(latest["mhc_health/global_attn_composite_amax_gain_bwd_max"], 1.0, places=5)

    def test_h_post_structure_keys_recorded(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            # h_post is constant 1.0 -> equal across streams and across tokens.
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_stream_concentration"], 1.0, places=4)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_post_token_std"], 0.0, places=5)


class MHCLogitsTest(_MHCFixture, unittest.TestCase):
    def test_logits_gradient_extrema_are_unscaled_without_changing_backward(self):
        n, s, b = 2, 1, 1
        source = torch.zeros(s, b, n * n, requires_grad=True)
        logits = source * 1.0
        hc = FakeHC(
            n=n,
            h_pre=torch.full((s, b, n), 0.5),
            h_post=torch.ones(s, b, n),
            h_res=torch.eye(n).reshape(s, b, n, n),
            h_res_logits=logits,
        )
        model = _mhc_model([FakeLayer(attn=hc, mlp=self._identity_layer(n, s, b).mlp_hyper_connection)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        self._prepare_and_attach(monitor, model)

        _, _, captured = hc._compute_h(None, None)
        weight = torch.tensor([[[2.0, -7.0, 4.0, 1.5]]])
        loss_scale = 16.0
        (captured * weight * loss_scale).sum().backward()

        # The hook observes but never modifies the autograd gradient.
        self.assertTrue(torch.allclose(source.grad, weight * loss_scale))
        monitor.finalize_scaled_grad_metrics(SimpleNamespace(scale=torch.tensor(loss_scale)))
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for key in (
            "mhc_health/layer_0/attn_h_res_logits_grad_min",
            "mhc_health/global_attn_h_res_logits_grad_min",
        ):
            self.assertAlmostEqual(latest[key], -7.0, places=5)
        for key in (
            "mhc_health/layer_0/attn_h_res_logits_grad_max",
            "mhc_health/global_attn_h_res_logits_grad_max",
        ):
            self.assertAlmostEqual(latest[key], 4.0, places=5)

    def test_logits_gradient_hook_skips_no_grad_recompute_forward(self):
        n, s, b = 2, 1, 1
        source = torch.zeros(s, b, n * n, requires_grad=True)
        hc = FakeHC(
            n=n,
            h_pre=torch.full((s, b, n), 0.5),
            h_post=torch.ones(s, b, n),
            h_res=torch.eye(n).reshape(s, b, n, n),
            h_res_logits=source * 1.0,
        )
        model = _mhc_model([FakeLayer(attn=hc, mlp=self._identity_layer(n, s, b).mlp_hyper_connection)])
        monitor = MHCHealthMonitor()
        self._prepare_and_attach(monitor, model)

        with torch.no_grad():
            hc._compute_h(None, None)
        min_key = "mhc_health/layer_0/attn_h_res_logits_grad_min"
        max_key = "mhc_health/layer_0/attn_h_res_logits_grad_max"
        self.assertEqual(monitor._gpu_cnt[min_key], 0)
        self.assertEqual(monitor._gpu_cnt[max_key], 0)

        _, _, replay_logits = hc._compute_h(None, None)
        replay_logits.sum().backward()
        self.assertEqual(monitor._gpu_cnt[min_key], 1)
        self.assertEqual(monitor._gpu_cnt[max_key], 1)
        monitor.step()
        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest[min_key], 1.0, places=5)
        self.assertAlmostEqual(latest[max_key], 1.0, places=5)

    def test_logits_extrema_are_recorded_and_reduce_by_extremum_across_layers(self):
        # layer_0 has zero logits; layer_1 spans [-104, 7.5]. The global series
        # must reduce by MIN / MAX so an extreme layer is not averaged away.
        n, s, b = 4, 2, 3
        extreme = (
            torch.tensor(
                [[-25.7, 7.5, -100.0, -61.0], [-33.4, 0.0, -89.0, -48.0]]
                + [[-34.0, -24.9, -104.0, 0.0], [-79.4, -72.6, -104.0, 0.0]]
            )
            .reshape(1, 1, n * n)
            .expand(s, b, n * n)
        )

        def sick_hc():
            return FakeHC(
                n=n,
                h_pre=torch.full((s, b, n), 0.5),
                h_post=torch.ones(s, b, n),
                h_res=torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous(),
                h_res_logits=extreme.clone(),
            )

        model = _mhc_model(
            [
                self._identity_layer(n, s, b, layer_number=1),
                FakeLayer(attn=sick_hc(), mlp=sick_hc(), layer_number=2),
            ]
        )

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_logits_min"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_1/{comp}_h_res_logits_min"], -104.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/global_{comp}_h_res_logits_min"], -104.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_0/{comp}_h_res_logits_max"], 0.0, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/layer_1/{comp}_h_res_logits_max"], 7.5, places=5)
            self.assertAlmostEqual(latest[f"mhc_health/global_{comp}_h_res_logits_max"], 7.5, places=5)


class MHCCompositeTest(_MHCFixture, unittest.TestCase):
    def test_composite_is_built_in_layer_order_not_call_order(self):
        # The composite must be a layer-ordered product: under recompute the
        # hooks fire in the reverse-layer backward replay, so call order is not
        # layer order. Here the hooks are fired in reverse on purpose.
        n, s, b = 4, 1, 1
        mats = [
            torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ]

        def row_gain(m):
            return m.sum(dim=-1).abs().max().item()

        def col_gain(m):
            return m.sum(dim=-2).abs().max().item()

        components = ("attn", "mlp")
        branches = [
            (layer_idx, component, mat.transpose(0, 1))
            for layer_idx, mat in enumerate(mats)
            for component in components
        ]

        expected_fwd = {}
        prefix = None
        for layer_idx, component, op in branches:
            prefix = op if prefix is None else torch.matmul(op, prefix)
            expected_fwd[(layer_idx, component)] = row_gain(prefix)

        expected_bwd = {}
        suffix = None
        for layer_idx, component, op in reversed(branches):
            suffix = op if suffix is None else torch.matmul(suffix, op)
            expected_bwd[(layer_idx, component)] = col_gain(suffix)
        # Guard against a vacuous test: reversing the physical branch order differs.
        wrong_prefix = None
        for _, _, op in reversed(branches):
            wrong_prefix = op if wrong_prefix is None else torch.matmul(op, wrong_prefix)
        self.assertNotAlmostEqual(expected_fwd[(2, "mlp")], row_gain(wrong_prefix), places=4)

        def layer_for(mat, layer_number):
            def make_hc():
                return FakeHC(
                    n=n,
                    h_pre=torch.full((s, b, n), 0.5),
                    h_post=torch.ones(s, b, n),
                    h_res=mat.reshape(1, 1, n, n).expand(s, b, n, n).contiguous(),
                )

            return FakeLayer(attn=make_hc(), mlp=make_hc(), layer_number=layer_number)

        model = _mhc_model([layer_for(m, i + 1) for i, m in enumerate(mats)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(list(reversed(targets)), x_dim=n * 8)  # deliberately out of order
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in components:
            for idx in range(len(mats)):
                key = (idx, comp)
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"],
                    expected_fwd[key],
                    places=4,
                )
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_bwd_max"],
                    expected_bwd[key],
                    places=4,
                )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_fwd_max"],
                max(value for (idx, component), value in expected_fwd.items() if component == comp),
                places=4,
            )
            self.assertAlmostEqual(
                latest[f"mhc_health/global_{comp}_composite_amax_gain_bwd_max"],
                max(value for (idx, component), value in expected_bwd.items() if component == comp),
                places=4,
            )

    def test_composite_is_one_for_identity_chain(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b, layer_number=i + 1) for i in range(3)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        for comp in ("attn", "mlp"):
            for idx in range(3):
                self.assertAlmostEqual(
                    latest[f"mhc_health/layer_{idx}/{comp}_composite_amax_gain_fwd_max"], 1.0, places=5
                )

    def test_composite_snapshot_is_cleared_each_step(self):
        n, s, b = 4, 2, 3
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)
        self._drive(targets, x_dim=n * 8)
        self.assertTrue(monitor._h_res_snapshot)
        monitor.step()
        # Stale matrices must not survive into the next step's product.
        self.assertFalse(monitor._h_res_snapshot)

    def test_composite_aggregates_each_microbatch_before_snapshot_overwrite(self):
        n, s, b = 2, 1, 1
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        amplified = torch.diag(torch.tensor([3.0, 1.0])).reshape(1, 1, n, n)
        for _, _, mod in targets:
            mod._h_res = amplified
        self._drive(targets, x_dim=n * 8)
        monitor.finalize_composite_microbatch()
        self.assertFalse(monitor._h_res_snapshot)

        identity = torch.eye(n).reshape(1, 1, n, n)
        for _, _, mod in targets:
            mod._h_res = identity
        self._drive(targets, x_dim=n * 8)
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertAlmostEqual(latest["mhc_health/layer_0/attn_composite_amax_gain_fwd_max"], 3.0, places=5)
        self.assertAlmostEqual(latest["mhc_health/layer_0/mlp_composite_amax_gain_fwd_max"], 9.0, places=5)
        self.assertAlmostEqual(latest["mhc_health/layer_0/attn_composite_amax_gain_bwd_max"], 9.0, places=5)
        self.assertAlmostEqual(latest["mhc_health/layer_0/mlp_composite_amax_gain_bwd_max"], 3.0, places=5)


class MHCTeardownTest(_MHCFixture, unittest.TestCase):
    def test_branch_residual_share_is_recorded_from_bda(self):
        n, s, b, c = 4, 2, 3, 6
        model = _mhc_model([self._identity_layer(n, s, b)])

        monitor = MHCHealthMonitor(log_per_layer=True, log_global=True)
        targets = self._prepare_and_attach(monitor, model)

        streams = torch.randn(s, b, n * c)
        x = torch.randn(s, b, c)
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

    def test_branch_residual_share_is_recorded_from_positional_bda_call(self):
        # Megatron's TransformerLayer passes these positionally; the keyword-with-
        # positional-fallback reader must cover both.
        n, s, b, c = 4, 1, 1, 5
        model = _mhc_model([self._identity_layer(n, s, b)])
        monitor = MHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)

        for _, _, mod in targets:
            mod.fused_h_res_h_post_bda(
                mod._h_res, torch.randn(s, b, n * c), mod._h_post, (torch.randn(s, b, c), None), 0.0, True, False
            )
        monitor.step()

        latest = training_logs.get_latest(prefix="mhc_health")
        self.assertIn("mhc_health/layer_0/attn_branch_residual_share", latest)

    def test_remove_hooks_restores_all_wrapped_methods(self):
        n, s, b = 4, 2, 3
        layer = self._identity_layer(n, s, b)
        model = _mhc_model([layer])
        monitor = MHCHealthMonitor()
        hc = layer.self_attention_hyper_connection
        original = hc.compute_mappings
        self._prepare_and_attach(monitor, model)
        self.assertIsNot(hc.compute_mappings, original)
        self.assertIn("fused_h_res_h_post_bda", vars(hc))
        monitor.remove_hooks()
        self.assertNotIn("fused_h_res_h_post_bda", vars(hc))
        # falls back to the (bound) class methods after deleting the instance attrs
        self.assertEqual(hc.compute_mappings.__func__, FakeHC.compute_mappings)
        self.assertEqual(hc._compute_h.__func__, FakeHC._compute_h)
        self.assertEqual(hc.fused_h_res_h_post_bda.__func__, FakeHC.fused_h_res_h_post_bda)
        self.assertEqual(monitor._wrapped, [])
        self.assertEqual(monitor._h_res_snapshot, {})

    def test_no_graph_retention(self):
        n, s, b = 4, 2, 3
        ident = torch.eye(n).reshape(1, 1, n, n).expand(s, b, n, n).contiguous()
        # Outputs attached to a graph (require grad) — the wrapper must detach.
        leaf = torch.zeros(s, b, n, requires_grad=True)
        h_pre = leaf + 0.5
        h_post = leaf + 1.0
        h_res = ident * (leaf.sum() + 1.0)  # requires_grad, has grad_fn
        hc = FakeHC(n=n, h_pre=h_pre, h_post=h_post, h_res=h_res)
        model = _mhc_model([FakeLayer(attn=hc, mlp=hc)])

        monitor = MHCHealthMonitor()
        targets = self._prepare_and_attach(monitor, model)
        for _, _, mod in targets:
            mod.compute_mappings(torch.randn(4, 2, n * 8))

        # The 0-dim accumulators and the composite snapshots must not be graph-
        # tracked: if the wrapper forgot to detach, the graph stays pinned
        # through them until backward.
        for key, acc in monitor._gpu_acc.items():
            self.assertFalse(acc.requires_grad, msg=key)
        for key, mat in monitor._h_res_snapshot.items():
            self.assertFalse(mat.requires_grad, msg=str(key))
            self.assertIsNone(mat.grad_fn, msg=str(key))
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
