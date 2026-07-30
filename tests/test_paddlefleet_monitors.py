import importlib
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

PaddleMassiveActivationMonitor = importlib.import_module(
    "internal_medicine.backends.paddlefleet.massive_activation_monitor"
).PaddleMassiveActivationMonitor
moe_monitor_module = importlib.import_module("internal_medicine.backends.paddlefleet.moe_monitor")
PaddleMoEMonitor = moe_monitor_module.PaddleMoEMonitor
PaddleQKStatsMonitor = importlib.import_module("internal_medicine.backends.paddlefleet.qk_monitor").PaddleQKStatsMonitor
paddlefleet_backend = importlib.import_module("internal_medicine.backends.paddlefleet")
layer_discovery = importlib.import_module("internal_medicine.backends.paddlefleet.layer_discovery")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


class BrokenPaddleMoELayer:
    training = True

    @property
    def grouped_gemm_experts(self):
        raise RuntimeError("grouped expert read failed")


class DummyHook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyMonitor:
    def __init__(self):
        self.hooks = [DummyHook()]
        self.removed = False

    def remove_hooks(self):
        self.removed = True
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class PaddleFleetSetupTest(unittest.TestCase):
    def setUp(self):
        self.original_monitor_map = dict(paddlefleet_backend._MONITOR_MAP)

    def tearDown(self):
        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update(self.original_monitor_map)

    def test_setup_monitors_reuses_existing_monitor_for_same_config(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=first, monitor_interval=2)
        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=second, monitor_interval=2)

        self.assertEqual(len(created), 1)
        self.assertIs(second["dummy"], first["dummy"])
        self.assertFalse(first["dummy"].removed)

    def test_setup_monitors_replaces_existing_monitor_when_config_changes(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=first, monitor_interval=1)
        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=second, monitor_interval=2)

        self.assertEqual(len(created), 2)
        self.assertTrue(first["dummy"].removed)
        self.assertIs(second["dummy"], created[1])

    def test_all_contains_only_supported_forward_monitors(self):
        self.assertEqual(
            set(paddlefleet_backend._MONITOR_MAP),
            {"qk_stats", "moe_health", "massive_act", "mhc_health"},
        )


class PaddleLayerDiscoveryTest(unittest.TestCase):
    def test_get_decoder_layers_flattens_virtual_pipeline_chunks(self):
        layer0 = SimpleNamespace(layer_idx=0)
        layer1 = SimpleNamespace(layer_idx=1)
        layer2 = SimpleNamespace(layer_idx=2)
        model = SimpleNamespace(
            _model_chunks=[
                SimpleNamespace(run_function=[layer0, layer1]),
                SimpleNamespace(run_function=[layer2]),
            ]
        )

        self.assertEqual(layer_discovery.get_decoder_layers(model), [layer0, layer1, layer2])

    def test_get_decoder_layers_checks_wrapped_module_after_empty_layers_wrapper(self):
        layer = SimpleNamespace(layer_idx=0)
        model = SimpleNamespace(_layers=SimpleNamespace(), module=SimpleNamespace(run_function=[layer]))

        self.assertEqual(layer_discovery.get_decoder_layers(model), [layer])

    def test_classify_attn_type_reads_is_swa_flag(self):
        """classify_attn_type returns swa/full based on self_attn.is_swa, None if absent."""
        swa_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=True))
        full_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        untagged_layer = SimpleNamespace(self_attn=SimpleNamespace())  # no is_swa
        no_attn_layer = SimpleNamespace()  # no self_attn at all

        self.assertEqual(layer_discovery.classify_attn_type(swa_layer), "swa")
        self.assertEqual(layer_discovery.classify_attn_type(full_layer), "full")
        self.assertIsNone(layer_discovery.classify_attn_type(untagged_layer))
        self.assertIsNone(layer_discovery.classify_attn_type(no_attn_layer))

    def test_classify_attn_type_falls_back_to_self_attention(self):
        """Some models expose the module as ``self_attention`` instead of ``self_attn``."""
        swa_layer = SimpleNamespace(self_attention=SimpleNamespace(is_swa=True))
        self.assertEqual(layer_discovery.classify_attn_type(swa_layer), "swa")

    def test_iter_monitor_layers_populates_attn_type_for_swa_and_full(self):
        """MonitorLayer.attn_type is populated per layer for models that mix SWA and full."""
        # Simulate ernie-lite: layer 0 full, layer 1 SWA, layer 2 full
        full_layer_a = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        swa_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=True))
        full_layer_b = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))

        result = layer_discovery.iter_monitor_layers(
            [full_layer_a, swa_layer, full_layer_b], lambda layer: hasattr(layer, "self_attn")
        )

        self.assertEqual([item.attn_type for item in result], ["full", "swa", "full"])

    def test_iter_monitor_layers_attn_type_is_none_for_backward_compat_models(self):
        """Models without an is_swa flag (eb5_v2 / dsv4) must yield attn_type=None."""
        layer = SimpleNamespace(self_attn=SimpleNamespace())  # no is_swa
        result = layer_discovery.iter_monitor_layers([layer], lambda x: hasattr(x, "self_attn"))
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].attn_type)

    def test_iter_monitor_layers_marks_unwrapped_mtp_layer(self):
        main_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        mtp_layer = SimpleNamespace(self_attn=SimpleNamespace(is_swa=False))
        mtp_wrapper = SimpleNamespace(transformer_layer=mtp_layer)

        result = layer_discovery.iter_monitor_layers(
            [main_layer, mtp_wrapper],
            lambda layer: hasattr(layer, "self_attn"),
        )

        self.assertEqual([item.idx for item in result], [0, 1])
        self.assertEqual([item.is_mtp for item in result], [False, True])


class PaddleMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_gpu_buffer_gate_metrics_recorded(self):
        """Gate hook records metrics via GPU-buffer API without D2H sync."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        # Manually declare and allocate for layer 0
        for m in ["router_entropy", "score_sum_mean", "score_sum_min", "score_sum_max"]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        # Simulate recording
        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/layer_0/router_entropy"], 2.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_router_entropy"], 2.0, places=4)

    def test_hard_load_distribution_metrics_distinguish_balance(self):
        balanced = moe_monitor_module._distribution_metrics(paddle.to_tensor([1.0, 1.0, 1.0, 1.0]))
        imbalanced = moe_monitor_module._distribution_metrics(paddle.to_tensor([4.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(float(balanced["cv"]), 0.0, places=6)
        self.assertAlmostEqual(float(balanced["entropy_norm"]), 1.0, places=6)
        self.assertGreater(float(imbalanced["cv"]), float(balanced["cv"]))
        self.assertLess(float(imbalanced["entropy_norm"]), float(balanced["entropy_norm"]))

    def test_moe_gate_metrics_ignore_all_invalid_expert_ids(self):
        probabilities = paddle.to_tensor([[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]])
        invalid_assignments = paddle.to_tensor([[[-1, 3], [4, -2]]], dtype="int64")
        assignment, gate_mass = moe_monitor_module._assignment_mass(probabilities, invalid_assignments)

        self.assertEqual(float(assignment.sum()), 0.0)
        self.assertEqual(float(gate_mass.sum()), 0.0)

    def test_mtp_layer_marker_is_encoded_in_metric_key(self):
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        monitor.mark_mtp_layers([1])
        monitor.declare_layer_metric(0, "router_entropy")
        monitor.declare_layer_metric(1, "router_entropy")
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(4.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/layer_0/router_entropy"], 2.0, places=4)
        self.assertAlmostEqual(latest["moe_health/layer_1_mtp/router_entropy"], 4.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_router_entropy"], 3.0, places=4)

    def test_gpu_buffer_multi_layer_global_aggregation(self):
        """Global metrics are derived from layer accumulators at flush time."""
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True)
        for layer_idx in (0, 1):
            for m in ["router_entropy", "score_sum_max"]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(4.0))
        monitor.record_layer_metric(0, "score_sum_max", paddle.to_tensor(0.8))
        monitor.record_layer_metric(1, "score_sum_max", paddle.to_tensor(0.9))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/global_router_entropy"], 3.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_score_sum_max"], 0.9, places=4)

    def test_attn_type_tag_produces_typed_keys_and_split_global_aggregation(self):
        """When declare/record_layer_metric receives attn_type, keys are
        ``{prefix}/layer_N/{type}_{metric}`` and ``{prefix}/global_{type}_{metric}``.
        SWA and full aggregate independently -- window vs full never mix.

        MoE monitor is used here as a convenient concrete PaddleProbe subclass;
        in production attn_type only flows through attention-family monitors.
        """
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        # layer 0 is full, layer 1 is swa, layer 2 is full
        monitor.declare_layer_metric(0, "router_entropy", attn_type="full")
        monitor.declare_layer_metric(1, "router_entropy", attn_type="swa")
        monitor.declare_layer_metric(2, "router_entropy", attn_type="full")
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0), attn_type="full")
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(10.0), attn_type="swa")
        monitor.record_layer_metric(2, "router_entropy", paddle.to_tensor(4.0), attn_type="full")
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        # Per-layer keys carry the attn_type prefix on the metric name.
        self.assertAlmostEqual(latest["moe_health/layer_0/full_router_entropy"], 2.0, places=4)
        self.assertAlmostEqual(latest["moe_health/layer_1/swa_router_entropy"], 10.0, places=4)
        self.assertAlmostEqual(latest["moe_health/layer_2/full_router_entropy"], 4.0, places=4)
        # Global aggregations split by attn_type: SWA global uses only layer 1;
        # full global averages layer 0 and layer 2. There is no untagged
        # ``global_router_entropy`` when all layers are typed.
        self.assertAlmostEqual(latest["moe_health/global_swa_router_entropy"], 10.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_full_router_entropy"], 3.0, places=4)
        self.assertNotIn("moe_health/global_router_entropy", latest)

    def test_fused_expert_norm_matches_per_expert_concat_norm(self):
        """Vectorized per-expert norm equals the old concat([w1_i, w2_i]).norm()."""
        moe_monitor_mod = importlib.import_module("internal_medicine.backends.paddlefleet.moe_monitor")
        num_experts, h, i = 4, 8, 6
        w1 = paddle.randn([num_experts, h, i], dtype="float32")
        w2 = paddle.randn([num_experts, i, h], dtype="float32")

        got = moe_monitor_mod._per_expert_stacked_norms(w1, w2)
        expected = paddle.stack([paddle.concat([w1[e].flatten(), w2[e].flatten()]).norm() for e in range(num_experts)])
        self.assertEqual(list(got.shape), [num_experts])
        self.assertTrue(bool(paddle.allclose(got, expected, atol=1e-5)))

    def test_collect_expert_norms_fused_layout_records_metrics(self):
        """collect_expert_norms records expert + shared norms for a fused-expert MoE layer."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True, verbose=False)
        for m in [
            "expert_norm_mean",
            "expert_norm_std",
            "expert_norm_min",
            "expert_norm_max",
            "shared_expert_norm",
            "shared_routed_ratio",
        ]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        num_experts, h, i = 3, 8, 6
        fused_experts = SimpleNamespace(
            up_gate_proj=SimpleNamespace(weight=paddle.randn([num_experts, h, i], dtype="float32")),
            down_proj=SimpleNamespace(weight=paddle.randn([num_experts, i, h], dtype="float32")),
        )
        shared = nn.Linear(h, i)
        moe_layer = SimpleNamespace(experts=fused_experts, shared_experts=shared)
        moe_layer.grouped_gemm_experts = None

        monitor._expert_norm_layers = [(0, moe_layer)]
        monitor.collect_expert_norms()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/layer_0/shared_expert_norm", latest)
        self.assertGreater(latest["moe_health/layer_0/expert_norm_mean"], 0.0)

    def test_collect_expert_norms_respects_monitor_interval(self):
        """Expert-norm collection is gated by the global monitor_interval."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True, monitor_interval=2, verbose=False)
        for m in ["expert_norm_mean", "expert_norm_std", "expert_norm_min", "expert_norm_max"]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        num_experts, h, i = 2, 4, 4
        fused_experts = SimpleNamespace(
            up_gate_proj=SimpleNamespace(weight=paddle.randn([num_experts, h, i], dtype="float32")),
            down_proj=SimpleNamespace(weight=paddle.randn([num_experts, i, h], dtype="float32")),
        )
        moe_layer = SimpleNamespace(experts=fused_experts, shared_experts=None)
        moe_layer.grouped_gemm_experts = None
        monitor._expert_norm_layers = [(0, moe_layer)]

        # step_count=1 -> 1 % 2 != 0 -> should NOT record
        monitor.step_count = 1
        monitor.collect_expert_norms()
        # nothing recorded -> count stays 0, flush emits nothing for this key
        self.assertEqual(monitor._gpu_cnt["moe_health/layer_0/expert_norm_mean"], 0)

    def test_expert_norm_collect_exception_does_not_crash(self):
        """collect_expert_norms swallows per-layer read errors without crashing the step."""
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, verbose=False)
        for m in ["expert_norm_mean", "expert_norm_std", "expert_norm_min", "expert_norm_max"]:
            monitor.declare_layer_metric(2, m)
        monitor.allocate_buffers()

        # Expert norms are collected at step-begin from _expert_norm_layers,
        # not from a forward hook. A layer that raises on weight access must be
        # caught so the step still completes.
        monitor._expert_norm_layers = [(2, BrokenPaddleMoELayer())]
        monitor.collect_expert_norms()

        # Should not crash; step should still work
        monitor.step()

    def test_hash_routing_cache_supports_sqrtsoftplus(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, verbose=False)
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")

        def original_hash_routing(logits, flat_ids):
            return "ok"

        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=original_hash_routing,
            scoring_func="sqrtsoftplus",
        )
        monitor._patch_gate_cache(gate)

        self.assertEqual(gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")), "ok")
        expected = paddle.sqrt(paddle.nn.functional.softplus(logits) + 1e-20)
        self.assertTrue(bool(paddle.allclose(gate._cached_gates, expected)))

        monitor.remove_hooks()
        self.assertIs(gate._hash_routing, original_hash_routing)

    def test_hash_routing_cache_respects_monitor_interval(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, monitor_interval=2, verbose=False)
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")
        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=lambda logits, flat_ids: "ok",
            scoring_func="sigmoid",
        )
        monitor._patch_gate_cache(gate)
        monitor.step_count = 1

        self.assertEqual(gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")), "ok")
        self.assertIsNone(gate._cached_gates)


class PaddleMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_extract_hidden_states_supports_dict_and_positional_inputs(self):
        monitor = PaddleMassiveActivationMonitor()
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        self.assertIs(monitor._extract_hidden_states(({"hidden_states": hidden_states},)), hidden_states)
        self.assertIs(monitor._extract_hidden_states((hidden_states,)), hidden_states)
        self.assertIsNone(monitor._extract_hidden_states(()))
        self.assertIsNone(monitor._extract_hidden_states(({"other": hidden_states},)))

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = paddle.to_tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ],
            dtype="float32",
        )

        # Must declare + allocate before _compute_and_log
        metric_names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "topk_channel_norm",
            "activation_rms",
            "post_norm_sparsity",
            "post_norm_cosine",
            "channel_count_gt_2",
            "channel_count_gt_3",
        ]
        for m in metric_names:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        monitor._compute_and_log(0, hidden_states, nn.Layer())
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in (
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "channel_count_gt_2",
            "channel_count_gt_3",
            "topk_channel_norm",
            "activation_rms",
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)

    def test_count_metrics_are_max_aggregated_across_layers(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=False,
            log_global=True,
            absolute_thresholds=(2.0, 3.0),
        )

        # Declare and allocate for 2 layers
        for layer_idx in (0, 1):
            for m in ["massive_act_channel_count", "channel_count_gt_2", "channel_count_gt_3"]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "massive_act_channel_count", paddle.to_tensor(0.0))
        monitor.record_layer_metric(0, "channel_count_gt_2", paddle.to_tensor(1.0))
        monitor.record_layer_metric(0, "channel_count_gt_3", paddle.to_tensor(0.0))
        monitor.record_layer_metric(1, "massive_act_channel_count", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "channel_count_gt_2", paddle.to_tensor(3.0))
        monitor.record_layer_metric(1, "channel_count_gt_3", paddle.to_tensor(1.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertEqual(latest["massive_act/global_massive_act_channel_count"], 2.0)
        self.assertEqual(latest["massive_act/global_channel_count_gt_2"], 3.0)
        self.assertEqual(latest["massive_act/global_channel_count_gt_3"], 1.0)


class PaddleQKMonitorTest(unittest.TestCase):
    def test_resolve_layer_idx_uses_shared_base_logic(self):
        monitor = PaddleQKStatsMonitor()

        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(layer_idx=8), 0, 4), 8)
        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(layer_number=2), 0, 4), 1)

        monitor.pp_rank = 1
        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(), 2, 4), 6)

    def test_row_stride_must_be_positive(self):
        for bad in (0, -1, -32):
            with self.assertRaises(ValueError):
                PaddleQKStatsMonitor(row_stride=bad)

    def test_row_stride_default_is_exact_full_pass(self):
        self.assertEqual(PaddleQKStatsMonitor().row_stride, 1)


class PaddleQKKernelComputeTest(unittest.TestCase):
    """GPU numerical tests for the shared triton qk_stats kernel via paddle."""

    @classmethod
    def setUpClass(cls):
        if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
            raise unittest.SkipTest("qk_stats kernel requires a CUDA GPU")
        paddle.device.set_device("gpu:0")
        cls.qk = importlib.import_module("internal_medicine.backends.paddlefleet.qk_monitor")

    def _gqa_inputs(self, B=1, S=128, Hq=8, Hkv=2, D=32, seed=0):
        paddle.seed(seed)
        q = paddle.randn([B, S, Hq, D], dtype="float32")
        k = paddle.randn([B, S, Hkv, D], dtype="float32")
        return q, k

    def test_kernel_grouping_matches_explicit_repeat_interleave(self):
        """GQA via in-kernel head mapping == materialized repeat_interleave."""
        q, k = self._gqa_inputs()
        heads_per_group = q.shape[2] // k.shape[2]

        grouped = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)

        k_expanded = k.repeat_interleave(heads_per_group, axis=2)
        expanded = self.qk.compute_qk_stats_paddle(q, k_expanded, causal=True, row_stride=1)

        for key in ("max_global", "mean_global", "entropy_global", "sink_global"):
            self.assertTrue(
                paddle.allclose(grouped[key], expanded[key], atol=1e-4, rtol=1e-4).item(),
                f"{key} mismatch: grouped={grouped[key].item()} expanded={expanded[key].item()}",
            )

    def test_sliding_window_matches_dense_reference(self):
        q, k = self._gqa_inputs(S=64, Hq=4, Hkv=4, D=16, seed=7)
        window = [8, 0]
        actual = self.qk.compute_qk_stats_paddle(q, k, causal=True, sliding_window=window)

        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        logits = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) / (q.shape[-1] ** 0.5)
        query_pos = paddle.arange(q.shape[1]).reshape([-1, 1])
        key_pos = paddle.arange(k.shape[1]).reshape([1, -1])
        valid = (key_pos <= query_pos) & (key_pos >= query_pos - window[0])
        logits = paddle.where(
            valid.reshape([1, 1, q.shape[1], k.shape[1]]),
            logits,
            paddle.full_like(logits, float("-inf")),
        )
        probability = paddle.nn.functional.softmax(logits, axis=-1)
        entropy = -paddle.where(
            probability > 0,
            probability * paddle.log(probability),
            paddle.zeros_like(probability),
        ).sum(axis=-1)

        self.assertTrue(paddle.allclose(actual["max_global"], logits.max(), atol=5e-3, rtol=1e-4).item())
        self.assertTrue(paddle.allclose(actual["entropy_global"], entropy.mean(), atol=1e-3, rtol=1e-4).item())
        self.assertTrue(paddle.allclose(actual["sink_global"], probability[..., 0].mean(), atol=1e-4, rtol=1e-4).item())

    def test_row_stride_approximates_mean_class_metrics(self):
        """Subsampling query rows should keep row-averaged metrics close to
        the full pass for representative random inputs. Entropy has meaningful
        magnitude, so use a relative check; mean and sink sit near zero for
        N(0,1) inputs, so use an absolute check.
        """
        q, k = self._gqa_inputs(S=512, seed=1)
        full = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        sub = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=8)

        rel = abs(sub["entropy_global"].item() - full["entropy_global"].item()) / (
            abs(full["entropy_global"].item()) + 1e-6
        )
        self.assertLess(
            rel, 0.1, f"entropy_global drifted: full={full['entropy_global'].item()} sub={sub['entropy_global'].item()}"
        )

        for key in ("mean_global", "sink_global"):
            self.assertLess(
                abs(sub[key].item() - full[key].item()),
                0.05,
                f"{key} drifted: full={full[key].item()} sub={sub[key].item()}",
            )

        # max is an extremum -> subsample is a lower bound (<= full).
        self.assertLessEqual(sub["max_global"].item(), full["max_global"].item() + 1e-4)

    def test_row_stride_one_is_exact_full_sequence(self):
        q, k = self._gqa_inputs(Hkv=8, seed=2)  # MHA, no grouping
        a = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        b = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        for key in ("max_global", "mean_global", "entropy_global", "sink_global"):
            self.assertTrue(paddle.allclose(a[key], b[key], atol=1e-5).item())

    def test_q_row_offset_matches_full_pass_when_reassembled(self):
        """CP-Option-A equivalence: computing stats on Q_local + K_full with
        the correct q_row_offset, then reassembling per-head means across the
        simulated CP shards, must reproduce the full-pass stats.

        This is the single-GPU numerical proof that the Triton kernel's
        ``q_row_offset`` parameter + K-only gather correctly recovers the
        distributed-Q semantics. Actual multi-rank behaviour is validated
        end-to-end via a training run.
        """
        # MHA to keep the head math trivial and avoid GQA-alignment concerns
        # in the reassembly.
        q, k = self._gqa_inputs(B=1, S=128, Hq=8, Hkv=8, D=32, seed=3)
        # Full pass — reference.
        full = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)

        # Simulate CP=2: split Q along seq into two halves, keep K full.
        S = q.shape[1]
        assert S % 2 == 0, "test requires even S"
        half = S // 2
        q_a = q[:, :half, :, :]
        q_b = q[:, half:, :, :]

        sub_a = self.qk.compute_qk_stats_paddle(q_a, k, causal=True, row_stride=1, q_row_offset=0)
        sub_b = self.qk.compute_qk_stats_paddle(q_b, k, causal=True, row_stride=1, q_row_offset=half)

        # Per-head means: average the two halves (rows split evenly).
        for key in ("entropy_per_head", "sink_per_head", "mean_per_head"):
            reassembled = (sub_a[key] + sub_b[key]) / 2.0
            self.assertTrue(
                paddle.allclose(full[key], reassembled, atol=1e-4, rtol=1e-4).item(),
                f"{key} mismatch after CP-halves reassembly",
            )
        # Per-head max: take elementwise max across halves.
        reassembled_max_ph = paddle.maximum(sub_a["max_per_head"], sub_b["max_per_head"])
        self.assertTrue(
            paddle.allclose(full["max_per_head"], reassembled_max_ph, atol=1e-4, rtol=1e-4).item(),
            "max_per_head mismatch after CP-halves reassembly",
        )

        # Global scalars: max via max across halves; mean/entropy/sink via
        # mean across halves (row counts are equal).
        self.assertTrue(
            paddle.allclose(
                full["max_global"],
                paddle.maximum(sub_a["max_global"], sub_b["max_global"]),
                atol=1e-4,
                rtol=1e-4,
            ).item()
        )
        for key in ("mean_global", "entropy_global", "sink_global"):
            reassembled_scalar = (sub_a[key] + sub_b[key]) / 2.0
            self.assertTrue(
                paddle.allclose(full[key], reassembled_scalar, atol=1e-4, rtol=1e-4).item(),
                f"{key} mismatch after CP-halves reassembly",
            )

    def test_q_row_offset_zero_matches_no_offset(self):
        """Passing q_row_offset=0 must be a no-op vs. the default path."""
        q, k = self._gqa_inputs(seed=4)
        default_pass = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1)
        with_zero_offset = self.qk.compute_qk_stats_paddle(q, k, causal=True, row_stride=1, q_row_offset=0)
        for key in ("max_global", "mean_global", "entropy_global", "sink_global"):
            self.assertTrue(
                paddle.allclose(default_pass[key], with_zero_offset[key], atol=1e-6).item(),
                f"{key} diverged with q_row_offset=0",
            )


if __name__ == "__main__":
    unittest.main()
