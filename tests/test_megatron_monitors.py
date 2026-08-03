import importlib
import math
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

MassiveActivationMonitor = importlib.import_module(
    "internal_medicine.backends.megatron.massive_activation_monitor"
).MassiveActivationMonitor
activation_dump_module = importlib.import_module("internal_medicine.backends.megatron.activation_dump_monitor")
ActivationDumpMonitor = activation_dump_module.ActivationDumpMonitor
setup_activation_dump_monitor = activation_dump_module.setup_activation_dump_monitor
lar_module = importlib.import_module("internal_medicine.backends.megatron.lar_monitor")
LARMonitor = lar_module.LARMonitor
setup_lar_monitor = lar_module.setup_lar_monitor
MoESpecialistMonitor = importlib.import_module("internal_medicine.backends.megatron.moe_monitor").MoESpecialistMonitor
moe_monitor_module = importlib.import_module("internal_medicine.backends.megatron.moe_monitor")
PLEHealthMonitor = importlib.import_module("internal_medicine.backends.megatron.ple_monitor").PLEHealthMonitor
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs
massive_activation_metrics = importlib.import_module("internal_medicine.backends.megatron.massive_activation_metrics")
compute_sink_head_classification = importlib.import_module(
    "internal_medicine.backends.megatron.sink_head_metrics"
).compute_sink_head_classification


class FakePLESublayer:
    act_fn = F.gelu


class WeakRefable:
    """Attribute bag that (unlike SimpleNamespace) supports weakref, so it can
    stand in for a router module in _load_balance_routers."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class FakeMoELayer(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts
        self.shared_experts = None


class FakeLogitLensLayer(nn.Module):
    """Minimal Megatron-style layer: called all-keyword, returns (output, context)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states), None


class FakeDecoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([FakeLogitLensLayer(hidden_size) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(hidden_states=x)
        return x


class FakeGPTModel(nn.Module):
    """Fake GPTModel exposing decoder.layers/final_layernorm + output_layer (head stage)."""

    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)  # weight [vocab, hidden]

    def forward(self, x, labels=None):
        # labels is accepted (and ignored) so the monitor's label-capture pre-hook can
        # read it from kwargs, mirroring Megatron GPTModel.forward(..., labels=...).
        return self.decoder(x)


class FakeTiedOutputLayer(nn.Module):
    """ColumnParallelLinear built with ``skip_weight_param_allocation=True``.

    Megatron does this when ``share_embeddings_and_output_weights=True``: the module
    owns no weight (``self.weight is None``) and the caller passes the shared tensor
    in at forward time. The forward hook therefore sees ``module.weight is None``.
    """

    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x, weight=None):
        return F.linear(x, weight)


class FakeTiedGPTModel(nn.Module):
    """GPTModel with tied input/output embeddings at PP=1.

    ``output_layer`` holds no weight; the tied tensor is reachable only through
    ``shared_embedding_or_output_weight()`` (which real Megatron resolves to
    ``embedding.word_embeddings.weight`` when ``pre_process`` is True).
    """

    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)
        self.shared_weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.3)
        self.output_layer = FakeTiedOutputLayer()

    def shared_embedding_or_output_weight(self):
        return self.shared_weight

    def forward(self, x, labels=None):
        hidden = self.decoder(x)
        return self.output_layer(hidden, weight=self.shared_weight)


class FakeBatchGPTModel(nn.Module):
    """GPTModel whose forward is all-keyword, as Megatron-Bridge calls it.

    ``gpt_step.py`` does ``model(**forward_args)`` with input_ids / position_ids /
    attention_mask / labels (+ packed_seq_params when packing), so this fixture is the
    shape ActivationDumpMonitor's batch-capture pre-hook actually sees in production.
    The hidden states are supplied separately so a test can assert on known values.
    """

    def __init__(self, num_layers, hidden_size, hidden):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)
        self._hidden = hidden

    def forward(self, input_ids=None, position_ids=None, attention_mask=None, labels=None, packed_seq_params=None):
        return self.decoder(self._hidden)


class FakePackedSeqParams:
    """Stand-in for megatron.core.packed_seq_params.PackedSeqParams.

    ``cp_group`` is a non-serialisable ProcessGroup in the real class; it is set to a
    junk value here so a test can prove the monitor never tries to persist it.
    """

    qkv_format = "thd"
    max_seqlen_q = 4
    max_seqlen_kv = 4
    total_tokens = 12
    local_cp_size = 1
    cu_seqlens_q_padded = None
    cu_seqlens_kv_padded = None
    cp_group = "NOT-A-SERIALISABLE-TENSOR"

    def __init__(self):
        self.cu_seqlens_q = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        self.cu_seqlens_kv = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        self.seq_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32)


class FakeHeadlessGPTModel(nn.Module):
    """Non-head PP stage: decoder present but no output_layer / tied weight accessor."""

    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.decoder = FakeDecoder(num_layers, hidden_size)

    def forward(self, x, labels=None):
        # labels is accepted (and ignored) so the monitor's label-capture pre-hook can
        # read it from kwargs, mirroring Megatron GPTModel.forward(..., labels=...).
        return self.decoder(x)


class FakeLatentProj(nn.Module):
    """Stand-in for mcore's ``fc2_latent_proj`` (TELinear, latent -> hidden).

    ``TELinear.forward(self, x)`` takes a single positional tensor and returns
    ``(out, bias)``, which is how ``MoELayer.postprocess`` calls it.
    """

    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(latent_size, hidden_size, bias=False)

    def forward(self, x):
        return self.linear(x), None


class FakeLatentMoELayer(nn.Module):
    """Latent MoE layer reproducing mcore's combine -> fc2_latent_proj ordering.

    ``forward`` mimics ``MoELayer``: experts produce per-expert outputs in LATENT dim,
    those are combined k-way with router weights, and the combined latent tensor is fed
    to ``fc2_latent_proj``. The monitor's pre-hook on ``fc2_latent_proj`` must observe
    exactly that combined tensor — ``self.combined`` is stashed so a test can assert it.
    """

    def __init__(self, hidden_size, latent_size, num_experts, topk=2):
        super().__init__()
        self.router = _FakeRouter(latent_size, num_experts)
        self.fc1_latent_proj = nn.Linear(hidden_size, latent_size, bias=False)
        self.fc2_latent_proj = FakeLatentProj(latent_size, hidden_size)
        self.topk = topk
        self.num_experts = num_experts
        self.combined = None

    def forward(self, hidden_states, expert_outputs=None, probs=None):
        latent = self.fc1_latent_proj(hidden_states)
        if expert_outputs is None:
            # topk expert outputs in latent dim; combine with uniform weights.
            expert_outputs = [latent for _ in range(self.topk)]
            probs = [1.0 / self.topk] * self.topk
        combined = sum(p * e for p, e in zip(expert_outputs, probs, strict=True))
        self.combined = combined
        out, _ = self.fc2_latent_proj(combined)
        return out


class MegatronMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_metrics_flush_from_gpu_buffer(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._ROUTER_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        router = SimpleNamespace(
            topk=2,
            _cached_scores_for_aux_loss=torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            ),
        )
        monitor._compute_router_metrics(0, router, None, None)

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/router_entropy", latest)
        self.assertIn("moe_health/layer_0/score_sum_mean", latest)
        self.assertIn("moe_health/global_router_entropy", latest)
        self.assertIn("moe_health/global_score_sum_max", latest)

    def test_step_computes_expert_metrics_even_under_no_grad(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._EXPERT_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        hidden_size = 4
        ffn_hidden = 8
        num_experts = 2
        experts = SimpleNamespace(
            num_local_experts=num_experts,
            config=SimpleNamespace(hidden_size=hidden_size),
            weight1=torch.nn.Parameter(torch.ones(num_experts * hidden_size, ffn_hidden)),
            weight2=torch.nn.Parameter(torch.ones(num_experts * ffn_hidden, hidden_size)),
        )
        moe_layer = FakeMoELayer(experts)
        monitor._monitored_moe_layers = [(0, weakref.ref(moe_layer))]

        with torch.no_grad():
            monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/global_expert_norm_mean", latest)

    def test_load_balance_metrics_from_reduced_tokens_per_expert(self):
        # Two bias-enabled layers, recorded in the order finalize_model_grads
        # stacks them. tokens_per_expert is the ALREADY-reduced global count.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0, 1]
        for layer_idx in (0, 1):
            for name in moe_monitor_module._LOAD_BALANCE_METRICS:
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers(torch.device("cpu"))

        # layer 0: [5,3,2,2] -> max/min=5/2=2.5, max/median=5/2=2.5
        # layer 1: [10,0,4,6] -> max/min=10/1(clamp)=10, max/median=10/4=2.5
        reduced = torch.tensor([[5.0, 3.0, 2.0, 2.0], [10.0, 0.0, 4.0, 6.0]])
        monitor._record_load_balance_metrics(reduced, monitor._load_balance_layer_order)
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_min_ratio"], 10.0, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_median_ratio"], 2.5, places=5)
        # CV = population std / mean. layer 0 mean=3, std=sqrt(((2)+(0)+(1)+(1))/4)=sqrt(1.5)
        self.assertAlmostEqual(latest["moe_health/layer_0/load_cv"], (1.5**0.5) / 3.0, places=5)
        self.assertIn("moe_health/global_load_max_min_ratio", latest)
        self.assertIn("moe_health/global_load_cv", latest)

    def test_load_balance_metrics_absent_without_any_source(self):
        # When no router exposes global_tokens_per_expert and none has expert-bias
        # enabled, the metric must not be declared (schema stays clean) — nothing
        # to record or patch.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        self.assertFalse(monitor._expert_bias_enabled)
        self.assertFalse(monitor._global_lb_enabled)
        self.assertEqual(monitor._load_balance_layer_order, [])
        self.assertEqual(monitor._load_balance_routers, [])
        # Both global patches are idempotent no-ops when no source is present.
        monitor._patch_expert_bias_update()
        monitor._patch_reset_temporary_tensors()
        self.assertIsNone(monitor._orig_get_updated_expert_bias)
        self.assertIsNone(monitor._orig_reset_model_temporary_tensors)

    def test_prepare_layers_prefers_global_over_expert_bias(self):
        # A router exposing both global_tokens_per_expert and enable_expert_bias
        # must be routed to the global source, not the expert-bias fallback.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        router = WeakRefable(
            global_tokens_per_expert=torch.zeros(4),
            ga_steps=torch.tensor(0.0),
            enable_expert_bias=True,
        )
        moe_layer = SimpleNamespace(router=router)
        monitor._find_moe_layers = lambda _model: [(0, moe_layer)]
        monitor._prepare_layers(object())
        self.assertTrue(monitor._global_lb_enabled)
        self.assertEqual([idx for idx, _ in monitor._load_balance_routers], [0])
        self.assertFalse(monitor._expert_bias_enabled)
        self.assertEqual(monitor._load_balance_layer_order, [])

    def test_load_balance_metrics_from_global_tokens_per_expert(self):
        # Preferred path: counts sourced from each router's own
        # global_tokens_per_expert buffer (already TPxCPxDP-reduced).
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        routers = [
            WeakRefable(global_tokens_per_expert=torch.tensor([5.0, 3.0, 2.0, 2.0]), ga_steps=torch.tensor(4.0)),
            WeakRefable(global_tokens_per_expert=torch.tensor([10.0, 0.0, 4.0, 6.0]), ga_steps=torch.tensor(4.0)),
        ]
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [(idx, weakref.ref(r)) for idx, r in enumerate(routers)]
        for layer_idx in (0, 1):
            for name in moe_monitor_module._LOAD_BALANCE_METRICS:
                monitor.declare_layer_metric(layer_idx, name)
        monitor.allocate_buffers(torch.device("cpu"))

        monitor._record_global_load_balance_metrics()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        # Ratios are scale-invariant, so raw sums give the same values as the
        # expert-bias test rows: layer 0 -> 2.5, layer 1 -> 10 / 2.5.
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_min_ratio"], 10.0, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.5, places=5)
        self.assertAlmostEqual(latest["moe_health/layer_1/load_max_median_ratio"], 2.5, places=5)
        self.assertIn("moe_health/global_load_max_min_ratio", latest)

    def test_global_load_balance_skips_uninitialized_ga_steps(self):
        # A router whose global_aux_loss buffer has not accumulated yet
        # (ga_steps == 0) must be skipped, not divide-by-zero or emit garbage.
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        router = WeakRefable(global_tokens_per_expert=torch.zeros(4), ga_steps=torch.tensor(0.0))
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [(0, weakref.ref(router))]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        monitor._record_global_load_balance_metrics()
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertNotIn("moe_health/layer_0/load_max_min_ratio", latest)

    def test_reset_temporary_tensors_patch_reads_before_zero(self):
        # The wrapper must rebind reset_model_temporary_tensors in
        # finalize_model_grads and read global_tokens_per_expert BEFORE the
        # original zeroes it.
        fmg = moe_monitor_module._finalize_model_grads
        original = fmg.reset_model_temporary_tensors

        router = WeakRefable(
            global_tokens_per_expert=torch.tensor([6.0, 2.0, 3.0, 3.0]),  # max/min=3, max/median=2
            ga_steps=torch.tensor(4.0),
        )

        # Stub original to zero the buffer, proving the wrapper read it first.
        def zeroing_reset(*_a, **_k):
            router.global_tokens_per_expert.zero_()
            router.ga_steps.zero_()

        fmg.reset_model_temporary_tensors = zeroing_reset

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._global_lb_enabled = True
        monitor._load_balance_routers = [(0, weakref.ref(router))]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        try:
            monitor._patch_reset_temporary_tensors()
            self.assertTrue(getattr(fmg.reset_model_temporary_tensors, "_im_patched", False))

            fmg.reset_model_temporary_tensors(None, None)  # config, model
            # Buffer was zeroed by the original, proving order.
            self.assertEqual(float(router.global_tokens_per_expert.sum()), 0.0)
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 3.0, places=5)
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.0, places=5)
        finally:
            monitor.remove_hooks()
            fmg.reset_model_temporary_tensors = original

    def test_expert_bias_patch_rebinds_caller_and_fires(self):
        # The wrapper must rebind the name in finalize_model_grads (the caller),
        # observe the reduced tokens_per_expert, and unpatch cleanly.
        fmg = moe_monitor_module._finalize_model_grads
        original = fmg.get_updated_expert_bias
        # Stub the underlying update so the test needs no distributed group:
        # the real fn all-reduces tokens_per_expert, which requires dist init.
        # Our wrapper only cares that the tensor it receives is the reduced one.
        fmg.get_updated_expert_bias = lambda tpe, bias, rate, *a, **k: bias

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._expert_bias_enabled = True
        monitor._load_balance_layer_order = [0]
        for name in moe_monitor_module._LOAD_BALANCE_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        try:
            monitor._patch_expert_bias_update()
            self.assertTrue(getattr(fmg.get_updated_expert_bias, "_im_patched", False))

            # Caller hands the wrapper an ALREADY-reduced count row.
            tokens = torch.tensor([[6.0, 2.0, 3.0, 3.0]])  # max/min=6/2=3, max/median=6/3=2
            bias = torch.zeros_like(tokens)
            fmg.get_updated_expert_bias(tokens, bias, 0.0)
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_min_ratio"], 3.0, places=5)
            self.assertAlmostEqual(latest["moe_health/layer_0/load_max_median_ratio"], 2.0, places=5)
        finally:
            monitor.remove_hooks()
            fmg.get_updated_expert_bias = original

    # ---------------- latent-combine magnitude ----------------

    def test_latent_combine_metrics_measure_post_combine_pre_up_proj_tensor(self):
        """The pre-hook on fc2_latent_proj must see the k-way-combined LATENT tensor —
        after expert combine, before the up-projection back to hidden_size."""
        S, B, H, L, E = 4, 2, 8, 5, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            layer(torch.randn(S, B, H))
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            combined = layer.combined.detach().reshape(-1, L).float()
            want_rms = float(combined.square().mean().sqrt())
            per_channel_max = combined.abs().amax(dim=0)
            want_ratio = float(per_channel_max.max() / per_channel_max.median())
            self.assertAlmostEqual(latest["moe_health/layer_0/latent_combine_rms"], want_rms, places=5)
            self.assertAlmostEqual(
                latest["moe_health/layer_0/latent_combine_channel_max_median_ratio"], want_ratio, places=5
            )
            self.assertIn("moe_health/global_latent_combine_rms", latest)
            self.assertIn("moe_health/global_latent_combine_channel_max_median_ratio", latest)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_measures_latent_dim_not_hidden_dim(self):
        """Guard against hooking the wrong side: the measured tensor must have
        latent_size channels, so a per-channel stat over hidden_size would differ."""
        S, B, H, L, E = 3, 1, 8, 4, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            hidden = torch.randn(S, B, H)
            out = layer(hidden)
            monitor.step()
            self.assertEqual(layer.combined.shape[-1], L)
            self.assertEqual(out.shape[-1], H)
            latest = training_logs.get_latest(prefix="moe_health")
            got = latest["moe_health/layer_0/latent_combine_rms"]
            hidden_side_rms = float(out.detach().reshape(-1, H).float().square().mean().sqrt())
            self.assertNotAlmostEqual(got, hidden_side_rms, places=4)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_ratio_detects_dominant_channel(self):
        """channel_max/mean ratio ~1 for a flat combine, and large when one latent
        channel dominates — the massive-activation signature we want to catch."""
        S, B, H, L, E = 4, 1, 8, 4, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            flat = torch.ones(S, B, L)
            layer(torch.randn(S, B, H), expert_outputs=[flat], probs=[1.0])
            monitor.step()
            ratio_flat = training_logs.get_latest(prefix="moe_health")[
                "moe_health/layer_0/latent_combine_channel_max_median_ratio"
            ]
            self.assertAlmostEqual(ratio_flat, 1.0, places=5)

            training_logs.reset()
            spiked = torch.ones(S, B, L)
            spiked[..., 0] = 100.0
            layer(torch.randn(S, B, H), expert_outputs=[spiked], probs=[1.0])
            monitor.step()
            ratio_spiked = training_logs.get_latest(prefix="moe_health")[
                "moe_health/layer_0/latent_combine_channel_max_median_ratio"
            ]
            # per-channel maxima = [100, 1, 1, 1]; torch.median takes the lower middle
            # value of the sorted [1, 1, 1, 100] => 1.0, so the ratio is the full 100.
            # A mean denominator would give 100/25.75 ~= 3.88 — the spike inflating its
            # own denominator, which is exactly why median is the right choice here.
            self.assertAlmostEqual(ratio_spiked, 100.0, places=4)
            self.assertGreater(ratio_spiked, ratio_flat)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_metrics_absent_on_non_latent_moe(self):
        """Plain (non-latent) MoE has no fc2_latent_proj: the metrics must simply not
        be declared, not crash and not emit zeros."""
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        plain = _FakeMoELayer(hidden_size=8, num_experts=4)
        self.assertIsNone(monitor._latent_proj_of(plain))
        monitor._find_moe_layers = lambda _model: [(0, plain)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, plain)])
        try:
            plain(torch.randn(4, 1, 8))
            monitor.step()
            latest = training_logs.get_latest(prefix="moe_health")
            for name in moe_monitor_module._LATENT_COMBINE_METRICS:
                self.assertNotIn(f"moe_health/layer_0/{name}", latest)
                self.assertNotIn(f"moe_health/global_{name}", latest)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_metrics_are_max_aggregated_across_ranks(self):
        """Both latent-combine metrics compose with MAX, not mean: they exist to catch
        magnitude blow-up, and a mean over microbatches / layers / ranks would average a
        spike away. Neither ends in _max, so both must be listed explicitly in
        MAX_AGGREGATED (per-layer -> global) and MAX_AGGREGATED_SUFFIXES (cross-rank)."""
        for name in moe_monitor_module._LATENT_COMBINE_METRICS:
            with self.subTest(metric=name):
                self.assertIn(name, MoESpecialistMonitor.MAX_AGGREGATED)
                self.assertTrue(training_logs._is_max_metric(f"moe_health/layer_0/{name}"))
                self.assertTrue(training_logs._is_max_metric(f"moe_health/global_{name}"))

    def test_latent_combine_global_takes_max_over_layers(self):
        """The global key must be the worst layer, not the average of layers."""
        S, B, H, L, E = 4, 1, 8, 4, 4
        layers = [FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E) for _ in range(2)]
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: list(enumerate(layers))
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks(list(enumerate(layers)))
        try:
            # layer 0 quiet, layer 1 hot: global must follow layer 1.
            layers[0](torch.randn(S, B, H), expert_outputs=[torch.ones(S, B, L)], probs=[1.0])
            hot = torch.ones(S, B, L) * 3.0
            hot[..., 0] = 300.0
            layers[1](torch.randn(S, B, H), expert_outputs=[hot], probs=[1.0])
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            for name in moe_monitor_module._LATENT_COMBINE_METRICS:
                per_layer = [latest[f"moe_health/layer_{i}/{name}"] for i in range(2)]
                self.assertAlmostEqual(latest[f"moe_health/global_{name}"], max(per_layer), places=5)
                # A mean would sit strictly below the max given these two layers differ.
                self.assertGreater(max(per_layer), sum(per_layer) / 2)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_rms_keeps_worst_microbatch(self):
        """Two forwards in one step: the recorded RMS is the larger, not their mean."""
        S, B, H, L, E = 4, 1, 8, 4, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            layer(torch.randn(S, B, H), expert_outputs=[torch.ones(S, B, L)], probs=[1.0])
            layer(torch.randn(S, B, H), expert_outputs=[torch.ones(S, B, L) * 5.0], probs=[1.0])
            monitor.step()
            got = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/latent_combine_rms"]
            self.assertAlmostEqual(got, 5.0, places=5)  # max(1.0, 5.0), not mean 3.0
        finally:
            monitor.remove_hooks()

    def test_latent_combine_hook_respects_monitor_interval(self):
        S, B, H, L, E = 4, 1, 8, 4, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True, monitor_interval=2)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            monitor.step_count = 1  # 1 % 2 != 0 -> unmonitored step
            layer(torch.randn(S, B, H))
            monitor.step()
            latest = training_logs.get_latest(prefix="moe_health")
            self.assertNotIn("moe_health/layer_0/latent_combine_rms", latest)
        finally:
            monitor.remove_hooks()


class MegatronPLEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_global_hooks_are_disabled_when_log_global_is_false(self):
        monitor = PLEHealthMonitor(log_global=False)
        monitor._num_layers = 2
        monitor._hidden_size = 6
        monitor._hidden_size_ple = 3

        monitor._make_token_ple_hook()(None, None, torch.randn(2, 4, 6))
        monitor._make_proj_ple_hook()(None, None, torch.randn(2, 4, 6))

        self.assertIsNone(monitor._token_ple_buf)
        self.assertIsNone(monitor._proj_ple_buf)
        self.assertEqual(training_logs.get_latest(prefix="ple_health"), {})

    def test_layer_hook_records_residual_and_gate_metrics_as_one_observation(self):
        monitor = PLEHealthMonitor(log_per_layer=True, log_global=True, gate_sparsity_threshold=0.01)
        hidden_states = torch.ones(2, 3, 4)
        for name in ("residual_ratio", "gate_activation_mean", "gate_sparsity"):
            monitor.declare_layer_metric(5, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._gate_out_buf[5] = torch.ones(2, 3, 4)
        output = hidden_states * 1.5

        hook = monitor._make_ple_layer_hook(5, FakePLESublayer())
        hook(None, (hidden_states,), output)
        monitor.step()

        latest = training_logs.get_latest(prefix="ple_health")
        self.assertIn("ple_health/layer_5/residual_ratio", latest)
        self.assertIn("ple_health/layer_5/gate_activation_mean", latest)
        self.assertIn("ple_health/global_residual_ratio", latest)
        self.assertEqual(monitor._gate_out_buf, {})


class MegatronMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._compute_residual_metrics(0, hidden_states)
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
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)

    def test_massive_act_channel_count_is_per_token_mean_with_sqrt_h_threshold(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=False,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
        )
        hidden_states = torch.tensor(
            [
                [[100.0, 1.0, 1.0, 1.0]],
                [[100.0, 1.0, -1.0, 1.0]],
            ]
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._compute_residual_metrics(0, hidden_states)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertAlmostEqual(latest["massive_act/layer_0/massive_act_channel_count"], 1.0, places=5)

    def test_spectral_norm_bounds_record_per_token_rms_ratio(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
        )
        pre = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        # post = 2 * pre => per-token RMS ratio is exactly 2.0 for every token,
        # so both the max and min bound collapse to 2.0.
        post = pre * 2.0
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(pre.device)

        monitor._compute_spectral_norm(0, pre, post)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("spectral_norm_max", "spectral_norm_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(latest["massive_act/layer_0/spectral_norm_max"], 2.0, places=5)
        self.assertAlmostEqual(latest["massive_act/layer_0/spectral_norm_min"], 2.0, places=5)
        # activation_rms is derived from the shared per-token pre-RMS, not a second
        # square of the input: it must equal sqrt(mean(pre**2)) over the whole input.
        self.assertIn("massive_act/layer_0/activation_rms", latest)
        expected_rms = pre.reshape(-1, pre.shape[-1]).float().square().mean().sqrt().item()
        self.assertAlmostEqual(latest["massive_act/layer_0/activation_rms"], expected_rms, places=5)
        # activation_rms_std is the std of the per-token pre-RMS (dispersion).
        self.assertIn("massive_act/layer_0/activation_rms_std", latest)
        expected_rms_std = pre.reshape(-1, pre.shape[-1]).float().square().mean(dim=-1).sqrt().std().item()
        self.assertAlmostEqual(latest["massive_act/layer_0/activation_rms_std"], expected_rms_std, places=5)

    def test_derived_activation_rms_matches_original_formula(self):
        # Regression guard: the merged/derived activation_rms (from the spectral
        # hook's per-token pre-RMS) must be numerically identical to the original
        # standalone compute_activation_scale_stats(h) over the same tensor.
        torch.manual_seed(0)
        hidden_states = torch.randn(7, 3, 5) * 4.0 + 1.5  # [S, B, H], non-trivial scale

        original = massive_activation_metrics.compute_activation_scale_stats(hidden_states)["activation_rms"]
        # activation_rms depends only on the pre tensor; post is irrelevant to it.
        derived = massive_activation_metrics.compute_spectral_norm_bounds(
            hidden_states, torch.zeros_like(hidden_states), include_activation_rms=True
        )["activation_rms"]

        self.assertTrue(torch.allclose(original, derived, rtol=0, atol=1e-6), f"{original} != {derived}")

    def test_grad_gain_bounds_equal_scaled_gradients(self):
        # grad_in = 2 * grad_out per token => the per-token ratio ‖dx‖/‖dy‖ is exactly
        # 2.0 for every token, so both the Lipschitz max and min bound collapse to 2.0.
        torch.manual_seed(0)
        grad_out = torch.randn(5, 3, 4)  # [S, B, H]
        grad_in = grad_out * 2.0

        bounds = massive_activation_metrics.compute_grad_gain_bounds(grad_in, grad_out)

        self.assertAlmostEqual(bounds["lipschitz_max"].item(), 2.0, places=5)
        self.assertAlmostEqual(bounds["lipschitz_min"].item(), 2.0, places=5)

    def test_grad_gain_hook_records_lipschitz_from_backward(self):
        # End-to-end: a scalar-mul layer y = 2*x has ∂L/∂x = 2·∂L/∂y for any loss, so
        # the backward-captured gradient-gain ratio is exactly 2.0 regardless of the
        # gradient direction. This also proves the tensor grad-hook path fires and
        # records (a module full_backward_hook would see an empty grad_input here,
        # since the layer is called all-keyword like Megatron does).
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=True,
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        class ScalarMul(nn.Module):
            def forward(self, hidden_states):
                return hidden_states * 2.0, None  # (output, context) like a Megatron layer

        layer = ScalarMul()
        hook = layer.register_forward_hook(monitor._make_grad_gain_hook(0), with_kwargs=True)

        x = torch.randn(3, 2, 4, requires_grad=True)  # [S, B, H]
        out, _ = layer(hidden_states=x)  # all-keyword call, as the Megatron block does
        out.sum().backward()
        hook.remove()

        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in ("lipschitz_max", "lipschitz_min"):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertAlmostEqual(latest["massive_act/layer_0/lipschitz_max"], 2.0, places=5)
        self.assertAlmostEqual(latest["massive_act/layer_0/lipschitz_min"], 2.0, places=5)

    def test_logit_lens_entropy_uniform_is_log_vocab(self):
        # Zero weight => every logit is 0 => uniform softmax => H == log(vocab).
        import math

        vocab, hidden = 8, 4
        h = torch.randn(3, 2, hidden)
        weight = torch.zeros(vocab, hidden)
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, chunk_size=2)
        self.assertIn("logit_lens_entropy_mean", out)
        self.assertAlmostEqual(out["logit_lens_entropy_mean"].item(), math.log(vocab), places=5)

    def test_logit_lens_entropy_peaked_is_near_zero(self):
        # One vocab logit dominates => softmax ~ one-hot => H ~ 0.
        h = torch.tensor([[[1.0, 0.0]]])  # [s=1, b=1, hidden=2]
        weight = torch.tensor([[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # [vocab=3, hidden=2]
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight)
        self.assertLess(out["logit_lens_entropy_mean"].item(), 1e-3)
        self.assertGreaterEqual(out["logit_lens_entropy_mean"].item(), 0.0)

    def test_logit_lens_entropy_chunking_matches_single_pass(self):
        import math

        torch.manual_seed(0)
        vocab, hidden = 16, 5
        h = torch.randn(9, 2, hidden)
        weight = torch.randn(vocab, hidden)
        full = massive_activation_metrics.compute_logit_lens_entropy(h, weight, chunk_size=1000)
        chunked = massive_activation_metrics.compute_logit_lens_entropy(h, weight, chunk_size=4)
        for key in full:
            self.assertAlmostEqual(full[key].item(), chunked[key].item(), places=5)
        # Bounds: 0 <= mean H <= log(vocab).
        self.assertGreaterEqual(full["logit_lens_entropy_mean"].item(), -1e-5)
        self.assertLessEqual(full["logit_lens_entropy_mean"].item(), math.log(vocab) + 1e-4)

    def test_logit_lens_entropy_applies_final_norm(self):
        # A norm that zeros its input makes every logit 0 => uniform => log(vocab),
        # regardless of the (large) hidden values: proves final_norm is applied.
        import math

        vocab, hidden = 4, 3
        h = torch.randn(2, 2, hidden) * 10.0
        weight = torch.randn(vocab, hidden)
        zero_norm = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            zero_norm.weight.zero_()
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, final_norm=zero_norm)
        self.assertAlmostEqual(out["logit_lens_entropy_mean"].item(), math.log(vocab), places=5)

    def test_logit_lens_entropy_empty_input_is_zero(self):
        out = massive_activation_metrics.compute_logit_lens_entropy(torch.zeros(0, 4), torch.randn(6, 4))
        for key in ("logit_lens_entropy_mean", "logit_lens_logsumexp_mean"):
            self.assertEqual(out[key].item(), 0.0)

    def test_logit_lens_logsumexp_uniform_equals_c_plus_log_vocab(self):
        # Constant logit c over vocab V => logsumexp = c + log(V). With a one-hot
        # weight column of value c and a one-hot hidden, every logit == c.
        import math

        vocab, hidden = 8, 4
        c = 3.0
        h = torch.zeros(3, 2, hidden)
        h[..., 0] = 1.0  # select column 0 of the weight
        weight = torch.zeros(vocab, hidden)
        weight[:, 0] = c  # every vocab logit == c
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, chunk_size=2)
        expected = c + math.log(vocab)
        self.assertIn("logit_lens_logsumexp_mean", out)
        self.assertAlmostEqual(out["logit_lens_logsumexp_mean"].item(), expected, places=4)

    def test_logit_lens_entropy_end_to_end_records_keys(self):
        import math

        torch.manual_seed(0)
        num_layers, hidden, vocab = 2, 6, 10
        model = FakeGPTModel(num_layers, hidden, vocab)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
            logit_lens_chunk_size=3,
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, hidden))  # [S, B, H], grad enabled so _should_monitor fires
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for i in range(num_layers):
            for key in ("logit_lens_entropy_mean", "logit_lens_logsumexp_mean"):
                full_key = f"massive_act/layer_{i}/{key}"
                self.assertIn(full_key, latest)
                self.assertTrue(math.isfinite(latest[full_key]))
            mean_entropy = latest[f"massive_act/layer_{i}/logit_lens_entropy_mean"]
            self.assertGreaterEqual(mean_entropy, -1e-4)
            self.assertLessEqual(mean_entropy, math.log(vocab) + 1e-3)
        self.assertIn("massive_act/global_logit_lens_entropy_mean", latest)
        self.assertIn("massive_act/global_logit_lens_logsumexp_mean", latest)
        monitor.remove_hooks()

    def test_logit_lens_entropy_absent_when_disabled(self):
        model = FakeGPTModel(2, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=False,
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("logit_lens_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_logit_lens_entropy_no_head_stage_is_noop(self):
        # A PP stage without the LM head: _resolve_lm_head -> (None, None), no entropy
        # hook attaches, no entropy keys declared, and the forward runs without error.
        model = FakeHeadlessGPTModel(2, 6)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
        )
        self.assertEqual(monitor._resolve_lm_head(model), (None, None))
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("logit_lens_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_logit_lens_entropy_layer_filter_restricts_declared_layers(self):
        # logit_lens_layers restricts the entropy metric to the listed global indices.
        model = FakeGPTModel(3, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=True,
            logit_lens_layers=[1],
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertIn("massive_act/layer_1/logit_lens_entropy_mean", latest)
        self.assertNotIn("massive_act/layer_0/logit_lens_entropy_mean", latest)
        self.assertNotIn("massive_act/layer_2/logit_lens_entropy_mean", latest)
        monitor.remove_hooks()

    def test_logit_lens_cross_entropy_matches_reference(self):
        # CE via the logit lens must equal F.cross_entropy on the same logits.
        torch.manual_seed(0)
        tokens, hidden, vocab = 7, 5, 11
        h = torch.randn(tokens, hidden)
        weight = torch.randn(vocab, hidden)
        labels = torch.randint(0, vocab, (tokens,))
        out = massive_activation_metrics.compute_logit_lens_entropy(
            h, weight, labels=labels, want_entropy=False, chunk_size=3
        )
        self.assertIn("logit_lens_cross_entropy_mean", out)
        # want_entropy=False -> only the CE metric is produced.
        self.assertNotIn("logit_lens_entropy_mean", out)
        self.assertNotIn("logit_lens_logsumexp_mean", out)
        ref = F.cross_entropy((h @ weight.t()).float(), labels)
        self.assertAlmostEqual(out["logit_lens_cross_entropy_mean"].item(), ref.item(), places=4)

    def test_logit_lens_cross_entropy_peaked_correct_is_near_zero(self):
        # Logits sharply peaked at the correct label -> CE ~ 0.
        vocab = hidden = 8
        weight = torch.eye(vocab, hidden) * 50.0
        h = torch.eye(vocab, hidden)  # token i one-hot at coord i
        labels = torch.arange(vocab)
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, labels=labels, want_entropy=False)
        self.assertLess(out["logit_lens_cross_entropy_mean"].item(), 1e-3)
        self.assertGreaterEqual(out["logit_lens_cross_entropy_mean"].item(), 0.0)

    def test_logit_lens_all_three_metrics_together(self):
        # entropy + logsumexp + cross-entropy share one projection when both flags on.
        torch.manual_seed(1)
        h = torch.randn(6, 4)
        weight = torch.randn(9, 4)
        labels = torch.randint(0, 9, (6,))
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, labels=labels, want_entropy=True)
        for key in ("logit_lens_entropy_mean", "logit_lens_logsumexp_mean", "logit_lens_cross_entropy_mean"):
            self.assertIn(key, out)
            self.assertTrue(math.isfinite(out[key].item()))

    def test_logit_lens_cross_entropy_label_count_mismatch_skips_ce(self):
        # A label/token count mismatch drops CE (no crash) but keeps entropy.
        h = torch.randn(6, 4)
        weight = torch.randn(9, 4)
        labels = torch.randint(0, 9, (5,))  # wrong count
        out = massive_activation_metrics.compute_logit_lens_entropy(h, weight, labels=labels, want_entropy=True)
        self.assertNotIn("logit_lens_cross_entropy_mean", out)
        self.assertIn("logit_lens_entropy_mean", out)

    def test_logit_lens_cross_entropy_end_to_end_matches_lm_loss(self):
        # End-to-end: the final layer's logit-lens CE equals the LM loss (final_norm +
        # head + F.cross_entropy) computed by hand, validating label alignment.
        torch.manual_seed(0)
        num_layers, hidden, vocab = 2, 6, 10
        seq, batch = 4, 2
        model = FakeGPTModel(num_layers, hidden, vocab)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_entropy=False,
            log_logit_lens_cross_entropy=True,
            logit_lens_chunk_size=3,
        )
        monitor.register_hooks(model)
        x = torch.randn(seq, batch, hidden)  # [S, B, H]
        labels = torch.randint(0, vocab, (batch, seq))  # [B, S], like Megatron
        out_hidden = model(x, labels=labels)  # last layer output (== decoder return)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for i in range(num_layers):
            key = f"massive_act/layer_{i}/logit_lens_cross_entropy_mean"
            self.assertIn(key, latest)
            self.assertTrue(math.isfinite(latest[key]))
        with torch.no_grad():
            final = model.decoder.final_layernorm(out_hidden)
            logits = final.reshape(-1, hidden) @ model.output_layer.weight.t()
            labels_aligned = labels.transpose(0, 1).reshape(-1)  # seq-major, matches hidden
            ref = F.cross_entropy(logits.float(), labels_aligned)
        self.assertAlmostEqual(
            latest[f"massive_act/layer_{num_layers - 1}/logit_lens_cross_entropy_mean"],
            ref.item(),
            places=4,
        )
        self.assertIn("massive_act/global_logit_lens_cross_entropy_mean", latest)
        monitor.remove_hooks()

    def test_logit_lens_cross_entropy_absent_when_disabled(self):
        model = FakeGPTModel(2, 6, 10)
        monitor = MassiveActivationMonitor(
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_logit_lens_cross_entropy=False,
        )
        monitor.register_hooks(model)
        model(torch.randn(4, 2, 6), labels=torch.randint(0, 10, (2, 4)))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertFalse(any("cross_entropy" in key for key in latest))
        monitor.remove_hooks()

    def test_hidden_spectral_entropy_rank_one_is_near_zero(self):
        # A rank-1 token set (all tokens on one direction) spans one effective
        # direction -> spectral entropy ~ 0.
        direction = torch.randn(8)
        coeffs = torch.randn(16, 1)
        h = coeffs * direction  # [16, 8], rank 1
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertLess(entropy.item(), 1e-4)
        self.assertGreaterEqual(entropy.item(), -1e-6)

    def test_hidden_spectral_entropy_orthonormal_rows_is_log_k(self):
        import math

        # k orthonormal rows -> k equal singular values -> uniform p_i -> H = log(k).
        k = 4
        h = torch.eye(k, 8)  # 4 orthonormal rows in R^8
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertAlmostEqual(entropy.item(), math.log(k), places=5)

    def test_hidden_spectral_entropy_empty_input_is_zero(self):
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(torch.zeros(0, 4))
        self.assertEqual(entropy.item(), 0.0)

    def test_hidden_spectral_entropy_within_bounds(self):
        import math

        torch.manual_seed(0)
        n, d = 32, 8
        h = torch.randn(n, d)
        entropy = massive_activation_metrics.compute_hidden_spectral_entropy(h)
        self.assertGreaterEqual(entropy.item(), -1e-5)
        self.assertLessEqual(entropy.item(), math.log(min(n, d)) + 1e-4)

    def test_hidden_spectral_entropy_end_to_end_records_key(self):
        import math

        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            log_post_norm_metrics=False,
            log_activation_rms=False,
            log_lipschitz=False,
            log_hidden_spectral_entropy=True,
        )
        self.assertIn("hidden_spectral_entropy", monitor._layer_metric_names())
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        normalized = torch.randn(4, 2, 8)  # [S, B, H], post-RMSNorm hidden
        monitor._compute_spectral_entropy(0, normalized)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertIn("massive_act/layer_0/hidden_spectral_entropy", latest)
        self.assertIn("massive_act/global_hidden_spectral_entropy", latest)
        value = latest["massive_act/layer_0/hidden_spectral_entropy"]
        self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(value, -1e-5)
        self.assertLessEqual(value, math.log(8) + 1e-4)

    def test_hidden_spectral_entropy_absent_when_disabled(self):
        monitor = MassiveActivationMonitor(log_hidden_spectral_entropy=False)
        self.assertNotIn("hidden_spectral_entropy", monitor._layer_metric_names())


class SinkHeadClassificationTest(unittest.TestCase):
    """The gap computation is branchless to avoid a GPU->CPU sync on the hot
    path (Python comparisons on a tensor sink_count would .item()). These cases
    pin the branchless result against a readable branched reference.
    """

    THRESHOLD = 0.3

    def _reference_gap(self, sink_per_head):
        is_sink = sink_per_head > self.THRESHOLD
        num_heads = sink_per_head.numel()
        sink_count = int(is_sink.sum())
        if 0 < sink_count < num_heads:
            return (sink_per_head[is_sink].mean() - sink_per_head[~is_sink].mean()).item()
        if sink_count == num_heads:
            return sink_per_head.mean().item()
        return 0.0

    def _assert_gap(self, sink_per_head):
        result = compute_sink_head_classification(sink_per_head, threshold=self.THRESHOLD)
        self.assertAlmostEqual(result["sink_nonsink_gap"].item(), self._reference_gap(sink_per_head), places=5)

    def test_mixed_sink_and_nonsink(self):
        self._assert_gap(torch.tensor([0.5, 0.1, 0.8, 0.05]))

    def test_all_heads_are_sinks(self):
        self._assert_gap(torch.tensor([0.5, 0.6, 0.9]))

    def test_no_sinks(self):
        self._assert_gap(torch.tensor([0.1, 0.2, 0.05]))

    def test_single_head(self):
        self._assert_gap(torch.tensor([0.9]))
        self._assert_gap(torch.tensor([0.1]))

    def test_empty_input_is_zero(self):
        result = compute_sink_head_classification(torch.tensor([]), threshold=self.THRESHOLD)
        self.assertEqual(result["sink_nonsink_gap"].item(), 0.0)
        self.assertEqual(result["sink_head_ratio"].item(), 0.0)


class MegatronActivationDumpMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()
        import tempfile

        # Scratch under the repo, NOT /tmp: act_dump's dump_dir validation rejects
        # /tmp (small, shared — filling it takes down the node), and the project rule
        # is that no test output lands there either. Keeping the fixture on a real
        # volume also means these tests exercise the same path checks production does.
        scratch = Path(__file__).resolve().parent / ".scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=str(scratch))
        self.dump_dir = Path(self._tmp.name) / "act_dumps"

    def tearDown(self):
        training_logs.reset()
        self._tmp.cleanup()

    def _list_files(self, root):
        return sorted(str(p) for p in Path(root).rglob("*.safetensors"))

    def _load(self, path):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in list(f.keys())}
        return tensors, meta

    def test_dump_written_on_monitored_step(self):
        S, B, H = 7, 2, 8
        model = FakeGPTModel(num_layers=3, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=5, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 3, f"expected one file per layer, got {files}")
        tensors, meta = self._load(files[0])
        self.assertEqual(tuple(tensors["hidden"].shape), (5, H))
        self.assertEqual(tensors["token_index"].numel(), 5)
        self.assertEqual(meta["hidden_size"], str(H))
        self.assertEqual(meta["which"], "output")
        self.assertIn("step_0000000", files[0])

    def test_random_positions_not_first_k(self):
        S, B, H = 20, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=6, token_sample_seed=123, monitor_interval=1
        )
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        files = self._list_files(self.dump_dir)
        tensors, _ = self._load(files[0])
        idx = tensors["token_index"]
        self.assertFalse(torch.equal(idx, torch.arange(6, dtype=idx.dtype)), "should not be first-K positions")

    def test_same_positions_across_layers_same_step(self):
        S, B, H = 20, 1, 8
        model = FakeGPTModel(num_layers=3, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=6, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 3)
        idxs = [self._load(f)[0]["token_index"] for f in files]
        for other in idxs[1:]:
            self.assertTrue(torch.equal(idxs[0], other), "positions must match across layers within a step")

    def test_positions_differ_across_steps(self):
        S, B, H = 20, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=6, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        model(torch.randn(S, B, H))
        monitor.step()
        d0 = self._load(self._list_files(Path(self.dump_dir) / "step_0000000")[0])[0]["token_index"]
        d1 = self._load(self._list_files(Path(self.dump_dir) / "step_0000001")[0])[0]["token_index"]
        self.assertFalse(torch.equal(d0, d1), "positions should differ across steps (seed depends on step)")

    def test_interval_gate_skips_unmonitored_step(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=4, monitor_interval=2)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))  # step_count 0 -> monitored
        monitor.step()  # -> step_count 1
        n_after_first = len(self._list_files(self.dump_dir))
        model(torch.randn(S, B, H))  # step_count 1 -> 1 % 2 != 0 -> skipped
        monitor.step()  # -> step_count 2
        self.assertEqual(len(self._list_files(self.dump_dir)), n_after_first, "no new files on unmonitored step")
        self.assertFalse((Path(self.dump_dir) / "step_0000001").exists())

    def test_sample_layers_respected(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=4, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=4, sample_layers=[1, 3], monitor_interval=1
        )
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 2)
        layers = sorted(self._load(f)[1]["layer_idx"] for f in files)
        self.assertEqual(layers, ["1", "3"])

    def test_first_microbatch_only(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=2, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=4, first_microbatch_only=True, monitor_interval=1
        )
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))  # microbatch 1
        model(torch.randn(S, B, H))  # microbatch 2 (same step) -> skipped by first_microbatch_only
        monitor.step()
        self.assertEqual(len(self._list_files(self.dump_dir)), 2, "one file per layer despite two forwards")

    def test_rank_filter_disables_dump(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=2, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=4, dump_dp_ranks=[1], monitor_interval=1
        )
        monitor.register_hooks(model)  # test runs on dp_rank 0, filter wants rank 1
        model(torch.randn(S, B, H))
        monitor.step()
        self.assertEqual(self._list_files(self.dump_dir), [], "no files when this rank is not in dump_dp_ranks")

    def test_dump_uses_global_step_when_provided(self):
        S, B, H = 4, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=4, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step(global_step=137)
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 1)
        self.assertIn("step_0000137", files[0])
        _, meta = self._load(files[0])
        self.assertEqual(meta["step"], "137")

    def test_setup_helper_registers_and_dumps(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=2, hidden_size=H, vocab_size=16)
        monitor_dict = {}
        setup_activation_dump_monitor(
            model, dump_dir=str(self.dump_dir), n_sample_tokens=4, monitor_interval=1, monitor_dict=monitor_dict
        )
        self.assertIn("act_dump", monitor_dict)
        model(torch.randn(S, B, H))
        monitor_dict["act_dump"].step()
        self.assertEqual(len(self._list_files(self.dump_dir)), 2)

    def test_rotation_keeps_only_recent_steps(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=4, max_dump_steps=2, monitor_interval=1
        )
        monitor.register_hooks(model)
        for _ in range(5):
            model(torch.randn(S, B, H))
            monitor.step()
        step_dirs = sorted(p.name for p in Path(self.dump_dir).glob("step_*"))
        self.assertEqual(step_dirs, ["step_0000003", "step_0000004"], "only the 2 most-recent steps retained")

    def test_rotation_disabled_when_none(self):
        S, B, H = 8, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir), n_sample_tokens=4, max_dump_steps=None, monitor_interval=1
        )
        monitor.register_hooks(model)
        for _ in range(4):
            model(torch.randn(S, B, H))
            monitor.step()
        self.assertEqual(len(list(Path(self.dump_dir).glob("step_*"))), 4, "no pruning when max_dump_steps=None")

    def test_channel_max_ratio_recorded_in_metadata(self):
        S, B, H = 6, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=4, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        _, meta = self._load(self._list_files(self.dump_dir)[0])
        self.assertIn("channel_max_ratio", meta)
        self.assertGreater(float(meta["channel_max_ratio"]), 0.0)

    def test_min_channel_max_ratio_filters_healthy_states(self):
        S, B, H = 6, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir),
            n_sample_tokens=4,
            monitor_interval=1,
            min_channel_max_ratio=1e9,
        )
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        self.assertEqual(self._list_files(self.dump_dir), [], "healthy activations should not hit disk")

    def test_min_channel_max_ratio_admits_spike_states(self):
        S, B, H = 6, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(
            dump_dir=str(self.dump_dir),
            n_sample_tokens=4,
            monitor_interval=1,
            min_channel_max_ratio=10.0,
            which="input",  # measure ratio on layer input so our crafted spike is visible
        )
        monitor.register_hooks(model)
        h = torch.randn(S, B, H) * 0.01
        h[..., 0] = 500.0  # blow one channel far above the median
        model(h)
        monitor.step()
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 1)
        _, meta = self._load(files[0])
        self.assertGreater(float(meta["channel_max_ratio"]), 10.0)
        self.assertEqual(meta["layer_idx"], "0")
        self.assertEqual(meta["step"], "0")
        self.assertIn("global_rank", meta)

    # ---------------- dump_dir validation ----------------

    def test_relative_dump_dir_is_resolved_against_cwd(self):
        """A relative dump_dir is supported (the default) — it lands under the run
        directory. It is stored resolved so a later os.chdir cannot move the dump."""
        import os

        rel = os.path.relpath(str(self.dump_dir), os.getcwd())
        monitor = ActivationDumpMonitor(dump_dir=rel, monitor_interval=1)
        self.assertTrue(os.path.isabs(monitor.dump_dir))
        self.assertEqual(monitor.dump_dir, os.path.abspath(str(self.dump_dir)))

    def test_dump_dir_rejects_new_top_level_dir_in_root(self):
        """Writing to '/' directly is not allowed: a dump_dir whose top-level ancestor
        does not exist (e.g. '/outputs/...', what './outputs/...' becomes when the job is
        launched from '/') must fail at construction, not after filling the disk."""
        with self.assertRaises(ValueError) as ctx:
            ActivationDumpMonitor(dump_dir="/outputs/act_dumps")
        self.assertIn("top-level", str(ctx.exception))

    def test_dump_dir_rejects_filesystem_root(self):
        with self.assertRaises(ValueError) as ctx:
            ActivationDumpMonitor(dump_dir="/")
        self.assertIn("filesystem root", str(ctx.exception))

    def test_dump_dir_rejects_small_shared_volumes(self):
        for bad in ("/tmp/act_dumps", "/dev/shm/act_dumps", "/var/tmp/x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as ctx:
                ActivationDumpMonitor(dump_dir=bad)
            self.assertIn("small/shared", str(ctx.exception))

    def test_dump_root_env_pins_allowed_prefix(self):
        import os
        from unittest import mock

        root = str(Path(self._tmp.name) / "allowed")
        os.makedirs(root, exist_ok=True)
        with mock.patch.dict(os.environ, {"INTERNAL_MEDICINE_DUMP_ROOT": root}):
            inside = ActivationDumpMonitor(dump_dir=os.path.join(root, "dumps"))
            self.assertTrue(inside.dump_dir.startswith(root))
            with self.assertRaises(ValueError) as ctx:
                ActivationDumpMonitor(dump_dir=str(Path(self._tmp.name) / "elsewhere"))
            self.assertIn("INTERNAL_MEDICINE_DUMP_ROOT", str(ctx.exception))

    def test_setup_helper_also_validates_dump_dir(self):
        model = FakeGPTModel(num_layers=1, hidden_size=8, vocab_size=16)
        with self.assertRaises(ValueError):
            setup_activation_dump_monitor(model, dump_dir="/outputs/act_dumps")

    # ---------------- full-hidden dump (default) ----------------

    def test_full_hidden_dumped_by_default(self):
        """Default n_sample_tokens=None must dump every token row, not a 512 subsample:
        a subsample cannot be lined up against the batch's input_ids / labels."""
        S, B, H = 7, 3, 8
        self.assertIsNone(ActivationDumpMonitor(dump_dir=str(self.dump_dir)).n_sample_tokens)
        hidden = torch.randn(S, B, H)
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=hidden)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        model(input_ids=torch.randint(0, 16, (B, S)))
        monitor.step()
        act = [f for f in self._list_files(self.dump_dir) if "batch" not in Path(f).name]
        tensors, meta = self._load(act[0])
        self.assertEqual(tuple(tensors["hidden"].shape), (S * B, H))
        self.assertEqual(meta["full_dump"], "True")
        self.assertEqual(meta["n_tokens"], str(S * B))
        self.assertEqual(meta["n_sample_tokens"], str(S * B))
        # token_index is the identity permutation on a full dump.
        self.assertTrue(torch.equal(tensors["token_index"], torch.arange(S * B)))

    def test_hidden_values_match_layer_output_on_full_dump(self):
        """The dumped rows must be the actual layer output, seq-major flattened."""
        S, B, H = 5, 2, 8
        hidden = torch.randn(S, B, H)
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=hidden)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        with torch.no_grad():
            want, _ = model.decoder.layers[0](hidden_states=hidden)
        model(input_ids=torch.randint(0, 16, (B, S)))
        monitor.step()
        act = [f for f in self._list_files(self.dump_dir) if "batch" not in Path(f).name]
        got = self._load(act[0])[0]["hidden"]
        self.assertTrue(torch.allclose(got, want.reshape(-1, H), atol=1e-6))

    def test_explicit_n_sample_tokens_still_subsamples(self):
        S, B, H = 20, 1, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=16)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), n_sample_tokens=6, monitor_interval=1)
        monitor.register_hooks(model)
        model(torch.randn(S, B, H))
        monitor.step()
        tensors, meta = self._load(self._list_files(self.dump_dir)[0])
        self.assertEqual(tuple(tensors["hidden"].shape), (6, H))
        self.assertEqual(meta["full_dump"], "False")

    # ---------------- input-batch dump ----------------

    def test_input_batch_dumped_alongside_hidden(self):
        """input_ids / labels / position_ids land in a batch_*.safetensors next to the
        activations, byte-identical to what was fed to the forward."""
        S, B, H = 6, 2, 8
        hidden = torch.randn(S, B, H)
        model = FakeBatchGPTModel(num_layers=2, hidden_size=H, hidden=hidden)
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        ids = torch.randint(0, 16, (B, S))
        labels = torch.randint(0, 16, (B, S))
        pos = torch.arange(S).expand(B, S).contiguous()
        model(input_ids=ids, position_ids=pos, attention_mask=None, labels=labels)
        monitor.step()

        files = self._list_files(self.dump_dir)
        batch = [f for f in files if "batch" in Path(f).name]
        self.assertEqual(len(batch), 1, f"expected exactly one batch file, got {files}")
        self.assertEqual(len([f for f in files if "batch" not in Path(f).name]), 2, "one hidden file per layer")
        tensors, meta = self._load(batch[0])
        self.assertTrue(torch.equal(tensors["input_ids"], ids))
        self.assertTrue(torch.equal(tensors["labels"], labels))
        self.assertTrue(torch.equal(tensors["position_ids"], pos))
        self.assertEqual(meta["kind"], "input_batch")
        self.assertEqual(meta["step"], "0")
        self.assertEqual(meta["input_ids_shape"], str((B, S)))
        # attention_mask was None; must not appear as a key.
        self.assertNotIn("attention_mask", tensors)

    def test_packed_seq_params_dumped_without_cp_group(self):
        """PackedSeqParams tensors are persisted; the non-serialisable cp_group is not."""
        S, B, H = 6, 2, 8
        psp = FakePackedSeqParams()
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        model(input_ids=torch.randint(0, 16, (B, S)), packed_seq_params=psp)
        monitor.step()

        batch = [f for f in self._list_files(self.dump_dir) if "batch" in Path(f).name]
        tensors, meta = self._load(batch[0])
        self.assertTrue(torch.equal(tensors["packed_seq_params.cu_seqlens_q"], psp.cu_seqlens_q))
        self.assertTrue(torch.equal(tensors["packed_seq_params.seq_idx"], psp.seq_idx))
        self.assertEqual(meta["packed_seq_params_present"], "True")
        self.assertEqual(meta["packed_seq_params.qkv_format"], "thd")
        self.assertEqual(meta["packed_seq_params.total_tokens"], "12")
        for key in list(tensors) + list(meta):
            self.assertNotIn("cp_group", key, "cp_group is a ProcessGroup; must never be persisted")
        # cu_seqlens_q_padded is None on this batch -> absent, not a null entry.
        self.assertNotIn("packed_seq_params.cu_seqlens_q_padded", tensors)

    def test_packed_seq_params_tensor_max_seqlen_avoids_host_sync(self):
        """This repo's own get_packed_seq_params (src/trainers/gpt_step_fix_cp.py) sets
        max_seqlen_q/kv from ``batch["max_seqlen"].squeeze()`` — 0-dim TENSORS, not ints.
        str()/int() on those would D2H-sync inside the pre-hook, so they must be routed
        through the async copy and land as tensors, not metadata."""
        S, B, H = 6, 1, 8
        psp = FakePackedSeqParams()
        psp.max_seqlen_q = torch.tensor(4, dtype=torch.int32)  # 0-dim tensor, as in this repo
        psp.max_seqlen_kv = torch.tensor(4, dtype=torch.int32)
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        model(input_ids=torch.randint(0, 16, (B, S)), packed_seq_params=psp)
        monitor.step()

        batch = [f for f in self._list_files(self.dump_dir) if "batch" in Path(f).name]
        tensors, meta = self._load(batch[0])
        self.assertIn("packed_seq_params.max_seqlen_q", tensors)
        self.assertEqual(int(tensors["packed_seq_params.max_seqlen_q"]), 4)
        self.assertNotIn("packed_seq_params.max_seqlen_q", meta, "tensor must not be stringified")
        # qkv_format is a genuine python str -> still metadata.
        self.assertEqual(meta["packed_seq_params.qkv_format"], "thd")

    def test_batch_dump_disabled(self):
        S, B, H = 6, 1, 8
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1, dump_input_batch=False)
        monitor.register_hooks(model)
        model(input_ids=torch.randint(0, 16, (B, S)))
        monitor.step()
        files = self._list_files(self.dump_dir)
        self.assertEqual(len(files), 1)
        self.assertNotIn("batch", Path(files[0]).name)

    def test_batch_not_written_when_ratio_gate_drops_every_dump(self):
        """No hidden file survives the gate -> no orphan batch file either."""
        S, B, H = 6, 1, 8
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1, min_channel_max_ratio=1e9)
        monitor.register_hooks(model)
        model(input_ids=torch.randint(0, 16, (B, S)))
        monitor.step()
        self.assertEqual(self._list_files(self.dump_dir), [])

    def test_batch_captured_from_first_microbatch_only(self):
        """Two forwards in one step -> the batch on disk is the FIRST one, matching the
        first-microbatch hidden states it sits next to."""
        S, B, H = 6, 1, 8
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        first = torch.randint(0, 16, (B, S))
        second = torch.randint(100, 116, (B, S))
        model(input_ids=first)
        model(input_ids=second)
        monitor.step()
        batch = [f for f in self._list_files(self.dump_dir) if "batch" in Path(f).name]
        self.assertEqual(len(batch), 1)
        got = self._load(batch[0])[0]["input_ids"]
        self.assertTrue(torch.equal(got, first), "batch must come from the first microbatch")

    def test_batch_state_reset_between_steps(self):
        S, B, H = 6, 1, 8
        model = FakeBatchGPTModel(num_layers=1, hidden_size=H, hidden=torch.randn(S, B, H))
        monitor = ActivationDumpMonitor(dump_dir=str(self.dump_dir), monitor_interval=1)
        monitor.register_hooks(model)
        step0 = torch.randint(0, 16, (B, S))
        step1 = torch.randint(100, 116, (B, S))
        model(input_ids=step0)
        monitor.step()
        model(input_ids=step1)
        monitor.step()
        got0 = self._load(self._list_files(Path(self.dump_dir) / "step_0000000")[0])[0]
        b1 = [f for f in self._list_files(Path(self.dump_dir) / "step_0000001") if "batch" in Path(f).name]
        got1 = self._load(b1[0])[0]
        self.assertTrue(torch.equal(got0["input_ids"], step0))
        self.assertTrue(torch.equal(got1["input_ids"], step1), "step 1 must capture its own batch")


def _lar_analytical(hidden, weight, logits=None):
    """RMS-based LAR reference: hidden [T,H], weight [V,H], logits [T,V] (or None => hidden @ weight.T)."""
    H = weight.shape[1]
    x = hidden.reshape(-1, H).detach().float()
    w = weight.detach().float()
    z = (x @ w.t()) if logits is None else logits.reshape(-1, weight.shape[0]).detach().float()
    rms_w = w.pow(2).mean().sqrt()
    rms_x = x.pow(2).mean().sqrt()
    rms_z = z.pow(2).mean().sqrt()
    lar = float(torch.log(rms_z / (rms_w * rms_x)) / math.log(H))
    return lar, rms_w.item(), rms_x.item(), rms_z.item()


def _lar_svd(hidden, weight):
    """Spectral cross-check: LAR = 1 + 0.5 * log_n(Σ p_i q_i)  (spec §5)."""
    W = weight.detach().float()
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    p = S**2
    p = p / p.sum()
    H = W.shape[1]
    x = hidden.reshape(-1, H).float()
    proj = x @ Vh.t()
    q = proj.pow(2).sum(0)
    q = q / q.sum()
    return 1.0 + 0.5 * float(torch.log((p * q).sum()) / math.log(H))


class _FakeRouter(nn.Module):
    """Minimal MoE router: gating linear returning (probs, routing_map) — the real
    Megatron TopKRouter contract, from which LARMonitor must NOT read logits."""

    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))

    def forward(self, hidden):
        logits = F.linear(hidden, self.weight)
        probs = F.softmax(logits, dim=-1)
        routing_map = probs.argmax(dim=-1, keepdim=True)
        return probs, routing_map


class _FakeMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.router = _FakeRouter(hidden_size, num_experts)

    def forward(self, hidden):
        self.router(hidden)
        return hidden


class _FakeMoEBlock(nn.Module):
    """Wraps FakeGPTModel with an MoE-style .mlp on each decoder layer."""

    def __init__(self, num_layers, hidden_size, num_experts, vocab_size):
        super().__init__()
        decoder_layers = [SimpleNamespace(mlp=_FakeMoELayer(hidden_size, num_experts)) for _ in range(num_layers)]
        # register the router as a real submodule so parameters are visible.
        self._routers = nn.ModuleList([layer.mlp.router for layer in decoder_layers])
        self.decoder = SimpleNamespace(
            layers=decoder_layers,
            final_layernorm=nn.LayerNorm(hidden_size),
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden, labels=None):
        # run every router so their forward hooks fire
        for layer in self.decoder.layers:
            layer.mlp(hidden)
        return self.output_layer(hidden)


class MegatronLARMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_lar_lm_head_matches_analytical(self):
        S, B, H, V = 6, 1, 8, 12
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        # Drive output_layer directly with a known hidden so analytical matches exactly.
        model.output_layer(hidden)
        monitor.step()
        got = training_logs.get_latest(prefix="lar")
        want_lar, *_ = _lar_analytical(hidden, model.output_layer.weight, logits=model.output_layer(hidden))
        self.assertAlmostEqual(got["lar/lm_head/lar"], want_lar, places=4)
        self.assertAlmostEqual(got["lar/global_lm_head_lar"], want_lar, places=4)

    def test_lar_logs_only_lar(self):
        """The emitted schema is exactly {lar} per site, plus the lar globals.

        rms_w / rms_x / rms_z are flush-time intermediates — only their combination
        (lar) is meaningful, and raw activation scale belongs to massive_act.
        k = H**(2*(1-lar)) was removed as a monotone reparametrisation of lar carrying
        no extra information. valid_frac was removed: it reported nX/H clamped to 1.0,
        i.e. always exactly 1.0, and a truthful keep-rate needs a pre-mask token count
        the hooks discard."""
        S, B, H, V = 6, 1, 8, 12
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        model.output_layer(torch.randn(S, B, H))
        monitor.step()
        got = training_logs.get_latest(prefix="lar")
        self.assertEqual(set(got), {"lar/lm_head/lar", "lar/global_lm_head_lar"})

    def test_lar_svd_cross_check(self):
        S, B, H, V = 8, 1, 6, 10
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        model.output_layer(hidden)
        monitor.step()
        got_lar = training_logs.get_latest(prefix="lar")["lar/lm_head/lar"]
        svd_lar = _lar_svd(hidden.reshape(-1, H), model.output_layer.weight)
        self.assertAlmostEqual(got_lar, svd_lar, places=3)

    def test_lar_router_matches_analytical(self):
        S, B, H, E, V = 4, 1, 6, 5, 8
        model = _FakeMoEBlock(num_layers=1, hidden_size=H, num_experts=E, vocab_size=V)
        monitor = LARMonitor(hook_lm_head=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        model(hidden)
        monitor.step()
        got = training_logs.get_latest(prefix="lar")
        router_weight = model.decoder.layers[0].mlp.router.weight
        want_lar, *_ = _lar_analytical(hidden, router_weight, logits=hidden.reshape(-1, H) @ router_weight.t())
        self.assertAlmostEqual(got["lar/router_0/lar"], want_lar, places=4)
        self.assertAlmostEqual(got["lar/global_router_lar"], want_lar, places=4)

    def test_lar_multi_microbatch_accumulation(self):
        S, B, H, V = 4, 1, 6, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        h1 = torch.randn(S, B, H)
        h2 = torch.randn(S, B, H)
        h3 = torch.randn(S, B, H)
        model.output_layer(h1)
        model.output_layer(h2)
        model.output_layer(h3)
        monitor.step()
        got = training_logs.get_latest(prefix="lar")["lar/lm_head/lar"]
        big = torch.cat([h1.reshape(-1, H), h2.reshape(-1, H), h3.reshape(-1, H)], dim=0)
        want, *_ = _lar_analytical(big, model.output_layer.weight, logits=big @ model.output_layer.weight.t())
        self.assertAlmostEqual(got, want, places=4)

    def test_lar_interval_gate(self):
        S, B, H, V = 4, 1, 6, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=5, apply_loss_mask=False)
        monitor.register_hooks(model)
        # step_count=0 before the very first step(), so _should_monitor() would fire, but
        # we exercise a step that is NOT a multiple of 5.
        monitor.step_count = 1  # ensure next _should_monitor() check is 1 % 5 != 0
        model.output_layer(torch.randn(S, B, H))
        monitor.step()  # step_count 1 -> 2
        self.assertEqual(training_logs.get_latest(prefix="lar"), {})

    def test_lar_lm_head_absent_on_middle_stage(self):
        H = 6
        model = FakeHeadlessGPTModel(num_layers=1, hidden_size=H)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)  # must not crash and must produce no site
        model(torch.randn(4, 1, H))
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="lar"), {})

    def test_lar_tied_weights_uses_shared_tensor(self):
        S, B, H, V = 4, 1, 6, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        # Simulate tied embeddings: swap output_layer.weight with a shared tensor.
        shared = nn.Parameter(torch.randn(V, H) * 0.5)
        model.output_layer.weight = shared
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        logits = model.output_layer(hidden)
        monitor.step()
        # ``lar`` depends on ||W||_rms, so matching the analytical value computed from
        # ``shared`` pins that the shared tensor (not a stale one) fed the weight stats.
        want_lar, *_ = _lar_analytical(hidden, shared, logits=logits)
        self.assertAlmostEqual(training_logs.get_latest(prefix="lar")["lar/lm_head/lar"], want_lar, places=4)

    def test_lar_lm_head_resolves_via_shared_accessor_when_weight_none(self):
        """Tied embeddings + PP=1: ``output_layer.weight`` is None, so the monitor must
        resolve the weight through ``shared_embedding_or_output_weight()`` at attach
        time and close over it — reading ``module.weight`` in the hook yields None and
        would silently drop the site."""
        S, B, H, V = 4, 1, 6, 8
        model = FakeTiedGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        self.assertIsNone(model.output_layer.weight)  # guard: fixture models the real case

        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        self.assertIn("lm_head", monitor._sites)
        self.assertEqual(monitor._sites["lm_head"]["H"], H)

        # Drive the real forward so the hook fires with module.weight still None.
        hidden = torch.randn(S, B, H)
        logits = model(hidden)
        monitor.step()

        got = training_logs.get_latest(prefix="lar")
        # ``hidden`` is the decoder input; the head sees the decoder output.
        head_input = model.decoder(hidden)
        want_lar, *_ = _lar_analytical(head_input, model.shared_weight, logits=logits)
        self.assertAlmostEqual(got["lar/lm_head/lar"], want_lar, places=4)

    def test_lar_lm_head_masking_matches_manual_index(self):
        S, B, H, V = 5, 1, 4, 6
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=True)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        labels = torch.tensor([[1, 2, -100, 4, -100]])  # [B, S] -> valid at positions 0,1,3
        # Drive the head with the model so the label-capture pre-hook fires.
        model(hidden, labels=labels)  # note: FakeGPTModel(x) ignores labels but pre-hook still sees them
        # The pre-hook is on the model, but model(hidden, labels=...) doesn't call output_layer;
        # so drive output_layer separately using the same captured labels.
        model.output_layer(hidden)
        monitor.step()
        got = training_logs.get_latest(prefix="lar")
        mask = labels.transpose(0, 1).reshape(-1) != -100  # seq-major
        hidden_masked = hidden.reshape(-1, H)[mask]
        logits_masked = model.output_layer(hidden).reshape(-1, V)[mask]
        want_lar, *_ = _lar_analytical(hidden_masked, model.output_layer.weight, logits=logits_masked)
        self.assertAlmostEqual(got["lar/lm_head/lar"], want_lar, places=4)

    def test_lar_falls_back_when_labels_absent(self):
        S, B, H, V = 4, 1, 6, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=True)
        monitor.register_hooks(model)
        model.output_layer(torch.randn(S, B, H))  # no labels captured -> fall back
        monitor.step()
        got = training_logs.get_latest(prefix="lar")
        self.assertIn("lar/lm_head/lar", got)  # still emitted; no crash

    def test_lar_router_masking_uses_captured_labels(self):
        S, B, H, E, V = 4, 1, 6, 5, 8
        model = _FakeMoEBlock(num_layers=1, hidden_size=H, num_experts=E, vocab_size=V)
        monitor = LARMonitor(hook_lm_head=False, monitor_interval=1, apply_loss_mask=True)
        monitor.register_hooks(model)
        hidden = torch.randn(S, B, H)
        labels = torch.tensor([[7, -100, 3, -100]])  # valid at 0, 2
        model(hidden, labels=labels)
        monitor.step()
        got = training_logs.get_latest(prefix="lar")["lar/router_0/lar"]
        router_weight = model.decoder.layers[0].mlp.router.weight
        mask = labels.transpose(0, 1).reshape(-1) != -100
        hidden_masked = hidden.reshape(-1, H)[mask]
        want, *_ = _lar_analytical(hidden_masked, router_weight, logits=hidden_masked @ router_weight.t())
        self.assertAlmostEqual(got, want, places=4)


if __name__ == "__main__":
    unittest.main()
