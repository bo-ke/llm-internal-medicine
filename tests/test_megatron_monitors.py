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
optim_update_module = importlib.import_module("internal_medicine.backends.megatron.optim_update_monitor")
OptimUpdateMonitor = optim_update_module.OptimUpdateMonitor
setup_optim_update_monitor = optim_update_module.setup_optim_update_monitor
MoESpecialistMonitor = importlib.import_module("internal_medicine.backends.megatron.moe_monitor").MoESpecialistMonitor
moe_monitor_module = importlib.import_module("internal_medicine.backends.megatron.moe_monitor")
PLEHealthMonitor = importlib.import_module("internal_medicine.backends.megatron.ple_monitor").PLEHealthMonitor
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs
megatron_backend = importlib.import_module("internal_medicine.backends.megatron")
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


class FakeTokenDispatcher:
    """Stand-in for mcore's MoETokenDispatcher.

    Deliberately a PLAIN object, not an nn.Module — the real
    ``megatron.core.transformer.moe.token_dispatcher.MoETokenDispatcher`` is too, which
    is why the monitor patches ``combine_postprocess`` on the instance instead of
    registering a forward hook.
    """

    def __init__(self):
        self.combined = None

    def combine_postprocess(self, expert_outputs, probs):
        combined = sum(p * e for p, e in zip(expert_outputs, probs, strict=True))
        self.combined = combined
        return combined


class FakeLatentRMSNorm(nn.Module):
    """RMSNorm on the latent dim, as some models insert between combine and up-proj.

    Its output RMS is ~1 by construction, so a monitor that measured
    ``fc2_latent_proj``'s INPUT would read a constant regardless of how large the
    combined expert output actually was.
    """

    def __init__(self, latent_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(latent_size))
        self.eps = eps

    def forward(self, x):
        rms = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).type_as(x) * self.weight


class FakeLatentMoELayer(nn.Module):
    """Latent MoE layer reproducing mcore's MoELayer.postprocess ordering.

    ``forward`` mirrors the real path: experts produce per-expert outputs in LATENT dim,
    ``token_dispatcher.combine_postprocess`` sums them k-way with router probs, then
    ``fc2_latent_proj`` projects latent -> hidden. With ``latent_norm=True`` an RMSNorm
    sits between the two, which is the case that forced the measurement point to move to
    the combine's output.
    """

    def __init__(self, hidden_size, latent_size, num_experts, topk=2, latent_norm=False):
        super().__init__()
        self.router = _FakeRouter(latent_size, num_experts)
        self.fc1_latent_proj = nn.Linear(hidden_size, latent_size, bias=False)
        self.fc2_latent_proj = FakeLatentProj(latent_size, hidden_size)
        self.token_dispatcher = FakeTokenDispatcher()
        self.latent_norm = FakeLatentRMSNorm(latent_size) if latent_norm else None
        self.topk = topk
        self.num_experts = num_experts

    @property
    def combined(self):
        """The tensor combine_postprocess returned on the last forward."""
        return self.token_dispatcher.combined

    def forward(self, hidden_states, expert_outputs=None, probs=None):
        latent = self.fc1_latent_proj(hidden_states)
        self.router(latent)
        if expert_outputs is None:
            expert_outputs = [latent for _ in range(self.topk)]
            probs = [1.0 / self.topk] * self.topk
        combined = self.token_dispatcher.combine_postprocess(expert_outputs, probs)
        self.up_proj_input = self.latent_norm(combined) if self.latent_norm is not None else combined
        out, _ = self.fc2_latent_proj(self.up_proj_input)
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

    def _sharpness_monitor(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._ROUTER_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))
        return monitor

    def test_combine_coef_sharpness_is_mean_over_tokens_of_max_over_median(self):
        """Per-token max/median over the topk combine coefficients, then MEAN over
        tokens: token 0 is an even 2-way split (ratio 1.0), token 1 is dominated by one
        expert (0.9/0.1 = 9.0), so the reported value is their mean."""
        monitor = self._sharpness_monitor()
        probs = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        router = SimpleNamespace(topk=2, _cached_scores_for_aux_loss=None)
        monitor._compute_router_metrics(0, router, (probs, None), None)
        monitor.step()
        got = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/combine_coef_max_median_ratio"]
        self.assertAlmostEqual(got, (1.0 + 9.0) / 2, places=5)

    def test_combine_coef_sharpness_is_one_for_uniform_combine(self):
        """A perfectly even k-way blend must read exactly 1.0 — the calibration point."""
        monitor = self._sharpness_monitor()
        probs = torch.zeros(6, 8)
        probs[:, :4] = 0.25  # 4 experts, equal coefficients
        router = SimpleNamespace(topk=4, _cached_scores_for_aux_loss=None)
        monitor._compute_router_metrics(0, router, (probs, None), None)
        monitor.step()
        got = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/combine_coef_max_median_ratio"]
        self.assertAlmostEqual(got, 1.0, places=5)

    def test_combine_coef_sharpness_grows_as_routing_peaks(self):
        monitor = self._sharpness_monitor()
        soft = torch.tensor([[0.4, 0.35, 0.25, 0.0]])
        router = SimpleNamespace(topk=3, _cached_scores_for_aux_loss=None)
        monitor._compute_router_metrics(0, router, (soft, None), None)
        monitor.step()
        soft_val = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/combine_coef_max_median_ratio"]

        training_logs.reset()
        monitor = self._sharpness_monitor()
        peaked = torch.tensor([[0.96, 0.03, 0.01, 0.0]])
        monitor._compute_router_metrics(0, router, (peaked, None), None)
        monitor.step()
        peaked_val = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/combine_coef_max_median_ratio"]

        self.assertAlmostEqual(soft_val, 0.4 / 0.35, places=5)
        self.assertAlmostEqual(peaked_val, 0.96 / 0.03, places=4)
        self.assertGreater(peaked_val, soft_val)

    def test_combine_coef_sharpness_reads_router_probs_not_aux_scores(self):
        """The coefficients are the router's FINAL probs (outputs[0]). The cached
        aux-loss scores are pre-topk and skip renormalisation/scaling, so using them
        would report a different number — pin that we use the right tensor."""
        monitor = self._sharpness_monitor()
        probs = torch.tensor([[0.8, 0.2, 0.0, 0.0]])  # ratio 4.0
        aux_scores = torch.tensor([[0.3, 0.3, 0.2, 0.2]])  # would give ratio 1.0
        router = SimpleNamespace(topk=2, _cached_scores_for_aux_loss=aux_scores)
        monitor._compute_router_metrics(0, router, (probs, None), None)
        monitor.step()
        got = training_logs.get_latest(prefix="moe_health")["moe_health/layer_0/combine_coef_max_median_ratio"]
        self.assertAlmostEqual(got, 4.0, places=5)

    def test_combine_coef_sharpness_skipped_at_topk_1(self):
        """At topk == 1 max == median identically, so the metric would be a constant
        1.0 carrying no information — it must not be emitted at all."""
        monitor = self._sharpness_monitor()
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        router = SimpleNamespace(topk=1, _cached_scores_for_aux_loss=None)
        monitor._compute_router_metrics(0, router, (probs, None), None)
        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertNotIn("moe_health/layer_0/combine_coef_max_median_ratio", latest)

    def test_combine_coef_sharpness_is_mean_aggregated(self):
        """Unlike the latent-combine metrics, this one is MEAN-composed: it already
        summarises a distribution over tokens, and the ask is the typical token."""
        name = "combine_coef_max_median_ratio"
        self.assertNotIn(name, MoESpecialistMonitor.MAX_AGGREGATED)
        self.assertNotIn(name, MoESpecialistMonitor.MIN_AGGREGATED)
        self.assertFalse(training_logs._is_max_metric(f"moe_health/layer_0/{name}"))
        self.assertFalse(training_logs._is_min_metric(f"moe_health/layer_0/{name}"))

    def test_combine_coef_sharpness_end_to_end_through_router_hook(self):
        """Fires through the real forward-hook path, not a direct _compute call."""
        S, B, H, L, E = 4, 1, 8, 6, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        layer.router.topk = 2
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            layer(torch.randn(S, B, H))
            monitor.step()
            latest = training_logs.get_latest(prefix="moe_health")
            self.assertIn("moe_health/layer_0/combine_coef_max_median_ratio", latest)
            self.assertIn("moe_health/global_combine_coef_max_median_ratio", latest)
            self.assertGreaterEqual(latest["moe_health/layer_0/combine_coef_max_median_ratio"], 1.0)
        finally:
            monitor.remove_hooks()

    def _latent_eps_layer(self, hidden_size=8, latent_size=4, num_experts=4, eps=1e-5):
        layer = FakeLatentMoELayer(hidden_size=hidden_size, latent_size=latent_size, num_experts=num_experts)
        layer.config = SimpleNamespace(layernorm_epsilon=eps)
        return layer

    @staticmethod
    def _latent_eps_layer_resolved_eps(layer):
        return MoESpecialistMonitor._layernorm_eps_of(layer)

    def _run_latent_eps(self, layer, expert_out):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            layer(
                torch.randn(expert_out.shape[0], expert_out.shape[1], layer.fc1_latent_proj.in_features),
                expert_outputs=[expert_out],
                probs=[1.0],
            )
            monitor.step()
            return training_logs.get_latest(prefix="moe_health")
        finally:
            monitor.remove_hooks()

    def test_latent_eps_ratio_is_negligible_for_healthy_activations(self):
        """O(1) activations => eps contributes ~nothing to the RMSNorm denominator."""
        eps = 1e-5
        latest = self._run_latent_eps(self._latent_eps_layer(eps=eps), torch.ones(4, 2, 4))
        got = latest["moe_health/layer_0/latent_eps_ratio"]
        self.assertAlmostEqual(got, eps / (1.0 + eps), places=7)
        self.assertLess(got, 1e-4)

    def test_latent_eps_ratio_approaches_one_when_eps_dominates(self):
        """Tiny activations => the norm divides by eps, not by the token's own scale.
        This is the failure the metric exists to surface."""
        eps = 1e-5
        latest = self._run_latent_eps(self._latent_eps_layer(eps=eps), torch.full((4, 2, 4), 1e-4))
        got = latest["moe_health/layer_0/latent_eps_ratio"]
        self.assertAlmostEqual(got, eps / (1e-8 + eps), places=5)
        self.assertGreater(got, 0.99)

    def test_latent_eps_ratio_is_half_at_crossover(self):
        """mean(h**2) == eps is the 50/50 point — the readable calibration anchor."""
        eps = 1e-5
        coef = torch.full((4, 2, 4), eps**0.5)
        got = self._run_latent_eps(self._latent_eps_layer(eps=eps), coef)["moe_health/layer_0/latent_eps_ratio"]
        self.assertAlmostEqual(got, 0.5, places=5)

    def test_latent_eps_ratio_is_per_token_not_global(self):
        """One healthy token + one eps-floored token must average to ~0.5. A single
        GLOBAL mean(h**2) would instead read ~2e-5 and hide the floored token entirely —
        that is why the ratio is formed per token, matching how RMSNorm normalises."""
        eps = 1e-5
        coef = torch.zeros(2, 1, 4)
        coef[0] = 1.0  # healthy token
        coef[1] = 1e-5  # mean_sq 1e-10 -> eps-dominated
        got = self._run_latent_eps(self._latent_eps_layer(eps=eps), coef)["moe_health/layer_0/latent_eps_ratio"]
        want_per_token = (eps / (1.0 + eps) + eps / (1e-10 + eps)) / 2
        self.assertAlmostEqual(got, want_per_token, places=5)

        global_form = eps / (float(coef.reshape(-1, 4).square().mean()) + eps)
        self.assertLess(global_form, 1e-4)  # the form we did NOT use
        self.assertGreater(got, 0.4)

    def test_latent_eps_ratio_absent_without_layernorm_epsilon(self):
        """No config eps => no guessing a default; the metric is simply not emitted."""
        layer = FakeLatentMoELayer(hidden_size=8, latent_size=4, num_experts=4)  # no .config
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            self.assertIsNone(monitor._layernorm_eps_of(layer))
            layer(torch.randn(4, 1, 8))
            monitor.step()
            latest = training_logs.get_latest(prefix="moe_health")
            self.assertNotIn("moe_health/layer_0/latent_eps_ratio", latest)
            self.assertIn("moe_health/layer_0/latent_combine_rms", latest)
        finally:
            monitor.remove_hooks()

    def test_latent_eps_ratio_is_mean_aggregated(self):
        """Already a per-token average describing the typical token -> mean-composed,
        unlike its max-aggregated latent_combine_* siblings."""
        name = "latent_eps_ratio"
        self.assertNotIn(name, MoESpecialistMonitor.MAX_AGGREGATED)
        self.assertNotIn(name, MoESpecialistMonitor.MIN_AGGREGATED)
        self.assertFalse(training_logs._is_max_metric(f"moe_health/layer_0/{name}"))
        self.assertFalse(training_logs._is_min_metric(f"moe_health/layer_0/{name}"))

    def test_latent_eps_ratio_prefers_the_latent_norms_own_eps(self):
        """A latent norm may carry its own eps; the ratio must use THAT, not the global
        knob. Normalising u (natural scale ~1e-3) is a different regime from a unit-scale
        hidden state, and moving layernorm_epsilon would move five other norms with it —
        so a model that sets moe_latent_norm_eps is running a norm the global value does
        not describe, and a ratio computed from the global value would read identically
        to the un-modified baseline."""
        layer = self._latent_eps_layer(eps=1e-5)
        layer.config.moe_latent_norm_eps = 1e-8
        self.assertEqual(self._latent_eps_layer_resolved_eps(layer), 1e-8)
        latest = self._run_latent_eps(layer, torch.full((4, 2, 4), 1e-3))
        got = latest["moe_health/layer_0/latent_eps_ratio"]
        # mean(h**2) == 1e-6, so the eps that is actually in the denominator is decisive:
        # 1e-8 -> ~0.01 (the norm normalises), 1e-5 -> ~0.91 (eps swamps it).
        self.assertAlmostEqual(got, 1e-8 / (1e-6 + 1e-8), places=5)
        self.assertLess(got, 0.02)

    def test_latent_eps_ratio_falls_back_to_the_global_knob(self):
        """Regression guard: a model WITHOUT the per-norm field must read exactly as
        before — every model that does not set it is unaffected."""
        layer = self._latent_eps_layer(eps=1e-5)
        self.assertFalse(hasattr(layer.config, "moe_latent_norm_eps"))
        self.assertEqual(self._latent_eps_layer_resolved_eps(layer), 1e-5)
        latest = self._run_latent_eps(layer, torch.full((4, 2, 4), 1e-3))
        self.assertAlmostEqual(latest["moe_health/layer_0/latent_eps_ratio"], 1e-5 / (1e-6 + 1e-5), places=5)

        # An explicit None means "not materialised" and must behave like absent.
        layer_none = self._latent_eps_layer(eps=1e-5)
        layer_none.config.moe_latent_norm_eps = None
        self.assertEqual(self._latent_eps_layer_resolved_eps(layer_none), 1e-5)

        # 0.0 is NOT a fallback trigger: a norm running at eps=0 is described by 0, and
        # reporting the global 1e-5 instead would be the same silent mismatch the
        # prefers-the-real-eps rule exists to prevent.
        layer_zero = self._latent_eps_layer(eps=1e-5)
        layer_zero.config.moe_latent_norm_eps = 0.0
        self.assertEqual(self._latent_eps_layer_resolved_eps(layer_zero), 0.0)

    def test_latent_combine_metrics_measure_combine_postprocess_output(self):
        """The metrics must describe exactly what combine_postprocess returned — the
        k-way-combined expert output in LATENT dim."""
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

    def test_latent_combine_survives_rmsnorm_before_up_projection(self):
        """THE reason the measurement point is combine_postprocess and not
        fc2_latent_proj's input: with an RMSNorm in between, the up-proj input has RMS
        ~1 by construction. Measuring there would report ~1 no matter how large the
        combined expert output grew — a silently useless magnitude metric."""
        S, B, H, L, E = 4, 2, 8, 6, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E, latent_norm=True)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])
        try:
            big = torch.full((S, B, L), 40.0)
            big[..., 0] = 400.0
            layer(torch.randn(S, B, H), expert_outputs=[big], probs=[1.0])
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            got_rms = latest["moe_health/layer_0/latent_combine_rms"]

            combined = layer.combined.detach().reshape(-1, L).float()
            want_rms = float(combined.square().mean().sqrt())
            self.assertAlmostEqual(got_rms, want_rms, places=4)

            normed_rms = float(layer.up_proj_input.detach().reshape(-1, L).float().square().mean().sqrt())
            self.assertAlmostEqual(normed_rms, 1.0, places=3)
            self.assertGreater(got_rms, 100.0, "must report the pre-norm magnitude")

            got_ratio = latest["moe_health/layer_0/latent_combine_channel_max_median_ratio"]
            per_channel_max = combined.abs().amax(dim=0)
            self.assertAlmostEqual(got_ratio, float(per_channel_max.max() / per_channel_max.median()), places=4)
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
            layers[0](torch.randn(S, B, H), expert_outputs=[torch.ones(S, B, L)], probs=[1.0])
            hot = torch.ones(S, B, L) * 3.0
            hot[..., 0] = 300.0
            layers[1](torch.randn(S, B, H), expert_outputs=[hot], probs=[1.0])
            monitor.step()

            latest = training_logs.get_latest(prefix="moe_health")
            for name in moe_monitor_module._LATENT_COMBINE_METRICS:
                per_layer = [latest[f"moe_health/layer_{i}/{name}"] for i in range(2)]
                self.assertAlmostEqual(latest[f"moe_health/global_{name}"], max(per_layer), places=5)
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

    def test_latent_combine_respects_monitor_interval(self):
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

    def test_combine_postprocess_patch_is_restored_and_transparent(self):
        """The patch must return combine_postprocess's value untouched and be fully
        reverted by remove_hooks — a monkey-patch that leaks would corrupt the model for
        any later run in the same process."""
        S, B, H, L, E = 4, 1, 8, 4, 4
        layer = FakeLatentMoELayer(hidden_size=H, latent_size=L, num_experts=E)
        dispatcher = layer.token_dispatcher
        original = dispatcher.combine_postprocess

        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        monitor._attach_hooks([(0, layer)])

        self.assertIsNot(dispatcher.combine_postprocess, original, "patch installed")
        self.assertTrue(dispatcher._im_combine_patched)

        expert_out = torch.randn(S, B, L)
        out = layer(torch.randn(S, B, H), expert_outputs=[expert_out], probs=[1.0])
        self.assertTrue(torch.allclose(layer.combined, expert_out))
        self.assertEqual(out.shape[-1], H)
        monitor.step()
        self.assertIn("moe_health/layer_0/latent_combine_rms", training_logs.get_latest(prefix="moe_health"))

        monitor.remove_hooks()
        self.assertNotIn("combine_postprocess", vars(dispatcher))
        self.assertIs(dispatcher.combine_postprocess.__func__, original.__func__)
        self.assertFalse(hasattr(dispatcher, "_im_combine_patched"))
        self.assertFalse(hasattr(dispatcher, "_im_original_combine_postprocess"))

    def test_combine_postprocess_patch_is_not_applied_twice(self):
        layer = FakeLatentMoELayer(hidden_size=8, latent_size=4, num_experts=4)
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        try:
            monitor._attach_hooks([(0, layer)])
            patched_once = layer.token_dispatcher.combine_postprocess
            monitor._patch_combine_postprocess(0, layer)  # idempotent
            self.assertIs(layer.token_dispatcher.combine_postprocess, patched_once)
            self.assertEqual(len(monitor._patched_dispatchers), 1)
        finally:
            monitor.remove_hooks()

    def test_latent_combine_skipped_when_dispatcher_absent(self):
        """A latent MoE whose dispatcher lacks combine_postprocess must not crash setup;
        the metrics are simply not recorded."""
        layer = FakeLatentMoELayer(hidden_size=8, latent_size=4, num_experts=4)
        layer.token_dispatcher = None
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        monitor._find_moe_layers = lambda _model: [(0, layer)]
        monitor._prepare_layers(object())
        monitor.allocate_buffers(torch.device("cpu"))
        try:
            monitor._attach_hooks([(0, layer)])  # must not raise
            self.assertEqual(monitor._patched_dispatchers, [])
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

        hidden = torch.randn(S, B, H)
        logits = model(hidden)
        monitor.step()

        got = training_logs.get_latest(prefix="lar")
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

    def test_sum_of_squares_matches_naive_form_across_dtypes(self):
        """``vector_norm(dtype=fp32).square()`` replaced ``t.float().pow(2).sum()`` to
        avoid two full fp32 temporaries of the [T, vocab] logits. It must stay
        numerically equivalent for the dtypes training actually uses."""
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                t = (torch.randn(64, 128) * 3.0).to(dtype)
                got = lar_module._sum_of_squares(t)
                want = t.float().pow(2).sum()
                self.assertEqual(got.dtype, torch.float32, "must accumulate in fp32")
                self.assertEqual(got.dim(), 0)
                self.assertAlmostEqual(float(got) / float(want), 1.0, places=5, msg=f"{float(got)} vs {float(want)}")

    def test_sum_of_squares_masked_matches_indexed_form(self):
        """The masked path sums squared per-row norms instead of materialising
        ``t[mask]`` (which on the logits would copy ~valid_frac x [T, vocab]). Same sum,
        up to fp32 round-off."""
        t = torch.randn(32, 48)
        mask = torch.zeros(32, dtype=torch.bool)
        mask[::3] = True
        got = lar_module._sum_of_squares(t, mask)
        want = t[mask].float().pow(2).sum()
        self.assertAlmostEqual(float(got) / float(want), 1.0, places=5)

    def test_sum_of_squares_masked_all_false_is_zero(self):
        t = torch.randn(8, 4)
        got = lar_module._sum_of_squares(t, torch.zeros(8, dtype=torch.bool))
        self.assertEqual(float(got), 0.0)

    def test_masked_numel_counts_selected_rows_without_host_sync(self):
        t = torch.randn(10, 7)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:4] = True
        got = lar_module._masked_numel(t, mask)
        self.assertIsInstance(got, torch.Tensor, "stays a tensor: no .item() on the hot path")
        self.assertEqual(got.dim(), 0)
        self.assertEqual(float(got), 4 * 7)
        self.assertEqual(float(lar_module._masked_numel(t, None)), 10 * 7)

    def test_masked_numel_is_fp64_and_exact_at_logits_scale(self):
        """The logits element count is valid_tokens*vocab (~1e8-1e9 at seq 4096 / 200k
        vocab), past fp32's 2**24 exact-integer limit, so an fp32 count rounds for a
        general token count. Pin fp64 and exactness at realistic magnitude.

        Note a FULL 4096-row count (2**12 * 200019) happens to be exactly representable
        in fp32; it is the partially-masked counts — the normal case once padding is
        dropped — that round, so this test uses one of those.
        """
        n_rows, n_valid, vocab = 4096, 3277, 200019
        t = torch.empty(n_rows, vocab, device="meta")
        mask = torch.zeros(n_rows, dtype=torch.bool)
        mask[:n_valid] = True
        got = lar_module._masked_numel(t, mask)
        self.assertEqual(got.dtype, torch.float64)
        self.assertEqual(int(got), n_valid * vocab)
        as_fp32 = int(torch.tensor(float(n_valid * vocab), dtype=torch.float32))
        self.assertNotEqual(as_fp32, n_valid * vocab, "fp32 would have been exact here — pick another count")

    def test_accumulator_keeps_fp64_counts_across_microbatches(self):
        """The (ss, n) accumulators must adopt each value's dtype: a bare fp32 zero seed
        would downcast the fp64 counts on the first add and reintroduce the rounding."""
        S, B, H, V = 4, 1, 6, 8
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=False)
        monitor.register_hooks(model)
        model.output_layer(torch.randn(S, B, H))
        model.output_layer(torch.randn(S, B, H))
        site = monitor._sites["lm_head"]
        self.assertEqual(site["n"]["X"].dtype, torch.float64)
        self.assertEqual(site["n"]["Z"].dtype, torch.float64)
        self.assertEqual(site["ss"]["X"].dtype, torch.float32)
        self.assertEqual(float(site["n"]["X"]), 2 * S * B * H, "counts accumulate over microbatches")
        self.assertEqual(float(site["n"]["Z"]), 2 * S * B * V)

    def test_lar_masking_does_not_index_the_logits_tensor(self):
        """Regression guard for the allocation fix: the lm_head hook must hand the mask
        to _accumulate rather than indexing the logits. Indexing a [T, vocab] tensor is
        the expensive step, so assert __getitem__ is never called on it."""
        S, B, H, V = 5, 1, 4, 6
        model = FakeGPTModel(num_layers=1, hidden_size=H, vocab_size=V)
        monitor = LARMonitor(hook_moe_router=False, monitor_interval=1, apply_loss_mask=True)
        monitor.register_hooks(model)

        hidden = torch.randn(S, B, H)
        labels = torch.tensor([[1, 2, -100, 4, -100]])
        model(hidden, labels=labels)

        indexed: list[tuple[int, ...]] = []

        class TattlingTensor(torch.Tensor):
            """Records boolean-mask indexing of 2-D tensors, so the test can prove the
            full [T, vocab] logits are never copied. The subclass propagates through ops,
            so 1-D indexing (the intended ``row_sq[mask]``, shape [T]) is ignored — that
            is the cheap path we deliberately moved TO."""

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, False)

            def __getitem__(self, item):
                if isinstance(item, torch.Tensor) and item.dtype == torch.bool and self.dim() >= 2:
                    indexed.append(tuple(self.shape))
                return super().__getitem__(item)

        logits = TattlingTensor(model.output_layer(hidden).detach())
        for hook in model.output_layer._forward_hooks.values():
            hook(model.output_layer, (hidden,), logits)
        monitor.step()

        self.assertEqual(indexed, [], f"a 2-D tensor was mask-indexed: {indexed}")
        got = training_logs.get_latest(prefix="lar")
        mask = labels.transpose(0, 1).reshape(-1) != -100
        want, *_ = _lar_analytical(
            hidden.reshape(-1, H)[mask],
            model.output_layer.weight,
            logits=model.output_layer(hidden).reshape(-1, V)[mask],
        )
        self.assertAlmostEqual(got["lar/lm_head/lar"], want, places=4)


class _FakeDistOpt:
    """Stand-in for DistributedOptimizer's main/model param pairing.

    Mirrors the real invariant the monitor depends on: ``shard_fp32_from_float16_groups``
    holds the fp32 master (already stepped to ``theta_new``) while
    ``shard_float16_groups`` still holds the bf16 ``theta_old`` until
    ``_copy_main_params_to_model_params`` runs.
    """

    is_stub_optimizer = False

    def __init__(self, theta_old: "torch.Tensor", dtype=None):
        dtype = dtype or torch.bfloat16
        # The fp32 master is the authoritative value and is NOT on the bf16 grid — it
        # persists across steps and accumulates sub-bf16 updates. The model param is its
        # rounded copy. Seeding the master FROM the bf16 param instead would put it
        # exactly on-grid, removing the rounding error the debias exists to cancel and
        # making these tests measure a state training never reaches.
        self.main_param = theta_old.detach().clone().float()
        self.model_param = self.main_param.to(dtype)
        self.shard_float16_groups = [[self.model_param]]
        self.shard_fp32_from_float16_groups = [[self.main_param]]
        self.copy_calls = 0

    def apply_update(self, delta: "torch.Tensor") -> None:
        """The inner ``optimizer.step()``: fp32 master moves, model param does not."""
        self.main_param.add_(delta)

    def _copy_main_params_to_model_params(self):
        self.copy_calls += 1
        self.model_param.copy_(self.main_param)


class MegatronOptimUpdateMonitorTest(unittest.TestCase):
    """optim/update_rms + param_rms: measured between optimizer.step() and copy-back."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _run_step(self, theta_old, delta, **kwargs):
        opt = _FakeDistOpt(theta_old)
        monitor = OptimUpdateMonitor(**kwargs)
        self.assertTrue(monitor.attach_optimizer(opt), "no copy method was wrapped")
        opt.apply_update(delta)
        opt._copy_main_params_to_model_params()
        try:
            return opt, monitor.step()
        finally:
            monitor.remove_hooks()

    def test_reports_update_and_param_rms(self):
        torch.manual_seed(0)
        theta = torch.randn(50_000) * 0.02
        delta = torch.randn(50_000) * 3e-4
        opt, got = self._run_step(theta, delta)

        want_update = float(delta.pow(2).mean().sqrt())
        want_param = float(opt.main_param.pow(2).mean().sqrt())
        self.assertAlmostEqual(got["optim/update_rms"] / want_update, 1.0, places=2)
        self.assertAlmostEqual(got["optim/param_rms"] / want_param, 1.0, places=4)
        self.assertAlmostEqual(
            got["optim/update_param_ratio"],
            got["optim/update_rms"] / got["optim/param_rms"],
            places=9,
        )

    def test_metrics_land_in_training_logs(self):
        torch.manual_seed(1)
        self._run_step(torch.randn(4096) * 0.02, torch.randn(4096) * 1e-4)
        latest = training_logs.get_latest(prefix="optim")
        self.assertEqual(
            set(latest),
            {"optim/update_rms", "optim/param_rms", "optim/update_param_ratio"},
        )

    def test_debias_keeps_small_updates_accurate(self):
        """A bf16 theta_old rounds, which inflates the raw difference once the update is
        near the bf16 resolution — 11x at lr 3e-6. Debiasing must hold ~1.0x across the
        whole lr range, otherwise the metric is useless late in a cosine decay."""
        torch.manual_seed(2)
        for lr in (3e-4, 3e-5, 3e-6):
            theta = torch.randn(200_000) * 0.02
            delta = torch.randn(200_000) * lr
            want = float(delta.pow(2).mean().sqrt())
            with self.subTest(lr=lr, debias=True):
                _, got = self._run_step(theta, delta, debias_low_precision=True)
                self.assertAlmostEqual(got["optim/update_rms"] / want, 1.0, places=1)
        # And confirm the raw form really is badly biased at the small end, so the
        # debias branch is not dead weight.
        theta = torch.randn(200_000) * 0.02
        delta = torch.randn(200_000) * 3e-6
        want = float(delta.pow(2).mean().sqrt())
        _, raw = self._run_step(theta, delta, debias_low_precision=False)
        self.assertGreater(raw["optim/update_rms"] / want, 3.0)

    def test_zero_update_reports_zero(self):
        """No update at all must not produce a floor from bf16 rounding."""
        torch.manual_seed(3)
        theta = torch.randn(20_000) * 0.02
        _, got = self._run_step(theta, torch.zeros(20_000))
        self.assertLess(got["optim/update_rms"], 1e-6)
        self.assertGreater(got["optim/param_rms"], 0.0)

    def test_copy_back_still_happens(self):
        """The wrapper must be transparent: the real copy has to run, exactly once."""
        torch.manual_seed(4)
        theta = torch.randn(8192) * 0.02
        delta = torch.randn(8192) * 1e-4
        opt, _ = self._run_step(theta, delta)
        self.assertEqual(opt.copy_calls, 1)
        self.assertTrue(torch.equal(opt.model_param, opt.main_param.to(opt.model_param.dtype)))

    def test_patch_is_reverted_by_remove_hooks(self):
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        original = opt._copy_main_params_to_model_params
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        self.assertIsNot(opt._copy_main_params_to_model_params, original)
        monitor.remove_hooks()
        # Reverted by dropping the instance attribute so class lookup resumes; bound
        # methods are recreated per access, so compare __func__.
        self.assertNotIn("_copy_main_params_to_model_params", vars(opt))
        self.assertIs(opt._copy_main_params_to_model_params.__func__, original.__func__)

    def test_patch_is_idempotent(self):
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        patched_once = opt._copy_main_params_to_model_params
        monitor.attach_optimizer(opt)
        try:
            self.assertIs(opt._copy_main_params_to_model_params, patched_once)
            self.assertEqual(len(monitor._patched), 1)
        finally:
            monitor.remove_hooks()

    def test_accumulates_across_chained_optimizers(self):
        """ChainedOptimizer (Muon + Adam) must contribute both sub-optimizers' shards to
        one pooled RMS, not just the first."""
        torch.manual_seed(5)
        theta_a, theta_b = torch.randn(30_000) * 0.02, torch.randn(30_000) * 0.02
        delta_a, delta_b = torch.randn(30_000) * 3e-4, torch.randn(30_000) * 3e-4
        opt_a, opt_b = _FakeDistOpt(theta_a), _FakeDistOpt(theta_b)
        chained = SimpleNamespace(chained_optimizers=[opt_a, opt_b])

        monitor = OptimUpdateMonitor()
        self.assertTrue(monitor.attach_optimizer(chained))
        try:
            opt_a.apply_update(delta_a)
            opt_b.apply_update(delta_b)
            opt_a._copy_main_params_to_model_params()
            opt_b._copy_main_params_to_model_params()
            got = monitor.step()
        finally:
            monitor.remove_hooks()

        want = float(torch.cat([delta_a, delta_b]).pow(2).mean().sqrt())
        self.assertAlmostEqual(got["optim/update_rms"] / want, 1.0, places=2)

    def test_step_is_empty_without_a_step(self):
        """step() with nothing pending must emit nothing rather than 0/0."""
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            self.assertEqual(monitor.step(), {})
            self.assertEqual(training_logs.get_latest(prefix="optim"), {})
        finally:
            monitor.remove_hooks()

    def test_pending_is_cleared_between_steps(self):
        """Two consecutive steps must report each step's own update, not a running sum."""
        torch.manual_seed(6)
        opt = _FakeDistOpt(torch.randn(40_000) * 0.02)
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            big = torch.randn(40_000) * 1e-3
            opt.apply_update(big)
            opt._copy_main_params_to_model_params()
            first = monitor.step()["optim/update_rms"]

            small = torch.randn(40_000) * 1e-4
            opt.apply_update(small)
            opt._copy_main_params_to_model_params()
            second = monitor.step()["optim/update_rms"]
        finally:
            monitor.remove_hooks()

        self.assertAlmostEqual(first / float(big.pow(2).mean().sqrt()), 1.0, places=1)
        self.assertAlmostEqual(second / float(small.pow(2).mean().sqrt()), 1.0, places=1)
        self.assertLess(second, first / 5)

    def test_stub_optimizer_is_skipped(self):
        opt = _FakeDistOpt(torch.randn(512) * 0.02)
        opt.is_stub_optimizer = True
        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            opt.apply_update(torch.randn(512) * 1e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(monitor.step(), {})
        finally:
            monitor.remove_hooks()

    def test_setup_helper_registers_under_the_metric_prefix(self):
        """The monitor_dict key must equal METRIC_PREFIX ("optim"), because callers print
        training_logs by iterating monitor_dict keys as metric prefixes — a mismatched key
        would silently drop these metrics from the log. The helper also returns ``model``,
        matching the other setup_* functions' contract."""
        opt = _FakeDistOpt(torch.randn(1024) * 0.02)
        monitor_dict = {}
        sentinel = object()
        returned = setup_optim_update_monitor(sentinel, optimizer=opt, monitor_dict=monitor_dict)
        monitor = monitor_dict[optim_update_module.METRIC_PREFIX]
        try:
            self.assertIs(returned, sentinel, "must return the model, like the other setups")
            self.assertEqual(list(monitor_dict), [optim_update_module.METRIC_PREFIX])
            self.assertTrue(monitor._patched)

            opt.apply_update(torch.randn(1024) * 3e-4)
            opt._copy_main_params_to_model_params()
            monitor.step()
            emitted = training_logs.get_latest(prefix=optim_update_module.METRIC_PREFIX)
            self.assertEqual(len(emitted), 3)
            for key in emitted:
                self.assertTrue(key.startswith(f"{optim_update_module.METRIC_PREFIX}/"))
        finally:
            monitor.remove_hooks()

    def test_always_on_without_being_named_in_monitors(self):
        """The monitor must be installed by setup_monitors regardless of the ``monitors``
        spec — it is in _ALWAYS_ON_MONITORS, not _MONITOR_MAP — so no training script has
        to name it or thread the optimizer through a callback."""
        self.assertIn("optim", megatron_backend._ALWAYS_ON_MONITORS)
        self.assertNotIn("optim", megatron_backend._MONITOR_MAP)
        self.assertNotIn("optim", megatron_backend._expand_monitor_names(["all"]))

        monitor_dict = {}
        model = FakeGPTModel(num_layers=1, hidden_size=8, vocab_size=16)
        megatron_backend.setup_monitors(model, monitors=["qk_stats"], monitor_dict=monitor_dict)
        monitor = monitor_dict.get("optim")
        try:
            self.assertIsInstance(monitor, OptimUpdateMonitor)
            self.assertTrue(monitor._patched, "class-level patch should be installed")
        finally:
            if monitor is not None:
                monitor.remove_hooks()

    def test_class_patch_covers_optimizers_built_afterwards(self):
        """Patching the mcore optimizer CLASS is what removes the need for a trainer-side
        callback: an instance constructed after setup must still be measured."""
        monitor = OptimUpdateMonitor()
        if not monitor.attach_optimizer_classes():
            self.skipTest("megatron.core optimizer classes unavailable")
        try:
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

            patched = DistributedOptimizer.__dict__["_copy_main_params_to_model_params"]
            self.assertTrue(getattr(patched, "_im_update_patched", False))
        finally:
            monitor.remove_hooks()
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

            reverted = DistributedOptimizer.__dict__["_copy_main_params_to_model_params"]
            self.assertFalse(getattr(reverted, "_im_update_patched", False), "class must be restored")

    def test_chunked_sum_matches_single_pass(self):
        """The shard is walked in fixed-size chunks to bound the temporary; the result
        must equal a single-pass computation across a chunk boundary."""
        torch.manual_seed(7)
        n = optim_update_module._CHUNK + 12345
        main = torch.randn(n)
        model = main.to(torch.bfloat16)
        ss_u, ss_p, ss_q, count = optim_update_module._pair_sums(main, model)
        self.assertEqual(count, n)
        # fp32 chunk accumulation vs a single fp32 reduction over ~1M elements: agreement
        # is fp32 round-off, not exact.
        self.assertAlmostEqual(float(ss_u) / float((main - model.float()).pow(2).sum()), 1.0, places=3)
        self.assertAlmostEqual(float(ss_p) / float(main.pow(2).sum()), 1.0, places=4)

    def test_tp_duplicate_params_are_counted_once(self):
        """A TP-REPLICATED param exists identically on every TP rank, so only rank 0 may
        count it; a TP-SHARDED param is distinct per rank and always counts. Getting this
        backwards inflates the RMS by the TP size."""
        replicated = torch.nn.Parameter(torch.randn(64))
        sharded = torch.nn.Parameter(torch.randn(64))
        sharded.tensor_model_parallel = True

        rank0 = SimpleNamespace(rank=lambda: 0)
        rank1 = SimpleNamespace(rank=lambda: 1)
        not_dup = optim_update_module._param_is_not_tp_duplicate
        self.assertTrue(not_dup(sharded, rank1), "TP-sharded is distinct on every rank")
        self.assertTrue(not_dup(replicated, rank0))
        self.assertFalse(not_dup(replicated, rank1), "replicated must only count on TP rank 0")

    def test_shared_params_are_excluded(self):
        """Tied embeddings are marked ``shared`` and appear on two PP stages; counting
        both would double-count them."""
        plain = torch.nn.Parameter(torch.randn(8))
        shared = torch.nn.Parameter(torch.randn(8))
        shared.shared = True
        self.assertTrue(optim_update_module._param_is_not_shared(plain))
        self.assertFalse(optim_update_module._param_is_not_shared(shared))

    def test_expert_and_dense_shards_are_bucketed_separately(self):
        """Expert params reduce over different groups than dense ones, so they must land
        in separate buckets (``allreduce=False`` is mcore's expert marker)."""
        torch.manual_seed(8)
        opt = _FakeDistOpt(torch.randn(4096) * 0.02)
        expert_model = (torch.randn(4096) * 0.02).to(torch.bfloat16)
        expert_main = expert_model.float()
        expert_model.allreduce = False
        opt.shard_fp32_from_float16_groups[0].append(expert_main)
        opt.shard_float16_groups[0].append(expert_model)

        monitor = OptimUpdateMonitor()
        monitor.attach_optimizer(opt)
        try:
            opt.apply_update(torch.randn(4096) * 3e-4)
            expert_main.add_(torch.randn(4096) * 3e-4)
            opt._copy_main_params_to_model_params()
            self.assertEqual(len(monitor._pending_dense), 1)
            self.assertEqual(len(monitor._pending_expert), 1)
            got = monitor.step()
            self.assertIn("optim/update_rms", got)
            self.assertEqual(monitor._pending_dense, [])
            self.assertEqual(monitor._pending_expert, [])
        finally:
            monitor.remove_hooks()


class MegatronMonitorRegistryTest(unittest.TestCase):
    """``monitors=["all"]`` expansion — which monitors it does and does not include."""

    OPT_IN_ONLY = ("act_dump", "lar")

    def test_all_excludes_the_opt_in_only_monitors(self):
        """act_dump writes tensors to disk (tens of GB per step at the current
        full-hidden defaults) and lar reads the [T, vocab] logits every monitored step
        plus a gating matmul per router. Neither should be swept in by "all"; both stay
        reachable by name."""
        names = megatron_backend._expand_monitor_names(["all"])
        for name in self.OPT_IN_ONLY:
            with self.subTest(monitor=name):
                self.assertNotIn(name, names)
                self.assertIn(name, megatron_backend._MONITOR_MAP, "still reachable by name")

    def test_all_expands_to_the_all_monitors_set(self):
        names = set(megatron_backend._expand_monitor_names(["all"]))
        self.assertEqual(names, set(megatron_backend._ALL_MONITORS))
        self.assertEqual(
            set(megatron_backend._MONITOR_MAP) - set(megatron_backend._ALL_MONITORS),
            set(self.OPT_IN_ONLY),
        )

    def test_all_monitors_setup_fns_match_the_registry(self):
        """_MONITOR_MAP is built from _ALL_MONITORS, so every name must resolve to the
        same setup fn in both — a copy-paste divergence would silently run the wrong
        setup for a monitor."""
        for name, setup_fn in megatron_backend._ALL_MONITORS.items():
            with self.subTest(monitor=name):
                self.assertIs(megatron_backend._MONITOR_MAP[name], setup_fn)

    def test_none_defaults_to_all_and_still_excludes_opt_in_only(self):
        self.assertEqual(
            megatron_backend._expand_monitor_names(None),
            megatron_backend._expand_monitor_names(["all"]),
        )
        for name in self.OPT_IN_ONLY:
            with self.subTest(monitor=name):
                self.assertNotIn(name, megatron_backend._expand_monitor_names(None))

    def test_explicit_opt_in_alongside_all_is_honoured(self):
        """Naming an opt-in monitor next to "all" must survive the exclusion — otherwise
        there would be no way to get the cheap metrics plus lar/act_dump together."""
        for name in self.OPT_IN_ONLY:
            with self.subTest(monitor=name):
                names = megatron_backend._expand_monitor_names(["all", name])
                self.assertIn(name, names)
                self.assertIn("qk_stats", names)
                self.assertEqual(len(names), len(set(names)), "no duplicates")

    def test_explicit_opt_in_alone_is_honoured(self):
        for name in self.OPT_IN_ONLY:
            with self.subTest(monitor=name):
                self.assertEqual(megatron_backend._expand_monitor_names([name]), [name])

    def test_explicit_list_is_preserved_and_deduped(self):
        self.assertEqual(
            megatron_backend._expand_monitor_names(["qk_stats", "lar", "qk_stats"]),
            ["qk_stats", "lar"],
        )

    def test_bare_string_spec_is_accepted(self):
        self.assertEqual(megatron_backend._expand_monitor_names("lar"), ["lar"])
        for name in self.OPT_IN_ONLY:
            self.assertNotIn(name, megatron_backend._expand_monitor_names("all"))


if __name__ == "__main__":
    unittest.main()
