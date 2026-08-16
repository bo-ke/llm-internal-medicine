import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Probe = importlib.import_module("internal_medicine.core.base_monitor").Probe
AVAILABLE_MONITORS = importlib.import_module("internal_medicine.core.registry").AVAILABLE_MONITORS
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


class DummyProbe(Probe):
    METRIC_PREFIX = "dummy"
    MAX_AGGREGATED = {"peak"}
    MIN_AGGREGATED = {"floor"}

    def register_hooks(self, model) -> None:
        return None


class CoreMonitoringTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_record_metrics_logs_per_layer_and_global_once_per_observation(self):
        probe = DummyProbe(log_per_layer=True, log_global=True)

        probe._record_metrics(0, {"mean": 2.0, "peak": 5.0, "floor": 3.0})
        probe._record_metrics(1, {"mean": 4.0, "peak": 2.0, "floor": 1.0})

        self.assertEqual(probe._global_count, 2)
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/layer_0/mean"], 2.0)
        self.assertEqual(latest["dummy/layer_1/mean"], 4.0)
        self.assertEqual(latest["dummy/global_mean"], 3.0)
        self.assertEqual(latest["dummy/global_peak"], 5.0)
        self.assertEqual(latest["dummy/global_floor"], 1.0)
        self.assertEqual(probe._global_count, 0)
        self.assertEqual(probe._global_accum, {})
        self.assertEqual(probe._global_metric_counts, {})

    def test_sparse_global_metrics_use_per_metric_counts(self):
        probe = DummyProbe(log_per_layer=False, log_global=True)

        probe._accumulate_global({"common": 2.0, "sparse": 4.0})
        probe._count_global_observation({"common", "sparse"})
        probe._accumulate_global({"common": 6.0})
        probe._count_global_observation({"common"})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/global_common"], 4.0)
        self.assertEqual(latest["dummy/global_sparse"], 4.0)

    def test_gather_is_collective_even_when_this_rank_has_no_metrics(self):
        """A PP stage with no monitored layer must still enter the gather.

        ``_gather_fn`` wraps ``all_gather_object``; skipping it on an empty rank
        hangs every other rank in the collective (observed as a PP=4 deadlock).
        """
        calls = []

        def fake_gather(local_metrics):
            calls.append(dict(local_metrics))
            # Pretend another rank reported one layer.
            return [local_metrics, {"dummy/layer_0/mean": 4.0}]

        training_logs.set_gather_fn(fake_gather)
        try:
            self.assertEqual(training_logs.get_latest(), {})
            aggregated = training_logs.gather_and_aggregate()
        finally:
            training_logs.set_gather_fn(None)

        self.assertEqual(calls, [{}], "gather_fn must be called even with no local metrics")
        self.assertEqual(aggregated, {"dummy/layer_0/mean": 4.0})

    def test_log_flags_are_respected(self):
        probe = DummyProbe(log_per_layer=False, log_global=True)
        probe._record_metrics(0, {"mean": 2.0})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertNotIn("dummy/layer_0/mean", latest)
        self.assertEqual(latest["dummy/global_mean"], 2.0)

        training_logs.reset()
        probe = DummyProbe(log_per_layer=True, log_global=False)
        probe._record_metrics(0, {"mean": 7.0})
        probe.step()

        latest = training_logs.get_latest(prefix="dummy")
        self.assertEqual(latest["dummy/layer_0/mean"], 7.0)
        self.assertNotIn("dummy/global_mean", latest)
        self.assertEqual(probe._global_count, 0)

    def test_empty_metrics_do_not_count_or_log(self):
        probe = DummyProbe(log_per_layer=True, log_global=True)
        probe._record_metrics(0, {})
        probe.step()

        self.assertEqual(probe._global_count, 0)
        self.assertEqual(training_logs.get_latest(prefix="dummy"), {})

    def test_massive_activation_scale_keys_are_max_aggregated(self):
        for key in (
            "massive_act/layer_0/channel_max_ratio",
            "massive_act/layer_0/channel_median",
            "massive_act/layer_0/channel_p95",
            "massive_act/layer_0/channel_p99",
            "massive_act/layer_0/massive_act_channel_count",
            "massive_act/layer_0/channel_count_gt_100",
            "massive_act/layer_0/activation_rms",
            "massive_act/layer_0/spectral_norm_max",
            "massive_act/global_spectral_norm_max",
            "massive_act/layer_0/lipschitz_max",
            "massive_act/global_lipschitz_max",
            # attn_type-tagged keys (paddlefleet prepends mla_/hca_/csa_/...)
            "massive_act/layer_0/hca_massive_act_channel_count",
            "massive_act/layer_0/hca_channel_count_gt_10",
            "massive_act/global_hca_channel_count_gt_10",
        ):
            self.assertTrue(training_logs._is_max_metric(key), key)

    def test_spectral_norm_min_keys_are_min_aggregated(self):
        for key in (
            "massive_act/layer_0/spectral_norm_min",
            "massive_act/global_spectral_norm_min",
            "massive_act/layer_0/lipschitz_min",
            "massive_act/global_lipschitz_min",
        ):
            self.assertTrue(training_logs._is_min_metric(key), key)
            self.assertFalse(training_logs._is_max_metric(key), key)

    def test_resolve_layer_idx_prefers_explicit_attrs_then_layer_number_then_offset(self):
        probe = DummyProbe()

        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(layer_idx=9), 0, 4), 9)
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(layer_number=3), 0, 4), 2)
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(), 2, 4), 2)

        probe.pp_rank = 1
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(), 2, 4), 6)
        self.assertEqual(probe._resolve_layer_idx(SimpleNamespace(), 2, 4, layer_offset=8), 14)

    def test_paddlefleet_registry_lists_massive_activation_monitor(self):
        self.assertIn("massive_act", AVAILABLE_MONITORS["paddlefleet"])


if __name__ == "__main__":
    unittest.main()
