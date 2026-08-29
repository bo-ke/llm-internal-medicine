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

    def test_pre_aggregated_keys_skip_the_gather_payload(self):
        """Device-reduced keys must not be pickled per rank, but must still be returned.

        Vector metrics emit one key per element (512 experts x 43 layers), so leaving
        them in the ``all_gather_object`` dict dominates the payload while every rank
        already agrees on the value.
        """
        calls = []

        def fake_gather(local_metrics):
            calls.append(dict(local_metrics))
            return [local_metrics, {"dummy/layer_0/mean": 4.0}]

        training_logs.update(**{"dummy/layer_0/mean": 2.0, "dummy/layer_0/share_e0": 7.0})
        training_logs.mark_pre_aggregated(["dummy/layer_0/share_e0"])
        training_logs.set_gather_fn(fake_gather)
        try:
            aggregated = training_logs.gather_and_aggregate()
        finally:
            training_logs.set_gather_fn(None)
            training_logs._pre_aggregated_keys.clear()

        self.assertEqual(calls, [{"dummy/layer_0/mean": 2.0}], "pre-aggregated key must be excluded")
        self.assertEqual(aggregated["dummy/layer_0/mean"], 3.0)
        self.assertEqual(aggregated["dummy/layer_0/share_e0"], 7.0, "value must survive as-is")

    def _install_two_rank_reduce(self, other_metrics):
        """Fake the fixed-schema reduce against one more rank holding ``other_metrics``.

        Decodes the vectors with the schema training_logs just froze, so the test
        also pins the layout (offsets of the resync flag and the two checksums).
        """

        def schema_sync_fn(local_keys):
            return [list(local_keys), sorted(other_metrics)]

        def reduce_fn(sum_vec, max_vec, min_vec):
            # The fixed-size (length, checksum) probe: both ranks agree, so MAX
            # and MIN of it come back identical.
            if not sum_vec:
                return [], list(max_vec), list(min_vec)
            mean_keys = training_logs._schema_mean
            width = len(mean_keys)
            sums = list(sum_vec[:width])
            mask = list(sum_vec[width:])
            for i, k in enumerate(mean_keys):
                if k in other_metrics:
                    sums[i] += other_metrics[k]
                    mask[i] += 1.0
            maxes = list(max_vec)
            for i, k in enumerate(training_logs._schema_max):
                if k in other_metrics:
                    maxes[1 + i] = max(maxes[1 + i], other_metrics[k])
            mins = list(min_vec)
            for i, k in enumerate(training_logs._schema_min):
                if k in other_metrics:
                    mins[i] = min(mins[i], other_metrics[k])
            return sums + mask, maxes, mins

        training_logs.set_schema_sync_fn(schema_sync_fn)
        training_logs.set_reduce_fn(reduce_fn)

    def _clear_aggregation_fns(self):
        training_logs.set_reduce_fn(None)
        training_logs.set_schema_sync_fn(None)
        training_logs.set_gather_fn(None)

    def test_tensor_path_matches_the_legacy_gather_path(self):
        """The fixed-schema reduce must reproduce all_gather_object numerically.

        mean stays an unweighted mean over the ranks holding the key (the presence
        mask, not an observation count), and max/min keys present on only one rank
        must survive instead of collapsing to +-inf.
        """
        local = {"dummy/layer_0/mean": 2.0, "dummy/layer_0/peak_max": 5.0, "dummy/layer_0/floor_min": 3.0}
        other = {"dummy/layer_0/mean": 4.0, "dummy/layer_0/peak_max": 1.0, "dummy/layer_1/mean": 9.0}

        training_logs.update(**local)
        try:
            training_logs.set_gather_fn(lambda m: [dict(m), dict(other)])
            legacy = training_logs.gather_and_aggregate()

            self._install_two_rank_reduce(other)
            tensor = training_logs.gather_and_aggregate()
        finally:
            self._clear_aggregation_fns()

        self.assertEqual(set(tensor), set(legacy))
        for k in legacy:
            self.assertAlmostEqual(tensor[k], legacy[k], places=12, msg=k)
        self.assertAlmostEqual(tensor["dummy/layer_0/mean"], 3.0, places=12)
        self.assertAlmostEqual(tensor["dummy/layer_1/mean"], 9.0, places=12, msg="single-rank key keeps its value")
        self.assertAlmostEqual(tensor["dummy/layer_0/peak_max"], 5.0, places=12)
        self.assertAlmostEqual(tensor["dummy/layer_0/floor_min"], 3.0, places=12)

    def test_tensor_path_rebuilds_the_schema_for_a_late_key(self):
        """A key appearing after the schema froze triggers one collective rebuild."""
        training_logs.update(**{"dummy/layer_0/mean": 2.0})
        rebuilds = []
        try:
            self._install_two_rank_reduce({"dummy/layer_0/mean": 4.0})
            inner = training_logs._schema_sync_fn

            def counting_sync(local_keys):
                rebuilds.append(sorted(local_keys))
                return inner(local_keys)

            training_logs._schema_sync_fn = counting_sync
            training_logs.gather_and_aggregate()
            self.assertEqual(len(rebuilds), 1)

            # optim-style key written after the first aggregation.
            training_logs.update(**{"optim/update_rms": 8.0})
            aggregated = training_logs.gather_and_aggregate()
            disabled = training_logs._tensor_path_disabled
        finally:
            self._clear_aggregation_fns()

        self.assertEqual(len(rebuilds), 2, "the resync flag must trigger exactly one rebuild")
        self.assertAlmostEqual(aggregated["optim/update_rms"], 8.0, places=12)
        self.assertFalse(disabled)

    def test_tensor_path_falls_back_when_schemas_disagree(self):
        """A schema mismatch must be caught by the fixed-size probe, not by hanging.

        Two ranks with different schemas build payload vectors of different lengths;
        a mismatched all_reduce hangs, so the probe has to catch it first.
        """
        training_logs.update(**{"dummy/layer_0/mean": 2.0})
        try:
            training_logs.set_gather_fn(lambda m: [dict(m), {"dummy/layer_9/mean": 6.0}])
            training_logs.set_schema_sync_fn(lambda keys: [list(keys)])
            # Only the probe is answered, and with a checksum the other rank does
            # not share: MAX and MIN of it diverge.
            training_logs.set_reduce_fn(lambda s, mx, mn: ([], [mx[0], mx[1] + 1.0], list(mn)))
            aggregated = training_logs.gather_and_aggregate()
            disabled = training_logs._tensor_path_disabled
        finally:
            self._clear_aggregation_fns()

        self.assertTrue(disabled, "must latch off after a mismatch")
        self.assertAlmostEqual(aggregated["dummy/layer_0/mean"], 2.0, places=12)
        self.assertAlmostEqual(aggregated["dummy/layer_9/mean"], 6.0, places=12, msg="legacy path took over")

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

    def test_sampled_this_step_marks_exactly_the_steps_the_hooks_recorded(self):
        """The flag is what a trainer callback gates its cross-rank gather on.

        It has to agree with `_should_monitor()` as seen *inside* the hooks —
        i.e. the pre-increment `step_count` — or the callback either drops
        metrics or pays for an empty collective.
        """
        for interval in (1, 2, 3, 5, 200):
            probe = DummyProbe(monitor_interval=interval)
            recorded, flagged = [], []
            for global_step in range(1, 4 * interval + 2):
                if probe._should_monitor():  # forward: hooks see step_count pre-increment
                    recorded.append(global_step)
                probe.step()  # on_step_end
                if probe.sampled_this_step:  # on_log
                    flagged.append(global_step)
            self.assertEqual(recorded, flagged, f"interval={interval}")
            self.assertEqual(recorded[0], 1, f"interval={interval}")

    def test_sampled_this_step_survives_a_resume_at_an_unaligned_step(self):
        """`step_count` restarts at 0 on resume while `global_step` does not.

        Deriving the phase from `global_step % interval` drifts (and silently
        drops every sampled step); the flag cannot, because it only ever reads
        `step_count`.
        """
        interval = 200
        for resume_at in (0, 300, 5050, 999):
            probe = DummyProbe(monitor_interval=interval)
            recorded, flagged, by_global_step = [], [], []
            for global_step in range(resume_at + 1, resume_at + 1 + 3 * interval):
                if probe._should_monitor():
                    recorded.append(global_step)
                probe.step()
                if probe.sampled_this_step:
                    flagged.append(global_step)
                if global_step % interval == 1:
                    by_global_step.append(global_step)
            self.assertEqual(recorded, flagged, f"resume_at={resume_at}")
            if resume_at % interval:
                self.assertNotEqual(recorded, by_global_step, f"resume_at={resume_at}")

    def test_sampled_this_step_is_false_when_monitoring_is_disabled(self):
        probe = DummyProbe(monitor_interval=0)
        probe.step()
        self.assertFalse(probe.sampled_this_step)


if __name__ == "__main__":
    unittest.main()
