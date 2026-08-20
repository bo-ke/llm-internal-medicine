# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
"""Opt-in integration test: drive the *real* paddlefleet multimax lm_head.

The installed paddlefleet wheel predates the multimax feature, so this test is
skipped unless a multimax-capable checkout is importable (put its ``src`` ahead
of site-packages on ``PYTHONPATH``) and ``IM_MULTIMAX_INTEGRATION=1`` is set.
It needs a GPU and initializes fleet, which is why it is not part of the
default suite.

    IM_MULTIMAX_INTEGRATION=1 \
    PYTHONPATH=<PaddleFleet>/src:$PYTHONPATH \
    python -m pytest tests/test_paddlefleet_multimax_integration.py -q
"""

import functools
import math
import os
import sys
import unittest
from pathlib import Path

import paddle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from internal_medicine.backends.paddlefleet.multimax_monitor import PaddleMultiMaxMonitor  # noqa: E402
from internal_medicine.core import training_logs  # noqa: E402

_ENABLED = os.environ.get("IM_MULTIMAX_INTEGRATION") == "1"


def _multimax_available() -> bool:
    if not _ENABLED:
        return False
    try:
        import paddlefleet_ops

        # The sonicmoe ecosystem op is not loadable on every GPU arch and the
        # multimax path does not need it; neutralize the gate like upstream's
        # own single-card tests do.
        paddlefleet_ops.is_sonic_moe_available = lambda: False
        from paddlefleet.models.gpt.lm_head import (
            GPTLMHead,  # noqa: F401
            SegLU,  # noqa: F401
        )
    except Exception:
        return False
    return True


@unittest.skipUnless(_multimax_available(), "needs IM_MULTIMAX_INTEGRATION=1 and a multimax paddlefleet")
class MultiMaxRealHeadTest(unittest.TestCase):
    HIDDEN = 256
    VOCAB = 64

    @classmethod
    def setUpClass(cls):
        import paddlefleet.parallel_state as ps
        from paddle.distributed import fleet

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"],
        }
        fleet.init(is_collective=True, strategy=strategy)
        ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())

    def tearDown(self):
        training_logs.reset()

    def _build(self, **extra):
        from paddlefleet.gpt_builders import gpt_builder
        from paddlefleet.models.gpt import GPTConfig

        init = functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0)
        kwargs = dict(
            num_hidden_layers=2,
            hidden_size=self.HIDDEN,
            vocab_size=self.VOCAB,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=init,
            output_layer_init_method=init,
            tie_word_embeddings=True,
            multimax_modules=["lm_head"],
        )
        kwargs.update(extra)
        return gpt_builder(GPTConfig(**kwargs), num_stages=1)

    def _drive(self, model, predictions=1, hidden=None, ranges=None, ts=None):
        monitor = PaddleMultiMaxMonitor(sample_tokens=16, topk=10)
        monitor.register_hooks(model)
        heads = monitor._find_heads(model)
        self.assertTrue(heads, "no multimax head found in the built model")
        if hidden is None:
            hidden = paddle.randn([predictions * 8, 1, self.HIDDEN], dtype="float32")
        for _tag, head in heads:
            if ranges is not None:
                head.multimax_ranges.set_value(paddle.to_tensor(ranges, dtype=head.multimax_ranges.dtype))
            if ts is not None:
                head.multimax_ts.set_value(paddle.to_tensor(ts, dtype=head.multimax_ts.dtype))
            head.train()
            head({"hidden_states": hidden})
        monitor.step()
        monitor.remove_hooks()
        return heads, training_logs.get_latest(prefix="multimax/")

    def _assert_sane(self, logs, prefix=""):
        for name in ("entropy", "entropy_norm", "top10_prob", "top1_prob", "multi_modality", "sparsity", "range_0"):
            self.assertIn(f"multimax/global_{prefix}{name}", logs)
        entropy = logs[f"multimax/global_{prefix}entropy"]
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, math.log(self.VOCAB) + 1e-4)
        self.assertEqual(logs[f"multimax/global_{prefix}rows"], 8.0)
        # Cold start: SegLU is the identity, so every parameter reads back zero.
        self.assertEqual(logs[f"multimax/global_{prefix}range_0"], 0.0)
        self.assertEqual(logs[f"multimax/global_{prefix}t_3"], 0.0)

    def test_unfused_head(self):
        _heads, logs = self._drive(self._build())
        self._assert_sane(logs)

    def test_fused_ce_head_recomputes_the_tile(self):
        # fused_linear_ce_loss_chunk > 0 makes the head return
        # (hidden, weight, bias, ranges, ts) and never materialize the logits.
        _heads, logs = self._drive(self._build(fused_linear_ce_loss_chunk=1))
        self._assert_sane(logs)

    def test_mtp_list_output_takes_the_main_prediction(self):
        # A single GPTLMHead with MTP returns [main, mtp...]; the monitor reads
        # the main prediction and declares no mtp_ keys.
        _heads, logs = self._drive(self._build(num_nextn_predict_layers=1), predictions=2)
        self._assert_sane(logs)
        self.assertFalse([k for k in logs if "mtp_" in k])

    def test_separate_mtp_head_is_tagged_and_reads_mtp_logits(self):
        # separate_mtp_headloss=True builds GPTMTPLMHead, whose forward returns
        # the passthrough dict_args with predictions under "mtp_logits".
        model = self._build(
            num_nextn_predict_layers=1,
            separate_mtp_headloss=True,
            tie_word_embeddings=False,
        )
        heads, logs = self._drive(model, predictions=2)
        self.assertIn("mtp_", [tag for tag, _head in heads])
        self._assert_sane(logs, prefix="mtp_")

    def test_trained_parameter_values_are_reported_verbatim(self):
        # Cold start is all zeros, which cannot distinguish "reads the params"
        # from "reports zeros", so write known non-zero values onto the real
        # head and check every component comes back.
        ranges = [-2.0, 1.5, -3.0, 2.5]
        ts = [0.25, -0.125, 0.0625, -0.03125]
        hidden = paddle.randn([8, 1, self.HIDDEN], dtype="float32")

        _heads, identity_logs = self._drive(self._build(), hidden=hidden)
        training_logs.reset()
        _heads, logs = self._drive(self._build(), hidden=hidden, ranges=ranges, ts=ts)

        for i, want in enumerate(ranges):
            self.assertAlmostEqual(logs[f"multimax/global_range_{i}"], want, places=5)
        for i, want in enumerate(ts):
            self.assertAlmostEqual(logs[f"multimax/global_t_{i}"], want, places=5)
        # Non-zero params mean SegLU is no longer the identity, so the measured
        # distribution must move off the cold-start value for the same input.
        self.assertNotAlmostEqual(
            logs["multimax/global_entropy"],
            identity_logs["multimax/global_entropy"],
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
