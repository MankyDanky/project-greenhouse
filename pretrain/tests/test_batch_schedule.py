"""nanochat/batch_schedule.py: phases, the canonical clock and the exact-horizon last step (CPU, instant)."""
import unittest

from tests.schedule_code import batch_schedule_module

bs = batch_schedule_module()
T = 287_049_515_008


class BatchSchedule(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(bs.parse_batch_schedule("262144:10e9,524288"), [(262144, 10_000_000_000), (524288, None)])
        for bad in ("", "262144:10e9", "262144:20e9,524288:10e9,1", "262144:10e9,524288:10e9,1", "0:1e9,2"):
            with self.assertRaises(AssertionError, msg=bad):
                bs.parse_batch_schedule(bad)

    def test_d34_phases(self):
        s = bs.build_batch_schedule("262144:10e9,524288:30.4e9,1048576:92537618432,2097152", T, granularity=131072)
        self.assertEqual(s.phase_ranges(), [(262144, 0, 38146), (524288, 38147, 77056), (1048576, 77057, 136315),
                                            (2097152, 136316, 229065), (1048576, 229066, 229066)])
        self.assertEqual((s.num_steps, s.tokens_before[-1], s.canonical_horizon, s.overshoot), (229_067, T, 321_817, 0))

    def test_small_backbone_phases(self):
        s = bs.build_batch_schedule("262144:10e9,524288:30.4e9,1048576:92537823232,2097152", T, granularity=262144)
        self.assertEqual(s.phase_ranges(), [(262144, 0, 38146), (524288, 38147, 77056), (1048576, 77057, 136316),
                                            (2097152, 136317, 229066)])
        self.assertEqual((s.num_steps, s.tokens_before[-1], s.canonical_horizon, s.overshoot), (229_067, T, 321_817, 0))

    def test_exact_horizon_needs_whole_micro_batches(self):
        spec = "262144:10e9,524288:30.4e9,1048576:92537618432,2097152"
        whole = bs.build_batch_schedule(spec, T, granularity=8 * 8 * 2048)
        self.assertEqual((whole.batch[-1], whole.overshoot), (1_048_576, 0))
        odd = bs.build_batch_schedule(spec, T, granularity=3 * 8 * 2048)       # 3 GPUs: the half batch is not whole
        self.assertEqual((odd.batch[-1], odd.overshoot), (2_097_152, 1_048_576))
        legacy = bs.build_batch_schedule(spec, T)                             # no granularity: the old behaviour
        self.assertEqual((legacy.batch[-1], legacy.overshoot), (2_097_152, 1_048_576))

    def test_constant_batch_is_nanochat_steps(self):
        s = bs.build_batch_schedule("524288", 524288 * 1000)
        self.assertEqual((s.num_steps, s.canonical_horizon), (1000, 1000))
        self.assertEqual(s.canonical_before[:3], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
