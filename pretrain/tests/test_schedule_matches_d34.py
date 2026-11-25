"""The unified trainer reproduces the original d34 run's schedule at every step (CPU, a few seconds).

The original run (d34_original_run/) changed its batch by stopping and restarting in three stages with three trainers;
the unified trainer changes it inside one process. With the d34 config both give the same batch, token count,
canonical step, learning-rate multiplier, Muon momentum and Muon weight decay at all 229,067 steps. The 0.9B and 0.6B
runs used the 2M hand-off one step later; their difference from the d34 schedule is pinned here too.
    cd pretrain && python -m unittest discover -s tests -t .
"""
import unittest

from tests.schedule_code import config, d34_run_schedule, unified_schedule

FIELDS = ("batch", "tokens_before", "canonical_step", "lr_multiplier", "muon_momentum", "muon_weight_decay")


class ScheduleMatchesD34(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d34 = d34_run_schedule()

    def test_d34_config_is_the_run_at_every_step(self):
        ours, s = unified_schedule(config("d34_3.3b.yaml"))
        self.assertEqual(len(ours), len(self.d34))
        self.assertEqual(len(ours), 229_067)
        for k, name in enumerate(FIELDS):
            bad = [i for i, (a, b) in enumerate(zip(self.d34, ours)) if a[k] != b[k]]
            self.assertEqual(bad, [], f"{name} differs at {len(bad)} steps, first {bad[:3]}")
        self.assertEqual(s.tokens_before[-1], 287_049_515_008)
        self.assertEqual(s.batch[-1], 1_048_576)          # the run's half-batch last update

    def test_small_configs_hand_off_one_step_later(self):
        for name in ("d32x1152_0.9b.yaml", "d28x1024_0.6b.yaml"):
            ours, s = unified_schedule(config(name))
            self.assertEqual(len(ours), 229_067)
            self.assertEqual(s.overshoot, 0)
            self.assertEqual(s.tokens_before[-1], 287_049_515_008)
            first_2m = next(i for i, b in enumerate(s.batch) if b == 2_097_152)
            self.assertEqual(first_2m, 136_317)               # the d34 run: 136,316
            self.assertEqual(s.batch[-1], 2_097_152)          # no half batch at the end
            for i in range(136_316):                          # identical before the hand-off
                self.assertEqual(ours[i], self.d34[i])
            self.assertLess(max(abs(a[3] - b[3]) for a, b in zip(self.d34, ours)), 5.4e-6)   # LR multiplier
            self.assertLess(max(abs(a[4] - b[4]) for a, b in zip(self.d34, ours)), 4e-7)     # Muon momentum
            self.assertLess(max(abs(a[5] - b[5]) for a, b in zip(self.d34, ours)), 1e-5)     # weight decay (relative)


if __name__ == "__main__":
    unittest.main()
