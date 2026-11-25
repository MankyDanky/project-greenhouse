"""The three run configs: valid for the trainer, one ClimbMix pass each, and refused on a GPU count whose
micro-batch does not divide the batches (CPU, a few seconds)."""
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

from tests.schedule_code import PRETRAIN

sys.path.insert(0, PRETRAIN)
import full_args  # noqa: E402

CONFIGS = ("d34_3.3b.yaml", "d32x1152_0.9b.yaml", "d28x1024_0.6b.yaml")


class Configs(unittest.TestCase):
    def test_check_passes_on_the_released_layout(self):
        for name in CONFIGS:
            path = os.path.join(PRETRAIN, "configs", name)
            out = subprocess.run([sys.executable, os.path.join(PRETRAIN, "full_args.py"), path, "--check"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("287,049,515,008 = 287.05B over 229,067 steps", out.stdout)

    def test_bash_assignments(self):
        out = subprocess.run([sys.executable, os.path.join(PRETRAIN, "full_args.py"),
                              os.path.join(PRETRAIN, "configs", "d28x1024_0.6b.yaml"), "--ngpu", "4"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("RUN=d28x1024-climbmix", out.stdout)
        self.assertIn("NUM_ITERS=229067", out.stdout)
        self.assertIn("--ve-every 7", out.stdout)

    def test_bad_layouts_are_refused(self):
        _, errors = full_args.load(os.path.join(PRETRAIN, "configs", "d34_3.3b.yaml"), 3)
        self.assertTrue(any("not a multiple" in e or "overshoots" in e for e in errors), errors)
        cfg = yaml.safe_load(open(os.path.join(PRETRAIN, "configs", "d32x1152_0.9b.yaml")))
        cfg["train_args"]["model-tag"] = "x"
        cfg["train_args"]["not-a-flag"] = 1
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
        _, errors = full_args.load(f.name, 4)
        os.unlink(f.name)
        self.assertTrue(any("set by train.sh" in e for e in errors), errors)
        self.assertTrue(any("not-a-flag" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
