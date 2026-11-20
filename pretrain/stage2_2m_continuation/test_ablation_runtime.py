"""CPU-only instrumentation tests; they do not exercise PyTorch model training."""

import ast
import copy
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# The local Mac environment has no PyTorch. Only the logging/metadata helpers
# are exercised here; no model/optimizer numerical-validation claim is made.
torch_stub = types.ModuleType("torch")
dist_stub = types.ModuleType("torch.distributed")
dist_stub.is_initialized = lambda: False
torch_stub.distributed = dist_stub
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.distributed", dist_stub)

import ablation_runtime as runtime


class FakeOptimizer:
    def __init__(self):
        self.param_groups = []
        self.state = {}

    def load_state_dict(self, data):
        self.param_groups = copy.deepcopy(data["param_groups"])
        self.state = data["state"]


class FakeBatch:
    def __init__(self, value):
        self.array = np.full((8, 2048), value, dtype=np.int64)

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def numel(self): return self.array.size


def optimizer_data():
    return {"param_groups": [
        {"kind": "muon", "params": [0], "lr": 0.01414213562373095,
         "initial_lr": 0.01414213562373095, "weight_decay": 0.013,
         "momentum": 0.97, "ns_steps": 5},
        {"kind": "adamw", "params": [1], "lr": 0.126025,
         "initial_lr": 0.126025, "betas": (0.9, 0.95), "weight_decay": 0.0},
    ], "state": {0: {"momentum_buffer": object()}}}


def arguments(base, batch):
    return types.SimpleNamespace(
        resume_from_step=136316, num_iterations=321817, total_batch_size=batch,
        device_batch_size=8, max_seq_len=2048, depth=34, warmdown_ratio=0.5528,
        final_lr_frac=0.05, warmup_steps=40, eval_tokens=41943040,
        ablation_matrix_lr=0.02, ablation_token_budget=32 * 1048576,
        ablation_checkpoint_source_dir=str(base / "source"),
        ablation_metrics_path=str(base / "metrics" / "metrics.jsonl"),
        model_tag="isolated_test", run="dummy", fp8=True, fp8_recipe="tensorwise",
    )


class InstrumentationTests(unittest.TestCase):
    def test_only_muon_learning_rates_change_and_buffers_stay_loaded(self):
        original = optimizer_data()
        optimizer = FakeOptimizer()
        before = runtime.restore_optimizer_for_ablation(optimizer, original, 0.02)
        self.assertIs(optimizer.state, original["state"])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.97)
        self.assertEqual(runtime.group_settings(optimizer)[1], before[1])

    def run_branch(self, base, batch, complete=True, missing_shard=False):
        args = arguments(base, batch)
        runtime.validate_arguments(args, 8)
        self.assertEqual((args.core_metric_every, args.sample_every, args.save_every), (-1, -1, -1))
        optimizer = FakeOptimizer()
        before = runtime.restore_optimizer_for_ablation(optimizer, optimizer_data(), 0.02)
        state = {"pq_idx": 1945, "rg_idx": 24, "epoch": 1}
        meta = {"step": 136316, "total_batch_size": 1048576,
                "dataloader_state_dict": state, "model_config": {"n_layer": 34}}
        check_dir = base / "checkpoints"
        task = runtime.AblationRuntime(args, 0, 8, check_dir, meta, optimizer, before, state)
        start = 136316
        task.record_eval(start, 0.68, optimizer, state)
        micro_count = 0
        updates = args.ablation_token_budget // batch
        for update in range(updates):
            for _ in range(batch // (8 * 8 * 2048)):
                task.consumed_batch(FakeBatch(micro_count), FakeBatch(micro_count + 1))
                micro_count += 1
            task.record_step(start + update, 2.0, 0.1, optimizer, 1.0)
            step = start + update + 1
            if task.added_tokens(step) in task.eval_tokens:
                task.record_eval(step, 0.679, optimizer, state)
        self.assertEqual(micro_count, 256)
        self.assertEqual(task.canonical_step(step), 136348)
        if complete:
            check_dir.mkdir()
            (check_dir / f"model_{step:06d}.pt").write_text("test model placeholder")
            (check_dir / f"meta_{step:06d}.json").write_text(json.dumps({"step": step}))
            for rank in range(7 if missing_shard else 8):
                (check_dir / f"optim_{step:06d}_rank{rank}.pt").write_text("test optimizer placeholder")
            if missing_shard:
                with self.assertRaises(RuntimeError):
                    task.finish(step, optimizer)
                self.assertFalse((Path(args.ablation_metrics_path).parent / "COMPLETE.json").exists())
            else:
                task.finish(step, optimizer)
                summary = json.loads((Path(args.ablation_metrics_path).parent / "COMPLETE.json").read_text())
                self.assertEqual(summary["added_tokens"], args.ablation_token_budget)
                self.assertEqual(summary["completed_updates"], updates)
        else:
            self.assertFalse((Path(args.ablation_metrics_path).parent / "COMPLETE.json").exists())
        return [json.loads(line) for line in Path(args.ablation_metrics_path).read_text().splitlines()]

    def test_stream_hash_and_eval_grid_match_across_accumulation_groupings(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            one = self.run_branch(base / "one", 1048576)
            two = self.run_branch(base / "two", 2097152)
            eval_one = [r for r in one if r["event"] == "eval"]
            eval_two = [r for r in two if r["event"] == "eval"]
            self.assertEqual([r["added_tokens"] for r in eval_one], [0, 16 * 1048576, 32 * 1048576])
            for a, b in zip(eval_one, eval_two):
                self.assertEqual(a["consumed_streams"], b["consumed_streams"])
                self.assertEqual(a["interval_train_mean_loss"], b["interval_train_mean_loss"])

    def test_missing_shard_and_partial_run_cannot_mark_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.run_branch(base / "missing", 1048576, missing_shard=True)
            self.run_branch(base / "partial", 2097152, complete=False)

    def test_original_scheduler_at_equal_tokens_and_horizon(self):
        tree = ast.parse((Path(__file__).parent / "base_train.py").read_text())
        schedulers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in {"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"}],
                                type_ignores=[])
        scope = {"args": types.SimpleNamespace(warmup_steps=40, warmdown_ratio=0.5528, final_lr_frac=0.05),
                 "num_iterations": 321817, "math": math, "weight_decay_scaled": 0.021764}
        exec(compile(schedulers, "scheduler_test", "exec"), scope)
        for two_m_update in range(1025):
            canonical = 136316 + 2 * two_m_update
            self.assertEqual(scope["get_lr_multiplier"](canonical), 1.0)
            self.assertEqual(scope["get_muon_momentum"](canonical), 0.97)
            self.assertEqual(scope["get_weight_decay"](canonical),
                             scope["get_weight_decay"](136316 + (two_m_update * 2097152) / 1048576))


if __name__ == "__main__":
    unittest.main()
