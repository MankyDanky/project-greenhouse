"""Pure token-clock, actual scheduler-source, filesystem and chain-state tests."""
import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from checkpoint_state import complete_checkpoints
from production_clock import (REFERENCE_BATCH as B, RAMP_SOURCE_TOKENS, ABLATION_TOKENS,
                              HORIZON, FINAL_TOKENS, TokenClock, source_clock)
from production_chain import start_leg, finish_leg
from winner_manifest import create as create_winner, verify as verify_winner
from corpus_manifest import hash_records
from test_completion_evidence import ablation_fixture

HERE = Path(__file__).resolve().parent
START_TOKENS = RAMP_SOURCE_TOKENS + ABLATION_TOKENS


def actual_schedulers():
    tree = ast.parse((HERE / "base_train.py").read_text())
    names = {"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    namespace = {"args": SimpleNamespace(warmup_steps=40, warmdown_ratio=0.5528, final_lr_frac=0.05),
                 "num_iterations": HORIZON, "weight_decay_scaled": 0.021764, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_scheduler_functions", "exec"), namespace)
    return [namespace[name] for name in ("get_lr_multiplier", "get_muon_momentum", "get_weight_decay")]


def checkpoint_fixture(directory, step, canonical, missing_rank=None):
    meta = {"step": step, "production": {"version": 1, "actual_step": step, "canonical_step": canonical,
            "total_training_tokens": RAMP_SOURCE_TOKENS + (canonical - 136316) * B, "world_size": 8}}
    (directory / f"meta_{step:06d}.json").write_text(json.dumps(meta))
    (directory / f"model_{step:06d}.pt").write_bytes(b"test model fixture")
    for rank in range(8):
        if rank != missing_rank:
            (directory / f"optim_{step:06d}_rank{rank}.pt").write_bytes(b"test shard fixture")
    return meta


class ProductionClockTests(unittest.TestCase):
    def test_both_batch_sizes_finish_at_original_total_tokens(self):
        small = TokenClock(138364, 138364, START_TOKENS, B)
        large = TokenClock(137340, 138364, START_TOKENS, 2 * B)
        self.assertEqual(small.updates_to_finish, 183453)
        self.assertEqual(large.updates_to_finish, 91727)
        self.assertEqual(small.final_actual, 321817)
        self.assertEqual(large.final_actual, 229067)
        for clock in (small, large):
            self.assertEqual(clock.canonical(clock.final_actual), HORIZON)
            self.assertEqual(clock.total(clock.final_actual), FINAL_TOKENS)
        self.assertEqual(large.update_tokens(large.final_actual - 1), B)
        self.assertEqual(large.update_tokens(large.final_actual - 2), 2 * B)
        self.assertEqual(large.update_tokens(large.final_actual), 0)

    def test_checkpoint_leg_restart_preserves_token_schedule(self):
        clock = TokenClock(137340, 138364, START_TOKENS, 2 * B)
        cut = 140340
        resumed = TokenClock(cut, clock.canonical(cut), clock.total(cut), 2 * B)
        for step in (cut, cut + 100, cut + 10000, clock.final_actual - 1, clock.final_actual):
            self.assertEqual(clock.canonical(step), resumed.canonical(step))
            self.assertEqual(clock.total(step), resumed.total(step))
            self.assertEqual(clock.update_tokens(step), resumed.update_tokens(step))
            for scheduler in actual_schedulers():
                self.assertEqual(scheduler(clock.canonical(step)), scheduler(resumed.canonical(step)))

    def test_real_source_warmdown_boundary_and_final_values(self):
        lr, momentum, decay = actual_schedulers()
        self.assertEqual(lr(143917), 1)
        self.assertLess(lr(143918), 1)
        self.assertEqual(momentum(143917), 0.97)
        self.assertLess(momentum(143918), 0.97)
        self.assertAlmostEqual(lr(HORIZON), 0.05)
        self.assertAlmostEqual(momentum(HORIZON), 0.90)
        self.assertEqual(decay(HORIZON), 0)
        self.assertGreater(lr(HORIZON - 1), 0.05)
        self.assertGreater(decay(HORIZON - 1), 0)

    def test_source_clock_rejects_changed_tokens(self):
        meta = {"ablation": {"source_step": 136316, "added_tokens": ABLATION_TOKENS,
                            "token_budget": ABLATION_TOKENS, "canonical_step": 138364}}
        self.assertEqual(source_clock(meta), (138364, 94685102080))
        meta["ablation"]["canonical_step"] += 1
        with self.assertRaises(AssertionError):
            source_clock(meta)

    def test_complete_checkpoint_discovery_ignores_partial_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            checkpoint_fixture(directory, 140000, 143684)
            checkpoint_fixture(directory, 140100, 143884, missing_rank=7)
            (directory / ".pending_140200_job").mkdir()
            self.assertEqual([row["step"] for row in complete_checkpoints(directory)], [140000])

    def test_three_lost_legs_stop_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertTrue(start_leg(directory, "1", START_TOKENS))
            self.assertTrue(start_leg(directory, "2", START_TOKENS))
            self.assertTrue(start_leg(directory, "3", START_TOKENS))
            self.assertFalse(start_leg(directory, "4", START_TOKENS))
            self.assertTrue((directory / "CHAIN_STOP").is_file())

    def test_progress_resets_failures_and_no_progress_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            start_leg(directory, "1", START_TOKENS)
            state = finish_leg(directory, "1", 0, START_TOKENS)
            self.assertEqual(state["consecutive_failures"], 1)
            start_leg(directory, "2", START_TOKENS)
            state = finish_leg(directory, "2", 0, START_TOKENS + B)
            self.assertEqual(state["consecutive_failures"], 0)

    def test_per_rank_resume_uses_saved_rank_cursor(self):
        tree = ast.parse((HERE / "production_runtime.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resume_loader_state")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "resume_loader_state", "exec"), namespace)
        states = [{"rank": rank, "state": {"pq_idx": 1990 + rank, "rg_idx": 32 + rank, "epoch": 1}}
                  for rank in range(8)]
        meta = {"production": {"loader_states": list(reversed(states))}}
        for rank in range(8):
            self.assertEqual(namespace["resume_loader_state"](meta, rank), states[rank]["state"])

    def test_winner_manifest_detects_same_size_corruption_after_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            step = 138364
            meta, completion = ablation_fixture()
            completion_path = directory / "COMPLETE.json"
            completion_path.write_text(json.dumps(completion))
            (directory / f"meta_{step:06d}.json").write_text(json.dumps(meta))
            (directory / f"model_{step:06d}.pt").write_bytes(b"model-weights")
            for rank in range(8):
                (directory / f"optim_{step:06d}_rank{rank}.pt").write_bytes(b"optimizer")
            manifest, receipt = directory / "manifest.json", directory / "receipt.json"
            create_winner(directory, step, manifest, completion_path)
            self.assertGreater(verify_winner(directory, manifest, receipt, completion_path)["hashed_bytes"], 0)
            self.assertEqual(verify_winner(directory, manifest, receipt, completion_path)["hashed_bytes"], 0)
            (directory / f"model_{step:06d}.pt").write_bytes(b"wrong-weights")
            with self.assertRaises(AssertionError):
                verify_winner(directory, manifest, receipt, completion_path)

    def test_corpus_ledger_resumes_and_detects_changed_content(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            names = ["shard_00000.parquet", "shard_00001.parquet"]
            for name in names:
                (directory / name).write_bytes(b"original corpus")
            ledger = directory / "ledger.jsonl"
            rows, hashed = hash_records(directory, [{"path": name} for name in names], ledger, workers=2)
            self.assertEqual(hashed, 2 * len(b"original corpus"))
            expected = [{key: row[key] for key in ("path", "size", "sha256")} for row in rows]
            self.assertEqual(hash_records(directory, expected, ledger, workers=2)[1], 0)
            (directory / names[0]).write_bytes(b"modified corpus")
            with self.assertRaises(AssertionError):
                hash_records(directory, expected, ledger, workers=2)


if __name__ == "__main__":
    unittest.main()
