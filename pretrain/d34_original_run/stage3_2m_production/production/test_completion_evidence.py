"""First-resume regressions using the completed runs' distinct rank cursors."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from completion_evidence import (checkpoint_names, load_completion, released_initial_evidence,
                                 validate_completion)
from production_clock import ABLATION_TOKENS, REFERENCE_BATCH as B
from winner_manifest import create, digest, verify

HERE = Path(__file__).resolve().parent


def ablation_fixture(batch=B):
    step = 136316 + ABLATION_TOKENS // batch
    tag = "d34abl136316_b2m_lr020" if batch == 2 * B else "d34abl136316_b1m_lr014142"
    lr = 0.02 if batch == 2 * B else 0.01414213562373095
    cursors = [(1990, 64), (1991, 1), (1992, 58), (1995, 19),
               (1995, 20), (1995, 29), (1995, 30), (1995, 23)]
    streams = [{"rank": rank, "consumed_microbatches": 16384, "consumed_x_tokens": 268435456,
                "sha256": hashlib.sha256(str(rank).encode()).hexdigest(),
                "prefetched_loader_state": {"pq_idx": pq, "rg_idx": rg, "epoch": 1}}
               for rank, (pq, rg) in enumerate(cursors)]
    meta = {"step": step, "total_batch_size": batch, "val_bpb": 0.68,
            "ablation": {"source_step": 136316, "added_tokens": ABLATION_TOKENS,
                         "token_budget": ABLATION_TOKENS, "canonical_step": 138364.0,
                         "actual_matrix_lr": lr},
            "user_config": {"model_tag": tag, "total_batch_size": batch,
                            "ablation_matrix_lr": lr, "resume_from_step": 136316,
                            "ablation_token_budget": ABLATION_TOKENS},
            "dataloader_state_dict": copy.deepcopy(streams[0]["prefetched_loader_state"])}
    completion = {"status": "complete", "source_step": 136316, "terminal_step": step,
                  "added_tokens": ABLATION_TOKENS, "token_budget": ABLATION_TOKENS,
                  "canonical_step": 138364.0, "completed_updates": step - 136316,
                  "actual_matrix_lr": lr, "checkpoint_dir": f"/original/base_checkpoints/{tag}",
                  "checkpoint_files": checkpoint_names(step), "terminal_val_bpb": 0.68,
                  "consumed_streams": streams,
                  "optimizer": [{"kind": "muon", "lr": lr, "initial_lr": lr} for _ in range(4)]}
    return meta, completion


def published_fixture(directory, batch=B):
    meta, completion = ablation_fixture(batch)
    step = meta["step"]
    for name in checkpoint_names(step):
        (directory / name).write_bytes(b"model-weights" if name.startswith("model_") else b"optimizer")
    (directory / f"meta_{step:06d}.json").write_text(json.dumps(meta))
    complete_path = directory / "COMPLETE.json"
    complete_path.write_text(json.dumps(completion))
    manifest = directory / "winner.json"
    create(directory, step, manifest, complete_path)
    release = {"status": "approved_for_production", "validation_status": "all_four_arms_complete_and_comparable",
               "winner_completion": str(complete_path), "winner_completion_sha256": digest(complete_path),
               "winner_checkpoint_manifest": str(manifest), "winner_checkpoint_manifest_sha256": digest(manifest),
               "actual_checkpoint_step": step, "total_batch_size": batch,
               "actual_matrix_lr": meta["ablation"]["actual_matrix_lr"]}
    release_path = directory / "release.json"
    release_path.write_text(json.dumps(release))
    return meta, completion, complete_path, manifest, release_path, release


def runtime_resume():
    tree = ast.parse((HERE / "production_runtime.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resume_loader_state")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_resume_loader_state", "exec"), namespace)
    return namespace["resume_loader_state"]


class CompletionEvidenceTests(unittest.TestCase):
    @unittest.skipUnless((HERE.parent / "audits").is_dir(), "needs the original fleet's arm 2/4 records (not in this recipe)")
    def test_actual_completed_arm2_and_arm4_records(self):
        fleet = HERE.parent
        for arm, tag in ((2, "d34abl136316_b2m_lr020"), (4, "d34abl136316_b1m_lr014142")):
            with self.subTest(arm=arm):
                meta = json.loads((fleet / "audits" / f"20260914T1429_arm{arm}_terminal_files.json").read_text())["metadata"]
                value = json.loads((fleet / "collected" / "branches" / tag / "COMPLETE.json").read_text())
                states = validate_completion(meta, value)
                for rank in range(8):
                    self.assertEqual(runtime_resume()(meta, rank, {"loader_states": states}),
                                     value["consumed_streams"][rank]["prefetched_loader_state"])
                self.assertEqual([row["state"]["pq_idx"] for row in states], [1990, 1991, 1992, 1995, 1995, 1995, 1995, 1995])

    def test_first_resume_uses_each_actual_rank_cursor_for_both_batches(self):
        for batch in (B, 2 * B):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                meta, complete, path, manifest, release_path, release = published_fixture(directory, batch)
                evidence = released_initial_evidence(meta, directory, directory, release_path, path,
                                                     release["winner_completion_sha256"])
                for rank in range(8):
                    self.assertEqual(runtime_resume()(meta, rank, evidence),
                                     complete["consumed_streams"][rank]["prefetched_loader_state"])
                self.assertNotEqual(runtime_resume()(meta, 7, evidence), meta["dataloader_state_dict"])
                with self.assertRaises(AssertionError):
                    runtime_resume()(meta, 7)

    def test_missing_malformed_and_mismatched_completion_fields_fail(self):
        meta, valid = ablation_fixture()
        mutations = {
            "status": lambda v: v.update(status="partial_deadline"),
            "source": lambda v: v.update(source_step=154800),
            "step": lambda v: v.update(terminal_step=137340),
            "budget": lambda v: v.update(added_tokens=ABLATION_TOKENS - B),
            "declared_budget": lambda v: v.update(token_budget=ABLATION_TOKENS - B),
            "clock": lambda v: v.update(canonical_step=138365),
            "batch_updates": lambda v: v.update(completed_updates=1024),
            "lr": lambda v: v.update(actual_matrix_lr=0.02),
            "tag": lambda v: v.update(checkpoint_dir="/original/another_arm"),
            "file": lambda v: v["checkpoint_files"].pop(),
            "rank_missing": lambda v: v["consumed_streams"].pop(),
            "rank_duplicate": lambda v: v["consumed_streams"][7].update(rank=6),
            "count": lambda v: v["consumed_streams"][7].update(consumed_microbatches=1),
            "count_nonint": lambda v: v["consumed_streams"][7].update(consumed_microbatches=16384.0),
            "tokens": lambda v: v["consumed_streams"][7].update(consumed_x_tokens=1),
            "hash": lambda v: v["consumed_streams"][7].update(sha256="bad"),
            "cursor_absent": lambda v: v["consumed_streams"][7].pop("prefetched_loader_state"),
            "cursor_wrong_rank": lambda v: v["consumed_streams"][7]["prefetched_loader_state"].update(rg_idx=24),
            "cursor_val_shard": lambda v: v["consumed_streams"][7]["prefetched_loader_state"].update(pq_idx=6542),
            "cursor_negative": lambda v: v["consumed_streams"][7]["prefetched_loader_state"].update(rg_idx=-1),
            "cursor_nonint": lambda v: v["consumed_streams"][7]["prefetched_loader_state"].update(epoch=1.0),
            "cursor_epoch": lambda v: v["consumed_streams"][7]["prefetched_loader_state"].update(epoch=0),
            "rank_zero": lambda v: v["consumed_streams"][0]["prefetched_loader_state"].update(pq_idx=2000),
            "optimizer_lr": lambda v: v["optimizer"][0].update(initial_lr=0.02),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                altered = copy.deepcopy(valid)
                mutation(altered)
                with self.assertRaises((AssertionError, KeyError)):
                    validate_completion(meta, altered)

    def test_missing_completion_and_changed_bytes_fail_even_after_cached_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            meta, complete, path, manifest, release_path, release = published_fixture(directory)
            receipt = directory / "receipt.json"
            verify(directory, manifest, receipt, path)
            self.assertEqual(verify(directory, manifest, receipt, path)["hashed_bytes"], 0)
            with self.assertRaises(AssertionError):
                load_completion(meta, None, release["winner_completion_sha256"])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify(directory, manifest, receipt, path)
            complete["consumed_streams"][7]["prefetched_loader_state"]["pq_idx"] += 1
            path.write_text(json.dumps(complete))
            with self.assertRaises(AssertionError):
                verify(directory, manifest, receipt, path)
            with self.assertRaises(AssertionError):
                released_initial_evidence(meta, directory, directory, release_path, path,
                                          release["winner_completion_sha256"])

    def test_release_manifest_source_and_metadata_bindings_fail_closed(self):
        for alteration in ("release_completion_sha", "manifest_bytes", "source_dir", "metadata_bytes", "completion_pin"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                meta, complete, path, manifest, release_path, release = published_fixture(directory)
                source_dir = directory
                expected_sha = release["winner_completion_sha256"]
                if alteration == "release_completion_sha":
                    release["winner_completion_sha256"] = "0" * 64
                    release_path.write_text(json.dumps(release))
                elif alteration == "manifest_bytes":
                    manifest.write_text(manifest.read_text() + "\n")
                elif alteration == "source_dir":
                    source_dir = directory / "another_source"
                elif alteration == "metadata_bytes":
                    meta_path = directory / f"meta_{meta['step']:06d}.json"
                    meta_path.write_text(meta_path.read_text() + "\n")
                else:
                    value = json.loads(manifest.read_text())
                    value["completion"]["sha256"] = "0" * 64
                    manifest.write_text(json.dumps(value))
                    release["winner_checkpoint_manifest_sha256"] = digest(manifest)
                    release_path.write_text(json.dumps(release))
                with self.assertRaises(AssertionError):
                    released_initial_evidence(meta, source_dir, directory, release_path, path, expected_sha)

    def test_subsequent_production_resume_ignores_initial_evidence(self):
        states = [{"rank": rank, "state": {"pq_idx": 2200 + rank, "rg_idx": 40 + rank, "epoch": 1}}
                  for rank in range(8)]
        meta = {"production": {"loader_states": list(reversed(states))}}
        for rank in range(8):
            self.assertEqual(runtime_resume()(meta, rank, {"loader_states": []}), states[rank]["state"])


if __name__ == "__main__":
    unittest.main()
