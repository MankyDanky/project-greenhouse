"""Allocated production preflight after static ablation assets were verified."""
import json
import os
from pathlib import Path
import time

from checkpoint_state import complete_checkpoints
from completion_evidence import released_initial_evidence
from corpus_manifest import verify as verify_corpus
from production_clock import source_clock
from winner_manifest import atomic, digest, filenames, verify


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Run production preflight inside an allocation"
    env = os.environ
    release = json.loads(Path(env["D34_PROD_RELEASE"]).read_text())
    assert release["status"] == "approved_for_production"
    assert release["validation_status"] == "all_four_arms_complete_and_comparable"
    code_root = Path(env["D34_PRODUCTION_CODE"])
    code_manifest = code_root / "code_manifest.json"
    assert digest(code_manifest) == release["production_code_manifest_sha256"]
    code = json.loads(code_manifest.read_text())
    assert code["format"] == "d34-production-code-v1"
    expected_code = {p.name for p in code_root.iterdir() if p.suffix in (".py", ".sh", ".sbatch")
                     and not p.name.startswith("test_")}
    assert len(code["files"]) == len(expected_code)
    assert {row["path"] for row in code["files"]} == expected_code
    for row in code["files"]:
        assert Path(row["path"]).name == row["path"]
        path = code_root / row["path"]
        assert path.stat().st_size == row["size"] and digest(path) == row["sha256"]

    winner_path = Path(release["winner_checkpoint_manifest"])
    assert digest(winner_path) == release["winner_checkpoint_manifest_sha256"]
    winner = json.loads(winner_path.read_text())
    assert winner["actual_step"] == int(env["D34_WINNER_STEP"]) == release["actual_checkpoint_step"]
    assert winner["total_batch_size"] == int(env["D34_WINNER_BATCH"]) == release["total_batch_size"]
    assert winner["actual_matrix_lr"] == float(env["D34_WINNER_LR"]) == release["actual_matrix_lr"]
    state_dir = Path(env["D34_PROD_DIR"])
    completion_path = env["D34_WINNER_COMPLETION"]
    assert Path(release["winner_completion"]).resolve() == Path(completion_path).resolve()
    assert release["winner_completion_sha256"] == env["D34_WINNER_COMPLETION_SHA256"] == winner["completion"]["sha256"]
    winner_receipt = verify(env["D34_WINNER_CHECKPOINT_DIR"], winner_path,
                            state_dir / "preflight" / "winner_receipt.json", completion_path)

    source_dir = Path(env["D34_PROD_SOURCE_DIR"])
    source_step = int(env["D34_PROD_SOURCE_STEP"])
    for name in filenames(source_step):
        path = source_dir / name
        assert path.is_file() and path.stat().st_size > 0, f"Incomplete resume checkpoint: {path}"
    source = json.loads((source_dir / f"meta_{source_step:06d}.json").read_text())
    canonical, total_tokens = source_clock(source)
    assert source["total_batch_size"] == int(env["D34_WINNER_BATCH"])
    if "production" in source:
        entries = complete_checkpoints(source_dir)
        assert any(entry["step"] == source_step for entry in entries)
        cursors = source["production"]["loader_states"]
        assert len(cursors) == 8 and sorted(row["rank"] for row in cursors) == list(range(8))
    else:
        evidence = released_initial_evidence(
            source, source_dir, env["D34_WINNER_CHECKPOINT_DIR"], env["D34_PROD_RELEASE"],
            completion_path, env["D34_WINNER_COMPLETION_SHA256"])
        assert source_step == source["step"], "Initial requested step differs from the released winner"
        cursors = evidence["loader_states"]
    for row in cursors:
        cursor = row["state"]
        assert 0 <= cursor["pq_idx"] < 6542 and cursor["rg_idx"] >= 0

    shard_manifest = json.loads(Path(env["D34_SHARD_MANIFEST"]).read_text())
    names = [f"shard_{index:05d}.parquet" for index in range(6543)]
    assert shard_manifest["format"] == "d34-indexed-climbmix-v1"
    assert shard_manifest["parquet_basenames"] == names
    data_root = Path(env["D34_DATA_ROOT"])
    # Full-corpus hashing is performed in a CPU allocation before launch. GPU
    # preflight checks every verified file's current fingerprint without rereading
    # 558 GiB. A missing, modified or only partially staged corpus fails closed.
    full_manifest = Path(release["full_corpus_manifest"])
    assert digest(full_manifest) == release["full_corpus_manifest_sha256"]
    corpus_receipt = verify_corpus(data_root, full_manifest, env["D34_FULL_CORPUS_RECEIPT"], receipt_only=True)
    result = {"status": "passed", "job_id": env["SLURM_JOB_ID"], "verified_at": time.time(),
              "source_actual_step": source_step, "source_canonical_step": canonical,
              "source_total_training_tokens": total_tokens, "loader_cursors": cursors,
              "corpus_mode": "full_corpus_sha256_receipt", "parquet_files": len(names),
              "parquet_root": str(data_root.resolve()),
              "winner_manifest_sha256": winner_receipt["manifest_sha256"],
              "winner_completion_sha256": winner_receipt["completion_sha256"],
              "production_code_manifest_sha256": digest(code_manifest),
              "full_corpus_manifest_sha256": corpus_receipt["manifest_sha256"],
              "checksum_scope": "Static ablation assets, all ten initial winner files, and all 6543 full-corpus shards"}
    atomic(state_dir / "preflight" / env["SLURM_JOB_ID"] / "production.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
