"""Validate the original ablation completion record before its first resume."""
import hashlib
import json
import math
from pathlib import Path

from production_clock import ABLATION_TOKENS, REFERENCE_BATCH, RAMP_SOURCE_STEP, source_clock


def checkpoint_names(step):
    return [f"model_{step:06d}.pt", f"meta_{step:06d}.json"] + [
        f"optim_{step:06d}_rank{rank}.pt" for rank in range(8)]


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_completion(meta, value):
    """Return the eight recorded prefetched cursors; never infer absent ranks."""
    assert "production" not in meta and "ablation" in meta, "Expected an ablation terminal checkpoint"
    canonical, _ = source_clock(meta)
    assert canonical == 138364
    batch, step = meta["total_batch_size"], meta["step"]
    assert type(batch) is int and batch in (REFERENCE_BATCH, 2 * REFERENCE_BATCH)
    assert type(step) is int
    assert all(type(value[key]) is int for key in
               ("source_step", "terminal_step", "added_tokens", "token_budget", "completed_updates"))
    assert value["status"] == "complete"
    assert value["source_step"] == meta["ablation"]["source_step"] == RAMP_SOURCE_STEP
    assert value["terminal_step"] == step
    assert value["added_tokens"] == value["token_budget"] == ABLATION_TOKENS
    assert value["canonical_step"] == canonical
    updates = value["completed_updates"]
    assert type(updates) is int and updates == step - RAMP_SOURCE_STEP
    assert updates * batch == ABLATION_TOKENS, "Completion update count/batch does not match exact budget"
    actual_lr = meta["ablation"]["actual_matrix_lr"]
    assert actual_lr in (0.02, 0.01414213562373095) and value["actual_matrix_lr"] == actual_lr
    config = meta["user_config"]
    assert config["total_batch_size"] == batch
    assert config["ablation_matrix_lr"] == actual_lr
    assert config["resume_from_step"] == RAMP_SOURCE_STEP
    assert config["ablation_token_budget"] == ABLATION_TOKENS
    tag = config["model_tag"]
    assert isinstance(tag, str) and tag.startswith("d34abl136316_") and Path(tag).name == tag
    assert Path(value["checkpoint_dir"]).is_absolute() and Path(value["checkpoint_dir"]).name == tag
    names = value["checkpoint_files"]
    assert len(names) == 10 and set(names) == set(checkpoint_names(step))
    assert math.isfinite(value["terminal_val_bpb"]) and value["terminal_val_bpb"] == meta["val_bpb"]
    muon = [row for row in value["optimizer"] if row["kind"] == "muon"]
    assert len(muon) == 4 and all(row["lr"] == row["initial_lr"] == actual_lr for row in muon)

    streams = value["consumed_streams"]
    assert len(streams) == 8 and all(type(row["rank"]) is int for row in streams)
    assert sorted(row["rank"] for row in streams) == list(range(8)), "Missing or duplicate rank evidence"
    states = []
    for row in sorted(streams, key=lambda row: row["rank"]):
        rank, state = row["rank"], row["prefetched_loader_state"]
        assert type(row["consumed_microbatches"]) is int and type(row["consumed_x_tokens"]) is int
        assert row["consumed_microbatches"] == 16384 and row["consumed_x_tokens"] == ABLATION_TOKENS // 8
        assert valid_sha(row["sha256"]), "Missing or malformed consumed-stream hash"
        assert set(state) == {"pq_idx", "rg_idx", "epoch"}
        assert all(type(item) is int for item in state.values())
        assert 0 <= state["pq_idx"] < 6542 and state["rg_idx"] >= 0 and state["epoch"] >= 1
        assert state["rg_idx"] % 8 == rank, "Cursor does not belong to this rank"
        states.append({"rank": rank, "state": dict(state)})
    assert states[0]["state"] == meta["dataloader_state_dict"], "Rank-zero metadata/completion cursor mismatch"
    return states


def read_bound_json(path, expected_sha):
    assert path and valid_sha(expected_sha), "Missing path or SHA256 evidence pin"
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha, f"Evidence checksum mismatch: {path}"
    return json.loads(raw), len(raw)


def load_completion(meta, path, expected_sha):
    value, size = read_bound_json(path, expected_sha)
    return {"sha256": expected_sha, "size": size,
            "source_tag": meta["user_config"]["model_tag"],
            "loader_states": validate_completion(meta, value)}


def released_initial_evidence(meta, source_dir, winner_dir, release_path, completion_path, completion_sha):
    """Recheck release pins in both preflight and each first-resume trainer rank."""
    assert release_path and winner_dir, "First resume requires the production release and verified winner directory"
    release = json.loads(Path(release_path).read_text())
    assert release["status"] == "approved_for_production"
    assert release["validation_status"] == "all_four_arms_complete_and_comparable"
    assert completion_path and Path(release["winner_completion"]).resolve() == Path(completion_path).resolve()
    assert release["winner_completion_sha256"] == completion_sha
    winner, _ = read_bound_json(release["winner_checkpoint_manifest"], release["winner_checkpoint_manifest_sha256"])
    assert winner["format"] == "d34-winning-checkpoint-v2"
    assert Path(source_dir).resolve() == Path(winner_dir).resolve(), "Initial source is not the verified winner"
    assert meta["step"] == winner["actual_step"] == release["actual_checkpoint_step"]
    assert meta["total_batch_size"] == winner["total_batch_size"] == release["total_batch_size"]
    assert meta["ablation"]["actual_matrix_lr"] == winner["actual_matrix_lr"] == release["actual_matrix_lr"]
    assert source_clock(meta) == (winner["canonical_step"], winner["total_training_tokens"])
    names = checkpoint_names(meta["step"])
    assert len(winner["files"]) == 10 and {row["path"] for row in winner["files"]} == set(names)
    meta_name = f"meta_{meta['step']:06d}.json"
    record = next(row for row in winner["files"] if row["path"] == meta_name)
    bound_meta, size = read_bound_json(Path(source_dir) / meta_name, record["sha256"])
    assert size == record["size"] and bound_meta == meta, "Loaded metadata differs from the released winner"
    evidence = load_completion(meta, completion_path, completion_sha)
    assert winner["completion"] == {key: evidence[key] for key in ("sha256", "size", "source_tag")}
    evidence["winner_manifest_sha256"] = release["winner_checkpoint_manifest_sha256"]
    return evidence
