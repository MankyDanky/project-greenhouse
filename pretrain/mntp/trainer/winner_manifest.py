"""Create/verify a complete selected checkpoint's byte manifest inside an allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from production_clock import source_clock
from completion_evidence import checkpoint_names, load_completion


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stamp(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


filenames = checkpoint_names


def create(directory, step, output, completion_path):
    directory = Path(directory)
    meta = json.loads((directory / f"meta_{step:06d}.json").read_text())
    canonical, tokens = source_clock(meta)
    assert meta["step"] == step
    evidence = load_completion(meta, completion_path, digest(completion_path))
    records = []
    for name in filenames(step):
        path = directory / name
        before = stamp(path)
        assert path.is_file() and before["size"] > 0
        checksum = digest(path)
        assert stamp(path) == before, f"Checkpoint changed while hashing: {path}"
        records.append({"path": name, "size": before["size"], "sha256": checksum})
    value = {"format": "d34-winning-checkpoint-v2", "actual_step": step,
             "canonical_step": canonical, "total_training_tokens": tokens,
             "total_batch_size": meta["total_batch_size"],
             "actual_matrix_lr": meta.get("production", meta.get("ablation"))["actual_matrix_lr"],
             "completion": {key: evidence[key] for key in ("sha256", "size", "source_tag")},
             "files": records, "created_host": socket.gethostname(),
             "created_job": os.environ.get("SLURM_JOB_ID"), "created_at": time.time()}
    assert not Path(output).exists(), "Refusing to overwrite an existing checkpoint manifest"
    atomic(output, value)
    return value


def verify(directory, manifest_path, receipt_path, completion_path):
    directory, manifest_path = Path(directory), Path(manifest_path)
    value = json.loads(manifest_path.read_text())
    assert value["format"] == "d34-winning-checkpoint-v2"
    expected = filenames(value["actual_step"])
    assert len(value["files"]) == 10 and {row["path"] for row in value["files"]} == set(expected)
    manifest_sha = digest(manifest_path)
    previous = json.loads(Path(receipt_path).read_text()) if Path(receipt_path).exists() else {}
    cached = previous.get("files", {}) if previous.get("manifest_sha256") == manifest_sha else {}
    records, hashed = {}, 0
    for row in value["files"]:
        path = directory / row["path"]
        before = stamp(path)
        assert path.is_file() and row["size"] > 0 and before["size"] == row["size"]
        assert len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"])
        if cached.get(row["path"]) != {"stamp": before, "sha256": row["sha256"]}:
            assert digest(path) == row["sha256"], f"Winner checksum mismatch: {path}"
            assert stamp(path) == before
            hashed += row["size"]
        records[row["path"]] = {"stamp": before, "sha256": row["sha256"]}
    meta = json.loads((directory / f"meta_{value['actual_step']:06d}.json").read_text())
    assert meta["step"] == value["actual_step"]
    assert source_clock(meta) == (value["canonical_step"], value["total_training_tokens"])
    assert meta["total_batch_size"] == value["total_batch_size"]
    assert meta.get("production", meta.get("ablation"))["actual_matrix_lr"] == value["actual_matrix_lr"]
    evidence = load_completion(meta, completion_path, value["completion"]["sha256"])
    assert value["completion"] == {key: evidence[key] for key in ("sha256", "size", "source_tag")}
    receipt = {"completion_sha256": evidence["sha256"], "status": "verified", "manifest_sha256": manifest_sha, "files": records,
               "hashed_bytes": hashed, "verified_at": time.time(), "job_id": os.environ.get("SLURM_JOB_ID")}
    atomic(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "Run checkpoint hashing in a CPU/GPU allocation"
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("create")
    first.add_argument("--checkpoint-dir", required=True)
    first.add_argument("--step", type=int, required=True)
    first.add_argument("--output", required=True)
    first.add_argument("--completion", required=True)
    second = commands.add_parser("verify")
    second.add_argument("--checkpoint-dir", required=True)
    second.add_argument("--manifest", required=True)
    second.add_argument("--receipt", required=True)
    second.add_argument("--completion", required=True)
    args = parser.parse_args()
    if args.command == "create":
        value = create(args.checkpoint_dir, args.step, args.output, args.completion)
        print(json.dumps({"manifest": args.output, "sha256": digest(args.output),
                          "files": len(value["files"])}, sort_keys=True))
    else:
        value = verify(args.checkpoint_dir, args.manifest, args.receipt, args.completion)
        print(json.dumps({key: item for key, item in value.items() if key != "files"}, sort_keys=True))
