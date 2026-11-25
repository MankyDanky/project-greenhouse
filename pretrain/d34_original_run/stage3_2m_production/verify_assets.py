"""Verify a path-independent staged identity manifest, only inside an allocation."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time

ROOT_ENV = {
    "experiment": "D34_EXP_DIR", "checkpoint": "D34_CHECKPOINT_DIR",
    "base": "D34_BASE_DIR", "hf_home": "D34_HF_HOME",
    "runtime": "D34_RUNTIME_ROOT", "data": "D34_DATA_ROOT",
}
CHECKPOINT_NAMES = {"model_136316.pt", "meta_136316.json"} | {
    f"optim_136316_rank{rank}.pt" for rank in range(8)}
REQUIRED_SOURCES = {"source/base_train.py", "source/ablation_runtime.py", "fleet/data/indexed_shards.json"} | {
    f"upstream_snapshot/nanochat/{name}.py" for name in
    ("gpt", "optim", "tokenizer", "dataset", "dataloader", "checkpoint_manager", "common", "loss_eval", "fp8", "flash_attention")}
REQUIRED_DATA = {f"shard_{index:05d}.parquet" for index in list(range(1945, 2073)) + [6542]}
REQUIRED_TOKENIZER = {"tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"}


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_identity(manifest):
    files = sorted(manifest["files"], key=lambda item: (item["root"], item["path"]))
    body = {"version": manifest["version"], "files": files}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_record(record):
    root = record["root"]
    assert root in ROOT_ENV, f"Unknown asset root {root}"
    relative = PurePosixPath(record["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    assert str(relative) not in ("", ".")
    assert ROOT_ENV[root] in os.environ, f"Missing {ROOT_ENV[root]}"
    # Symlinked data/tokenizer/cache directories are legitimate staged layouts.
    return Path(os.environ[ROOT_ENV[root]]) / str(relative)


def fingerprint(path):
    info = path.stat()
    return {"path": str(path.resolve()), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "inode": info.st_ino}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_manifest(manifest):
    assert manifest["version"] == 1
    keys = [(item["root"], item["path"]) for item in manifest["files"]]
    assert len(keys) == len(set(keys)), "Duplicate asset entries"
    assert CHECKPOINT_NAMES <= {p for root, p in keys if root == "checkpoint"}, "All ten checkpoint files are required"
    assert REQUIRED_SOURCES <= {p for root, p in keys if root == "experiment"}, "Frozen training sources are missing"
    assert REQUIRED_DATA <= {p for root, p in keys if root == "data"}, "Staged training/validation corpus files are missing"
    assert REQUIRED_TOKENIZER <= {p for root, p in keys if root == "base"}, "Tokenizer assets are missing"
    for item in manifest["files"]:
        assert isinstance(item["size"], int) and item["size"] >= 0
        if item["root"] == "checkpoint":
            assert item["size"] > 0
        digest = item["sha256"]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def verify(manifest_path, receipt_path, provenance_path=None, receipt_only=False):
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    identity = canonical_identity(manifest)
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    old_records = old.get("verified_files", {}) if old.get("asset_identity") == identity else {}
    records = {}
    hashed_bytes = cached_files = 0
    started = time.time()
    for record in manifest["files"]:
        key = record["root"] + ":" + record["path"]
        path = resolve_record(record)
        assert path.is_file(), f"Missing asset: {path}"
        stamp = fingerprint(path)
        assert stamp["size"] == record["size"], f"Asset size differs: {path}"
        cached = old_records.get(key, {})
        if cached.get("fingerprint") == stamp and cached.get("sha256") == record["sha256"]:
            cached_files += 1
        else:
            assert not receipt_only, f"Asset receipt missing or stale: {path}"
            actual = sha256_file(path)
            assert actual == record["sha256"], f"Asset checksum differs: {path}"
            assert fingerprint(path) == stamp, f"Asset changed while hashing: {path}"
            hashed_bytes += record["size"]
        records[key] = {"sha256": record["sha256"], "fingerprint": stamp}
    result = {"status": "verified", "asset_identity": identity,
              "manifest_sha256": sha256_file(manifest_path), "verified_at": time.time(),
              "cluster": os.environ.get("D34_CLUSTER"), "job_id": os.environ.get("SLURM_JOB_ID"),
              "verified_files": records, "hashed_bytes_this_check": hashed_bytes,
              "cached_files_this_check": cached_files, "wall_seconds": time.time() - started}
    if not receipt_only:
        atomic_json(receipt_path, result)
    if provenance_path:
        atomic_json(provenance_path, result)
    return result


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "Hash staging assets inside a Slurm allocation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--receipt-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.manifest, args.receipt, args.provenance, args.receipt_only)
    print(json.dumps({key: value for key, value in result.items() if key != "verified_files"}, sort_keys=True))
