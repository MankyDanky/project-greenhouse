"""Resumable full-corpus SHA256 creation/verification; CLI requires an allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import time

from winner_manifest import atomic, digest, stamp

NAMES = [f"shard_{index:05d}.parquet" for index in range(6543)]


def read_ledger(path):
    result = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
                result[row["path"]] = row
            except (ValueError, KeyError):
                continue  # A killed writer may leave a partial final line.
    return result


def hash_records(data_dir, expected, ledger_path, workers, receipt_only=False):
    assert 1 <= workers <= 4
    data_dir, ledger_path = Path(data_dir), Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    cached = read_ledger(ledger_path)

    def one(record):
        name = record["path"]
        assert name in NAMES
        path = data_dir / name
        before = stamp(path)
        assert path.is_file() and before["size"] > 0
        if "size" in record:
            assert before["size"] == record["size"], f"Corpus size mismatch: {path}"
        prior = cached.get(name)
        if prior and prior.get("stamp") == before and len(prior.get("sha256", "")) == 64:
            checksum = prior["sha256"]
            reused = True
        else:
            assert not receipt_only, f"Full corpus receipt missing/stale: {path}; verify in a CPU allocation"
            checksum = digest(path)
            assert stamp(path) == before, f"Corpus changed while hashing: {path}"
            reused = False
        if "sha256" in record:
            assert checksum == record["sha256"], f"Corpus checksum mismatch: {path}"
        return {"path": name, "size": before["size"], "sha256": checksum, "stamp": before}, reused

    rows, hashed_bytes = [], 0
    # Write each completed result so interruptions can resume without rereading
    # unchanged shards. The executor's bounded worker count caps open streams.
    with ThreadPoolExecutor(max_workers=workers) as executor, ledger_path.open("a") as ledger:
        for row, reused in executor.map(one, expected):
            rows.append(row)
            if not reused:
                hashed_bytes += row["size"]
                ledger.write(json.dumps(row, sort_keys=True) + "\n")
                ledger.flush()
            if len(rows) % 100 == 0:
                print(json.dumps({"verified_shards": len(rows), "of": len(expected),
                                  "hashed_bytes_this_run": hashed_bytes}), flush=True)
    return rows, hashed_bytes


def create(data_dir, output, workers=4):
    output = Path(output)
    assert not output.exists(), "Refusing to overwrite a complete corpus manifest"
    rows, hashed = hash_records(data_dir, [{"path": name} for name in NAMES],
                                output.with_name(output.name + ".partial.jsonl"), workers)
    value = {"format": "d34-full-climbmix-v1", "shard_count": len(NAMES),
             "validation_index": 6542,
             "source_dataset": "karpathy/climbmix-400b-shuffle",
             "identity_note": "Content-addressed frozen corpus; no unverified upstream revision claim",
             "files": [{k: row[k] for k in ("path", "size", "sha256")} for row in rows],
             "total_bytes": sum(row["size"] for row in rows),
             "source_host": socket.gethostname(), "source_job": os.environ.get("SLURM_JOB_ID"),
             "created_at": time.time()}
    atomic(output, value)
    return {"manifest": str(output), "manifest_sha256": digest(output),
            "files": len(rows), "total_bytes": value["total_bytes"], "hashed_bytes_this_run": hashed}


def verify(data_dir, manifest_path, receipt_path, workers=4, receipt_only=False):
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    value = json.loads(manifest_path.read_text())
    assert value["format"] == "d34-full-climbmix-v1"
    assert value["shard_count"] == 6543 and value["validation_index"] == 6542
    assert [row["path"] for row in value["files"]] == NAMES
    assert sum(row["size"] for row in value["files"]) == value["total_bytes"]
    for row in value["files"]:
        assert isinstance(row["size"], int) and row["size"] > 0
        assert len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"])
    manifest_sha = digest(manifest_path)
    if receipt_only:
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "verified" and receipt["manifest_sha256"] == manifest_sha
        assert set(receipt["files"]) == set(NAMES)
        for row in value["files"]:
            current = stamp(Path(data_dir) / row["path"])
            prior = receipt["files"][row["path"]]
            assert prior["sha256"] == row["sha256"] and prior["stamp"] == current, f"Stale corpus receipt: {row['path']}"
        return {"status": "verified", "manifest_sha256": manifest_sha, "files": 6543,
                "receipt_only": True, "hashed_bytes_this_run": 0}
    rows, hashed = hash_records(data_dir, value["files"],
                                receipt_path.with_name(receipt_path.name + ".partial.jsonl"), workers)
    receipt = {"status": "verified", "manifest_sha256": manifest_sha,
               "data_root": str(Path(data_dir).resolve()), "verified_at": time.time(),
               "job_id": os.environ.get("SLURM_JOB_ID"),
               "files": {row["path"]: {"sha256": row["sha256"], "stamp": row["stamp"]} for row in rows}}
    atomic(receipt_path, receipt)
    return {"status": "verified", "manifest_sha256": manifest_sha, "files": len(rows),
            "receipt_only": False, "hashed_bytes_this_run": hashed}


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "Hash/verify full corpus only inside a CPU/GPU allocation"
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("create")
    first.add_argument("--data-dir", required=True)
    first.add_argument("--output", required=True)
    first.add_argument("--workers", type=int, default=4)
    second = commands.add_parser("verify")
    second.add_argument("--data-dir", required=True)
    second.add_argument("--manifest", required=True)
    second.add_argument("--receipt", required=True)
    second.add_argument("--workers", type=int, default=4)
    second.add_argument("--receipt-only", action="store_true")
    args = parser.parse_args()
    result = (create(args.data_dir, args.output, args.workers) if args.command == "create" else
              verify(args.data_dir, args.manifest, args.receipt, args.workers, args.receipt_only))
    print(json.dumps(result, sort_keys=True), flush=True)
