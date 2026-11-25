"""Create the canonical asset manifest on the authoritative source allocation."""

import argparse
import json
import os
from pathlib import Path

from verify_assets import CHECKPOINT_NAMES, ROOT_ENV, atomic_json, sha256_file, validate_manifest


def make_records(include_runtime=False, include_hf=False):
    entries = set()
    experiment = Path(os.environ["D34_EXP_DIR"])
    for directory in ("source", "upstream_snapshot", "fleet"):
        for path in (experiment / directory).rglob("*"):
            relative = path.relative_to(experiment)
            if not path.is_file() or any(part in {"__pycache__", ".git"} for part in relative.parts):
                continue
            if path.suffix in {".py", ".sh", ".sbatch"} or path.name in {
                "pyproject.toml", "uv.lock", "indexed_shards.json"}:
                entries.add(("experiment", relative.as_posix()))
    entries.update(("checkpoint", name) for name in CHECKPOINT_NAMES)
    base = Path(os.environ["D34_BASE_DIR"])
    for path in (base / "tokenizer").rglob("*"):
        if path.is_file():
            entries.add(("base", path.relative_to(base).as_posix()))
    assert any(root == "base" for root, _ in entries), "Tokenizer files missing"
    for index in list(range(1945, 2073)) + [6542]:
        entries.add(("data", f"shard_{index:05d}.parquet"))
    for root, include in (("runtime", include_runtime), ("hf_home", include_hf)):
        if not include:
            continue
        directory = Path(os.environ[ROOT_ENV[root]])
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.stat().st_size:
                entries.add((root, path.relative_to(directory).as_posix()))
    records = []
    for root, relative in sorted(entries):
        path = Path(os.environ[ROOT_ENV[root]]) / relative
        size = path.stat().st_size
        if size == 0 and root == "experiment":
            # Empty package __init__.py is still source identity; record format allows it.
            pass
        records.append({"root": root, "path": relative, "size": size, "sha256": sha256_file(path)})
    return records


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), "Hash source assets inside an allocation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-runtime", action="store_true")
    parser.add_argument("--include-hf", action="store_true")
    args = parser.parse_args()
    manifest = {"version": 1, "files": make_records(args.include_runtime, args.include_hf)}
    validate_manifest(manifest)
    atomic_json(args.output, manifest)
    print(json.dumps({"manifest": args.output, "files": len(manifest["files"]),
                      "bytes": sum(item["size"] for item in manifest["files"])}))
