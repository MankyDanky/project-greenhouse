"""Allocated GPU/runtime provenance and an eight-rank NCCL smoke test."""

import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import torch
import torch.distributed as dist

from verify_assets import atomic_json, sha256_file, verify


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    expected_local = int(os.environ["D34_GPUS_PER_NODE"])
    assert world == 8 and local_world == expected_local
    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__ == "2.9.1+cu128", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.device_count() == expected_local
    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    assert any(model in properties.name for model in ("H100", "H200")), properties.name
    assert (properties.major, properties.minor) == (9, 0), "Require the same Hopper architecture"
    assert properties.total_memory >= 79 * 1024**3, "Require full >=80GB-class devices"
    if local_rank == 0:
        inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        assert "MIG" not in inventory, "MIG allocations/nodes are excluded"
    else:
        inventory = None

    from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
    from nanochat.flash_attention import USE_FA3
    assert COMPUTE_DTYPE == torch.bfloat16, str(COMPUTE_DTYPE)
    assert USE_FA3, "The original Flash Attention 3 backend must be available offline"

    # Heavy checksums completed in the preceding single-node Slurm step.
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    if local_rank == 0:
        receipt = verify(os.environ["D34_ASSET_MANIFEST"], os.environ["D34_ASSET_RECEIPT"], receipt_only=True)
        identity = receipt["asset_identity"]
    else:
        identity = None
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    assert value.item() == 36.0
    packages = {}
    for name in ("torch", "triton", "numpy", "pyarrow", "tokenizers", "huggingface-hub", "kernels", "wandb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    parquet_manifest = os.environ["D34_SHARD_MANIFEST"]
    current = {
        "rank": rank, "local_rank": local_rank, "world_size": world,
        "hostname": socket.gethostname(), "cluster": os.environ.get("D34_CLUSTER"),
        "job_id": os.environ.get("SLURM_JOB_ID"), "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "gpu_memory_bytes": properties.total_memory, "sm": [properties.major, properties.minor],
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "compute_dtype": str(COMPUTE_DTYPE),
        "compute_dtype_reason": COMPUTE_DTYPE_REASON, "use_fa3": bool(USE_FA3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "asset_identity": identity, "node_gpu_inventory": inventory,
        "parquet_manifest": parquet_manifest,
        "parquet_manifest_sha256": sha256_file(parquet_manifest) if parquet_manifest else None,
        "nccl_allreduce_sum": value.item(),
    }
    all_ranks = [None] * world
    dist.all_gather_object(all_ranks, current)
    if rank == 0:
        assert sorted(row["rank"] for row in all_ranks) == list(range(8))
        assert len({row["hostname"] for row in all_ranks}) == int(os.environ["D34_NNODES"])
        assert len({row["asset_identity"] for row in all_ranks if row["asset_identity"]}) == 1
        assert len({row["parquet_manifest_sha256"] for row in all_ranks}) == 1
        atomic_json(Path(os.environ["D34_BRANCH_DIR"]) / "provenance" / "runtime.json",
                    {"status": "passed", "ranks": all_ranks})
        print("FLEET_PREFLIGHT_PASSED " + json.dumps(all_ranks, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
