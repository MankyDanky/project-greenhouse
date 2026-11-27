"""Run the Qwen3-0.6B-Base control for Gaggle 0.6B RLHN-250K SFT on Modal.

Use a Modal profile for the intended workspace. Stage and preflight on CPU,
probe six updates on an H200, then deploy and spawn the train function from the
deployed app only after checking the workspace's remaining budget. Passing 200
to train creates a checkpointed learning-rate pilot; spawning train again
without that limit resumes the same one-epoch run.
Training saves resumable checkpoints to its own Volume; rerunning train resumes.
"""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_COMMIT = "9b611eabd880de20dd03bac6409dc78abeee95da"
BASE_REPO = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
BASE_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
DATA_SHA256 = "c3743e83920a54d260483573d9d3176e165dcc3a34e398413ecc49f75b9c06d0"
PLAN_SHA256 = "858fd1cd772a3f29903b72342fcf7335f93def8119aab41f231755a90de1e136"
EXPECTED_STEPS = 1928
RUN_NAME = "qwen3_06b_lce_rlhn250k_20261127"
MAX_PROCESS_SECONDS = 10 * 3600
CONFIG = "configs/experiments/qwen3_06b_lce.yaml"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git init /opt/project-greenhouse",
        "git -C /opt/project-greenhouse remote add origin https://github.com/MankyDanky/project-greenhouse.git",
        f"git -C /opt/project-greenhouse fetch --depth 1 origin {REPO_COMMIT}",
        "git -C /opt/project-greenhouse checkout --detach FETCH_HEAD",
        "python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130",
        "python -m pip install -r /opt/project-greenhouse/reranker/requirements.txt",
    )
    .env({"HF_HOME": "/qwen/hf", "WANDB_MODE": "offline"})
)
app = modal.App("greenhouse-qwen3-06b-rlhn-sft", image=image)
shared_data = modal.Volume.from_name("greenhouse-sft-bounded-probe")
model_volume = modal.Volume.from_name("greenhouse-qwen3-06b-inputs", create_if_missing=True)
outputs = modal.Volume.from_name("greenhouse-qwen3-06b-sft", create_if_missing=True)

reranker_dir = Path("/opt/project-greenhouse/reranker")
model_source = Path("/qwen/model")
model_dir = reranker_dir / "models/qwen3-0.6b-base"
data_file = reranker_dir / "data/rlhn250k_prepared_climbmix_lens/train.parquet"
run_dir = Path("/results") / RUN_NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def link_inputs() -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not model_dir.exists():
        model_dir.symlink_to(model_source, target_is_directory=True)
    data_link = reranker_dir / "data"
    if not data_link.exists():
        data_link.symlink_to("/probe/data", target_is_directory=True)


def check_inputs() -> dict:
    import yaml
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    link_inputs()
    if sha256(model_source / "model.safetensors") != BASE_SHA256:
        raise RuntimeError("Qwen3-0.6B-Base weight SHA256 mismatch")
    if sha256(data_file) != DATA_SHA256:
        raise RuntimeError("Prepared RLHN-250K SHA256 mismatch")
    if pq.ParquetFile(data_file).metadata.num_rows != 247534:
        raise RuntimeError("Prepared RLHN-250K row count mismatch")
    config = AutoConfig.from_pretrained(model_dir)
    if (config.model_type, config.num_hidden_layers, config.hidden_size, config.tie_word_embeddings) != (
        "qwen3", 28, 1024, True
    ):
        raise RuntimeError("Pinned checkpoint is not the expected Qwen3-0.6B base architecture")
    qwen = yaml.safe_load((reranker_dir / CONFIG).read_text())
    gaggle = yaml.safe_load((reranker_dir / "configs/experiments/gaggle06b_lce.yaml").read_text())
    controlled = (
        "train_parquet", "max_query_len", "max_doc_len", "max_tokens_per_batch", "max_groups",
        "global_batch_groups", "k_choices", "precision", "grad_checkpointing", "attn_implementation",
        "tau", "epochs", "lr", "weight_decay", "warmup_steps", "clip", "seed", "num_workers",
        "resume", "save_every", "log_every",
    )
    changed = {key: (gaggle.get(key), qwen.get(key)) for key in controlled
               if gaggle.get(key) != qwen.get(key)}
    if changed or qwen["pack_world"] != 1 or qwen["packing_scaffold"] != 14:
        raise RuntimeError(f"SFT control differs from Gaggle: {changed}")
    sys.path.insert(0, str(reranker_dir / "engines/multi"))
    from prompts import PairBuilder, answer_ids, setup_tokenizer
    from sampling import load_lengths, make_sampler, plan_of, summarize

    tok = setup_tokenizer(AutoTokenizer.from_pretrained(model_dir), "qwen")
    builder = PairBuilder(tok, "qwen", "caps", 128, 512)
    if builder.overhead != 12 or answer_ids(tok) != (830, 895):
        raise RuntimeError("Qwen prompt scaffold or answer tokens changed")
    avail, neg_lens, query_lens = load_lengths(data_file)
    sampler = make_sampler(qwen, avail, neg_lens, query_lens)
    plan = summarize(plan_of(sampler, qwen))
    if plan["sha256"] != PLAN_SHA256 or plan["steps"] != EXPECTED_STEPS:
        raise RuntimeError(f"Qwen plan differs from Gaggle: {plan}")
    return {
        "repo_commit": REPO_COMMIT, "base_revision": BASE_REVISION, "base_sha256": BASE_SHA256,
        "data_sha256": DATA_SHA256, "data_rows": 247534, "plan": plan,
        "config_sha256": sha256(reranker_dir / CONFIG),
        "qwen_scaffold": builder.overhead, "packing_scaffold": qwen["packing_scaffold"],
        "answer_ids": list(answer_ids(tok)), "peak_lr": qwen["lr"], "train_gpu": "H200",
    }


@app.function(cpu=4, memory=16384, timeout=3600,
              volumes={"/probe": shared_data, "/qwen": model_volume})
def stage_inputs() -> dict:
    from huggingface_hub import snapshot_download

    model_source.mkdir(parents=True, exist_ok=True)
    snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=model_source,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json"])
    receipt = check_inputs()
    model_volume.commit()
    print("STAGE_OK", json.dumps(receipt), flush=True)
    return receipt


@app.function(cpu=4, memory=16384, timeout=1200,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs})
def preflight() -> dict:
    receipt = check_inputs()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "preflight.json"
    if (run_dir / "DONE").exists() or any(run_dir.glob("resume_*.pt")) or (
        run_dir / "resolved_config.yaml"
    ).exists():
        raise RuntimeError("Output already contains training state; do not replace its preflight")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    outputs.commit()
    print("PREFLIGHT_OK", json.dumps(receipt), flush=True)
    return receipt


def run_training(probe: bool, stop_after_steps: int = 0, probe_chunk_tokens: int = 0,
                 probe_gpu: str = "H200") -> dict:
    receipt = check_inputs()
    if json.loads((run_dir / "preflight.json").read_text()) != receipt:
        raise RuntimeError("Missing or changed preflight receipt")
    target = run_dir / f"probe_{probe_gpu.lower().replace('-', '_')}_{probe_chunk_tokens}" if probe else run_dir
    if not probe and (run_dir / "DONE").exists():
        return {"status": "already_complete", **receipt}
    env = os.environ.copy()
    command = ["bash", "launch/train.sh", CONFIG, "1", "--override",
               f"model_name={model_dir}", f"output_dir={target}"]
    if probe:
        env["LCE_STOP_AFTER_STEPS"] = "6"
        env["LCE_TEST_NOSAVE"] = "1"
        command.append("log_every=1")
        if probe_chunk_tokens:
            if not 1 <= probe_chunk_tokens <= 120000:
                raise ValueError("probe_chunk_tokens must be between 1 and 120000")
            command.append(f"exec_chunk_tokens={probe_chunk_tokens}")
    else:
        env["LCE_STOP_AFTER_SEC"] = str(MAX_PROCESS_SECONDS)
        if stop_after_steps:
            if not 1 <= stop_after_steps < EXPECTED_STEPS:
                raise ValueError("stop_after_steps must be inside the one-epoch plan")
            env["LCE_STOP_AFTER_STEPS"] = str(stop_after_steps)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=reranker_dir, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    committed = set()
    for line in process.stdout:
        print(line, end="", flush=True)
        if not probe:
            for path in target.glob("resume_*.pt"):
                if path.name not in committed:
                    outputs.commit()
                    committed.add(path.name)
    code = process.wait()
    if code:
        outputs.commit()
        raise RuntimeError(f"Trainer exited with code {code}")
    outputs.commit()
    elapsed = round(time.monotonic() - started, 1)
    if probe:
        rows = []
        for path in target.glob("train_*_rank0.jsonl"):
            rows += [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 6 or [r["step"] for r in rows] != list(range(1, 7)):
            raise RuntimeError(f"Expected exactly six probe updates; got {[r['step'] for r in rows]}")
        import torch
        result = {"status": "probed", "steps": 6, "gpu_requested": probe_gpu,
                  "gpu_actual": torch.cuda.get_device_name(0),
                  "median_update_seconds": statistics.median(r["step_sec"] for r in rows[1:]),
                  "wall_seconds": elapsed, "updates": rows, **receipt}
    else:
        done = run_dir / "DONE"
        resumes = sorted(target.glob("resume_*.pt"))
        if done.exists() and json.loads(done.read_text())["step"] == EXPECTED_STEPS:
            status = "complete"
        elif resumes:
            status = "paused"
        else:
            raise RuntimeError("Trainer exited without a final model or resume point")
        result = {"status": status, "wall_seconds": elapsed,
                  "latest_resume": resumes[-1].name if resumes else None, **receipt}
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        outputs.commit()
    print("QWEN_SFT_RESULT", json.dumps(result), flush=True)
    return result


@app.function(gpu="H200", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def probe(chunk_tokens: int = 0) -> dict:
    return run_training(True, probe_chunk_tokens=chunk_tokens, probe_gpu="H200")


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def probe_rtx(chunk_tokens: int = 120000) -> dict:
    return run_training(True, probe_chunk_tokens=chunk_tokens, probe_gpu="RTX-PRO-6000")


@app.function(gpu="A100-80GB", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def probe_a100(chunk_tokens: int = 120000) -> dict:
    return run_training(True, probe_chunk_tokens=chunk_tokens, probe_gpu="A100-80GB")


@app.function(gpu="L40S", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def probe_l40s(chunk_tokens: int = 120000) -> dict:
    return run_training(True, probe_chunk_tokens=chunk_tokens, probe_gpu="L40S")


@app.function(gpu="B200", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def probe_b200(chunk_tokens: int = 120000) -> dict:
    return run_training(True, probe_chunk_tokens=chunk_tokens, probe_gpu="B200")


@app.function(gpu="H200", cpu=8, memory=32768, timeout=11 * 3600,
              volumes={"/probe": shared_data, "/qwen": model_volume, "/results": outputs},
              single_use_containers=True)
def train(stop_after_steps: int = 0) -> dict:
    return run_training(False, stop_after_steps)


@app.local_entrypoint()
def main(stage: str = "preflight", chunk_tokens: int = 0) -> None:
    if stage == "stage":
        print(json.dumps(stage_inputs.remote(), indent=2))
    elif stage == "preflight":
        print(json.dumps(preflight.remote(), indent=2))
    elif stage == "probe":
        print(json.dumps(probe.remote(chunk_tokens), indent=2))
    elif stage == "probe-rtx":
        print(json.dumps(probe_rtx.remote(chunk_tokens or 120000), indent=2))
    elif stage == "probe-a100":
        print(json.dumps(probe_a100.remote(chunk_tokens or 120000), indent=2))
    elif stage == "probe-l40s":
        print(json.dumps(probe_l40s.remote(chunk_tokens or 120000), indent=2))
    elif stage == "probe-b200":
        print(json.dumps(probe_b200.remote(chunk_tokens or 120000), indent=2))
    else:
        raise ValueError("Training must be spawned from the deployed app; this entrypoint is for staging and probes")
