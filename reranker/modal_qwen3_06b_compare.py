"""Compare final Qwen3-0.6B SFT with our final Gaggle 0.6B SFT on Modal.

The BM25 candidate files and Gaggle scores come from the earlier reproduction's
Volume. Candidate text stays on Modal. Run prepare while training is in progress;
run compare only after the final Qwen checkpoint exists.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path

import modal

REPO_COMMIT = "9b611eabd880de20dd03bac6409dc78abeee95da"
QWEN_RUN = "qwen3_06b_lce_rlhn250k_20261127"
GAGGLE_RUN = "gaggle06b_lce_rlhn250k_20261125"
DL = ("dl19", "dl20", "dl21", "dl22", "dl23")
BEIR = ("covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa")
QRELS = {
    **{d: f"qrels.{d}-passage.txt" for d in DL},
    "covid": "qrels.beir-v1.0.0-trec-covid.test.txt",
    "news": "qrels.beir-v1.0.0-trec-news.test.txt",
    "robust04": "qrels.beir-v1.0.0-robust04.test.txt",
    "nfc": "qrels.beir-v1.0.0-nfcorpus.test.txt",
    "scifact": "qrels.beir-v1.0.0-scifact.test.txt",
    "scidocs": "qrels.beir-v1.0.0-scidocs-test.txt",
    "fiqa": "qrels.beir-v1.0.0-fiqa-test.txt",
}

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
    .apt_install("build-essential")
    .run_commands(
        "git clone --depth 1 --branch v9.0.8 https://github.com/usnistgov/trec_eval.git /opt/trec_eval",
        "make -C /opt/trec_eval",
        "make -C /opt/trec_eval quicktest",
    )
    .env({"HF_HOME": "/qwen/hf", "TOKENIZERS_PARALLELISM": "false",
          "EVAL_TREC_EVAL": "/opt/trec_eval/trec_eval"})
)
app = modal.App("greenhouse-qwen3-06b-sft-compare", image=image)
gaggle = modal.Volume.from_name("greenhouse-gaggle06b-sft")
qwen_inputs = modal.Volume.from_name("greenhouse-qwen3-06b-inputs")
qwen_outputs = modal.Volume.from_name("greenhouse-qwen3-06b-sft")
cache = modal.Volume.from_name("gaggle-hf-cache")

repo = Path("/opt/project-greenhouse")
qwen_dir = Path("/results") / QWEN_RUN
qwen_checkpoint = qwen_dir / "milestones/step01928_q247449.pt"
comparison = qwen_dir / "comparison_gaggle"
gaggle_comparison = Path("/gaggle") / GAGGLE_RUN / "comparison"
candidates = gaggle_comparison / "candidates"
qrels_dir = Path("/cache/pyserini/qrels")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def same_first_stage_run(dataset: str, trec: list[str]) -> bool:
    reference = repo / "eval/candidates/runs" / f"{dataset}.bm25.trec.gz"
    with gzip.open(reference, "rt") as stream:
        for actual, expected in zip_longest(trec, stream):
            if actual is None or expected is None:
                return False
            a, e = actual.split(), expected.split()
            if (a[0], a[2], a[3]) != (e[0], e[2], e[3]):
                return False
    return True


def datasets(suite: str) -> tuple[str, ...]:
    if suite == "dl":
        return DL
    if suite == "beir":
        return BEIR
    raise ValueError("suite must be dl or beir")


def verify_inputs(suite: str) -> tuple[dict, dict]:
    sys.path.insert(0, str(repo / "eval/candidates"))
    from digest import digest

    expected_candidates = dict(reversed(line.split()) for line in
                               (repo / "eval/candidates/DIGESTS").read_text().splitlines())
    expected_qrels = dict(reversed(line.split()) for line in
                          (repo / "eval/qrels.sha256").read_text().splitlines())
    old_suite = "full" if suite == "dl" else "beir_full"
    old_summary = json.loads((gaggle_comparison / old_suite / "summary.json").read_text())
    if old_summary["final_step"] != 1928 or old_summary["limit_queries"] != 0:
        raise RuntimeError("Gaggle scores are not from the full final checkpoint")
    receipt = {}
    for dataset in datasets(suite):
        path = candidates / f"retrieve_results_{dataset}_top100.jsonl"
        actual_digest, trec = digest(path)
        paper_match = actual_digest == expected_candidates[dataset]
        if dataset in DL and not paper_match:
            raise RuntimeError(f"{dataset} candidates differ from the paper digest")
        if dataset in BEIR and not same_first_stage_run(dataset, trec):
            raise RuntimeError(f"{dataset} candidate IDs differ from the paper's BM25 run")
        qrels_hash = sha256(qrels_dir / QRELS[dataset])
        if qrels_hash != expected_qrels[QRELS[dataset]]:
            raise RuntimeError(f"{dataset} qrels differ from the paper")
        receipt[dataset] = {"candidate_digest": actual_digest,
                            "paper_digest_match": paper_match,
                            "qrels_sha256": qrels_hash,
                            "queries": sum(1 for _ in path.open())}
        saved = old_summary["inputs"][dataset]
        if any(receipt[dataset][key] != saved[key]
               for key in ("candidate_digest", "qrels_sha256", "queries")):
            raise RuntimeError(f"{dataset} inputs differ from the saved Gaggle evaluation")
        if "paper_digest_match" in saved and receipt[dataset]["paper_digest_match"] != saved["paper_digest_match"]:
            raise RuntimeError(f"{dataset} paper digest status changed")
        if dataset not in old_summary["datasets"]:
            raise RuntimeError(f"Missing final Gaggle result for {dataset}")
    return receipt, old_summary


def link_qwen() -> None:
    model = repo / "reranker/models/qwen3-0.6b-base"
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.exists():
        model.symlink_to("/qwen/model", target_is_directory=True)


@app.function(cpu=2, memory=8192, timeout=1200,
              volumes={"/gaggle": gaggle, "/cache": cache})
def prepare() -> dict:
    receipts = {suite: verify_inputs(suite)[0] for suite in ("dl", "beir")}
    print("EVAL_INPUTS_OK", json.dumps(receipts), flush=True)
    return receipts


@app.function(gpu="H200", cpu=8, memory=32768, timeout=6 * 3600,
              volumes={"/gaggle": gaggle, "/cache": cache, "/qwen": qwen_inputs,
                       "/results": qwen_outputs}, single_use_containers=True)
def compare(suite: str = "dl") -> dict:
    import torch

    if json.loads((qwen_dir / "DONE").read_text())["step"] != 1928:
        raise RuntimeError("Qwen SFT has not finished its final step")
    state = torch.load(qwen_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state["step"] != 1928 or state["kind"] != "final":
        raise RuntimeError("Qwen checkpoint is not the final export")
    del state
    link_qwen()
    receipt, old_summary = verify_inputs(suite)
    old_suite = "full" if suite == "dl" else "beir_full"
    tag = "qwen3_06b_sft"
    target = comparison / suite
    runs = target / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EVAL_RUNS": str(candidates),
           "EVAL_QRELS_DIR": str(qrels_dir)}
    result = {"repo_commit": REPO_COMMIT, "qwen_final_step": 1928,
              "qwen_checkpoint_sha256": sha256(qwen_checkpoint),
              "gaggle_final_step": 1928,
              "gaggle_summary_sha256": sha256(gaggle_comparison / old_suite / "summary.json"),
              "suite": suite, "inputs": receipt, "datasets": {}}
    for dataset in datasets(suite):
        saved_result = target / f"{tag}_{dataset}.json"
        saved_run = runs / f"{tag}_{dataset}.trec"
        if saved_result.exists() and saved_run.exists():
            row = json.loads(saved_result.read_text())
            if row["model"] != str(qwen_checkpoint) or row["backend"] != "lce":
                raise RuntimeError(f"Stale Qwen score file for {dataset}")
        else:
            command = [sys.executable, "eval/backends/multi.py", "--dataset", dataset,
                       "--backend", "lce", "--model", str(qwen_checkpoint),
                       "--tag", tag, "--out", str(runs), "--batch-size", "32"]
            subprocess.run(command, cwd=repo, env=env, check=True)
            row = json.loads((repo / "eval/results" / f"{tag}_{dataset}.json").read_text())
            shutil.copy2(repo / "eval/results" / f"{tag}_{dataset}.json", saved_result)
            qwen_outputs.commit()
        gaggle_ndcg = old_summary["datasets"][dataset]["reproduced_ndcg10"]
        result["datasets"][dataset] = {
            "gaggle_ndcg10": gaggle_ndcg,
            "qwen_ndcg10": row["ndcg_cut_10"],
            "delta_qwen_minus_gaggle": round(row["ndcg_cut_10"] - gaggle_ndcg, 4),
            "qwen_seconds": row["seconds"],
        }
        qwen_outputs.commit()
    gaggle_mean = sum(r["gaggle_ndcg10"] for r in result["datasets"].values()) / len(result["datasets"])
    qwen_mean = sum(r["qwen_ndcg10"] for r in result["datasets"].values()) / len(result["datasets"])
    result["gaggle_mean"] = round(gaggle_mean, 4)
    result["qwen_mean"] = round(qwen_mean, 4)
    result["delta_qwen_minus_gaggle"] = round(qwen_mean - gaggle_mean, 4)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    qwen_outputs.commit()
    print("QWEN_GAGGLE_COMPARISON_COMPLETE", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "prepare", suite: str = "dl") -> None:
    if stage == "prepare":
        print(json.dumps(prepare.remote(), indent=2))
    elif stage == "compare":
        print(json.dumps(compare.remote(suite), indent=2))
    else:
        raise ValueError("stage must be prepare or compare")
