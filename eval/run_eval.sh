#!/bin/bash
# Rerank BM25 top-100 on the paper's 12 test sets with one model, and score nDCG@10. Run from the repo root.
#
#   bash eval/run_eval.sh <scorer> <model> <tag> [dataset ...]
#
# scorer     model                                             used for
#   ours       reranker/checkpoints/<run>/ckpt_final.pt          rows trained by the nanochat engine (and run 4)
#   multi      reranker/checkpoints/<run>/milestones/<step>.pt   rows trained by the multi engine (MiniCPM, Gemma at 1e-5)
#   gemma4     reranker/checkpoints/<run>/...                    the native Gemma-4-E2B row
#   qwen35     reranker/checkpoints/<run>/...                    the native Qwen3.5-4B-Base row
#   ce         reranker/checkpoints/<run>/final                  the pointwise-CE row
#   gaggle_hf  castorini/gaggle-reranker-20261005 (or a run)     the released transformers models
#   monot5 | qwen3 | jinav3                                      baselines: see eval/baselines/run_baselines.sh
#
# Datasets default to all 12: dl19 dl20 dl21 dl22 dl23 covid news robust04 nfc scifact scidocs fiqa.
# Needs eval/qrels (bash eval/fetch_qrels.sh) and data/bm25_top100 (bash eval/candidates/build_candidates.sh).
# Results: eval/results/<tag>_<dataset>.json; run files: eval/runs/. Summarise with python eval/make_report.py.
# DTYPE=fp32 scores in strict fp32 (the paper used the default: bf16 backbone, fp32 scoring rows).
set -euo pipefail
SCORER=${1:?usage: bash eval/run_eval.sh <scorer> <model> <tag> [dataset ...]}; MODEL=${2:?model}; TAG=${3:?tag}; shift 3
DS=("$@"); [ ${#DS[@]} -gt 0 ] || DS=(dl19 dl20 dl21 dl22 dl23 covid news robust04 nfc scifact scidocs fiqa)
export EVAL_RUNS=${EVAL_RUNS:-$PWD/data/bm25_top100}
export EVAL_QRELS_DIR=${EVAL_QRELS_DIR:-$PWD/eval/qrels}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d "$EVAL_RUNS" ] || { echo "no candidates in $EVAL_RUNS: run bash eval/candidates/build_candidates.sh"; exit 1; }
mkdir -p eval/runs eval/results
for d in "${DS[@]}"; do
  echo "=== $SCORER $TAG $d"
  case "$SCORER" in
    ours|gaggle_hf|monot5|qwen3|jinav3)
      python eval/rerank_eval.py --dataset "$d" --backend "$SCORER" --model "$MODEL" --tag "$TAG" --out eval/runs \
        ${DTYPE:+--dtype "$DTYPE"} ${BATCH:+--batch-size "$BATCH"} ;;
    gemma4|qwen35)
      python "eval/backends/$SCORER.py" --dataset "$d" --backend ours --model "$MODEL" --tag "$TAG" --out eval/runs \
        ${DTYPE:+--dtype "$DTYPE"} ;;
    multi)
      python eval/backends/multi.py --dataset "$d" --backend lce --model "$MODEL" --tag "$TAG" --out eval/runs \
        ${DTYPE:+--dtype "$DTYPE"} ;;
    ce)
      python eval/backends/ce.py --dataset "$d" --model "$MODEL" --tag "$TAG" --out eval/runs ${DTYPE:+--dtype "$DTYPE"} ;;
    *) echo "unknown scorer: $SCORER"; exit 1 ;;
  esac
done
