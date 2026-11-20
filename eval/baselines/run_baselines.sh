#!/bin/bash
# The fine-tuned baselines of Tables 1-2 that we evaluated ourselves. Run from the repo root.
#
#   bash eval/baselines/run_baselines.sh [monot5] [qwen3-4b] [qwen3-8b] [jina-v3]
#
# All rerank the same BM25 top-100 candidates as Gaggle. Settings follow the paper:
#   * MonoT5-3B (castorini/monot5-3b-msmarco-10k): 512-token input limit, pointwise.
#   * Qwen3-Reranker-4B/8B: query and document capped separately at 128 / 512 tokens, the same caps as Gaggle
#     (EVAL_MAX_QUERY_LEN / EVAL_MAX_DOC_LEN; without them the assembled prompt is head-truncated instead).
#     The paper's two rows came from a separate matched-text implementation; this one reproduces them within
#     ~0.004 on the benchmark means (DL23 moves most, up to 0.012), see paper/results/public_rerun.json.
#   * jina-reranker-v3: all 100 candidates in one rerank() call (the model groups up to 64 per window),
#     documents truncated at 512 tokens, queries at the API default of 256.
# The listwise LLM rows (Gemma-4-26B-A4B, Qwen3.8-27B) are in eval/baselines/listwise/. BM25 and the oracle are
# eval/candidates/score_first_stage.sh and eval/oracle.py. GPT-6.1 Sol, Jev and RLHN Qwen2.5-3B are not included.
set -euo pipefail
[ $# -gt 0 ] || set -- monot5 qwen3-4b qwen3-8b jina-v3
for b in "$@"; do
  case "$b" in
    monot5)   BATCH=32 bash eval/run_eval.sh monot5 castorini/monot5-3b-msmarco-10k monot5-3b-10k ;;
    qwen3-4b) EVAL_MAX_QUERY_LEN=128 EVAL_MAX_DOC_LEN=512 BATCH=4 bash eval/run_eval.sh qwen3 Qwen/Qwen3-Reranker-4B qwen3-rr-4b ;;
    qwen3-8b) EVAL_MAX_QUERY_LEN=128 EVAL_MAX_DOC_LEN=512 BATCH=2 bash eval/run_eval.sh qwen3 Qwen/Qwen3-Reranker-8B qwen3-rr-8b ;;
    jina-v3)  BATCH=1 bash eval/run_eval.sh jinav3 jinaai/jina-reranker-v3 jina-v3 ;;
    *) echo "unknown baseline: $b"; exit 1 ;;
  esac
done
