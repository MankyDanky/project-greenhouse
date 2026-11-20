#!/bin/bash
# Headline-table listwise rows: Gemma-4-26B-A4B and Qwen3.8-27B as zero-shot listwise rerankers with RankLLM.
# Run from the repo root.
#
#   RANK_LLM=<rank_llm checkout at 443adf3> bash eval/baselines/listwise/run_listwise.sh gemma4 [dataset ...]
#   RANK_LLM=<rank_llm checkout at 443adf3> bash eval/baselines/listwise/run_listwise.sh qwen38 [dataset ...]
#
# Protocol (paper Sec. 3 and Appendix A): BM25 top-100, sliding windows of 20 with stride 10, one pass, non-thinking
# generation, 16,384-token context. Each passage is first cut to 512 tokens of the model's own tokenizer, from the
# same normalised text the pointwise rerankers read (pretruncate_requests.py), and --max-passage-words 512 never
# cuts further. Qwen3.8 uses RankLLM's qwen_non_thinking_template.yaml as is; Gemma-4 uses
# gemma_non_thinking_template.yaml, the same template without the Qwen-only "/no_think" line, with thinking turned
# off through the chat template. Needs vLLM; NUM_GPUS (default 1) sets tensor parallelism.
set -euo pipefail
WHICH=${1:?usage: run_listwise.sh gemma4|qwen38 [dataset ...]}; shift
RANK_LLM=${RANK_LLM:?set RANK_LLM to a rank_llm checkout at 443adf3}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
case "$WHICH" in
  gemma4) MODEL=google/gemma-4-26B-A4B-it; TAG=gemma4-26b-a4b-listwise; TEMPLATE=(--template "$HERE/gemma_non_thinking_template.yaml") ;;
  qwen38) MODEL=Qwen/Qwen3.8-27B;          TAG=qwen3.8-27b-listwise;    TEMPLATE=() ;;
  *) echo "unknown model: $WHICH"; exit 1 ;;
esac
DS=("$@"); [ ${#DS[@]} -gt 0 ] || DS=(dl19 dl20 dl21 dl22 dl23 covid news robust04 nfc scifact scidocs fiqa)
SRC=${EVAL_RUNS:-$REPO/data/bm25_top100}; REQ=$REPO/data/bm25_${WHICH}_512tok
mkdir -p "$REQ" "$REPO/eval/runs/listwise/$TAG" "$REPO/eval/results"
python "$HERE/pretruncate_requests.py" --tokenizer "$MODEL" --src "$SRC" --dst "$REQ" --datasets "${DS[@]}" --max-tokens 512
export PYTHONPATH=$RANK_LLM/src${PYTHONPATH:+:$PYTHONPATH} VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
cd "$RANK_LLM"
python "$HERE/listwise_driver.py" --model "$MODEL" --tag "$TAG" --datasets "${DS[@]}" --num-gpus "${NUM_GPUS:-1}" \
  --eval-dir "$REPO/eval" --out-dir "$REPO/eval/runs/listwise/$TAG" --results-dir "$REPO/eval/results" \
  --requests-dir "$REQ" --context-size 16384 --max-passage-words 512 --window-size 20 --stride 10 --top-k 100 \
  "${TEMPLATE[@]}"
