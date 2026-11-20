#!/bin/bash
# Score one base model as the paper's pre-training table does. Run from the repo root.
#
#   bash pretrain_eval/run_eval.sh nanochat <checkpoint dir> <step> <tag>          # our causal model
#   bash pretrain_eval/run_eval.sh mntp     <checkpoint dir> <step> <tag>          # the bidirectional variant
#   bash pretrain_eval/run_eval.sh hf       <hub id>         -      <tag>          # a transformers base model
#
#   CORE      nanochat's 22-task suite, 500 examples per task, the model's own prefix token, left truncation at 2,048
#   MMLU      lm-eval 0.4.9 `mmlu`, 5-shot, answer-letter log-likelihood
#   MMLU-Pro  lm-eval 0.4.9 `leaderboard_mmlu_pro`, the same scoring over 10 options
# A nanochat checkpoint dir holds model_<step>.pt, meta_<step>.json and tokenizer/ (tokenizer.pkl). For `mntp` the
# answer tokens are replaced by the MNTP mask (id 32767) and predicted from the position before, under bidirectional
# attention; CORE is not scored for it (that needs the masked CORE path, not included).
# Environment: EVAL_BUNDLE (nanochat's eval_bundle, which pretrain/setup.sh downloads), PY (python with lm-eval 0.4.9,
# rustbpe and tiktoken; models newer than transformers 4.x need a transformers >= 5 env for CORE).
# Results: paper/results/pretraining/<benchmark>_<tag>.json
set -euo pipefail
KIND=${1:?usage: run_eval.sh nanochat|mntp|hf <model> <step> <tag>}; MODEL=${2:?}; STEP=${3:?}; TAG=${4:?}
PY=${PY:-python}; BUNDLE=${EVAL_BUNDLE:?set EVAL_BUNDLE to nanochat\'s eval_bundle dir}
R=$(cd "$(dirname "$0")/.." && pwd); OUT=$R/paper/results/pretraining; mkdir -p "$OUT"
export NANOCHAT_ATTN=sdpa TOKENIZERS_PARALLELISM=false
case $KIND in
  nanochat) M=(--kind nanochat --src "$R/pretrain/nanochat" --ckpt "$MODEL" --step "$STEP"); MASK=() ;;
  mntp)     M=(--kind nanochat --src "$R/pretrain/mntp/upstream_snapshot_bidir" --ckpt "$MODEL" --step "$STEP")
            MASK=(--mntp-mask-id 32767); export NANOCHAT_BIDIRECTIONAL=1 ;;
  hf)       M=(--kind hf --hf "$MODEL"); MASK=() ;;
  *) echo "unknown kind $KIND"; exit 1 ;;
esac
cd "$R/pretrain_eval"
[ "$KIND" = mntp ] || $PY core_batched.py "${M[@]}" --bundle "$BUNDLE" --core-eval core_eval_head.py \
    --max-per-task 500 --max-seq-len 2048 --out "$OUT/core_$TAG.json"
for T in mmlu leaderboard_mmlu_pro; do
  $PY mmlu_any.py "${M[@]}" --task $T --max-len -1 "${MASK[@]}" --out "$OUT/${T}_$TAG.json"
done
