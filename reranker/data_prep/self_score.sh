#!/bin/bash
# Sec. 6.4: build the self-filtered training set from run 1's checkpoint. Run from reranker/.
#
#   bash data_prep/self_score.sh checkpoints/sft_run1_seed17/ckpt_final.pt [n_gpus]
#
# 1. Scores every positive and negative of RLHN-250K with the checkpoint (12 shards spread over the GPUs;
#    about 2 GPU-hours in total).
# 2. Drops a positive when more than max(9, 0.375 * n_negatives) negatives outrank it, drops a query with no
#    positive left, and leaves ArguAna untouched. Expected: 14,887 positives and 1,484 queries removed,
#    246,050 queries kept.
# The output carries the checkpoint's raw scores (pos_teacher / neg_teacher), which the self-distillation row
# reads; the self-filtering row ignores those two columns.
set -euo pipefail
CKPT=${1:?usage: bash data_prep/self_score.sh <run 1 ckpt_final.pt> [n_gpus]}
NGPU=${2:-$(nvidia-smi -L | wc -l)}
SHARDS=${SHARDS:-12}
WORK=data/_self_scores OUT=data/rlhn250k_self_filtered_scores
mkdir -p "$WORK"
for ((s = 0; s < SHARDS; s++)); do
  [ -f "$WORK/scores_$s.npz" ] && continue
  CUDA_VISIBLE_DEVICES=$(( s % NGPU )) python data_prep/self_filter.py score --ckpt "$CKPT" \
    --shard $s --shards "$SHARDS" --src data/rlhn250k_prepared_climbmix_lens/train.parquet --work "$WORK" &
  if (( (s + 1) % NGPU == 0 )); then wait; fi
done
wait
python data_prep/self_filter.py merge --work "$WORK" --out "$OUT"
ln -sfn rlhn250k_self_filtered_scores data/rlhn250k_self_filtered
want=$(awk '$2 == "rlhn250k_self_filtered_scores" {print $1}' data_prep/SHA256SUMS)
got=$(sha256sum "$OUT/train.parquet" | cut -d' ' -f1)
if [ "$want" = "$got" ]; then echo "matches the paper's file"
else echo "sha256 $got differs from the paper's $want (expected unless your run 1 is bit-identical)"; fi
