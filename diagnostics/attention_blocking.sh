#!/bin/bash
# Sec. 6.3: do the MNTP-adapted rerankers depend on attention to later tokens? Run from the repo root.
#
#   bash diagnostics/attention_blocking.sh reranker/checkpoints/attention_mntp_standard/ckpt_final.pt mntp_standard
#   bash diagnostics/attention_blocking.sh reranker/checkpoints/attention_mntp_mask/ckpt_final.pt mntp_mask
#
#   full    the mask it was trained with: must reproduce the row's own scores (validates the mask patch)
#   causal  same weights, attention to later tokens removed: the intervention
#   random  same weights, an equal number of in-window pairs removed at random: the control
# Paper: over DL19, DL20 and SciFact, blocking later tokens drops mean nDCG@10 from 0.753 to 0.700 (standard
# prompt) and from 0.750 to 0.688 (appended mask). One process per variant: BIDIR_MASK_VARIANT is read when
# reranker/src/bidirectional.py is imported.
set -euo pipefail
CKPT=${1:?usage: bash diagnostics/attention_blocking.sh <ckpt> <name>}; NAME=${2:?name}
for variant in full causal random; do
  BIDIR_MASK_VARIANT=$variant bash eval/run_eval.sh ours "$CKPT" "attnblock-$NAME-$variant" dl19 dl20 scifact
done
python eval/report.py "attnblock-$NAME-full" "attnblock-$NAME-causal" "attnblock-$NAME-random"
