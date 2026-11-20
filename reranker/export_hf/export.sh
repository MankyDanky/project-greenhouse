#!/bin/bash
# Convert a reranker checkpoint into a transformers repo (trust_remote_code), check that it scores like the
# checkpoint, and evaluate it end to end. Run from the repo root.
#
#   bash reranker/export_hf/export.sh reranker/checkpoints/sft_soup/ckpt_final.pt exports/gaggle-reranker
#
# Steps: convert_to_hf.py (sharded safetensors + config + tokenizer + modeling code + SHA256SUMS);
# parity_check.py (the HF model against the checkpoint on 12 SciFact queries); then DL19, NFCorpus and SciFact
# through eval/run_eval.sh with the gaggle_hf scorer.
set -euo pipefail
CKPT=${1:?usage: bash reranker/export_hf/export.sh <ckpt_final.pt> <out dir> [hub name]}; OUT=${2:?out dir}
NAME=${3:-castorini/gaggle-reranker-20261005}
python reranker/export_hf/convert_to_hf.py --ckpt "$CKPT" --out "$OUT" --name "$NAME"
python reranker/export_hf/parity_check.py --ckpt "$CKPT" --repo "$OUT" --dataset scifact --queries 12
bash eval/run_eval.sh gaggle_hf "$OUT" "$(basename "$OUT")" dl19 nfc scifact
