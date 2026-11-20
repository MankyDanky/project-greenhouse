#!/bin/bash
# Build the training parquets every paper row reads, and check them against the SHA256 of the files the runs
# used (data_prep/SHA256SUMS). CPU only; run from reranker/ after `bash data_prep/fetch_models.sh gaggle`.
#
#   bash data_prep/prepare_data.sh rlhn250k        # Sec. 5 and most of Sec. 6     (~1 h with 16 processes)
#   bash data_prep/prepare_data.sh rlhn680k        # Sec. 6.2 RLHN-680K
#   bash data_prep/prepare_data.sh tevatron        # Sec. 6.2 Tevatron MS MARCO
#   bash data_prep/prepare_data.sh rlhn250k-qwen   # Sec. 6.1 nanochat-d34 row (lengths from the Qwen3-0.6B tokenizer)
#
# Stored passage lengths only decide how query groups are packed into optimizer steps; the trainers re-tokenize
# and truncate with each backbone's own tokenizer. They still have to match exactly, because the packing plan
# (and so which queries share a step) is part of what each run reproduces.
#
# The self-filtered sets of Sec. 6.4 are built from run 1's checkpoint: see data_prep/self_score.sh.
set -euo pipefail
PROCS=${PROCS:-16}
TOK=models/gaggle-nanochat-pretrained-climbmix-20260924
export TOKENIZERS_PARALLELISM=false
mkdir -p data
need_tok() { [ -f $TOK/tokenizer.json ] || { echo "run: bash data_prep/fetch_models.sh gaggle"; exit 1; }; }
check() {  # $1 = data/<name>
  local want; want=$(awk -v n="$(basename "$1")" '$2 == n {print $1}' data_prep/SHA256SUMS)
  local got; got=$(sha256sum "$1/train.parquet" | cut -d' ' -f1)
  if [ "$want" = "$got" ]; then echo "  $1: matches the paper's file"; else
    echo "  $1: sha256 $got differs from the paper's $want."
    echo "  The upstream dataset may have changed since 2026-09; see ONBOARDING.md (\"Training data\")."; fi
}
# RLHN-250K with Qwen3-0.6B lengths: the base file. Every other RLHN-250K variant is derived from it.
base250k() {
  [ -f data/rlhn250k_prepared/train.parquet ] || \
    python data_prep/prepare_rlhn.py --dataset rlhn/rlhn-250K --out data/rlhn250k_prepared --num-proc "$PROCS"
  check data/rlhn250k_prepared
}
for what in "${@:-rlhn250k}"; do
  case "$what" in
    rlhn250k-qwen) base250k ;;
    rlhn250k)
      need_tok; base250k
      [ -f data/rlhn250k_prepared_climbmix_lens/train.parquet ] || \
        python data_prep/relength_nanochat.py --src data/rlhn250k_prepared/train.parquet \
          --out data/rlhn250k_prepared_climbmix_lens --tokenizer $TOK --procs "$PROCS"
      check data/rlhn250k_prepared_climbmix_lens ;;
    rlhn680k)
      need_tok
      # --min-negatives 8: the paper's 680K file was built when that was the script's default. It drops three FiQA
      # queries with exactly 7 negatives (648,762 rows, as in the paper); the current default of 7 keeps them.
      [ -f data/rlhn680k_prepared/train.parquet ] || \
        python data_prep/prepare_rlhn.py --dataset rlhn/rlhn-680K --out data/rlhn680k_prepared --min-negatives 8 \
          --num-proc "$PROCS"
      # The intermediate itself is not hash-checked: a rebuild is identical in every row, column and metadata field,
      # but the parquet writer's byte layout differs from the 2026-09 file. Configs read only the file below.
      [ -f data/rlhn680k_prepared_climbmix_lens/train.parquet ] || \
        python data_prep/relength_nanochat.py --src data/rlhn680k_prepared/train.parquet \
          --out data/rlhn680k_prepared_climbmix_lens --tokenizer $TOK --procs "$PROCS"
      check data/rlhn680k_prepared_climbmix_lens ;;
    tevatron)
      # 1 positive and 30 negatives per query; lengths measured directly with the ClimbMix tokenizer.
      need_tok
      SRC=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('Tevatron/msmarco-passage', \
              'train.jsonl.gz', repo_type='dataset', revision='330a160b40a739d4c07d52c7ce562e2c549214c0'))")
      [ -f data/msmarco_tevatron_prepared_climbmix_lens/train.parquet ] || \
        python data_prep/prepare_rlhn.py --json-file "$SRC" --subset-name msmarco_passage --tokenizer $TOK \
          --out data/msmarco_tevatron_prepared_climbmix_lens --min-negatives 7 --total-budget 0 --seed 17 --num-proc 8
      check data/msmarco_tevatron_prepared_climbmix_lens ;;
    *) echo "unknown dataset: $what"; exit 1 ;;
  esac
done
