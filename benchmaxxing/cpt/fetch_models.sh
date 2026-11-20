#!/bin/bash
# Download the starting checkpoints for continued pretraining into models/ and verify them (run from this folder).
#   bash fetch_models.sh                 # every released starting model (the 3B; the 0.6B once released)
#   bash fetch_models.sh 3b              # one of them: 3b | 0.6b
#   bash fetch_models.sh --tokenizer     # only the tokenizer (enough for data preparation and the smoke tests)
# Needs the `hf` CLI (pip install -U huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p models
get() {  # repo, step
  local repo=$1 step=$2 dir=models/${1#castorini/}
  hf download "$repo" "model_${step}.pt" "meta_${step}.json" tokenizer.json tokenizer_config.json \
    tokenizer/tokenizer.pkl tokenizer/token_bytes.pt SHA256SUMS --local-dir "$dir"
  (cd "$dir" && sed 's#  \./#  #' SHA256SUMS | sha256sum --quiet --ignore-missing -c -) && echo "$dir: checksums OK"
  # the reranker finds backbones in reranker/models/<name>: link this copy there unless one exists already
  local link=../../reranker/models/${1#castorini/}
  if [ ! -e "$link" ]; then mkdir -p ../../reranker/models && ln -s "../../benchmaxxing/cpt/$dir" "$link"; fi
}
# The 0.6B model is not released yet: when it is, set its repo and step here (and models.gaggle06b in configs/train.yaml)
REPO_06B=""
STEP_06B=""
get_06b() { [ -n "$REPO_06B" ] || { echo "the 0.6B Gaggle model is not released yet (REPO_06B is empty)"; return 1; }
            get "$REPO_06B" "$STEP_06B"; }
case "${1:-all}" in
  --tokenizer) hf download castorini/gaggle-nanochat-pretrained-climbmix-20260924 tokenizer.json tokenizer_config.json \
                 tokenizer/tokenizer.pkl tokenizer/token_bytes.pt --local-dir models/gaggle-nanochat-pretrained-climbmix-20260924 ;;
  3b)   get castorini/gaggle-nanochat-pretrained-climbmix-20260924 227158 ;;
  0.6b) get_06b ;;
  all)  get castorini/gaggle-nanochat-pretrained-climbmix-20260924 227158
        if [ -n "$REPO_06B" ]; then get_06b; fi ;;
  *) echo "usage: bash fetch_models.sh [3b|0.6b|all|--tokenizer]"; exit 1 ;;
esac
