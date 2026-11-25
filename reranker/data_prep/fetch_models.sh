#!/bin/bash
# Download backbones into reranker/models/ at the revisions the paper's runs used, and verify them.
# Run from reranker/. Needs the `hf` CLI (pip install -U huggingface_hub).
#
#   bash data_prep/fetch_models.sh gaggle                 # the causal pre-trained model: Sec. 5, most of Sec. 6
#   bash data_prep/fetch_models.sh gaggle gaggle-bidir    # + the MNTP model: Sec. 6.3 MNTP rows
#   bash data_prep/fetch_models.sh gemma4 minicpm5        # Sec. 6.1 rows trained by the multi engine
#   bash data_prep/fetch_models.sh nanochat-d34           # Sec. 6.1 nanochat-d34 row (needs a conversion, see below)
#   bash data_prep/fetch_models.sh gaggle-0.9b gaggle-0.6b   # the 0.9B and 0.6B Gaggle backbones (configs/experiments/)
#
# The native Gemma-4-E2B and Qwen3.5-4B-Base rows (configs/backbones/*.json) load straight from the Hub at the
# revision pinned in their config, so they need nothing here.
set -euo pipefail
mkdir -p models
G=gaggle-nanochat-pretrained-climbmix-20260924
B=gaggle-nanochat-pretrained-climbmix-bidirectional-20260929
G09=gaggle-nanochat-0.9b-pretrained-climbmix-20261105
G06=gaggle-nanochat-0.6b-pretrained-climbmix-20261121

verify() {  # check every file listed in the repo's SHA256SUMS that we downloaded
  (cd "models/$1" && [ -f SHA256SUMS ] && sha256sum --quiet --ignore-missing -c SHA256SUMS && echo "  $1: checksums OK") \
    || { echo "  $1: CHECKSUM MISMATCH"; exit 1; }
}

for m in "${@:-gaggle}"; do
  case "$m" in
    gaggle)
      # model_227158.pt is the released step (283B tokens). The HF tokenizer files were added to the repo after
      # the runs; they are a lossless conversion of tokenizer/tokenizer.pkl.
      hf download castorini/$G model_227158.pt meta_227158.json tokenizer.json tokenizer_config.json \
        tokenizer/tokenizer.pkl tokenizer/token_bytes.pt SHA256SUMS --local-dir models/$G
      verify $G ;;
    gaggle-bidir)
      [ -f models/$G/tokenizer.json ] || { echo "fetch 'gaggle' first: the bidirectional model reuses its tokenizer"; exit 1; }
      hf download castorini/$B model_229067.pt meta_229067.json tokenizer/tokenizer.pkl tokenizer/token_bytes.pt \
        SHA256SUMS --local-dir models/$B
      cp models/$G/tokenizer.json models/$G/tokenizer_config.json models/$B/
      verify $B ;;
    gaggle-0.9b|gaggle-0.6b)
      # The smaller Gaggle backbones: the final step (229,067) of each 287B-token ClimbMix pass, with the same tokenizer.
      R=$G09; [ "$m" = gaggle-0.6b ] && R=$G06
      hf download castorini/$R model_229067.pt meta_229067.json tokenizer.json tokenizer_config.json \
        tokenizer/tokenizer.pkl tokenizer/token_bytes.pt SHA256SUMS --local-dir models/$R
      verify $R ;;
    gemma4)   hf download google/gemma-4-E2B        --revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f --local-dir models/gemma-4-e2b ;;
    qwen35)   hf download Qwen/Qwen3.5-4B-Base      --revision 1001bb4d826a52d1f399e183466143f4da7b741b --local-dir models/qwen3.5-4b-base ;;
    minicpm5) hf download openbmb/MiniCPM5-2B-Base  --revision 96a57cd572a02506b4500f54427dca24970c1bac --local-dir models/minicpm5-2b-base ;;
    nanochat-d34)
      # Karpathy's base checkpoint (step 169,150), converted to an fp32 HF model; ~40 GB of RAM. The conversion's
      # reference check needs nanochat at f37d45c, the commit that checkpoint was trained with.
      hf download karpathy/nanochat-d34 --local-dir models/karpathy-nanochat-d34
      [ -d third_party/nanochat-f37d45c ] || git clone -q https://github.com/karpathy/nanochat third_party/nanochat-f37d45c
      git -C third_party/nanochat-f37d45c checkout -q f37d45c
      python src/convert_karpathy_base.py --src models/karpathy-nanochat-d34 --dst models/nanochat-d34-base-hf \
        --nanochat-src third_party/nanochat-f37d45c ;;
    *) echo "unknown model: $m"; exit 1 ;;
  esac
done
