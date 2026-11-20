#!/usr/bin/env bash
# CPU smoke test of the training side (a few minutes, no GPU): a 2-layer model is pre-trained for 3 steps on the
# smoke mix, then continued with train.py, scored with evaluate.py and exported with export_for_reranker.py.
# Needs: bash setup.sh; bash fetch_models.sh --tokenizer; nanochat's environment (PYTHON=... to choose one).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
PY=${PYTHON:-python}
[ -f data/smoke/manifest.json ] || { echo "build the smoke mix first, in the data environment:" \
  "python prepare_data.py tests/smoke_data.yaml --limit 3000"; exit 1; }
mkdir -p work/smoke_base
[ -e work/smoke_base/tokenizer ] || ln -s ../../models/gaggle-nanochat-pretrained-climbmix-20260924/tokenizer work/smoke_base/tokenizer
rm -rf work/smoke_base/base_checkpoints work/smoke_export results/smoke_cpt
(cd third_party/nanochat && NANOCHAT_BASE_DIR="$PWD/../../work/smoke_base" NANOCHAT_DATA_DIR="$PWD/../../data/smoke/train" \
  $PY -m scripts.base_train --depth 2 --aspect-ratio 64 --head-dim 64 --ve-every 4 --window-pattern SL \
    --max-seq-len 256 --device-batch-size 2 --total-batch-size 1024 --num-iterations 3 --eval-every -1 \
    --eval-tokens 1024 --core-metric-every -1 --sample-every -1 --model-tag smoke_init)
cp models/gaggle-nanochat-pretrained-climbmix-20260924/tokenizer{.json,_config.json} work/smoke_base/base_checkpoints/smoke_init/
$PY train.py tests/smoke_train.yaml --check
$PY train.py tests/smoke_train.yaml
$PY evaluate.py tests/smoke_train.yaml --init
$PY evaluate.py tests/smoke_train.yaml
$PY export_for_reranker.py tests/smoke_train.yaml --dest work/smoke_export
echo "smoke_train: OK"
