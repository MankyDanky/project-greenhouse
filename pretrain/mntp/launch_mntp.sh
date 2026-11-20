#!/bin/bash
# Bidirectional MNTP continuation of the causal d34 run (paper Sec. 4, "Bidirectional variant"). One 8-GPU node.
#
#   SRC_DIR=<dir with model_209158.pt + meta_209158.json>  NANOCHAT_BASE_DIR=<pretrain base dir: tokenizer + data>
#   OUT_DIR=<logs/metrics>  bash pretrain/mntp/launch_mntp.sh
#
# Starts from causal step 209,158 (245.3B tokens) with a fresh optimizer and runs to the one-pass horizon
# (actual step 229,067, 287.05B tokens): 19,909 updates of 2,097,152 tokens. The data stream continues from the
# checkpoint's per-rank cursors, i.e. exactly the tokens the causal run consumed after step 209,158.
# NANOCHAT_BASE_DIR is the same base directory the causal recipe (pretrain/README.md) writes; the run's checkpoints
# land under its base_checkpoints/d34bidir_mntp_from209158. Re-run the same command to resume after a time limit
# (--deadline-margin-s checkpoints and exits cleanly before SLURM_JOB_END_TIME, when set).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${SRC_DIR:?}" "${NANOCHAT_BASE_DIR:?}" "${OUT_DIR:?}"
mkdir -p "$OUT_DIR"
cd "$HERE/upstream_snapshot_bidir"
export PYTHONPATH="$HERE/upstream_snapshot_bidir:$HERE/trainer${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_BASE_DIR NANOCHAT_BIDIRECTIONAL=1 WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -m torch.distributed.run --standalone --nproc-per-node="${GPUS_PER_NODE:-8}" --max-restarts=0 \
    "$HERE/trainer/base_train.py" \
    --depth=34 --device-batch-size=8 --max-seq-len=2048 --fp8 --fp8-recipe=tensorwise \
    --num-iterations=321817 --warmdown-ratio=0.5528 --final-lr-frac=0.05 \
    --total-batch-size=2097152 --resume-from-step=209158 --model-tag=d34bidir_mntp_from209158 \
    --eval-every=250 --eval-tokens=41943040 --core-metric-every=-1 --sample-every=-1 --save-every=1200 \
    --deadline-margin-s=1800 --production-matrix-lr=0.01414213562373095 \
    --production-checkpoint-source-dir="$SRC_DIR" \
    --production-metrics-path="$OUT_DIR/metrics.jsonl" --production-keep-latest=3 \
    --bidir-origin-step=209158 --bidir-mask-prob=0.2 --bidir-rewarmup-steps=200 \
    2>&1 | tee -a "$OUT_DIR/train-$(date +%Y%m%d_%H%M%S).log"
