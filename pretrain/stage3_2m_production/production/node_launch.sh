#!/bin/bash
set -euo pipefail
: "${SLURM_PROCID:?Run once per allocated node}"
if [[ -n "${D34_ENV_SETUP:-}" ]]; then source "$D34_ENV_SETUP"; fi
cd "$D34_EXP_DIR/upstream_snapshot"
export PYTHONPATH="$D34_EXP_DIR/upstream_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_BASE_DIR="$D34_BASE_DIR"
export HF_HOME="$D34_HF_HOME" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="$TMPDIR/d34prod-${SLURM_JOB_ID}-triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/d34prod-${SLURM_JOB_ID}-inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
"$D34_PYTHON" -m torch.distributed.run \
    --nnodes="$D34_NNODES" --nproc-per-node="$D34_GPUS_PER_NODE" \
    --node-rank="$SLURM_PROCID" --master-addr="$D34_MASTER_ADDR" \
    --master-port="$D34_MASTER_PORT" --max-restarts=0 --monitor-interval=5 \
    "$D34_PRODUCTION_CODE/base_train.py" \
    --depth=34 --device-batch-size=8 --max-seq-len=2048 --fp8 --fp8-recipe=tensorwise \
    --num-iterations=321817 --warmdown-ratio=0.5528 --final-lr-frac=0.05 \
    --total-batch-size="$D34_WINNER_BATCH" --resume-from-step="$D34_PROD_SOURCE_STEP" \
    --model-tag="$D34_PROD_TAG" --eval-every=250 --eval-tokens=41943040 \
    --core-metric-every="${D34_PROD_CORE_EVERY:--1}" --sample-every=-1 --save-every=1200 \
    --deadline-margin-s="${D34_DEADLINE_MARGIN_S:-1800}" \
    --production-matrix-lr="$D34_WINNER_LR" \
    --production-release="$D34_PROD_RELEASE" \
    --production-winner-checkpoint-dir="$D34_WINNER_CHECKPOINT_DIR" \
    --production-winner-completion="$D34_WINNER_COMPLETION" \
    --production-winner-completion-sha256="$D34_WINNER_COMPLETION_SHA256" \
    --production-checkpoint-source-dir="$D34_PROD_SOURCE_DIR" \
    --production-metrics-path="$D34_PROD_DIR/metrics.jsonl" --production-keep-latest=3 \
    2>&1 | tee "$D34_PROD_DIR/train-${SLURM_JOB_ID}-node-${SLURM_PROCID}.log"
