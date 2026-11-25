#!/bin/bash
set -euo pipefail
: "${SLURM_PROCID:?Expected one Slurm process per node}"
if [[ -n "${D34_ENV_SETUP:-}" ]]; then
    # Explicitly configured, trusted module/LD_LIBRARY_PATH setup; no package installs.
    source "$D34_ENV_SETUP"
fi
[[ -x "$D34_PYTHON" ]] || { echo "Missing Python executable: $D34_PYTHON" >&2; exit 2; }
cd "$D34_EXP_DIR/upstream_snapshot"
export PYTHONPATH="$D34_EXP_DIR/upstream_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_BASE_DIR="$D34_BASE_DIR"
export HF_HOME="$D34_HF_HOME" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="$TMPDIR/d34abl-${SLURM_JOB_ID}-triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/d34abl-${SLURM_JOB_ID}-inductor"
export D34_SHARD_MANIFEST="${D34_SHARD_MANIFEST:?Indexed corpus manifest is required}"
export D34_DATA_ROOT="${D34_DATA_ROOT:?Local parquet corpus directory is required}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$D34_BRANCH_DIR/provenance"

# Hash checks run inside the allocation. Node zero validates once against a
# cluster-local receipt; every node's rank zero validates the receipt before training.
if [[ "${1:-}" == --verify-only ]]; then
    "$D34_PYTHON" "$D34_FLEET_DIR/verify_assets.py" \
        --manifest "$D34_ASSET_MANIFEST" --receipt "$D34_ASSET_RECEIPT" \
        --provenance "$D34_BRANCH_DIR/provenance/assets.json"
    exit 0
fi

distributed=("$D34_PYTHON" -m torch.distributed.run
    --nnodes="$D34_NNODES" --nproc-per-node="$D34_GPUS_PER_NODE"
    --node-rank="$SLURM_PROCID" --master-addr="$D34_MASTER_ADDR"
    --max-restarts=0 --monitor-interval=5)

# The heavy asset check completed in an earlier Slurm step, before rendezvous.
"${distributed[@]}" --master-port="$D34_MASTER_PORT" "$D34_FLEET_DIR/collective_probe.py"
[[ "$D34_RUN_MODE" == train ]] || exit 0

"${distributed[@]}" --master-port="$D34_TRAIN_PORT" "$D34_EXP_DIR/source/base_train.py" \
    --depth=34 --device-batch-size=8 --max-seq-len=2048 --fp8 --fp8-recipe=tensorwise \
    --num-iterations=321817 --warmdown-ratio=0.5528 --final-lr-frac=0.05 \
    --total-batch-size="$D34_BATCH" --resume-from-step=136316 --model-tag="$D34_TAG" \
    --eval-tokens=41943040 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
    --deadline-margin-s="$D34_DEADLINE_MARGIN_S" \
    --ablation-matrix-lr="$D34_LR" --ablation-token-budget="$D34_TOKEN_BUDGET" \
    --ablation-checkpoint-source-dir="$D34_CHECKPOINT_DIR" \
    --ablation-metrics-path="$D34_BRANCH_DIR/metrics.jsonl" \
    2>&1 | tee "$D34_BRANCH_DIR/train-node-${SLURM_PROCID}.log"
