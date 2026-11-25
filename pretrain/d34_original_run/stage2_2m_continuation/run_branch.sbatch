#!/bin/bash
#SBATCH --job-name=d34-2m-from-136316
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
# Stage 2: resume the stage-1 step-136,316 checkpoint at batch 2,097,152 and actual Muon LR 0.01414213562373095 for exactly
# 1,024 updates (2,147,483,648 tokens), the configuration the released model used. Writes the step-137,340 checkpoint and
# its completion record (COMPLETE.json, with all eight ranks' loader cursors), which stage 3 verifies by SHA256.
# submit: bash submit.sh gpu stage2_2m_continuation/run_branch.sbatch
# The run cannot resume: if it dies, move $D34_ROOT/stage2/<tag> and the output checkpoint dir aside and resubmit.

set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
EXP=$D34_EXP_DIR
TAG=$D34_STAGE2_TAG; BATCH=$D34_WINNER_BATCH; LR=$D34_WINNER_LR
BRANCH=$D34_ROOT/stage2/$TAG
mkdir -p "$BRANCH"
exec > >(tee -a "$BRANCH/runner.log") 2>&1
echo "BRANCH_START $(date -Is) job=$SLURM_JOB_ID node=$(hostname) tag=$TAG batch=$BATCH matrix_lr=$LR"
trap 'rc=$?; echo "BRANCH_EXIT $(date -Is) rc=$rc"; echo "$rc" > "$BRANCH/exit_code"' EXIT
# Read-only source: the stage-1 hand-off checkpoint, hard-linked (no bytes copied) so stage-1 retention can never touch it.
# Stage 3 hashes the same ten files as its static assets.
SRC_CK=$D34_BASE_DIR/base_checkpoints/$D34_RAMP_TAG
if [ ! -e "$D34_CHECKPOINT_DIR/meta_136316.json" ]; then
  mkdir -p "$D34_CHECKPOINT_DIR"
  for f in model_136316.pt optim_136316_rank{0..7}.pt meta_136316.json; do  # meta last: it marks the copy complete
    ln -f "$SRC_CK/$f" "$D34_CHECKPOINT_DIR/$f" 2>/dev/null || cp "$SRC_CK/$f" "$D34_CHECKPOINT_DIR/$f"
  done
fi
if [ -n "${D34_ENV_SETUP:-}" ]; then source "$D34_ENV_SETUP"; fi
source "$D34_VENV/bin/activate"
cd "$EXP/upstream_snapshot"
export PYTHONPATH="$EXP/upstream_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_BASE_DIR="$D34_BASE_DIR"
export HF_HOME="$D34_HF_HOME" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="$TMPDIR/d34abl-triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/d34abl-inductor"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
test "$(nvidia-smi -L | wc -l)" -eq 8
if nvidia-smi -L | grep -qi MIG; then echo 'FATAL: MIG is not supported'; exit 3; fi
python -c 'import torch; assert torch.cuda.device_count() == 8; assert all(any(m in torch.cuda.get_device_name(i) for m in ("H100", "H200")) for i in range(8)); print("RUNTIME", torch.__version__, torch.version.cuda)'
sha256sum "$EXP/source/"*.py
torchrun --standalone --nproc_per_node=8 "$EXP/source/base_train.py" \
  --depth=34 --device-batch-size=8 --max-seq-len=2048 --fp8 \
  --num-iterations=321817 --warmdown-ratio=0.5528 --final-lr-frac=0.05 \
  --total-batch-size="$BATCH" --resume-from-step=136316 --model-tag="$TAG" \
  --eval-tokens=41943040 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --deadline-margin-s=600 \
  --ablation-matrix-lr="$LR" --ablation-token-budget=2147483648 \
  --ablation-checkpoint-source-dir="$D34_CHECKPOINT_DIR" \
  --ablation-metrics-path="$BRANCH/metrics.jsonl" \
  2>&1 | tee "$BRANCH/train.log"
test -f "$BRANCH/COMPLETE.json"
