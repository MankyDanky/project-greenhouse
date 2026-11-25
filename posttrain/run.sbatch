#!/bin/bash
#SBATCH --job-name=gaggle-posttrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
# Midtraining -> SFT -> chat_eval of a Gaggle base checkpoint (POST_MODEL in config.sh): the 3B on one 80 GB GPU,
# the 0.9B on two.
#   bash submit.sh gpu run.sbatch                       full run (~7 h on an H100; midtraining alone 3 h 10 min)
#   bash submit.sh gpu run.sbatch --export=ALL,SMOKE=1  4 midtraining + 4 SFT steps + chat_eval on 8 problems, in a
#                                                       throwaway base dir (~15 min); nothing is kept
# The recipe's batches are kept by gradient accumulation (midtraining 524,288 tokens/step, SFT 32 conversations/step);
# POST_MID_DBS / POST_SFT_DBS are only the per-GPU micro-batches. A stage whose checkpoint already exists is skipped, so resubmitting
# after a failure continues from the last finished stage.
set -uo pipefail
source "${POST_CONFIG:?export POST_CONFIG=/absolute/path/to/config.sh}"
[ -z "$POST_ENV_SETUP" ] || source "$POST_ENV_SETUP"
SMOKE=${SMOKE:-0}; TAG=$POST_TAG; STEP=$POST_BASE_STEP; NPROC=$POST_NPROC; MID_DBS=$POST_MID_DBS; SFT_DBS=$POST_SFT_DBS
export NANOCHAT_BASE_DIR=$POST_BASE_DIR HF_HOME=$POST_HF_HOME HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=disabled OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${SLURM_TMPDIR:-/tmp} TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/inductor
LOG=$POST_LOGS/run-${SLURM_JOB_ID:-local}; mkdir -p "$LOG"
log(){ echo "[$(date -Is)] $*"; }
run(){ local name=$1; shift; log "$name: $*"
       "$POST_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=$NPROC -m "$@" > "$LOG/$name.log" 2>&1
       local rc=$?; grep -E "accuracy:|ChatCORE|Validation|Peak memory|Total training time" "$LOG/$name.log" | tail -8
       log "$name exit $rc (full log: $LOG/$name.log)"; return $rc; }
cd "$POST_NANOCHAT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# a GPU stuck in a bad state can be counted by NVML while CUDA init fails, and torch then silently runs on CPU
"$POST_PYTHON" -c "import torch; torch.cuda.init(); print('CUDA OK', torch.cuda.get_device_name(0))" \
  || { log "FATAL: CUDA unusable on $(hostname)"; exit 5; }

if [ "$SMOKE" = 1 ]; then
  SB=$POST_ROOT/smoke_base; rm -rf "$SB"; mkdir -p "$SB"
  for f in base_checkpoints tokenizer task_data identity_conversations.jsonl words_alpha.txt; do ln -s "$POST_BASE_DIR/$f" "$SB/$f"; done
  # --num-iterations > 0 makes the old mid_train's LR schedule count micro-batches: fine for a dry run, never for a real one
  NANOCHAT_BASE_DIR=$SB run smoke_mid scripts.mid_train -- --model-tag=$TAG --model-step=$STEP --device-batch-size=$MID_DBS \
      --num-iterations=4 --eval-every=2 --eval-tokens=$((MID_DBS * 2048 * NPROC * 4)) --dry-run || exit 2
  NANOCHAT_BASE_DIR=$SB run smoke_sft scripts.chat_sft_348 -- --source=base --model-tag=$TAG --model-step=$STEP \
      --device-batch-size=$SFT_DBS --num-iterations=4 --eval-every=2 --eval-steps=2 --eval-metrics-every=2 \
      --eval-metrics-max-problems=16 || exit 2
  NANOCHAT_BASE_DIR=$SB run smoke_eval scripts.chat_eval -- -i sft -g $TAG -x 8 || exit 2
  rm -rf "$SB"; log "SMOKE_OK"; exit 0
fi

ls "$POST_BASE_DIR/mid_checkpoints/$TAG"/meta_*.json >/dev/null 2>&1 \
  || run mid_train scripts.mid_train -- --model-tag=$TAG --model-step=$STEP --device-batch-size=$MID_DBS || exit 3
ls "$POST_BASE_DIR/chatsft_checkpoints/$TAG"/meta_*.json >/dev/null 2>&1 \
  || run chat_sft scripts.chat_sft_348 -- --source=mid --model-tag=$TAG --device-batch-size=$SFT_DBS || exit 4
run chat_eval_sft scripts.chat_eval -- -i sft -g $TAG
run chat_eval_mid scripts.chat_eval -- -i mid -g $TAG
run chat_eval_mmlupro scripts.chat_eval -- -i sft -g $TAG -a "MMLU|MMLU-Pro"
log "ALL_DONE"
