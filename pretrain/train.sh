#!/bin/bash
# Train (or resume) one backbone on the GPUs of this node: torchrun over nanochat/scripts/base_train.py with the
# flags of a run config. Run it inside a GPU allocation (train.sbatch does that for Slurm).
#
#   PT_CONFIG=/abs/path/config.sh bash train.sh configs/<model>.yaml
#
# - The first start copies the config into the run's checkpoint directory (run_config.yaml); a later start refuses a
#   config that differs from it, so a resumed run cannot silently change its recipe.
# - Every start resumes from the newest complete checkpoint (--auto-resume); a start that finds the final checkpoint
#   exits at once. The trainer saves and exits 15 minutes before a Slurm job's time limit (--deadline-margin-s).
# - The run refuses to start without all 6,543 ClimbMix shards: nanochat's loader lists them once and then cycles.
# - SMOKE=1: 12 steps on whatever GPUs are visible, no evaluation, no checkpoint, scratch tag; checks FA3 and memory.
set -euo pipefail
source "${PT_CONFIG:?export PT_CONFIG=/absolute/path/to/config.sh}"
[ -n "${PT_ENV_SETUP:-}" ] && source "$PT_ENV_SETUP"
CFG=$(realpath "${1:?usage: train.sh configs/<model>.yaml}")
SMOKE=${SMOKE:-0}
NGPU=${PT_NGPU:-$(nvidia-smi -L | wc -l)}
export NANOCHAT_BASE_DIR=$PT_BASE_DIR HF_HOME=$PT_HF_HOME HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
PY=$PT_PYTHON
echo "################ $(date '+%F %T %Z')  job ${SLURM_JOB_ID:-none} on $(hostname), $NGPU GPU(s), config $CFG$([ "$SMOKE" = 1 ] && echo ' SMOKE')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c

"$PY" "$PT_RECIPE/full_args.py" "$CFG" --check --ngpu "$NGPU"
eval "$("$PY" "$PT_RECIPE/full_args.py" "$CFG" --ngpu "$NGPU")"
export NANOCHAT_ATTN=$ATTN
cd "$PT_NANOCHAT"

if [ "$SMOKE" = 1 ]; then
  TAG=smoke-$RUN; CK=$NANOCHAT_BASE_DIR/base_checkpoints/$TAG
  rm -rf "${CK:?}"
  status=0
  torchrun --standalone --nproc_per_node="$NGPU" -m scripts.base_train -- "${ARGS[@]}" \
    --num-iterations 12 --eval-every -1 --core-metric-every -1 --save-every -1 --log-every 1 --model-tag="$TAG" --run=dummy \
    || status=$?
  rm -rf "${CK:?}"; exit $status
fi

[ -f "$PT_DATA_ROOT/.all_shards_present" ] || { echo "STOP: ClimbMix incomplete (run setup.sh)"; exit 1; }
CK=$NANOCHAT_BASE_DIR/base_checkpoints/$RUN
if [ -f "$CK/$(printf 'complete_%06d' "$NUM_ITERS")" ]; then echo "final checkpoint already complete -- nothing to do"; exit 0; fi
mkdir -p "$CK"
if [ -f "$CK/run_config.yaml" ]; then
  cmp -s "$CFG" "$CK/run_config.yaml" || { echo "STOP: $CFG differs from the config this run started with ($CK/run_config.yaml)"; exit 1; }
else
  cp "$CFG" "$CK/run_config.yaml"
fi
WANDB_NAME=dummy
if [ -n "${PT_WANDB_PROJECT:-}" ]; then
  export WANDB_PROJECT=$PT_WANDB_PROJECT WANDB_RUN_ID=$RUN WANDB_RESUME=allow WANDB_DIR=$PT_ROOT/wandb
  mkdir -p "$WANDB_DIR"; WANDB_NAME=$RUN
fi
torchrun --standalone --nproc_per_node="$NGPU" -m scripts.base_train -- "${ARGS[@]}" \
  --model-tag="$RUN" --run="$WANDB_NAME" --auto-resume
