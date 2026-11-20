#!/bin/bash
#SBATCH --job-name=d34ve-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
# One leg of the d34 (value embeddings) x ClimbMix chain on a whole 8-GPU Hopper node; launch_ramp.sh starts it, and each
# leg queues the next one until the ramp is done. --depth=34, everything else derived by base_train unless STAGES/PIN_LR_BATCH
# override it; --fp8 (speedrun default on Hopper), device batch 8 (miniseries rule for depth>=28).
# Env knobs (inherited by the next leg): TAG (d34ve), NB_DIR ($D34_BASE_DIR), LEG_TIME (24:00:00), NUM_ITERS, DBS (8), FP8 (1),
# NO_CHAIN=1, EXTRA_ARGS (appended last; later flags win).
set -uo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
S1=$D34_ROOT/stage1; R=$D34_EXP_DIR/upstream_snapshot; HERE=$D34_RECIPE/stage1_ramp_to_92B
TAG=${TAG:-d34ve}; NB_DIR=${NB_DIR:-$D34_BASE_DIR}; STATE=$S1/state/$TAG; mkdir -p $STATE
CK=$NB_DIR/base_checkpoints/$TAG; mkdir -p $CK
LEG_TIME=${LEG_TIME:-24:00:00}; DBS=${DBS:-8}; FP8=${FP8:-1}; KEEP_EVERY=${KEEP_EVERY:-96000}
# Batch-size ramp (optional). STAGES = comma list of end_step:batch:save_every:core_every, ascending; the last end_step is the
# final step of the run (overrides NUM_ITERS). WARMDOWN = --warmdown-ratio for the whole run in global steps (the LR/wd/
# momentum schedules are functions of the global step and num_iterations, so every stage passes the same final horizon).
STAGES=${STAGES:-}; WARMDOWN=${WARMDOWN:-0.65}
NUM_ITERS=${NUM_ITERS:--1}
# DRY_RUN=1: resolve everything exactly as a leg would (stage, resume point, args) and print it without queueing or training;
# runnable on a login node: DRY_RUN=1 SLURM_JOB_ID=0 SLURM_JOB_NAME=<tag>-base TAG=... STAGES=... bash head_base.sbatch
DRY_RUN=${DRY_RUN:-0}; [ "$DRY_RUN" = "1" ] && NO_CHAIN=1
log(){ echo "[$(date '+%F %T')] $*"; }
log "leg job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) TAG=$TAG NB_DIR=$NB_DIR LEG_TIME=$LEG_TIME NUM_ITERS=$NUM_ITERS DBS=$DBS FP8=$FP8 end=$(date -d @${SLURM_JOB_END_TIME:-0} '+%F %T' 2>/dev/null)"
[ "$NUM_ITERS" -gt 0 ] || [ -n "$STAGES" ] || { log "FATAL: set NUM_ITERS or STAGES"; exit 2; }
if [ -f $STATE/BASE_DONE ]; then log "BASE_DONE present; nothing to do"; exit 0; fi
if [ -f $STATE/CHAIN_STOP ]; then log "CHAIN_STOP present; not running"; exit 0; fi
cd $D34_ROOT
if [ "${NO_CHAIN:-0}" != "1" ]; then
  PENDING=$(squeue -h -u $USER -n $SLURM_JOB_NAME -t PD -o %i | wc -l)
  if [ "$PENDING" -eq 0 ]; then
    NEXT=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$LEG_TIME $D34_GPU_SBATCH --output=$D34_LOGS/%x-%j.out --dependency=afterany:$SLURM_JOB_ID $HERE/head_base.sbatch) && log "queued next leg $NEXT (afterany:$SLURM_JOB_ID)"
  else log "a pending leg already exists ($PENDING); not queuing another"; fi
fi
if [ "$DRY_RUN" != "1" ]; then
  RUNNING_OTHERS=$(squeue -h -u $USER -n $SLURM_JOB_NAME -t R -o %i | grep -v "^$SLURM_JOB_ID$" | wc -l)
  [ "$RUNNING_OTHERS" -eq 0 ] || { log "FATAL: another $SLURM_JOB_NAME leg is RUNNING"; exit 6; }
fi
if [ -n "${D34_ENV_SETUP:-}" ]; then source "$D34_ENV_SETUP"; fi
source $D34_VENV/bin/activate; cd $R
export NANOCHAT_BASE_DIR=$NB_DIR HF_HOME=$D34_HF_HOME HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export TMPDIR=${SLURM_TMPDIR:-/tmp} TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/inductor
if [ "$DRY_RUN" != "1" ]; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
  if nvidia-smi -L | grep -qi MIG; then log "FATAL: MIG slice"; exit 3; fi
  NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -eq 8 ] || { log "FATAL: expected 8 GPUs, got $NGPU"; exit 3; }
fi
SKIP=$(cat $STATE/skip_steps 2>/dev/null | paste -sd, -)
RESUME=$(python $HERE/latest_ckpt.py $CK 8 "${SKIP:-}" 0)
STAGE_ARGS=""
if [ -n "$STAGES" ]; then
  # pick the first stage whose end_step is beyond the resume point (a resume exactly AT a boundary starts the next stage)
  START=$(( RESUME < 0 ? 0 : RESUME )); FINAL=${STAGES##*,}; FINAL=${FINAL%%:*}; NUM_ITERS=$FINAL
  for st in ${STAGES//,/ }; do IFS=: read -r END BATCH SAVE CORE <<< "$st"; if [ "$START" -lt "$END" ]; then STAGE_ARGS="--total-batch-size=$BATCH --stop-at-step=$END --save-every=$SAVE --core-metric-every=$CORE"; break; fi; done
  [ -n "$STAGE_ARGS" ] || { log "resume step $START is at/after the final step $FINAL; nothing to run"; touch $STATE/BASE_DONE; exit 0; }
  # STOP_RAMP_AT: end the whole ramp at this global step (the stage that contains it stops there instead of at its end)
  if [ -n "${STOP_RAMP_AT:-}" ]; then
    if [ "$START" -ge "$STOP_RAMP_AT" ]; then log "resume step $START is at/after STOP_RAMP_AT=$STOP_RAMP_AT; ramp done"; touch $STATE/BASE_DONE; exit 0; fi
    [ "$END" -gt "$STOP_RAMP_AT" ] && STAGE_ARGS=$(echo "$STAGE_ARGS" | sed -E "s/--stop-at-step=[0-9]+/--stop-at-step=$STOP_RAMP_AT/")
  fi
  # PIN_LR_BATCH=262144 reproduces the released model: every stage trains at the learning rates base_train computes for
  # batch 262,144 (Muon 0.0141). The original run got these rates because an upstream resume bug kept stage 1's rates
  # after each batch change (d34cm_head.patch fixes that bug, so the rates must now be pinned explicitly). Weight decay
  # is recomputed every step from the current batch, as in the original run.
  if [ -n "${PIN_LR_BATCH:-}" ]; then
    CUR_BATCH=$(echo "$STAGE_ARGS" | sed -E 's/.*--total-batch-size=([0-9]+).*/\1/')
    STAGE_ARGS="$STAGE_ARGS $(python3 -c "import math; f=math.sqrt($PIN_LR_BATCH/$CUR_BATCH); print(f'--matrix-lr={0.02*f:.12g} --embedding-lr={0.3*f:.12g} --unembedding-lr={0.008*f:.12g} --scalar-lr={0.5*f:.12g}')")"
  fi
  log "ramp stage: $STAGE_ARGS (final step $FINAL, warmdown ratio $WARMDOWN)"
  # stage-boundary checkpoints (every end_step but the final) are kept in full so any stage can be restarted from its start
  KEEP_FULL=$(for st in ${STAGES//,/ }; do echo ${st%%:*}; done | grep -vx "$FINAL" | paste -sd, -)
fi
KEEP_FULL=${KEEP_FULL:-}
ARGS="--depth=34 --device-batch-size=$DBS --num-iterations=$NUM_ITERS --save-every=600 --core-metric-every=2000 --model-tag=$TAG --deadline-margin-s=900 --warmdown-ratio=$WARMDOWN $STAGE_ARGS"
[ "$FP8" = "1" ] && ARGS="$ARGS --fp8"
ARGS="$ARGS ${EXTRA_ARGS:-}"
[ "$RESUME" -ge 0 ] && ARGS="$ARGS --resume-from-step=$RESUME"
log "resume_from_step=$RESUME  args: $ARGS"
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- $ARGS"
  log "DRY_RUN: prune_ckpts.py $CK --keep-last 3 --keep-every $KEEP_EVERY --keep-full '$KEEP_FULL' --final $NUM_ITERS; NB_DIR=$NB_DIR CK=$CK STATE=$STATE"
  exit 0
fi
T_START=$(date +%s)
# In-leg retention: a 24 GB checkpoint lands every ~52 min and the 1000 GiB scratch quota was hit mid-leg three times
# (legs died with "Disk quota exceeded"). Prune every 10 min while training runs: keep the newest 3 full checkpoints
# + model-only milestones every KEEP_EVERY steps (must be a multiple of every stage's save_every: 96000 for the ramp) + the
# final step + the stage boundaries in full. Never touches the newest 3, so the resume point is safe.
( while true; do sleep 600; python $HERE/prune_ckpts.py $CK --keep-last 3 --keep-every $KEEP_EVERY --keep-full "$KEEP_FULL" --final $NUM_ITERS >/dev/null 2>&1; done ) &
PRUNE_PID=$!
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- $ARGS
rc=$?
kill $PRUNE_PID 2>/dev/null; wait $PRUNE_PID 2>/dev/null
T_RUN=$(( $(date +%s) - T_START ))
NEWEST=$(python $HERE/latest_ckpt.py $CK 8 "${SKIP:-}" 0)
log "torchrun exit=$rc after ${T_RUN}s; newest complete checkpoint now $NEWEST (was $RESUME)"
echo "$(date '+%F %T') job=$SLURM_JOB_ID node=$(hostname) resume=$RESUME newest=$NEWEST rc=$rc secs=$T_RUN" >> $STATE/legs.log
if [ $rc -ne 0 ] && [ $T_RUN -lt 1800 ]; then
  N=$(( $(cat $STATE/fail_count 2>/dev/null || echo 0) + 1 )); echo $N > $STATE/fail_count
  if [ "$RESUME" -ge 0 ]; then R2=$(( $(cat $STATE/fail_$RESUME 2>/dev/null || echo 0) + 1 )); echo $R2 > $STATE/fail_$RESUME; [ $R2 -ge 2 ] && { echo $RESUME >> $STATE/skip_steps; log "step $RESUME failed to resume twice; will skip it"; }; fi
  [ $N -ge 3 ] && { touch $STATE/CHAIN_STOP; log "3 consecutive failures: CHAIN_STOP set"; }
else echo 0 > $STATE/fail_count; fi
python $HERE/prune_ckpts.py $CK --keep-last 3 --keep-every $KEEP_EVERY --keep-full "$KEEP_FULL" --final $NUM_ITERS | tail -1
if [ -f $CK/model_$(printf %06d $NUM_ITERS).pt ]; then
  touch $STATE/BASE_DONE; log "BASE_DONE: final step $NUM_ITERS reached"
fi
exit $rc
