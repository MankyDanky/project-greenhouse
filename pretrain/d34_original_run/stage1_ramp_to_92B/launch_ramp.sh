#!/bin/bash
# Launch the ramp run of the released model (stage 1). usage: D34_CONFIG=... [RESUME=1] bash launch_ramp.sh [first-leg-time]
# RESUME=1 restarts a stopped chain from its newest complete checkpoint (after CHAIN_STOP: fix the cause, delete CHAIN_STOP and fail_count).
# Stages: 262,144 to step 38,147 (10.0B tokens), 524,288 to 77,057 (30.4B), 1,048,576 onwards; the run is stopped at step 136,316
# (92.54B tokens, STOP_RAMP_AT below) and continued at batch 2,097,152 by stage2_2m_continuation/. LRs pinned at the 262K values.
# STAGES entries: end_step:batch:save_every:core_every (save every ~55 min at each stage's step time). One pass = 287.05B training tokens.
set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
T=${1:-24:00:00}
STAGES="38147:262144:4800:16000,77057:524288:2400:8000,321817:1048576:1200:4000"; WARMDOWN=0.5528
TAG=$D34_RAMP_TAG; S1=$D34_ROOT/stage1; mkdir -p $S1/state/$TAG $D34_LOGS
[ -e $D34_BASE_DIR/base_checkpoints/$TAG ] && [ "${RESUME:-0}" != 1 ] && { echo "checkpoint dir for $TAG already exists; refusing to relaunch over it (RESUME=1 continues it)"; exit 3; }
[ -e $S1/state/$TAG/CHAIN_STOP ] && { echo "CHAIN_STOP present in $S1/state/$TAG; fix the cause, then delete CHAIN_STOP and fail_count there"; exit 3; }
[ -f $D34_BASE_DIR/tokenizer/tokenizer.pkl ] || { echo "no tokenizer in $D34_BASE_DIR/tokenizer: run stage0_tokenizer/tok_train.sbatch first"; exit 3; }
echo "$STAGES" > $S1/state/$TAG/STAGES; echo "$WARMDOWN" > $S1/state/$TAG/WARMDOWN; echo "tag=$TAG stages=$STAGES warmdown=$WARMDOWN launched $(date)" >> $S1/LAUNCHES.txt
cd $D34_ROOT
# STOP_RAMP_AT ends the ramp at the released model's hand-off point (step 136,316, 92.54B tokens); the horizon (final step
# 321,817 = one pass at batch 1,048,576) still defines the LR/wd/momentum schedules, as in the original run.
J=$(TAG=$TAG STAGES=$STAGES WARMDOWN=$WARMDOWN PIN_LR_BATCH=262144 STOP_RAMP_AT=136316 KEEP_EVERY=96000 \
    sbatch --parsable --job-name=${TAG}-base --time=$T $D34_GPU_SBATCH --output=$D34_LOGS/%x-%j.out \
    $D34_RECIPE/stage1_ramp_to_92B/head_base.sbatch) && echo "ramp run leg 1 = job $J"
sleep 3; squeue -u $USER -h -o "%.10i %.16j %.2t %.10l %R"; squeue -h -j $J --start -o "est. start %S (%R)"
