#!/bin/bash
# Coordinator supplies account, nodes, GPUs, CPUs, memory and time explicitly.
#SBATCH --job-name=d34-production
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside an allocation}"
: "${D34_PROD_CONFIG:?Trusted coordinator environment file is required}"
set -a
source "$D34_PROD_CONFIG"
set +a
for key in D34_PROD_DIR D34_PRODUCTION_CODE D34_EXP_DIR D34_BASE_DIR D34_PYTHON \
    D34_HF_HOME D34_PROD_TAG D34_WINNER_CHECKPOINT_DIR D34_WINNER_STEP \
    D34_WINNER_BATCH D34_WINNER_LR D34_GPUS_PER_NODE D34_PROD_RELEASE \
    D34_WINNER_COMPLETION D34_WINNER_COMPLETION_SHA256 \
    D34_SHARD_MANIFEST D34_DATA_ROOT D34_FLEET_DIR D34_CHECKPOINT_DIR \
    D34_RUNTIME_ROOT D34_ASSET_MANIFEST D34_ASSET_RECEIPT D34_FULL_CORPUS_RECEIPT; do
    [[ -n "${!key:-}" ]] || { echo "Missing $key" >&2; exit 2; }
    export "$key"
done
mkdir -p "$D34_PROD_DIR"
[[ ! -e "$D34_PROD_DIR/BASE_DONE" && ! -e "$D34_PROD_DIR/CHAIN_STOP" ]] || exit 0
exec 9>"$D34_PROD_DIR/chain.lock"
flock -n 9 || { echo 'Another production leg owns this directory' >&2; exit 3; }
exec > >(tee -a "$D34_PROD_DIR/chain.log") 2>&1
if [[ -n "${D34_ENV_SETUP:-}" ]]; then source "$D34_ENV_SETUP"; export D34_ENV_SETUP; fi
export D34_NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:?}}"
export D34_CLUSTER="${SLURM_CLUSTER_NAME:-${D34_CLUSTER:-unknown}}"
if ! { [[ "$D34_NNODES" == 1 && "$D34_GPUS_PER_NODE" == 8 ]] || \
       [[ "$D34_NNODES" == 2 && "$D34_GPUS_PER_NODE" == 4 ]]; }; then exit 2; fi
export D34_PROD_CHECKPOINT_DIR="$D34_BASE_DIR/base_checkpoints/$D34_PROD_TAG"
"$D34_PYTHON" "$D34_PRODUCTION_CODE/production_chain.py" check-release \
    "$D34_PROD_RELEASE" "$D34_WINNER_CHECKPOINT_DIR" "$D34_WINNER_STEP" \
    "$D34_WINNER_BATCH" "$D34_WINNER_LR"
mapfile -t resume < <("$D34_PYTHON" "$D34_PRODUCTION_CODE/checkpoint_state.py" \
    "$D34_PROD_CHECKPOINT_DIR" --initial-dir "$D34_WINNER_CHECKPOINT_DIR" --initial-step "$D34_WINNER_STEP")
[[ "${#resume[@]}" == 2 ]] || exit 2
export D34_PROD_SOURCE_DIR="${resume[0]}" D34_PROD_SOURCE_STEP="${resume[1]}"
"$D34_PYTHON" "$D34_PRODUCTION_CODE/production_chain.py" start \
    "$D34_PROD_DIR" "$SLURM_JOB_ID" "$D34_PROD_SOURCE_DIR" "$D34_PROD_SOURCE_STEP"
next_job=""
on_exit() {
    local result=$?
    trap - EXIT
    "$D34_PYTHON" "$D34_PRODUCTION_CODE/production_chain.py" finish \
        "$D34_PROD_DIR" "$SLURM_JOB_ID" "$result" "$D34_PROD_CHECKPOINT_DIR" || result=1
    if [[ -n "$next_job" && ( -e "$D34_PROD_DIR/BASE_DONE" || -e "$D34_PROD_DIR/CHAIN_STOP" ) ]]; then
        # This ID was created by this exact leg, never discovered from a user-wide query.
        scancel "$next_job" || true
    fi
    exit "$result"
}
trap on_exit EXIT
if [[ "${D34_PROD_NO_CHAIN:-0}" != 1 ]]; then
    : "${D34_PROD_SBATCH_ARGS_FILE:?One explicitly approved placement argument per line}"
    placement=()
    while IFS= read -r argument; do
        [[ -z "$argument" || "$argument" == \#* ]] || placement+=("$argument")
    done < "$D34_PROD_SBATCH_ARGS_FILE"
    [[ "$D34_PROD_CONFIG" != *,* ]] || exit 2
    submitted=$(sbatch --parsable "${placement[@]}" --dependency="afterany:$SLURM_JOB_ID" \
        --export="ALL,D34_PROD_CONFIG=$D34_PROD_CONFIG" "$D34_PRODUCTION_CODE/production_leg.sbatch")
    next_job="${submitted%%;*}"
    [[ "$next_job" =~ ^[0-9]+$ ]] || exit 2
    printf '%s\t%s\t%s\n' "$SLURM_JOB_ID" "$next_job" "$(date -Is)" >> "$D34_PROD_DIR/legs.tsv"
fi
scontrol show job "$SLURM_JOB_ID" -o > "$D34_PROD_DIR/job-${SLURM_JOB_ID}.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$D34_PROD_DIR/nodes-${SLURM_JOB_ID}.txt"
read -r D34_MASTER_ADDR < "$D34_PROD_DIR/nodes-${SLURM_JOB_ID}.txt"
export D34_MASTER_ADDR D34_MASTER_PORT="$((20000 + 2 * (SLURM_JOB_ID % 15000)))"
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
    end_text=$(sed -n 's/.* EndTime=\([^ ]*\).*/\1/p' "$D34_PROD_DIR/job-${SLURM_JOB_ID}.txt")
    export SLURM_JOB_END_TIME="$(date -d "$end_text" +%s)"
fi
# Hash and filesystem checks finish before distributed rendezvous starts.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$D34_MASTER_ADDR" \
    --kill-on-bad-exit=1 --export=ALL bash "$D34_PRODUCTION_CODE/preflight.sh"
# Reuse the validated eight-rank Hopper/runtime/NCCL probe from the fleet.
D34_BRANCH_DIR="$D34_PROD_DIR/preflight/$SLURM_JOB_ID" D34_RUN_MODE=preflight \
    srun --nodes="$D34_NNODES" --ntasks="$D34_NNODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --export=ALL bash "$D34_FLEET_DIR/node_launch.sh"
export D34_MASTER_PORT="$((D34_MASTER_PORT + 1))"
srun --nodes="$D34_NNODES" --ntasks="$D34_NNODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --export=ALL bash "$D34_PRODUCTION_CODE/node_launch.sh"
