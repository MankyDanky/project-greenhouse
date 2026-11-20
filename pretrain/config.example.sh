# Site configuration: the only file in this recipe that names a path, account or partition.
#   cp config.example.sh config.sh; edit the first block; export D34_CONFIG=$PWD/config.sh
# Every launcher and Slurm job sources it (stage 3 with `set -a`, so it must stay valid under `set -euo pipefail`).

# ---- edit -----------------------------------------------------------------------------------------------------
# Work directory on a shared filesystem visible to login and compute nodes. Needs ~1.5 TB: the corpus is 558 GiB,
# a checkpoint with optimizer state 24 GB (each stage keeps its newest 3, plus stage boundaries and milestones).
D34_ROOT=/path/to/shared/scratch/d34cm
# Extra sbatch arguments for the 8-GPU jobs (one node, 8x 80 GB Hopper: H100 or H200) and for the CPU jobs.
# Put the account, partition, QoS and GPU type here; the self-chaining legs resubmit with the same arguments.
D34_GPU_SBATCH="--gres=gpu:h100:8"            # e.g. "--account=def-mypi --partition=gpu_h100 --gres=gpu:h100:8"
D34_CPU_SBATCH=""                             # e.g. "--account=def-mypi"
# Optional trusted shell file sourced on every compute node before Python starts (module loads, library paths).
# The original run used one line: `module load gcc/12.3` (a C compiler for Triton/Inductor).
D34_ENV_SETUP=""

# ---- derived (change only if you know why) ------------------------------------------------------------------------
D34_RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this pretrain/ folder (keep config.sh inside it)
D34_EXP_DIR=$D34_ROOT/exp                     # frozen code written by setup.sh: upstream_snapshot/ source/ fleet/
D34_BASE_DIR=$D34_ROOT/base                   # NANOCHAT_BASE_DIR: base_data_climbmix/ tokenizer/ eval_bundle/ base_checkpoints/
D34_VENV=$D34_ROOT/venv
D34_PYTHON=$D34_VENV/bin/python
D34_HF_HOME=$D34_ROOT/hf_home                 # offline Flash-Attention-3 kernel cache
D34_LOGS=$D34_ROOT/logs
D34_RECEIPTS=$D34_ROOT/receipts               # stage-3 manifests, receipts and release
D34_DATA_ROOT=$D34_BASE_DIR/base_data_climbmix
D34_SHARD_MANIFEST=$D34_EXP_DIR/fleet/data/indexed_shards.json

# Checkpoint tags of the original run. Stages 2 and 3 check them (stage 2 must start with d34abl136316_,
# stage 3 with d34prod), so keep them.
D34_RAMP_TAG=d34veramp
D34_STAGE2_TAG=d34abl136316_b2m_lr014142
D34_PROD_TAG=d34prod_arm3_b2m_lr014142

# Stage 3 (production_leg.sbatch reads these names).
D34_FLEET_DIR=$D34_EXP_DIR/fleet
D34_PRODUCTION_CODE=$D34_FLEET_DIR/production
D34_CHECKPOINT_DIR=$D34_ROOT/protected/136316         # stage-1 hand-off checkpoint, hard-linked by stage 2
D34_RUNTIME_ROOT=$D34_VENV
D34_ASSET_MANIFEST=$D34_RECEIPTS/asset_manifest.json
D34_ASSET_RECEIPT=$D34_RECEIPTS/verified_assets.json
D34_FULL_CORPUS_RECEIPT=$D34_RECEIPTS/full_corpus_receipt.json
D34_WINNER_CHECKPOINT_DIR=$D34_BASE_DIR/base_checkpoints/$D34_STAGE2_TAG
D34_WINNER_STEP=137340
D34_WINNER_BATCH=2097152
D34_WINNER_LR=0.01414213562373095
D34_WINNER_COMPLETION=$D34_ROOT/stage2/$D34_STAGE2_TAG/COMPLETE.json
D34_WINNER_COMPLETION_SHA256=$(cat "$D34_RECEIPTS/winner_completion.sha256" 2>/dev/null || true)  # pinned by prepare_release.sbatch
D34_PROD_RELEASE=$D34_RECEIPTS/release.json
D34_PROD_DIR=$D34_ROOT/production/$D34_PROD_TAG
D34_PROD_SBATCH_ARGS_FILE=$D34_RECEIPTS/production_sbatch_args.txt
D34_PROD_NODES=1                              # 1 node x 8 GPUs, or 2 nodes x 4 (then also D34_GPUS_PER_NODE=4)
D34_GPUS_PER_NODE=8
D34_DEADLINE_MARGIN_S=1800
D34_PROD_CORE_EVERY=1200                      # CORE every 1,200 steps: the released checkpoint is the best-CORE one
D34_PROD_NO_CHAIN=0
