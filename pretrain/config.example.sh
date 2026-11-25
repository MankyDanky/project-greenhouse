# Site configuration of the pre-training recipe: the only file that names a path, account or partition.
#   cp config.example.sh config.sh; edit the first block; export PT_CONFIG=$PWD/config.sh
# Every script here sources it (with `set -euo pipefail`, so keep it valid under that).

# ---- edit -----------------------------------------------------------------------------------------------------
# Work directory on a filesystem shared by login and compute nodes. Needs ~0.9 TB: the corpus is 558 GiB, and a
# checkpoint with optimizer state is 24 GB (3.3B), 6.2 GB (0.9B) or 4.4 GB (0.6B); runs keep their newest few.
PT_ROOT=/path/to/shared/scratch/gaggle-pretrain
# Extra sbatch arguments for the GPU jobs (one node; the configs name the GPU count they were run with) and CPU jobs.
PT_GPU_SBATCH="--gres=gpu:h100:8"            # e.g. "--account=def-mypi --partition=gpu_h100 --gres=gpu:h100:8"
PT_CPU_SBATCH=""                             # e.g. "--account=def-mypi"
PT_LEG_TIME=24:00:00                         # wall time of one job in a chain (train.sbatch); runs resume across jobs
# Optional trusted shell file sourced on every compute node before Python starts (module loads, library paths).
PT_ENV_SETUP=""
# Weights & Biases project for the runs ("" = no wandb logging).
PT_WANDB_PROJECT=""

# ---- derived (change only if you know why) ------------------------------------------------------------------------
PT_RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this pretrain/ folder (keep config.sh inside it)
PT_NANOCHAT=$PT_RECIPE/nanochat              # the trainer (nanochat 92d63d4 + the changes listed in README.md)
PT_BASE_DIR=$PT_ROOT/base                    # NANOCHAT_BASE_DIR: base_data_climbmix/ tokenizer/ eval_bundle/ base_checkpoints/
PT_DATA_ROOT=$PT_BASE_DIR/base_data_climbmix
PT_VENV=$PT_ROOT/venv
PT_PYTHON=$PT_VENV/bin/python
PT_HF_HOME=$PT_ROOT/hf_home                  # offline Flash-Attention-3 kernel cache
PT_LOGS=$PT_ROOT/logs
