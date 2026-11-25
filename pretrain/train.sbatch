#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --mail-type=NONE
# One job of a training chain: runs train.sh on this node's GPUs. Submit a chain with submit_chain.sh, which adds
# this site's placement ($PT_GPU_SBATCH), the wall time ($PT_LEG_TIME) and the dependencies; every job resumes from
# the newest complete checkpoint, and one that finds the final checkpoint exits at once.
#   (from submit_chain.sh)  sbatch ... train.sbatch configs/<model>.yaml
set -euo pipefail
source "${PT_CONFIG:?export PT_CONFIG=/absolute/path/to/config.sh}"
bash "$PT_RECIPE/train.sh" "${1:?config}"
