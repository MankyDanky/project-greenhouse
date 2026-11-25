#!/bin/bash
# Submit a chain of N Slurm jobs that train one backbone; each starts after the previous one ends (afterany) and
# resumes from its newest complete checkpoint.
#   PT_CONFIG=/abs/path/config.sh bash submit_chain.sh configs/<model>.yaml N [extra sbatch args...]
# The released runs took about 2.4 node-days (0.6B, 8 x H100), 6 days (0.9B, 4 x H200) and 8.4 node-days (3.3B,
# 8 x H100) of training; with 24-hour jobs, N = 4, 8 and 10 leave room for restarts.
set -euo pipefail
source "${PT_CONFIG:?export PT_CONFIG=/absolute/path/to/config.sh}"
export PT_CONFIG
CFG=$(realpath "${1:?config}"); N=${2:?number of jobs}; shift 2
"$PT_PYTHON" "$PT_RECIPE/full_args.py" "$CFG" --check >/dev/null
mkdir -p "$PT_LOGS"
dep=""
for i in $(seq 1 "$N"); do
  J=$(sbatch --parsable $PT_GPU_SBATCH --time="$PT_LEG_TIME" --output="$PT_LOGS/%x-%j.out" \
        --job-name="pt-$(basename "$CFG" .yaml)" $dep "$@" "$PT_RECIPE/train.sbatch" "$CFG")
  echo "job $i/$N: $J"
  dep="--dependency=afterany:$J"
done
