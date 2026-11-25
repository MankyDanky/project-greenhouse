#!/bin/bash
# sbatch with this site's placement from config.sh, logging to $D34_LOGS.
# usage: D34_CONFIG=... bash submit.sh gpu|cpu SCRIPT [sbatch args...]     (SCRIPT relative to this folder)
set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
export D34_CONFIG
kind=${1:?gpu or cpu}; script=${2:?script}; shift 2
case $kind in gpu) placement=$D34_GPU_SBATCH ;; cpu) placement=$D34_CPU_SBATCH ;; *) echo "usage: submit.sh gpu|cpu SCRIPT"; exit 2 ;; esac
[[ $script == /* ]] || script=$D34_RECIPE/$script
mkdir -p "$D34_LOGS"
sbatch $placement --output="$D34_LOGS/%x-%j.out" "$@" "$script"
