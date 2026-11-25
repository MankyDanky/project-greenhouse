#!/bin/bash
# sbatch with this site's placement from config.sh, logging to $POST_LOGS.
# usage: POST_CONFIG=... bash submit.sh gpu|cpu SCRIPT [sbatch args...]     (SCRIPT relative to this folder)
set -euo pipefail
source "${POST_CONFIG:?export POST_CONFIG=/absolute/path/to/config.sh}"
export POST_CONFIG
kind=${1:?gpu or cpu}; script=${2:?script}; shift 2
case $kind in gpu) placement=$POST_GPU_SBATCH ;; cpu) placement=$POST_CPU_SBATCH ;; *) echo "usage: submit.sh gpu|cpu SCRIPT"; exit 2 ;; esac
[[ $script == /* ]] || script=$POST_RECIPE/$script
mkdir -p "$POST_LOGS"
sbatch $placement --output="$POST_LOGS/%x-%j.out" "$@" "$script"
