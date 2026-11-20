#!/bin/bash
# Train one paper row. Run from reranker/:
#
#   bash launch/train.sh configs/sft/run1_seed17.yaml 2          # 2 GPUs on this machine
#   sbatch --gres=gpu:2 --wrap "bash launch/train.sh configs/sft/run1_seed17.yaml"   # under Slurm
#
# The engine is read from the config's "# Engine:" header line (JSON configs: from the file name):
#   nanochat  src/train_nc.py               nanochat backbones: LCE, self-KD, QPQ, bidirectional, mean pooling
#   multi     engines/multi/train.py        any backbone on run 1's frozen plan, with chunked / GradCache execution
#   gemma4    engines/gemma4/train.py       native Gemma-4-E2B (tied, soft-capped readout)
#   qwen35    engines/qwen35/train.py       native Qwen3.5-4B-Base
#   ce        engines/ce/train_nc.py        pointwise cross-entropy over the full vocabulary
# All engines share reranker/src (data, packing plan, LCE loss, LR schedule, fp32 guard).
#
# n_gpus defaults to every visible GPU. Extra arguments go to the trainer (e.g. --override seed=18).
# Environment: PY_BIN (python env bin dir, prepended to PATH), HF_HOME, WANDB_MODE, NCCL_P2P_DISABLE=1 (no NVLink).
set -euo pipefail
CONFIG=${1:?usage: bash launch/train.sh <config> [n_gpus] [trainer args...]}; shift
NGPU=${1:-}; [ $# -gt 0 ] && shift
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG (run from reranker/)"; exit 1; }
[ -n "$NGPU" ] || NGPU=$(nvidia-smi -L | wc -l)

ENGINE=$(sed -n 's/^# Engine: *\([a-z0-9]*\).*/\1/p' "$CONFIG" | head -1)
case "$CONFIG" in *gemma4*.json) ENGINE=gemma4 ;; *qwen35*.json) ENGINE=qwen35 ;; esac
case "$ENGINE" in
  nanochat) ENTRY=src/train_nc.py ;;
  multi)    ENTRY=engines/multi/train.py ;;
  gemma4)   ENTRY=engines/gemma4/train.py ;;
  qwen35)   ENTRY=engines/qwen35/train.py ;;
  ce)       ENTRY=engines/ce/train_nc.py ;;
  *) echo "cannot tell the engine of $CONFIG (no '# Engine:' line)"; exit 1 ;;
esac

mkdir -p logs checkpoints tmp
export PYTHONNOUSERSITE=1
[ -n "${PY_BIN:-}" ] && export PATH="$PY_BIN:$PATH"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# DataLoader workers pass tensors through unix sockets under TMPDIR; keep it off a shared /tmp that can fill up.
export TMPDIR=${TMPDIR_OVERRIDE:-$PWD/tmp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_DIR=$PWD/wandb WANDB_MODE=${WANDB_MODE:-offline}; mkdir -p "$WANDB_DIR"
# The repo root on PYTHONPATH lets engines/ce import the shared core as reranker.src.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"

[ "$ENGINE" = nanochat ] && python src/test_symbols.py
# A static loopback master: --standalone can publish the store under an FQDN that some nodes cannot reach.
PORT=$(( 29500 + (${SLURM_JOB_ID:-$$} % 20000) ))
LOG=logs/$(basename "${CONFIG%.*}")_$(date +%Y%m%d_%H%M%S).log
echo "engine=$ENGINE entry=$ENTRY gpus=$NGPU config=$CONFIG log=$LOG"
torchrun --nnodes=1 --nproc_per_node="$NGPU" --master-addr=127.0.0.1 --master-port="$PORT" \
  "$ENTRY" --config "$CONFIG" "$@" 2>&1 | tee "$LOG"
