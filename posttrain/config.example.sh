# Site configuration: the only file in this recipe that names a path, account or partition.
#   cp config.example.sh config.sh; edit the first block; export POST_CONFIG=$PWD/config.sh

# ---- edit -----------------------------------------------------------------------------------------------------
# Work directory on a filesystem visible to login and compute nodes. Needs ~60 GB: the base checkpoint (10.6 GB), the
# Python env (~6 GB), the midtraining checkpoint with its optimizer state (24 GB) and the SFT checkpoint (10.6 GB).
POST_ROOT=/path/to/shared/scratch/gaggle-posttrain
# Extra sbatch arguments for the GPU job (one 80 GB GPU: H100 or H200) and for CPU jobs.
POST_GPU_SBATCH="--gres=gpu:h100:1"           # e.g. "--account=def-mypi --gres=gpu:h100:1"
POST_CPU_SBATCH=""                            # e.g. "--account=def-mypi"
# Which base model to post-train: 3b (the released Gaggle chat model, one 80 GB GPU) or 0.9b (two GPUs; set
# POST_GPU_SBATCH to ask for two, e.g. "--gres=gpu:h200:2").
POST_MODEL=${POST_MODEL:-3b}
# Optional trusted shell file sourced on the compute node before Python starts (module loads, library paths).
# The original run used one line: `module load gcc/12.3` (a C compiler for Triton/Inductor).
POST_ENV_SETUP=""

# ---- derived (change only if you know why) ------------------------------------------------------------------------
POST_RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this posttrain/ folder (keep config.sh inside it)
POST_NANOCHAT=$POST_ROOT/nanochat             # working copy: ../pretrain/nanochat (nanochat 92d63d4) + nanochat_overlay/
POST_BASE_DIR=$POST_ROOT/base                 # NANOCHAT_BASE_DIR: inputs, task data, mid_checkpoints/, chatsft_checkpoints/
POST_VENV=$POST_ROOT/venv
POST_PYTHON=$POST_VENV/bin/python
POST_HF_HOME=$POST_ROOT/hf_home               # base checkpoint download + offline Flash-Attention-3 kernel cache
POST_LOGS=$POST_ROOT/logs
# Per model: the released base (Hub repo, step, SHA256 of model_<step>.pt and meta_<step>.json), the checkpoint tag
# (base_checkpoints/, mid_checkpoints/, chatsft_checkpoints/), GPUs per job and the per-GPU micro-batches of
# midtraining (rows of 2,048 tokens) and SFT (conversations). The scripts accumulate gradients up to the recipe's
# batches (524,288 tokens and 32 conversations per step) whatever the GPU count and micro-batch.
case $POST_MODEL in
  3b)   POST_BASE_REPO=castorini/gaggle-nanochat-pretrained-climbmix-20260924; POST_BASE_STEP=227158; POST_TAG=gaggle227158
        POST_BASE_MODEL_SHA256=0d314c2a3441fc81fbecb0e58e9040b6758a1094afbb35621c059aca1f30baf6
        POST_BASE_META_SHA256=1a5edd78716655f5156c63c4be9b988ed1335ce043b27ae0bc1e2e1afd2d97ad
        POST_NPROC=1; POST_MID_DBS=4; POST_SFT_DBS=4 ;;
  0.9b) POST_BASE_REPO=castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105; POST_BASE_STEP=229067; POST_TAG=gaggle09b229067
        POST_BASE_MODEL_SHA256=eb60b077519dfb8c29ef225ef4ab6927e95d73f9f55aaf61c9574bee8bc9e201
        POST_BASE_META_SHA256=fc9c4ba651d494dcd02ee0c39d04427bfae54177f3920298e54c049b162872d3
        POST_NPROC=2; POST_MID_DBS=16; POST_SFT_DBS=4 ;;
  *) echo "POST_MODEL must be 3b or 0.9b, got '$POST_MODEL'" >&2; return 2 2>/dev/null || exit 2 ;;
esac
