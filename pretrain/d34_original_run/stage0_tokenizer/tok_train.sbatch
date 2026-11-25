#!/bin/bash
#SBATCH --job-name=d34-tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=NONE
# nanochat's own tokenizer step: python -m scripts.tok_train (defaults: 2B chars, doc cap 10k, vocab 32,768) on ClimbMix, then tok_eval. CPU-only.
# submit: bash submit.sh cpu stage0_tokenizer/tok_train.sbatch
set -uo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
R=$D34_EXP_DIR/upstream_snapshot; NB=$D34_BASE_DIR
if [ -n "${D34_ENV_SETUP:-}" ]; then source "$D34_ENV_SETUP"; fi
source "$D34_VENV/bin/activate"; cd $R
export NANOCHAT_BASE_DIR=$NB HF_HOME=$D34_HF_HOME HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1 RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "[$(date '+%F %T')] tok_train on $(hostname) ($(nproc) cpus); data dir $(python -c 'import nanochat.dataset as d; print(d.DATA_DIR)')"
mkdir -p $NB/tokenizer
python -m scripts.tok_train || { echo "tok_train FAILED"; exit 4; }
python -m scripts.tok_eval || echo "tok_eval failed (non-fatal)"
python - <<'PY'
import os, pickle, torch
nb = os.environ["NANOCHAT_BASE_DIR"]
enc = pickle.load(open(f"{nb}/tokenizer/tokenizer.pkl", "rb")); tb = torch.load(f"{nb}/tokenizer/token_bytes.pt")
print(f"new tokenizer: n_vocab {enc.n_vocab}  token_bytes {tuple(tb.shape)}")
assert enc.n_vocab == 32768
PY
sha256sum $NB/tokenizer/tokenizer.pkl; echo "[$(date '+%F %T')] tok_train done"
