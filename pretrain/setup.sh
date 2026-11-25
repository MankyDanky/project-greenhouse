#!/bin/bash
# One-time setup on a login node (needs internet; compute nodes run offline). Idempotent; rerun to resume downloads.
#   1. Python 3.11 env from nanochat's uv.lock (torch 2.9.1+cu128) plus pyyaml, at $PT_VENV
#   2. Flash-Attention-3 kernels into $PT_HF_HOME, CORE eval bundle into $PT_BASE_DIR
#   3. the Gaggle tokenizer (the released one; d34_original_run/stage0_tokenizer reproduces it byte for byte)
#   4. ClimbMix: all 6,543 shards (558 GiB) into $PT_DATA_ROOT; SKIP_DATA=1 skips this step
# usage: PT_CONFIG=/abs/path/to/config.sh bash setup.sh
set -euo pipefail
source "${PT_CONFIG:?export PT_CONFIG=/absolute/path/to/config.sh}"
log(){ echo "[$(date '+%F %T')] $*"; }
case $PT_ROOT in
  /path/to/*) log "FATAL: set PT_ROOT in $PT_CONFIG"; exit 2 ;;
  /*) ;;
  *) log "FATAL: PT_ROOT must be an absolute path"; exit 2 ;;
esac
command -v uv >/dev/null || { log "FATAL: uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 2; }
mkdir -p "$PT_BASE_DIR" "$PT_HF_HOME" "$PT_LOGS"

log "1. Python env $PT_VENV"
export UV_PROJECT_ENVIRONMENT="$PT_VENV" UV_CACHE_DIR="${UV_CACHE_DIR:-$PT_ROOT/uv_cache}"
(cd "$PT_NANOCHAT" && uv sync --extra gpu --frozen --python 3.11 2>&1 | tail -3)
uv pip install --quiet --python "$PT_PYTHON" pyyaml
"$PT_PYTHON" - <<'PY'
import sys, torch, kernels, rustbpe, tiktoken, pyarrow, yaml
print(f"python {sys.version.split()[0]} | torch {torch.__version__} cuda {torch.version.cuda} | kernels {kernels.__version__} | tiktoken {tiktoken.__version__} | pyarrow {pyarrow.__version__}")
assert sys.version_info[:2] == (3, 11) and torch.__version__ == "2.9.1+cu128"
PY

log "2a. Flash-Attention-3 kernels into $PT_HF_HOME (nanochat asks for varunneal/flash-attention-3 on Hopper)"
HF_HOME="$PT_HF_HOME" "$PT_PYTHON" - <<'PY'
import os; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from kernels import get_kernel
for repo in ("varunneal/flash-attention-3", "kernels-community/flash-attn3"):
    try:
        k = get_kernel(repo); print(" ", repo, "cached, loaded:", "flash_attn_interface" in dir(k))
    except Exception as e:  # a login node without a GPU may fail to load the .so after downloading it
        print(" ", repo, "get_kernel:", type(e).__name__, str(e)[:200])
PY
ls -d "$PT_HF_HOME"/hub/models--varunneal--flash-attention-3 >/dev/null \
  || { log "FATAL: FA3 kernel not in $PT_HF_HOME; training would fall back to SDPA (the configs refuse that)"; exit 2; }

log "2b. CORE eval bundle into $PT_BASE_DIR/eval_bundle"
(cd "$PT_NANOCHAT" && NANOCHAT_BASE_DIR="$PT_BASE_DIR" PYTHONDONTWRITEBYTECODE=1 "$PT_PYTHON" -c "
from nanochat.common import download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
import os
if not os.path.isdir(os.path.join(os.environ['NANOCHAT_BASE_DIR'], 'eval_bundle')):
    download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)
")
ls "$PT_BASE_DIR/eval_bundle/core.yaml" >/dev/null

log "3. tokenizer into $PT_BASE_DIR/tokenizer"
HF_HOME="$PT_HF_HOME" "$PT_PYTHON" - "$PT_BASE_DIR" <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download
dest = os.path.join(sys.argv[1], "tokenizer"); os.makedirs(dest, exist_ok=True)
for f in ("tokenizer.pkl", "token_bytes.pt"):
    if not os.path.exists(os.path.join(dest, f)):
        shutil.copy(hf_hub_download("castorini/gaggle-nanochat-pretrained-climbmix-20260924", f"tokenizer/{f}"), dest)
h = hashlib.sha256(open(os.path.join(dest, "tokenizer.pkl"), "rb").read()).hexdigest()
assert h == "ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2", h
print("  tokenizer.pkl sha256 ae73c5f7... (the released one)")
PY

if [ "${SKIP_DATA:-0}" != 1 ]; then
  log "4. ClimbMix (karpathy/climbmix-400b-shuffle) into $PT_DATA_ROOT; shard 06542 is the validation shard"
  (cd "$PT_NANOCHAT" && NANOCHAT_BASE_DIR="$PT_BASE_DIR" PYTHONDONTWRITEBYTECODE=1 \
    "$PT_PYTHON" -m nanochat.dataset -n -1 -w "${DL_WORKERS:-8}" | grep -v '^Skipping' | tail -3)
  N=$(find "$PT_DATA_ROOT" -maxdepth 1 -name 'shard_*.parquet' | wc -l)
  [ "$N" -eq 6543 ] || { log "only $N/6543 shards present; rerun setup.sh to fetch the rest"; exit 1; }
  # nanochat's loader lists the shards once and then cycles, so train.sh refuses to start on a partial corpus
  touch "$PT_DATA_ROOT/.all_shards_present"
  log "  6543/6543 shards"
fi
log "setup done"
