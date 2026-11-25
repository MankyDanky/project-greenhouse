#!/bin/bash
# One-time setup on a login node (needs internet; compute nodes run offline). Idempotent; rerun to resume downloads.
#   1. rebuild the run's nanochat (upstream 92d63d4 + d34cm_head.patch + ../data/dataset_indexed_manifest.patch, checked
#      against records/original_nanochat.sha256) and freeze it with the stage-2/3 code into $D34_EXP_DIR (stages 2-3 run
#      from this copy and stage 3 hashes it, so it is never edited). The repository's ../nanochat is the unified trainer
#      of ../README.md, not this run's code.
#   2. Python 3.11 env from nanochat's uv.lock (torch 2.9.1+cu128) at $D34_VENV
#   3. Flash-Attention-3 kernels into $D34_HF_HOME, CORE eval bundle into $D34_BASE_DIR
#   4. ClimbMix: all 6,543 shards (558 GiB) into $D34_DATA_ROOT; SKIP_DATA=1 skips this step
# usage: D34_CONFIG=/abs/path/to/config.sh bash setup.sh
set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
log(){ echo "[$(date '+%F %T')] $*"; }
case $D34_ROOT in
  /path/to/*) log "FATAL: set D34_ROOT in $D34_CONFIG"; exit 2 ;;
  /*) ;;
  *) log "FATAL: D34_ROOT must be an absolute path"; exit 2 ;;
esac
command -v uv >/dev/null || { log "FATAL: uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 2; }
mkdir -p "$D34_EXP_DIR" "$D34_BASE_DIR" "$D34_HF_HOME" "$D34_LOGS" "$D34_RECEIPTS"

log "1. frozen code in $D34_EXP_DIR"
T=$(mktemp -d "$D34_ROOT/.freeze.XXXXXX"); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/upstream_snapshot" "$T/source" "$T/fleet/data" "$T/fleet/production"
# The run's nanochat: upstream 92d63d4 plus the two patches this recipe carries, rebuilt and checked file by file.
ORIG=$D34_ROOT/nanochat_original
[ -d "$ORIG/.git" ] || git clone --quiet https://github.com/karpathy/nanochat.git "$ORIG"
git -C "$ORIG" checkout --quiet --force 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
git -C "$ORIG" clean --quiet -fdx
git -C "$ORIG" apply -p1 "$D34_RECIPE/stage1_ramp_to_92B/d34cm_head.patch"
git -C "$ORIG" apply -p1 "$D34_RECIPE/../data/dataset_indexed_manifest.patch"
(cd "$ORIG" && sha256sum --quiet -c "$D34_RECIPE/records/original_nanochat.sha256") \
  || { log "FATAL: the rebuilt nanochat differs from records/original_nanochat.sha256"; exit 2; }
N=$(cd "$ORIG" && find . -type f -not -path "./.git/*" | wc -l)
[ "$N" -eq "$(wc -l < "$D34_RECIPE/records/original_nanochat.sha256")" ] \
  || { log "FATAL: the rebuilt nanochat has $N files, records/original_nanochat.sha256 lists a different number"; exit 2; }
log "  nanochat 92d63d4 + d34cm_head.patch + dataset_indexed_manifest.patch: identical to the run's ($N files)"
rsync -a --exclude .git --exclude __pycache__ --exclude .venv "$ORIG/" "$T/upstream_snapshot/"
cp "$D34_RECIPE"/stage2_2m_continuation/{base_train.py,ablation_runtime.py} "$T/source/"
cp "$D34_RECIPE"/stage3_2m_production/{node_launch.sh,collective_probe.py,verify_assets.py,make_asset_manifest.py} "$T/fleet/"
cp "$D34_RECIPE"/stage3_2m_production/production/*.{py,sh,sbatch} "$T/fleet/production/"
cp "$D34_RECIPE/../data/indexed_shards.json" "$T/fleet/data/"
for d in upstream_snapshot source fleet; do
  if [ -e "$D34_EXP_DIR/$d" ]; then
    # code_manifest.json is written later by prepare_release.sbatch
    diff -r -x __pycache__ -x code_manifest.json "$T/$d" "$D34_EXP_DIR/$d" >/dev/null \
      || { log "FATAL: $D34_EXP_DIR/$d differs from the recipe; frozen code is never edited (use a new D34_ROOT)"; exit 2; }
    log "  $d: present, identical"
  else
    mv "$T/$d" "$D34_EXP_DIR/$d"; log "  $d: written"
  fi
done

log "2. Python env $D34_VENV"
export UV_PROJECT_ENVIRONMENT="$D34_VENV" UV_CACHE_DIR="${UV_CACHE_DIR:-$D34_ROOT/uv_cache}"
(cd "$D34_EXP_DIR/upstream_snapshot" && uv sync --extra gpu --frozen --python 3.11 2>&1 | tail -3)
"$D34_PYTHON" - <<'PY'
import sys, torch, kernels, rustbpe, tiktoken, pyarrow
print(f"python {sys.version.split()[0]} | torch {torch.__version__} cuda {torch.version.cuda} | kernels {kernels.__version__} | tiktoken {tiktoken.__version__} | pyarrow {pyarrow.__version__}")
assert sys.version_info[:2] == (3, 11) and torch.__version__ == "2.9.1+cu128"
PY

log "3a. Flash-Attention-3 kernels into $D34_HF_HOME (nanochat asks for varunneal/flash-attention-3 on Hopper)"
HF_HOME="$D34_HF_HOME" "$D34_PYTHON" - <<'PY'
import os; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from kernels import get_kernel
for repo in ("varunneal/flash-attention-3", "kernels-community/flash-attn3"):
    try:
        k = get_kernel(repo); print(" ", repo, "cached, loaded:", "flash_attn_interface" in dir(k))
    except Exception as e:  # a login node without a GPU may fail to load the .so after downloading it
        print(" ", repo, "get_kernel:", type(e).__name__, str(e)[:200])
PY
ls -d "$D34_HF_HOME"/hub/models--varunneal--flash-attention-3 >/dev/null \
  || { log "FATAL: FA3 kernel not in $D34_HF_HOME; training would fall back to SDPA and stage 3 preflight would refuse"; exit 2; }

log "3b. CORE eval bundle into $D34_BASE_DIR/eval_bundle"
(cd "$D34_EXP_DIR/upstream_snapshot" && NANOCHAT_BASE_DIR="$D34_BASE_DIR" PYTHONDONTWRITEBYTECODE=1 "$D34_PYTHON" -c "
from nanochat.common import download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
import os
if not os.path.isdir(os.path.join(os.environ['NANOCHAT_BASE_DIR'], 'eval_bundle')):
    download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)
")
ls "$D34_BASE_DIR/eval_bundle/core.yaml" >/dev/null

if [ "${SKIP_DATA:-0}" != 1 ]; then
  log "4. ClimbMix (karpathy/climbmix-400b-shuffle) into $D34_DATA_ROOT; shard 06542 is the validation shard"
  (cd "$D34_EXP_DIR/upstream_snapshot" && NANOCHAT_BASE_DIR="$D34_BASE_DIR" PYTHONDONTWRITEBYTECODE=1 \
    "$D34_PYTHON" -m nanochat.dataset -n -1 -w "${DL_WORKERS:-8}" | grep -v '^Skipping' | tail -3)
  N=$(find "$D34_DATA_ROOT" -maxdepth 1 -name 'shard_*.parquet' | wc -l)
  [ "$N" -eq 6543 ] || { log "only $N/6543 shards present; rerun setup.sh to fetch the rest"; exit 1; }
  log "  6543/6543 shards"
fi
log "setup done"
