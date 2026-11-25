#!/bin/bash
# One-time setup on a login node (needs internet; compute nodes may run offline). Idempotent; rerun to resume.
#   1. working copy of nanochat in $POST_NANOCHAT: ../pretrain/nanochat (nanochat 92d63d4) + nanochat_overlay/
#   2. Python 3.11 env from nanochat's uv.lock (torch 2.9.1+cu128) at $POST_VENV
#   3. Flash-Attention-3 kernels into $POST_HF_HOME
#   4. inputs in $POST_BASE_DIR: the released base checkpoint and tokenizer (SHA256-checked), nanochat's identity
#      conversations, and every dataset the recipe and chat_eval read (row counts checked)
# usage: POST_CONFIG=/abs/path/to/config.sh bash setup.sh
set -euo pipefail
source "${POST_CONFIG:?export POST_CONFIG=/absolute/path/to/config.sh}"
log(){ echo "[$(date '+%F %T')] $*"; }
case $POST_ROOT in
  /path/to/*) log "FATAL: set POST_ROOT in $POST_CONFIG"; exit 2 ;;
  /*) ;;
  *) log "FATAL: POST_ROOT must be an absolute path"; exit 2 ;;
esac
command -v uv >/dev/null || { log "FATAL: uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 2; }
mkdir -p "$POST_NANOCHAT" "$POST_BASE_DIR/base_checkpoints/$POST_TAG" "$POST_HF_HOME" "$POST_LOGS"

log "1. nanochat working copy in $POST_NANOCHAT"
rsync -a --exclude __pycache__ --exclude .venv "$POST_RECIPE/../pretrain/nanochat/" "$POST_NANOCHAT/"
rsync -a "$POST_RECIPE/nanochat_overlay/" "$POST_NANOCHAT/"
grep -q '"mid": "mid_checkpoints"' "$POST_NANOCHAT/nanochat/checkpoint_manager.py"

log "2. Python env $POST_VENV"
export UV_PROJECT_ENVIRONMENT="$POST_VENV" UV_CACHE_DIR="${UV_CACHE_DIR:-$POST_ROOT/uv_cache}"
(cd "$POST_NANOCHAT" && uv sync --extra gpu --frozen --python 3.11 2>&1 | tail -3)
"$POST_PYTHON" - <<'PY'
import sys, torch, kernels, rustbpe, tiktoken, pyarrow, huggingface_hub
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | kernels {kernels.__version__} | hub {huggingface_hub.__version__}")
assert sys.version_info[:2] == (3, 11) and torch.__version__ == "2.9.1+cu128"
PY

log "3. Flash-Attention-3 kernels into $POST_HF_HOME (nanochat asks for varunneal/flash-attention-3 on Hopper)"
HF_HOME="$POST_HF_HOME" "$POST_PYTHON" - <<'PY'
import os; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from kernels import get_kernel
for repo in ("varunneal/flash-attention-3", "kernels-community/flash-attn3"):
    try:
        k = get_kernel(repo); print(" ", repo, "cached, loaded:", "flash_attn_interface" in dir(k))
    except Exception as e:  # a login node without a GPU may fail to load the .so after downloading it
        print(" ", repo, "get_kernel:", type(e).__name__, str(e)[:200])
PY
ls -d "$POST_HF_HOME"/hub/models--varunneal--flash-attention-3 >/dev/null \
  || { log "FATAL: FA3 kernel not in $POST_HF_HOME; training would fall back to SDPA (slower, not the original kernel)"; exit 2; }

log "4a. base checkpoint + tokenizer ($POST_BASE_REPO, step $POST_BASE_STEP)"
HF_HOME="$POST_HF_HOME" HF_HUB_DISABLE_PROGRESS_BARS=1 NANOCHAT_BASE_DIR="$POST_BASE_DIR" POST_TAG="$POST_TAG" \
  POST_BASE_REPO="$POST_BASE_REPO" POST_BASE_STEP="$POST_BASE_STEP" \
  POST_BASE_MODEL_SHA256="$POST_BASE_MODEL_SHA256" POST_BASE_META_SHA256="$POST_BASE_META_SHA256" \
  "$POST_PYTHON" - <<'PY'
import hashlib, os
from huggingface_hub import snapshot_download
b, tag, step = os.environ["NANOCHAT_BASE_DIR"], os.environ["POST_TAG"], os.environ["POST_BASE_STEP"]
p = snapshot_download(os.environ["POST_BASE_REPO"],
                      allow_patterns=[f"model_{step}.pt", f"meta_{step}.json", "tokenizer/*"])
links = {f"base_checkpoints/{tag}/model_{step}.pt": f"model_{step}.pt",
         f"base_checkpoints/{tag}/meta_{step}.json": f"meta_{step}.json", "tokenizer": "tokenizer"}
for dst, src in links.items():
    d = os.path.join(b, dst)
    if os.path.lexists(d):
        os.remove(d)
    os.symlink(os.path.join(p, src), d)
expected = {f"base_checkpoints/{tag}/model_{step}.pt": os.environ["POST_BASE_MODEL_SHA256"],
            f"base_checkpoints/{tag}/meta_{step}.json": os.environ["POST_BASE_META_SHA256"],
            "tokenizer/tokenizer.pkl": "ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2"}  # every Gaggle model's
for f, want in expected.items():
    h = hashlib.sha256()
    with open(os.path.join(b, f), "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    assert h.hexdigest() == want, f"{f}: sha256 {h.hexdigest()} != {want}"
    print("  sha256 ok", f)
PY

log "4b. identity conversations (nanochat's, 1,000 rows)"
[ -s "$POST_BASE_DIR/identity_conversations.jsonl" ] || curl -sfL -o "$POST_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
[ "$(wc -l < "$POST_BASE_DIR/identity_conversations.jsonl")" -eq 1000 ]

log "4c. task data into $POST_BASE_DIR/task_data and the SpellingBee word list (what the recipe and chat_eval read)"
(cd "$POST_NANOCHAT" && NANOCHAT_BASE_DIR="$POST_BASE_DIR" PYTHONDONTWRITEBYTECODE=1 "$POST_PYTHON" - <<'PY'
from tasks.smoltalk import SmolTalk; from tasks.mmlu import MMLU; from tasks.gsm8k import GSM8K; from tasks.arc import ARC
from tasks.humaneval import HumanEval; from tasks.mmlu_pro import MMLUPro; from tasks.spellingbee import SimpleSpelling
rows = {"smol-smoltalk train": len(SmolTalk(split="train")), "smol-smoltalk test": len(SmolTalk(split="test")),
        "MMLU auxiliary_train": len(MMLU(subset="all", split="auxiliary_train")), "MMLU test": len(MMLU(subset="all", split="test")),
        "GSM8K train": len(GSM8K(subset="main", split="train")), "GSM8K test": len(GSM8K(subset="main", split="test")),
        "ARC-Easy train": len(ARC(subset="ARC-Easy", split="train")), "ARC-Easy test": len(ARC(subset="ARC-Easy", split="test")),
        "ARC-Challenge train": len(ARC(subset="ARC-Challenge", split="train")),
        "ARC-Challenge test": len(ARC(subset="ARC-Challenge", split="test")),
        "HumanEval": len(HumanEval()), "MMLU-Pro test": len(MMLUPro(split="test"))}
SimpleSpelling(size=1, split="train")   # downloads words_alpha.txt
want = {"smol-smoltalk train": 460341, "smol-smoltalk test": 24229, "MMLU auxiliary_train": 99842, "MMLU test": 14042,
        "GSM8K train": 7473, "GSM8K test": 1319, "ARC-Easy train": 2251, "ARC-Easy test": 2376, "ARC-Challenge train": 1119,
        "ARC-Challenge test": 1172, "HumanEval": 164, "MMLU-Pro test": 12032}
bad = {k: (rows[k], v) for k, v in want.items() if rows[k] != v}
assert not bad, f"row counts differ (got, want): {bad}"
print("  row counts as in the original run:", rows)
PY
)
log "setup done"
