#!/usr/bin/env bash
# Build the CPT trainer in third_party/nanochat: karpathy/nanochat at 92d63d4 (the commit the Gaggle models were
# pre-trained from) plus nanochat_cpt.patch. Idempotent.
#
#   bash setup.sh          # clone + patch
#   bash setup.sh --env    # also create the training environment (uv): nanochat's pinned GPU stack + pyyaml
# Data preparation runs in its own environment: pip install -r requirements-data.txt (plus Java 21 for pyserini).
#
# What nanochat_cpt.patch changes (every hunk is commented in the code):
#   nanochat/gpt.py              `ve_every` in GPTConfig: value embeddings on every k-th layer. 2 (the default) is
#                                upstream's alternating rule, so the 3B model is unaffected; smaller Gaggle models
#                                may use another spacing.
#   nanochat/batch_schedule.py   token-based batch-size ramp (used in pre-training; CPT runs a fixed batch)
#   nanochat/flash_attention.py  FA3 only on sm80/sm90 (it crashes on Blackwell); NANOCHAT_ATTN=fa3|sdpa overrides
#   nanochat/dataset.py          NANOCHAT_DATA_DIR selects the shard directory (default: upstream's ClimbMix dir)
#   scripts/base_train.py        --init-from-dir/--init-from-step (start from released weights, fresh optimizer),
#                                --literal-hparams (use the given LRs and weight decay without nanochat's batch-size
#                                and horizon rescaling), --ve-every, --batch-schedule, --keep-checkpoints,
#                                --auto-resume
#   scripts/chat_sft.py          records ve_every in SFT checkpoints
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/nanochat"
COMMIT=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
PATCH="$HERE/nanochat_cpt.patch"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$HERE/third_party"
  git clone --quiet https://github.com/karpathy/nanochat.git "$DEST"
fi
cd "$DEST"
if git apply -p1 --reverse --check "$PATCH" 2>/dev/null; then
  echo "third_party/nanochat: already patched ($(git rev-parse --short HEAD))"
else
  [ -z "$(git status --porcelain)" ] || { echo "third_party/nanochat has local changes; remove it and re-run" >&2; exit 1; }
  git checkout --quiet "$COMMIT"
  git apply -p1 "$PATCH"
  echo "third_party/nanochat: $COMMIT + nanochat_cpt.patch"
fi

if [ "${1:-}" = "--env" ]; then
  command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
  uv sync --extra gpu                                    # creates third_party/nanochat/.venv (torch, kernels, ...)
  uv pip install --python .venv/bin/python pyyaml
  echo "activate with: source $DEST/.venv/bin/activate"
fi
