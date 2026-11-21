#!/usr/bin/env bash
# Build the continued-pretraining trainer in third_party/nanochat: karpathy/nanochat at 92d63d4 (the commit the
# Gaggle models were pre-trained from) plus nanochat_continue.patch. Idempotent.
#
#   bash setup.sh          # clone + patch
#   bash setup.sh --env    # also create its Python environment (uv): nanochat's pinned GPU stack + pyyaml
#
# What nanochat_continue.patch changes (every hunk is marked "(continue)" or commented in the code):
#   scripts/base_train.py        --config FILE.yaml (flag values from a file; the command line overrides it);
#                                --init-from-dir/--init-from-step (start from a checkpoint's weights, fresh optimizer;
#                                its architecture flags are checked against its meta, and its tokenizer is used and
#                                copied into the run's checkpoint dir); --literal-hparams (LRs and weight decay as
#                                given, without nanochat's batch-size and horizon rescaling); --data-dir;
#                                --tokenizer-dir; --ve-every; --batch-schedule; --keep-checkpoints; --auto-resume.
#                                ve_every is written to the meta only when it is not 2 (upstream's format otherwise).
#   scripts/base_eval.py         --checkpoint-dir (score any checkpoint dir, e.g. a released one), --tokenizer-dir
#   nanochat/gpt.py              `ve_every` in GPTConfig: value embeddings on every k-th layer (2 = upstream's rule)
#   nanochat/tokenizer.py        NANOCHAT_TOKENIZER_DIR selects the tokenizer
#   nanochat/dataset.py          NANOCHAT_DATA_DIR selects the shard directory
#   nanochat/batch_schedule.py   token-based batch-size ramp (used in pre-training)
#   nanochat/flash_attention.py  FA3 only on sm80/sm90 (it crashes on Blackwell); NANOCHAT_ATTN=fa3|sdpa overrides
#   scripts/chat_sft.py          records ve_every in SFT checkpoints
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/nanochat"
COMMIT=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
PATCH="$HERE/nanochat_continue.patch"

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
  echo "third_party/nanochat: $COMMIT + nanochat_continue.patch"
fi

if [ "${1:-}" = "--env" ]; then
  command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
  uv sync --extra gpu                                    # creates third_party/nanochat/.venv (torch, kernels, ...)
  uv pip install --python .venv/bin/python pyyaml
  echo "activate with: source $DEST/.venv/bin/activate"
fi
