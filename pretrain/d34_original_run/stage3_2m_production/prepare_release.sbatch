#!/bin/bash
#SBATCH --job-name=d34-release
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --mail-type=NONE
# After stage 2 and prepare_corpus.sbatch: write everything production_leg.sbatch checks before it trains.
#   code_manifest.json      SHA256 of the frozen production code ($D34_PRODUCTION_CODE)
#   winner_manifest.json    SHA256 of the ten stage-2 checkpoint files, bound to its COMPLETE.json
#   asset_manifest.json     static assets: frozen sources, the step-136,316 checkpoint, tokenizer, the stage-2 shards
#   verified_assets.json    receipt for the above (so the first production leg does not hash them on a GPU node)
#   release.json            the release record that binds all of these by SHA256
# CPU-only. submit: bash submit.sh cpu stage3_2m_production/prepare_release.sbatch
set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
export D34_EXP_DIR D34_BASE_DIR D34_CHECKPOINT_DIR D34_DATA_ROOT D34_HF_HOME D34_RUNTIME_ROOT PYTHONDONTWRITEBYTECODE=1
P=$D34_PRODUCTION_CODE; RC=$D34_RECEIPTS
CORPUS=$RC/full_corpus_manifest.json; WINNER=$RC/winner_manifest.json
[ -f "$D34_FULL_CORPUS_RECEIPT" ] || { echo "run prepare_corpus.sbatch first"; exit 2; }
[ -f "$D34_WINNER_COMPLETION" ] || { echo "no stage-2 COMPLETE.json at $D34_WINNER_COMPLETION"; exit 2; }
[ ! -e "$D34_PROD_RELEASE" ] || { echo "$D34_PROD_RELEASE exists; a release is written once"; exit 2; }
"$D34_PYTHON" "$P/freeze_code.py"
[ -e "$WINNER" ] || "$D34_PYTHON" "$P/winner_manifest.py" create --checkpoint-dir "$D34_WINNER_CHECKPOINT_DIR" \
    --step "$D34_WINNER_STEP" --completion "$D34_WINNER_COMPLETION" --output "$WINNER"
[ -e "$D34_ASSET_MANIFEST" ] || "$D34_PYTHON" "$D34_FLEET_DIR/make_asset_manifest.py" --output "$D34_ASSET_MANIFEST"
"$D34_PYTHON" "$D34_FLEET_DIR/verify_assets.py" --manifest "$D34_ASSET_MANIFEST" --receipt "$D34_ASSET_RECEIPT"
sha256sum "$D34_WINNER_COMPLETION" | cut -d' ' -f1 > "$RC/winner_completion.sha256"
"$D34_PYTHON" - "$D34_PROD_RELEASE" "$WINNER" "$D34_WINNER_COMPLETION" "$P/code_manifest.json" "$CORPUS" \
    "$D34_WINNER_CHECKPOINT_DIR" "$D34_WINNER_STEP" "$D34_WINNER_BATCH" "$D34_WINNER_LR" "$D34_STAGE2_TAG" "$D34_PROD_TAG" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, winner, completion, code, corpus, winner_dir, step, batch, lr, tag, prod_tag = sys.argv[1:]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
release = {
    "status": "approved_for_production",
    # Literal gate string of the original run, where arm 3 won a four-arm screen; this recipe runs only arm 3.
    "validation_status": "all_four_arms_complete_and_comparable",
    "decision": "Recipe reproduction of the released configuration: arm 3, batch 2,097,152, actual Muon LR 0.01414213562373095",
    "selected_arm": 3, "selected_tag": tag,
    "actual_checkpoint_step": int(step), "canonical_step": 138364,
    "total_batch_size": int(batch), "actual_matrix_lr": float(lr),
    "winner_checkpoint_dir": winner_dir,
    "winner_checkpoint_manifest": winner, "winner_checkpoint_manifest_sha256": sha(winner),
    "winner_completion": completion, "winner_completion_sha256": sha(completion),
    "production_code_manifest_sha256": sha(code),
    "full_corpus_manifest": corpus, "full_corpus_manifest_sha256": sha(corpus),
    "production_tag": prod_tag,
}
Path(out).write_text(json.dumps(release, indent=2) + "\n")
print(json.dumps(release, indent=2))
PY
