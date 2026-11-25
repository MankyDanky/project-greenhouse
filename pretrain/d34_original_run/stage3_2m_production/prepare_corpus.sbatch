#!/bin/bash
#SBATCH --job-name=d34-corpus-sha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
# SHA256 of all 6,543 ClimbMix shards: the full-corpus manifest and receipt that stage-3 preflight requires (it then only
# checks file fingerprints, instead of rereading 558 GiB while holding 8 GPUs). CPU-only, resumable: resubmit if it times out.
# Run any time after setup.sh finished the download, e.g. while stage 1 trains. Never modify the corpus afterwards.
# submit: bash submit.sh cpu stage3_2m_production/prepare_corpus.sbatch
set -euo pipefail
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
P=$D34_PRODUCTION_CODE; M=$D34_RECEIPTS/full_corpus_manifest.json
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$D34_RECEIPTS"
[ -e "$M" ] || "$D34_PYTHON" "$P/corpus_manifest.py" create --data-dir "$D34_DATA_ROOT" --output "$M" --workers 4
# One site: `create` has just hashed these same files, so seed the verifier's resumable ledger with those hashes rather
# than reading the corpus a second time. (Across sites, verify a copied manifest from scratch instead.)
[ -e "$D34_FULL_CORPUS_RECEIPT.partial.jsonl" ] || cp "$M.partial.jsonl" "$D34_FULL_CORPUS_RECEIPT.partial.jsonl"
"$D34_PYTHON" "$P/corpus_manifest.py" verify --data-dir "$D34_DATA_ROOT" --manifest "$M" \
    --receipt "$D34_FULL_CORPUS_RECEIPT" --workers 4
sha256sum "$M"
