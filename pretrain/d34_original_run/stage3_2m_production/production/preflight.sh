#!/bin/bash
set -euo pipefail
if [[ -n "${D34_ENV_SETUP:-}" ]]; then source "$D34_ENV_SETUP"; fi
export PYTHONPATH="$D34_EXP_DIR/upstream_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export NANOCHAT_BASE_DIR="$D34_BASE_DIR"
export HF_HOME="$D34_HF_HOME" HF_HUB_OFFLINE=1
mkdir -p "$D34_PROD_DIR/preflight/$SLURM_JOB_ID"
"$D34_PYTHON" "$D34_FLEET_DIR/verify_assets.py" \
    --manifest "$D34_ASSET_MANIFEST" --receipt "$D34_ASSET_RECEIPT" \
    --provenance "$D34_PROD_DIR/preflight/$SLURM_JOB_ID/static_assets.json"
"$D34_PYTHON" "$D34_PRODUCTION_CODE/production_preflight.py"
