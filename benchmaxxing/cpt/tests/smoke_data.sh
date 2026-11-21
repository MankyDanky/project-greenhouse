#!/usr/bin/env bash
# CPU smoke test of prepare_data.py (a minute): the smoke mix, then the exclusion and replay-only options through
# --set overrides. Run in the data environment (PYTHON=... to choose the interpreter).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
T=tests/smoke_data.yaml
$PY prepare_data.py $T --limit 3000                                                      # groups, a partial pass
$PY prepare_data.py $T --limit 3000 --set name=smoke_minus exclude_sources=[scifact]    # nfcorpus contributes the same
$PY prepare_data.py $T --limit 3000 --set name=smoke_no_ood exclude_ood=true            # nfcorpus (ood) never read
$PY prepare_data.py $T --limit 3000 --set name=smoke_control corpus=[] replay.tokens=from:data/smoke   # replay only
$PY - <<'PYEOF'
import json
m = {n: json.load(open(f"data/{n}/manifest.json")) for n in ("smoke", "smoke_minus", "smoke_no_ood", "smoke_control")}
nf = lambda n: m[n]["sources"].get("nfcorpus", {}).get("tokens_in_mix")
assert nf("smoke_minus") == nf("smoke") and "scifact" not in m["smoke_minus"]["sources"], "exclude_sources"
assert list(m["smoke_no_ood"]["sources"]) == ["scifact"] and m["smoke_no_ood"]["sources"]["scifact"]["passes"] == 1
assert m["smoke_control"]["totals"]["replay_share"] == 1.0 and not m["smoke_control"]["sources"]
assert abs(m["smoke"]["totals"]["replay_share"] - 0.25) < 0.01
print("smoke_data: OK")
PYEOF
