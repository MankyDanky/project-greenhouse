"""Training-loss comparison on identical batches: a validation run (lower LR) vs the original 5e-5 run of the same
backbone, in step windows. Every run follows run G's frozen plan, so step s sees the same queries in every run.

    python tools/compare_lr.py results/trainlogs
"""
import glob, json, statistics as st, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/trainlogs")
PAIRS = [("R0_G_repro", "X_R0_G_repro_lr1.0e-5"), ("C_minicpm", "X_C_minicpm_lr1.0e-5"),
         ("C_minicpm", "X_C_minicpm_lr2.0e-5"), ("C_gemma", "X_C_gemma_lr1.0e-5")]
WIN = [(1, 100), (101, 150), (151, 200), (201, 250), (251, 300), (301, 350), (351, 391)]


def load(d):
    rows = {}
    for f in glob.glob(str(root / d / "train_*_rank0.jsonl")):
        for l in open(f):
            r = json.loads(l); rows[r["step"]] = r
    return rows


for base, exp in PAIRS:
    b, e = load(base), load(exp)
    if not e:
        print(f"{exp}: no steps yet"); continue
    print(f"== {exp} vs {base} @ 5e-5 (identical batches; loss / acc)")
    for lo, hi in WIN:
        wb = [b[s] for s in b if lo <= s <= hi]; we = [e[s] for s in e if lo <= s <= hi]
        if not we: continue
        mb = lambda k: st.mean(x[k] for x in wb); me = lambda k: st.mean(x[k] for x in we)
        print(f"   steps {lo:>3d}-{hi:<3d}  5e-5: {mb('loss'):.3f} / {mb('acc'):.3f}   new: {me('loss'):.3f} / {me('acc'):.3f}"
              f"   d_loss {me('loss') - mb('loss'):+.3f}")
