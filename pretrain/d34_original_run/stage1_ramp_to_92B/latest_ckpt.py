#!/usr/bin/env python3
"""Print the latest COMPLETE checkpoint step in a nanochat base_checkpoints dir, or -1.
Complete = model_S.pt + meta_S.json (valid JSON with step==S) + optim_S_rank{0..W-1}.pt, all non-empty,
none modified in the last MIN_AGE s (default 120; 0 = no age check). usage: latest_ckpt.py DIR [WORLD=8] [SKIP=s1,s2] [MIN_AGE=120]"""
import json, os, re, sys, time
d = sys.argv[1]; world = int(sys.argv[2]) if len(sys.argv) > 2 else 8
skip = {int(x) for x in sys.argv[3].split(",") if x} if len(sys.argv) > 3 else set()
min_age = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
steps = []
if os.path.isdir(d):
    steps = sorted({int(m.group(1)) for f in os.listdir(d) if (m := re.match(r"model_(\d+)\.pt$", f))}, reverse=True)
now = time.time()
for s in steps:
    if s in skip:
        continue
    names = [f"model_{s:06d}.pt", f"meta_{s:06d}.json"] + [f"optim_{s:06d}_rank{r}.pt" for r in range(world)]
    paths = [os.path.join(d, n) for n in names]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths):
        continue
    if min_age > 0 and any(now - os.path.getmtime(p) < min_age for p in paths):
        continue
    try:
        if json.load(open(paths[1])).get("step") != s:
            continue
    except Exception:
        continue
    print(s); sys.exit(0)
print(-1)
