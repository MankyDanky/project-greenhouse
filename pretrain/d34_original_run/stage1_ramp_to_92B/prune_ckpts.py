#!/usr/bin/env python3
"""Checkpoint retention for the chain. Keeps model+meta+optimizer for the newest --keep-last complete steps
(and --final, and every step listed in --keep-full, e.g. the batch-ramp stage boundaries); keeps model+meta only (drops
optimizer shards) for steps that are multiples of --keep-every (must be a multiple of every stage's --save-every, or no
checkpoint ever lands on it); removes everything else.
usage: prune_ckpts.py DIR [--keep-last 3] [--keep-every 96000] [--keep-full s1,s2] [--final N] [--dry-run]"""
import argparse, os, re
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--keep-last", type=int, default=3)
ap.add_argument("--keep-every", type=int, default=96000); ap.add_argument("--keep-full", type=str, default="")
ap.add_argument("--final", type=int, default=-1); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if not os.path.isdir(a.dir): raise SystemExit(0)
files = os.listdir(a.dir)
def step_of(f):
    m = re.match(r"(?:model|meta|optim)_(\d+)", f); return int(m.group(1)) if m else None
complete = sorted({s for s in {step_of(f) for f in files} if s is not None and f"model_{s:06d}.pt" in files and f"meta_{s:06d}.json" in files})
keep_full = set(complete[-a.keep_last:]) | ({a.final} if a.final >= 0 else set()) | {int(x) for x in a.keep_full.split(",") if x.strip()}
keep_model = {s for s in complete if s % a.keep_every == 0}
removed = 0
for f in files:
    s = step_of(f)
    if s is None or s in keep_full: continue
    is_optim = f.startswith("optim_")
    if s in keep_model and not is_optim: continue
    p = os.path.join(a.dir, f); print(("would remove " if a.dry_run else "removing ") + p); removed += 1
    if not a.dry_run: os.remove(p)
print(f"kept full: {sorted(keep_full)}  model-only milestones: {sorted(keep_model - keep_full)}  removed {removed} files")
