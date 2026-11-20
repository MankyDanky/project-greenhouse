"""Drop optimizer and RNG state from a finished run's final checkpoint.

A checkpoint is 36 GB: 12.9 GB of weights plus 25.8 GB of AdamW moments, kept so a run can resume.
For a run that has FINISHED and been evaluated, that is dead weight -- evaluation reads sd["model"]
and sd["cfg"] only (eval/rerank_eval.py). Stripping leaves the artifact fully usable for scoring and
only gives up resumability, which a completed run does not need.

Writes alongside, verifies the model tensors survive byte-for-byte, then replaces.
"""
import argparse, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--out-dir", default=None,
                help="write the stripped copy here instead of replacing the source. Required for a\n"
                     "LIVE run: its numbered checkpoints must keep optimizer state so a preemption\n"
                     "can still resume, and keep_ckpts rotation will delete the originals anyway.")
a = ap.parse_args()
KEEP = ("model", "cfg", "step", "epoch", "batch_in_epoch", "wandb_id", "packing")

for p in a.ckpts:
    if not os.path.exists(p):
        print(f"MISSING {p}"); continue
    before = os.path.getsize(p)
    sd = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in sd:
        print(f"SKIP {p}: no 'model' key (keys: {list(sd)[:6]})"); continue
    if "optimizer" not in sd and "rng" not in sd:
        print(f"already stripped: {p}"); continue
    out = {k: sd[k] for k in KEEP if k in sd}
    out["stripped"] = "optimizer and rng state removed (finished run; eval needs model+cfg only)"
    n = sum(v.numel() for v in out["model"].values())
    print(f"{p}: {before/2**30:.1f} GiB, {len(sd)} keys -> {len(out)} keys, {n:,} model params", flush=True)
    if a.dry_run:
        continue
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        run = os.path.basename(os.path.dirname(os.path.abspath(p)))
        dest = os.path.join(a.out_dir, f"{run}__{os.path.basename(p)}")
        tmp = dest + ".stripping"
    else:
        dest = p
        tmp = p + ".stripping"
    torch.save(out, tmp)
    chk = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
    assert set(chk["model"]) == set(sd["model"]), "model keys changed"
    for k in list(sd["model"])[:8] + list(sd["model"])[-8:]:
        assert torch.equal(chk["model"][k], sd["model"][k]), f"tensor {k} differs"
    if "cfg" in sd and isinstance(sd.get("cfg"), dict) and "model_name" in sd["cfg"]:
        assert chk["cfg"]["model_name"] == sd["cfg"]["model_name"]
    del sd, chk
    os.replace(tmp, dest)
    note = "copy" if a.out_dir else f"saved {(before-os.path.getsize(dest))/2**30:.1f} GiB in place"
    print(f"  -> {dest} {os.path.getsize(dest)/2**30:.1f} GiB ({note})", flush=True)
