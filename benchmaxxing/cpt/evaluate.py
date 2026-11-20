"""Language-model evaluation of a CPT run, or of the checkpoint it started from, on the mix's evaluation sets. Give
the train config with the same overrides the run was trained with.

    python evaluate.py configs/train.yaml --init                    # the starting checkpoint (the baseline row)
    python evaluate.py configs/train.yaml                           # the run's last checkpoint
    python evaluate.py configs/train.yaml --set model=gaggle06b --step 1000 --core

Bits per byte (nanochat's tokenizer-independent loss) on the three sets prepare_data.py writes:
  heldout_corpus   corpus pieces never trained on              lower after CPT = gain on the domain
  seen_corpus      as many trained-on pieces, per source       its drop beyond heldout_corpus's = memorisation
  heldout_replay   pre-training validation documents           should stay flat; a rise = forgetting
The sets come from `eval_data` in the train config (default: the run's own data). Each document is scored exactly once (BOS + text, in windows of the model's context,
never packed with other documents), so the numbers do not depend on batch size or GPU count.
--core also runs nanochat's CORE benchmark (scripts/base_eval.py --eval core), the general-capability check.
Writes results/<run>/<init|step_N>.json.
"""
import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import prepare_base_dir, rel, resolve, visible_gpus  # noqa: E402

SETS = ("heldout_corpus", "seen_corpus", "heldout_replay")


def resolve_checkpoint(cfg, a, base):
    if a.init:
        src = rel(cfg["init"]["dir"]); tag = f"init_{src.name}"; step = int(cfg["init"]["step"])
        link = base / "base_checkpoints" / tag
        if not (link.exists() or link.is_symlink()):
            link.symlink_to(src, target_is_directory=True)
        return tag, step, "init"
    tag = cfg["run"]
    steps = sorted(int(m.group(1)) for p in (base / "base_checkpoints" / tag).glob("model_*.pt")
                   if (m := re.fullmatch(r"model_(\d+)\.pt", p.name)))
    assert steps, f"no checkpoints of {tag} in {base / 'base_checkpoints' / tag}"
    step = a.step if a.step is not None else steps[-1]
    assert step in steps, f"step {step} not among {steps}"
    return tag, step, f"step_{step}"


def bpb_of(model, tokenizer, token_bytes, texts, seq_len, batch_rows, device):
    import torch
    bos = tokenizer.get_bos_token_id()
    windows = []
    for ids in tokenizer.encode(texts, prepend=bos):
        for s in range(0, len(ids) - 1, seq_len):
            windows.append(ids[s:s + seq_len + 1])
    windows.sort(key=len)
    nats = 0.0; nbytes = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch_rows):
            chunk = windows[i:i + batch_rows]
            T = max(len(w) for w in chunk) - 1
            x = torch.full((len(chunk), T), bos, dtype=torch.long)
            y = torch.full((len(chunk), T), -1, dtype=torch.long)
            for r, w in enumerate(chunk):
                x[r, :len(w) - 1] = torch.tensor(w[:-1]); y[r, :len(w) - 1] = torch.tensor(w[1:])
            x, y = x.to(device), y.to(device)
            loss = model(x, y, loss_reduction="none").view(-1).float()
            y = y.view(-1)
            b = torch.where(y >= 0, token_bytes[y.clamp(min=0)], torch.zeros_like(y, dtype=token_bytes.dtype))
            nats += float((loss * (b > 0)).sum()); nbytes += int(b.sum())
    return nats / (math.log(2) * nbytes), nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--init", action="store_true", help="evaluate the starting checkpoint instead of the run")
    ap.add_argument("--step", type=int, help="a specific saved step of the run (default: the last)")
    ap.add_argument("--core", action="store_true", help="also run CORE (downloads nanochat's eval bundle once)")
    ap.add_argument("--batch-rows", type=int, default=8, help="sequences per forward pass")
    ap.add_argument("--ngpu", type=int, default=0, help="GPUs for CORE (bpb runs on one device)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="the overrides the run was trained with")
    a = ap.parse_args()
    cfg = resolve(a.config, a.set)
    nc, base = prepare_base_dir(cfg)
    tag, step, label = resolve_checkpoint(cfg, a, base)
    os.environ["NANOCHAT_BASE_DIR"] = str(base)
    sys.path.insert(0, str(nc))
    import torch
    import pyarrow.parquet as pq
    from nanochat.checkpoint_manager import build_model
    from nanochat.tokenizer import get_token_bytes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, meta = build_model(str(base / "base_checkpoints" / tag), step, device, phase="eval")
    token_bytes = get_token_bytes(device=device)
    seq_len = meta["model_config"]["sequence_len"]
    data = rel(cfg.get("eval_data") or cfg["data"])
    out = {"run": cfg["run"], "checkpoint": f"{tag} step {step}", "eval_data": data.name, "bpb": {}, "bytes": {}}
    for name in SETS:
        files = sorted((data / name).glob("*.parquet"))
        if not files:
            continue
        texts = [t for f in files for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()]
        t0 = time.time()
        out["bpb"][name], out["bytes"][name] = bpb_of(model, tokenizer, token_bytes, texts, seq_len, a.batch_rows, device)
        print(f"{name}: bpb {out['bpb'][name]:.5f} over {len(texts):,} docs ({time.time() - t0:.0f}s)", flush=True)
    if "heldout_corpus" in out["bpb"] and "seen_corpus" in out["bpb"]:
        out["bpb"]["seen_minus_heldout"] = out["bpb"]["seen_corpus"] - out["bpb"]["heldout_corpus"]
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if a.core:
        ngpu = a.ngpu or visible_gpus() or 1
        launcher = ["torchrun", "--standalone", f"--nproc_per_node={ngpu}"] if ngpu > 1 else [sys.executable]
        cmd = [*launcher, "-m", "scripts.base_eval", "--eval", "core", "--model-tag", tag, "--step", str(step)]
        res = subprocess.run(cmd, cwd=nc, env=dict(os.environ, NANOCHAT_BASE_DIR=str(base)),
                             capture_output=True, text=True)
        sys.stdout.write(res.stdout[-3000:]); sys.stderr.write(res.stderr[-3000:])
        m = re.search(r"CORE metric: ([0-9.]+)", res.stdout)
        assert res.returncode == 0 and m, "CORE evaluation failed"
        out["core"] = float(m.group(1))
    dest = HERE / "results" / cfg["run"] / f"{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["bpb"] | ({"core": out["core"]} if "core" in out else {}), indent=2), f"\n-> {dest}")


if __name__ == "__main__":
    main()
