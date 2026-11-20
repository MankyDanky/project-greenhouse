"""Hand a CPT checkpoint to the reranker: write reranker/models/cpt-<run>/ in the layout the paper's backbones use
(model_<step>.pt + meta_<step>.json + tokenizer.json + tokenizer_config.json; no optimizer state), and print the
fine-tuning and evaluation commands.

    python export_for_reranker.py configs/train/gaggle3b_ir_r25.yaml            # the run's last checkpoint
    python export_for_reranker.py configs/train/gaggle3b_ir_r25.yaml --step 1500

The reranker recipe is the train config's `rerank_config`: the paper's run 1 for the 3B model,
configs/rerank/gaggle06b_lce.yaml for the 0.6B model. `ve_every: 2` is dropped from the meta: it is upstream's
default, so the 3B export loads with the reranker's pinned nanochat snapshot exactly as the paper's backbone does. A
model with another `ve_every` needs this folder's patched nanochat (third_party/nanochat); the printed commands then
point to it.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfgload import load_config  # noqa: E402
from train import rel  # noqa: E402

REPO = HERE.parent.parent
RERANKER = REPO / "reranker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--step", type=int, help="saved step to export (default: the last)")
    ap.add_argument("--dest", default=str(RERANKER / "models"), help="parent folder (default: reranker/models)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = cfg["run"]
    src = rel(cfg.get("base_dir", "work/nanochat_base")) / "base_checkpoints" / run
    steps = sorted(int(p.stem.split("_")[1]) for p in src.glob("model_*.pt"))
    assert steps, f"no checkpoints in {src}"
    step = a.step if a.step is not None else steps[-1]
    assert step in steps, f"step {step} not among {steps}"
    dest = Path(a.dest) / f"cpt-{run}"
    if dest.exists():
        assert not list(dest.glob("model_*.pt")) or (dest / f"model_{step:06d}.pt").exists(), \
            f"{dest} holds another step; remove it first"
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / f"meta_{step:06d}.json").read_text())
    if meta["model_config"].get("ve_every") == 2:
        del meta["model_config"]["ve_every"]
    meta["cpt"] = {"run": run, "step": step, "init": f"{Path(cfg['init']['dir']).name} step {cfg['init']['step']}",
                   "data": Path(cfg["data"]).name}
    print(f"copying {src / f'model_{step:06d}.pt'} -> {dest}")
    shutil.copy2(src / f"model_{step:06d}.pt", dest / f"model_{step:06d}.pt")
    (dest / f"meta_{step:06d}.json").write_text(json.dumps(meta, indent=2))
    init = rel(cfg["init"]["dir"])
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(init / f, dest / f)

    model_name = f"models/{dest.name}" if dest.parent.resolve() == (RERANKER / "models").resolve() else str(dest.resolve())
    tag = f"cpt_{run}"
    recipe = os.path.relpath(REPO / cfg["rerank_config"], RERANKER)
    patched = "ve_every" in meta["model_config"]        # needs the patched nanochat, in training and evaluation
    nc_src = os.path.relpath(rel(cfg.get("nanochat", "third_party/nanochat")), RERANKER)
    print(f"\nexported {run} step {step} -> {dest}\n")
    print(f"Fine-tune with {cfg['rerank_config']}, backbone swapped (from reranker/):")
    print(f"  bash launch/train.sh {recipe} 2 --override model_name={model_name} output_dir=checkpoints/{tag} "
          f"wandb_run_name={tag}" + (f" nanochat_src={nc_src}" if patched else ""))
    print("Evaluate (from the repository root):")
    print("  " + ("EVAL_NANOCHAT_SRC=" + os.path.relpath(rel(cfg.get("nanochat", "third_party/nanochat")), REPO) + " "
                  if patched else "") + f"bash eval/run_eval.sh ours reranker/checkpoints/{tag}/ckpt_final.pt {tag}")
    print(f"  python eval/report.py {tag}")


if __name__ == "__main__":
    main()
