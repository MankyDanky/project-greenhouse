"""Continued pretraining (CPT) from a released Gaggle checkpoint, configured by one YAML file (see configs/train/).

    python train.py configs/train/gaggle3b_ir_r25.yaml --check        # print the derived settings, validate, exit
    python train.py configs/train/gaggle3b_ir_r25.yaml --ngpu 8       # run (torchrun on this node; 1 GPU: plain python)

It drives nanochat's own trainer, `scripts/base_train.py` at karpathy/nanochat@92d63d4 plus `nanochat_cpt.patch`
(setup.sh builds it in third_party/nanochat), with three CPT options the patch adds:
  --init-from-dir / --init-from-step   start from the released weights (architecture checked against their meta),
                                       fresh optimizer; a preempted run resumes from its own checkpoints instead
  --literal-hparams                    learning rates and weight decay are used as given (nanochat otherwise rescales
                                       them for batch size and training horizon, which is wrong for a short CPT)
  NANOCHAT_DATA_DIR                    the shard directory written by prepare_data.py

What this file derives from the YAML and the checkpoint's meta:
  * architecture flags (depth, width via aspect ratio, heads, value-embedding spacing, window pattern, context);
  * learning rates: the original run's effective peak LRs (its configured LRs times the batch-size factor it ran
    with) times `lr.scale`;
  * steps: `tokens` / `batch_tokens`, by default 95% of the mix's training tokens, so the run never wraps into a
    second pass (nanochat's loader cycles and its best-fit packing crops a few percent of tokens);
  * warmup steps and the final linear decay from `lr.warmup_frac` / `lr.decay_frac`.
Every key under `extra_args` must be a real flag of the trainer and is passed through unchanged.
"""
import argparse, json, math, os, re, shlex, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfgload import load_config  # noqa: E402

B_REF = 2 ** 19          # nanochat's reference batch (524,288 tokens) for its sqrt(B) LR scaling
LR_KEYS = ("matrix_lr", "embedding_lr", "unembedding_lr", "scalar_lr")


def rel(p):
    return (HERE / p).resolve() if not os.path.isabs(str(p)) else Path(p)


def original_lr_factor(meta):
    """The batch-size factor the original run applied to its configured LRs (constant for the whole run)."""
    uc = meta["user_config"]
    prod = meta.get("production") or {}
    if prod.get("actual_matrix_lr"):
        return prod["actual_matrix_lr"] / uc["matrix_lr"]
    if uc.get("batch_schedule"):
        first = int(str(uc["batch_schedule"]).split(",")[0].split(":")[0])
        return math.sqrt(first / B_REF)
    return math.sqrt(int(uc["total_batch_size"]) / B_REF)


def require(cfg):
    """Placeholders left empty (a model not released yet) stop the run with a clear message."""
    missing = [k for k, v in (("init.dir", cfg["init"].get("dir")), ("init.step", cfg["init"].get("step")),
                              ("weight_decay", cfg.get("weight_decay"))) if v in (None, "")]
    if missing:
        sys.exit(f"{cfg['run']}: fill in {', '.join(missing)} in the config (the starting model may not be released yet)")
    init = rel(cfg["init"]["dir"])
    if not (init / f"meta_{int(cfg['init']['step']):06d}.json").exists():
        sys.exit(f"{init}/meta_{int(cfg['init']['step']):06d}.json not found: run fetch_models.sh")


def derive(cfg):
    require(cfg)
    init = rel(cfg["init"]["dir"]); step = int(cfg["init"]["step"])
    meta = json.loads((init / f"meta_{step:06d}.json").read_text())
    mc = meta["model_config"]
    head_dim = mc["n_embd"] // mc["n_head"]
    ar = mc["n_embd"] // mc["n_layer"]
    assert ((mc["n_layer"] * ar + head_dim - 1) // head_dim) * head_dim == mc["n_embd"], "cannot express width"
    data = rel(cfg["data"])
    man = json.loads((data / "manifest.json").read_text())
    tokens = int(cfg.get("tokens") or 0.95 * man["totals"]["train_tokens"])
    B = int(cfg["batch_tokens"])
    steps = tokens // B
    lr = cfg["lr"]; f = original_lr_factor(meta)
    lrs = {k: meta["user_config"][k] * f * float(lr["scale"]) for k in LR_KEYS}
    flags = {
        "run": cfg["run"] if cfg.get("wandb") else "dummy", "model-tag": cfg["run"],   # "dummy": no W&B
        "init-from-dir": str(init), "init-from-step": step, "literal-hparams": True, "auto-resume": True,
        "depth": mc["n_layer"], "aspect-ratio": ar, "head-dim": head_dim, "ve-every": mc.get("ve_every", 2),
        "window-pattern": mc["window_pattern"], "max-seq-len": mc["sequence_len"],
        "num-iterations": steps, "total-batch-size": B, "device-batch-size": int(cfg["device_batch_size"]),
        **{k.replace("_", "-"): round(v, 8) for k, v in lrs.items()},
        "weight-decay": float(cfg["weight_decay"]),
        "warmup-steps": max(1, round(float(lr["warmup_frac"]) * steps)),
        "warmdown-ratio": float(lr["decay_frac"]), "final-lr-frac": float(lr["final_frac"]),
        "eval-every": int(cfg.get("eval_every", 250)), "eval-tokens": int(cfg.get("eval_tokens", 20 * 2 ** 20)),
        "core-metric-every": int(cfg.get("core_metric_every", -1)), "sample-every": -1,
        "save-every": int(cfg.get("save_every", 500)), "keep-checkpoints": int(cfg.get("keep_checkpoints", 2)),
    }
    if cfg.get("fp8"):
        flags["fp8"] = True
    flags.update(cfg.get("extra_args") or {})
    info = {"init": f"{init.name} step {step}", "arch": mc, "original_lr_factor": round(f, 6),
            "original_peak_lrs": {k: round(meta["user_config"][k] * f, 6) for k in LR_KEYS},
            "mix": f"{data.name}: {man['totals']['train_tokens']:,} tokens, replay share {man['totals']['replay_share']:.3f}",
            "tokens": tokens, "steps": steps}
    # what the starting model's data loader had consumed at release: replay from later shards is unseen
    # pre-training-distribution text rather than true replay
    states = [x["state"] for x in (meta.get("production") or {}).get("loader_states", [])] or \
             [meta.get("dataloader_state_dict") or {}]
    seen_upto = min((x.get("pq_idx", -1) for x in states), default=-1)
    info["loader_shard_at_release"] = seen_upto
    info["_replay_shards"] = man["config"]["replay"]["source"]["shards"]
    info["_tokenizer_json"] = str(HERE / man["config"]["tokenizer"])
    return flags, info


def visible_gpus():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
    except FileNotFoundError:
        return 0


def to_argv(flags):
    argv = []
    for k, v in flags.items():
        if v is True:
            argv.append(f"--{k}")
        elif v is not False and v is not None:
            argv += [f"--{k}", str(v)]
    return argv


def check(cfg, flags, ngpu, info):
    nc = rel(cfg.get("nanochat", "third_party/nanochat"))
    src = nc / "scripts" / "base_train.py"
    errors = []
    mine, theirs = rel(cfg["init"]["dir"]) / "tokenizer.json", Path(info.pop("_tokenizer_json"))
    if mine.exists() and theirs.exists() and mine.read_bytes() != theirs.read_bytes():
        errors.append(f"{mine} differs from the tokenizer the mix was measured with ({theirs}): the replay share "
                      f"and token counts would be wrong; rebuild the mix with this model's tokenizer")
    hi = info.pop("_replay_shards")[1]
    if 0 <= info["loader_shard_at_release"] < hi:
        print(f"WARNING: this model's data loader was at shard {info['loader_shard_at_release']} at release, but replay "
              f"draws from shards up to {hi}: part of the replay is pre-training-distribution text it never saw.")
    if not src.exists():
        return [f"{src} not found: run setup.sh"]
    known = set(re.findall(r'add_argument\("--([a-z0-9-]+)"', src.read_text()))
    errors += [f"--{k} is not a flag of {src} (is nanochat_cpt.patch applied?)" for k in flags if k not in known]
    B, dbs, T = flags["total-batch-size"], flags["device-batch-size"], flags["max-seq-len"]
    if B % (dbs * T * ngpu):
        errors.append(f"batch_tokens {B} is not a multiple of device_batch_size x seq_len x GPUs = {dbs * T * ngpu}")
    if flags["num-iterations"] < 10:
        errors.append(f"only {flags['num-iterations']} steps")
    return errors


def prepare_base_dir(cfg):
    """nanochat keeps checkpoints in $NANOCHAT_BASE_DIR/base_checkpoints/<tag> and reads its tokenizer from
    $NANOCHAT_BASE_DIR/tokenizer; the latter is linked to the init checkpoint's (both Gaggle models share it)."""
    nc = rel(cfg.get("nanochat", "third_party/nanochat"))
    base = rel(cfg.get("base_dir", "work/nanochat_base"))
    (base / "base_checkpoints").mkdir(parents=True, exist_ok=True)
    tok = base / "tokenizer"
    if not (tok.exists() or tok.is_symlink()):
        tok.symlink_to(rel(cfg["init"]["dir"]) / "tokenizer", target_is_directory=True)
    assert (tok / "tokenizer.pkl").exists(), f"{tok}/tokenizer.pkl missing: run fetch_models.sh"
    return nc, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ngpu", type=int, default=0, help="GPUs on this node (0 = all visible)")
    ap.add_argument("--check", action="store_true", help="print the derived settings and validate; do not train")
    a = ap.parse_args()
    cfg = load_config(a.config)
    flags, info = derive(cfg)
    ngpu = a.ngpu or visible_gpus() or 1
    errors = check(cfg, flags, ngpu, info)
    print(json.dumps(info, indent=2))
    print("base_train flags:", " ".join(shlex.quote(x) for x in to_argv(flags)))
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors)); sys.exit(1)
    if a.check:
        print(f"OK for {ngpu} GPU(s)"); return
    nc, base = prepare_base_dir(cfg)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), NANOCHAT_DATA_DIR=str(rel(cfg["data"]) / "train"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    launcher = ["torchrun", "--standalone", f"--nproc_per_node={ngpu}"] if ngpu > 1 else [sys.executable]
    cmd = [*launcher, "-m", "scripts.base_train", *to_argv(flags)]
    print("running:", " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=nc, env=env).returncode)


if __name__ == "__main__":
    main()
