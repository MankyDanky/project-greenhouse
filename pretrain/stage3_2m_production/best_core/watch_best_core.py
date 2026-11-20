"""Keep a hard-linked copy of the production checkpoint with the highest measured CORE.

Read-only toward training: parses chain.log CORE lines, hard-links complete checkpoint sets
(same filesystem, no bytes copied) into BEST/step_<N>, and replaces it only on a strictly higher
CORE. Retention pruning in the checkpoint dir only unlinks names, so the kept copy survives.
Only steps that have BOTH a CORE measurement and a saved checkpoint can be kept.
"""
import argparse, fcntl, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

PD = CKPT = BEST = STATE = LOG = None  # set from the command line in main()
PAT = re.compile(r"Step (\d+) \| CORE metric: ([0-9.]+)")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def core_values():
    out = subprocess.run(["grep", "-aoE", r"Step [0-9]+ \| CORE metric: [0-9.]+", str(PD / "chain.log")],
                         capture_output=True, text=True).stdout
    values = {}
    for m in PAT.finditer(out):
        values[int(m.group(1))] = float(m.group(2))  # last occurrence wins (resume re-evaluation)
    return values


def names(step):
    s = f"{step:06d}"
    return [f"model_{s}.pt", f"meta_{s}.json"] + [f"optim_{s}_rank{r}.pt" for r in range(8)]


def complete(step):
    try:
        if not all((CKPT / n).is_file() and (CKPT / n).stat().st_size > 0 for n in names(step)):
            return False
        return json.loads((CKPT / f"meta_{step:06d}.json").read_text())["step"] == step
    except (OSError, ValueError, KeyError):
        return False


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def keep(step, core, previous):
    target = BEST / f"step_{step:06d}"
    tmp = BEST / f".tmp_step_{step:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    files = []
    for n in names(step):
        os.link(CKPT / n, tmp / n)
        a, b = (CKPT / n).stat(), (tmp / n).stat()
        assert a.st_ino == b.st_ino and a.st_size == b.st_size, n
        files.append({"name": n, "size": b.st_size})
    meta = json.loads((tmp / f"meta_{step:06d}.json").read_text())
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    state = {"step": step, "core": core, "dir": str(target), "val_bpb": meta.get("val_bpb"),
             "canonical_step": meta.get("production", {}).get("canonical_step"),
             "total_training_tokens": meta.get("production", {}).get("total_training_tokens"),
             "kept_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files}
    t = STATE.with_suffix(".tmp")
    t.write_text(json.dumps(state, indent=2) + "\n")
    t.replace(STATE)
    if previous and Path(previous["dir"]) != target and Path(previous["dir"]).exists():
        shutil.rmtree(previous["dir"])
    log(f"KEPT step {step} CORE {core:.4f} (previous: {previous['step'] if previous else None} "
        f"{previous['core'] if previous else None}) val_bpb {state['val_bpb']}")


def once():
    best = load_state()
    cores = core_values()
    cands = [(c, s) for s, c in cores.items() if (best is None or c > best["core"]) and complete(s)]
    if cands:
        c, s = max(cands)
        keep(s, c, best)
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--prod-dir", required=True, help="production state dir with chain.log and BASE_DONE ($D34_PROD_DIR)")
    p.add_argument("--ckpt-dir", required=True, help="production checkpoint dir (base_checkpoints/$D34_PROD_TAG)")
    p.add_argument("--best-dir", required=True, help="where the best-CORE checkpoint is hard-linked (same filesystem)")
    a = p.parse_args()
    global PD, CKPT, BEST, STATE, LOG
    PD, CKPT, BEST = Path(a.prod_dir), Path(a.ckpt_dir), Path(a.best_dir)
    STATE, LOG = BEST / "best.json", BEST / "watch.log"
    BEST.mkdir(parents=True, exist_ok=True)
    lock = open(BEST / "watch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another watcher holds the lock; exiting")
        return 0
    start = time.time()
    log(f"watcher start pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID')} loop={a.loop}")
    while True:
        try:
            once()
        except Exception as e:  # never crash the loop on a transient FS error
            log(f"error: {type(e).__name__}: {e}")
        if not a.loop:
            break
        if (PD / "BASE_DONE").exists():
            once()
            log("BASE_DONE present; final check done, watcher exiting")
            return 3
        if a.max_seconds and time.time() - start > a.max_seconds:
            log("time budget reached; exiting for successor")
            break
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
