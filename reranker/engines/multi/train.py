"""LCE pointwise-reranker trainer for every backbone.

R0 is greenhouse run G; the C runs are the fair four-backbone comparison. Derived from greenhouse
reranker/src/train_nc.py; its group dataset/collator, loss, LR schedule, step meter and fp32
guard are imported unmodified. Differences, all execution-only (the per-step gradient is the same):
  * the frozen logical plan (max_tokens_per_batch, pack_world) is separate from execution: a microbatch
    with more than exec_chunk_tokens padded tokens runs as length-sorted chunks, either kept in one graph
    ("chunked") or via GradCache ("gradcache"), the latter as greenhouse/Nour do;
  * gradients are averaged with an explicit all-reduce at the step boundary instead of DDP (same mean);
  * optional ZeRO-1 AdamW (engines/gemma4/optimizer_sharding.py) for 4B+ models on 80 GB;
  * milestone exports every `milestone_every` training queries, a proxy mode (`stop_at_groups`) for the
    LR sweep, and a wall-clock stop that saves a resume point and exits 0 for chained Slurm legs.

    torchrun --standalone --nproc_per_node=2 src/train.py --config configs/R0_G_repro.yaml
"""
import argparse, contextlib, datetime, gc, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # reranker/: relative config paths resolve here
sys.path.insert(0, str(HERE))
from sampling import make_sampler, plan_of, summarize, milestones, step_cursor, PLAN_KEYS  # noqa: E402  (adds LCE_SRC)
sys.path.insert(0, str(ROOT / "engines/gemma4"))
from stepping import StepMeter, setup_precision, assert_fp32_params, FP32Guard, lr_at  # noqa: E402
from dataset_nc import PairGroups, GroupCollator  # noqa: E402
from losses_nc import total_loss  # noqa: E402
from optimizer_sharding import build_optimizer, prepare_checkpoint, finish_checkpoint, ownership_report  # noqa: E402
import prompts  # noqa: E402
from backbones import build_model, enable_checkpointing  # noqa: E402

RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))


def is_main(): return RANK == 0
def log(m):
    if is_main(): print(m, flush=True)


# ------------------------------------------------------------------ config
def load_cfg(path, overrides):
    import yaml
    cfg = yaml.safe_load(open(path))
    for ov in overrides:
        k, v = ov.split("=", 1); cfg[k] = yaml.safe_load(v)
    for k in ("model_name", "train_parquet", "output_dir", "nanochat_src"):
        if cfg.get(k) and not os.path.isabs(cfg[k]): cfg[k] = str(ROOT / cfg[k])
    assert cfg["backbone"] in ("nanochat", "qwen", "gemma", "minicpm", "gravity", "marin"), cfg["backbone"]
    assert cfg.get("lora", False) is False and cfg.get("w_kd", 0) == 0 and cfg.get("w_rkd", 0) == 0
    assert cfg.get("lr") is not None, "lr is required (C runs: --override lr=<sweep winner>)"
    return cfg


def make_builder(cfg, tok):
    if cfg.get("builder") == "greenhouse":          # R0: greenhouse's own PairBuilder, token-identical to G
        from prompt_builder_nc import PairBuilder as GB, setup_tokenizer as gst
        gst(tok)
        assert cfg["length_rule"] == "caps"
        return GB(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"], prompt="nochat")
    prompts.setup_tokenizer(tok, cfg["backbone"])
    return prompts.PairBuilder(tok, cfg["backbone"], cfg["length_rule"], cfg["max_query_len"],
                               cfg.get("max_doc_len"), cfg.get("seq_budget"))


# ------------------------------------------------------------------ execution
def chunk_rows(lens, chunk_tokens):
    """Length-sorted chunks of row indices, each with (rows x longest row) <= chunk_tokens (a single row
    longer than that still forms its own chunk; a sequence is never split)."""
    order = torch.argsort(lens, descending=True).tolist()
    L = lens.tolist()
    chunks, cur, cmax = [], [], 0
    for i in order:
        m = max(cmax, L[i])
        if cur and m * (len(cur) + 1) > chunk_tokens:
            chunks.append(cur); cur, m = [], L[i]
        cur.append(i); cmax = m
    chunks.append(cur)
    return chunks


def run_microbatch(model, ids, am, B, G, loss_fn, w, cfg):
    """Forward + backward of one microbatch; returns (scores [B,G] detached, loss, parts, n_chunks, mode)."""
    budget = int(cfg["exec_chunk_tokens"])
    if ids.numel() <= budget:
        scores = model(ids, am).scores.view(B, G)
        loss, parts = loss_fn(scores)
        (loss * w).backward()
        return scores.detach(), loss.detach(), parts, 1, "single"
    lens = am.sum(1)
    chunks = chunk_rows(lens, budget)
    idx = [torch.tensor(c, device=ids.device) for c in chunks]
    tmax = [int(lens[c].max()) for c in idx]
    stored = sum(len(c) * t for c, t in zip(chunks, tmax))
    mode = cfg.get("exec_mode", "gradcache")
    if mode == "auto":
        mode = "chunked" if stored <= int(cfg.get("exec_store_tokens", 0)) else "gradcache"
    fwd = lambda c, t: model(ids[c, :t], am[c, :t]).scores
    inv = torch.argsort(torch.cat(idx))
    if mode == "chunked":                      # all chunks in one graph, one backward
        scores = torch.cat([fwd(c, t) for c, t in zip(idx, tmax)])[inv].view(B, G)
        loss, parts = loss_fn(scores)
        (loss * w).backward()
        return scores.detach(), loss.detach(), parts, len(chunks), mode
    with torch.no_grad():                      # GradCache: scores, d loss / d score, then chunk backwards
        s = torch.cat([fwd(c, t) for c, t in zip(idx, tmax)])[inv]
    s = s.view(B, G).detach().requires_grad_(True)
    loss, parts = loss_fn(s)
    (loss * w).backward()
    g = s.grad.reshape(-1)
    for c, t in zip(idx, tmax):
        fwd(c, t).backward(g[c])
    return s.detach(), loss.detach(), parts, len(chunks), "gradcache"


def allreduce_grads(params, bucket_bytes=256 << 20):
    if WORLD == 1: return
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    buf, size = [], 0
    def flush():
        flat = _flatten_dense_tensors(buf); dist.all_reduce(flat); flat.div_(WORLD)
        for g, f in zip(buf, _unflatten_dense_tensors(flat, buf)): g.copy_(f)
    for p in params:
        buf.append(p.grad); size += p.grad.numel() * p.grad.element_size()
        if size >= bucket_bytes: flush(); buf, size = [], 0
    if buf: flush()


# ------------------------------------------------------------------ checkpoints
def signature(cfg, summ):
    return {**{k: cfg.get(k) for k in PLAN_KEYS}, "plan_sha256": summ["sha256"], "world": WORLD,
            "backbone": cfg["backbone"], "model_name": Path(cfg["model_name"]).name, "lr": cfg["lr"],
            "stop_at_groups": cfg.get("stop_at_groups")}


def save_resume(path, model, opt, gstep, epoch, bie, cfg, sig):
    prepare_checkpoint(opt, path)
    if is_main():
        tmp = str(path) + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": gstep, "epoch": epoch,
                    "batch_in_epoch": bie, "cfg": cfg, "packing": sig,
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                            "python": random.getstate(), "numpy": np.random.get_state()}}, tmp)
        os.replace(tmp, path)
    finish_checkpoint(opt, path)
    if WORLD > 1: dist.barrier()


def export_model(path, model, cfg, gstep, groups, sig, kind):
    """Model-only fp32 export (milestones, final). Carries the keys greenhouse rerank_eval expects."""
    if is_main():
        tmp = str(path) + ".tmp"
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": gstep, "groups": groups,
                    "packing": sig, "kind": kind}, tmp)
        os.replace(tmp, path)
    if WORLD > 1: dist.barrier()


def slurm_seconds_left():
    """Seconds left in this Slurm allocation (None outside Slurm). Read at startup so a job whose time limit was
    shortened after submission (scontrol update TimeLimit=...) still stops cleanly and saves a resume point."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        import subprocess
        left = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        days, _, hms = left.rpartition("-")
        parts = [int(x) for x in hms.split(":")]
        while len(parts) < 3: parts.insert(0, 0)
        return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return None


def resume_points(out):
    return sorted(Path(out).glob("resume_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.override)
    t_start = time.time()
    stop_after = float(os.environ.get("LCE_STOP_AFTER_SEC", cfg.get("stop_after_sec", 0)) or 0)
    stop_steps = int(os.environ.get("LCE_STOP_AFTER_STEPS", 0) or 0)       # tests only (resume exactness)
    left = slurm_seconds_left()
    if left:                                   # never outlive the allocation: stop 40 min before its end
        cap = max(600.0, left - 2400.0)
        stop_after = min(stop_after, cap) if stop_after else cap
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "milestones").mkdir(exist_ok=True)
    if (out / "DONE").exists():
        log(f"{out}/DONE exists: run finished, nothing to do"); return
    if WORLD > 1:
        torch.cuda.set_device(LOCAL)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30),
                                device_id=torch.device(f"cuda:{LOCAL}"))
    dev = f"cuda:{LOCAL}"
    if os.environ.get("LCE_MEM_FRACTION"):     # tests: emulate a smaller GPU; AMD: keep room for the HIP runtime
        torch.cuda.set_per_process_memory_fraction(float(os.environ["LCE_MEM_FRACTION"]), LOCAL)
    torch.manual_seed(cfg["seed"] + RANK); random.seed(cfg["seed"] + RANK); np.random.seed(cfg["seed"] + RANK)
    dtype = setup_precision(cfg, log)
    if cfg["backbone"] == "nanochat":
        os.environ.setdefault("NANOCHAT_DTYPE", "float32")
    log(json.dumps(cfg, indent=1))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    builder = make_builder(cfg, tok)
    yes_id, no_id = prompts.answer_ids(tok)
    if cfg.get("answer_ids"): assert [yes_id, no_id] == list(cfg["answer_ids"]), (yes_id, no_id)
    log(f"backbone={cfg['backbone']} answers {yes_id}/{no_id} scaffold={builder.overhead} | "
        f"example: {builder.render('QUERY', 'PASSAGE')!r}")

    ds = PairGroups(cfg["train_parquet"], builder, seed=cfg["seed"])
    sampler = make_sampler(cfg, ds.avail, ds.neg_lens, ds.query_lens, rank=RANK, world=WORLD)
    plan = plan_of(sampler, cfg)
    summ = summarize(plan, cfg.get("milestone_every"))
    if cfg.get("expected_plan_sha256"):
        assert summ["sha256"] == cfg["expected_plan_sha256"], f"plan {summ['sha256']} != expected"
    if cfg.get("stop_at_groups"):              # LR-sweep proxy: train the plan's first N queries only
        total_steps = milestones(plan, int(cfg["stop_at_groups"]))[0]["step"]
        ms_steps = {total_steps}
    else:
        total_steps = len(plan)
        ms_steps = {m["step"] for m in summ.get("milestones", [])} | {total_steps}
    cum = np.cumsum([len(s) for s in plan]).tolist()
    del plan
    sig = signature(cfg, summ)
    if is_main():
        import yaml
        (out / "resolved_config.yaml").write_text(yaml.safe_dump({**cfg, "_signature": sig}, sort_keys=True))
        (out / "plan_summary.json").write_text(json.dumps({**summ, "total_steps": total_steps,
                                                            "export_steps": sorted(ms_steps)}, indent=1))
        tok.save_pretrained(out / "tokenizer")
    log(f"plan {summ['sha256'][:12]}: {summ['steps']} steps / {summ['groups']:,} groups; training "
        f"{total_steps} steps; exports at {sorted(ms_steps)}")

    model = build_model(cfg, yes_id, no_id, dtype).to(dev)
    assert_fp32_params(model)
    if cfg["grad_checkpointing"]:
        enable_checkpointing(model)
    model.train()
    drop = [(n, m.p) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    assert not drop, f"active dropout would make runs stochastic: {drop[:4]}"
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:                           # every rank holds every grad, so all-reduce buckets line up
        p.grad = torch.zeros_like(p)
    log(f"trainable {sum(p.numel() for p in params):,} params in {len(params)} tensors | "
        f"{getattr(model, 'trainable_report', lambda: {})()}")

    opt = build_optimizer(model, cfg)
    log(f"optimizer: {ownership_report(opt) if WORLD > 1 else 'AdamW (unsharded)'}")
    start_step = start_epoch = start_batch = 0
    rps = resume_points(out)
    if rps and cfg.get("resume", "auto") == "auto":
        ck = torch.load(rps[-1], map_location="cpu", weights_only=False, mmap=True)
        start_step, start_epoch, start_batch = ck["step"], ck["epoch"], ck["batch_in_epoch"]
        saved_world = ck["packing"]["world"]
        if ck["packing"] != sig:
            # A different world size is allowed on request: the plan's steps are packed for pack_world whatever the
            # world size, and loss weights world*len/G with the all-reduce mean give the same step gradient.
            # Only the microbatch cursor is per rank (below). The optimizer state is saved in PyTorch's global
            # format either way, so ZeRO-1 may differ too.
            if not (cfg.get("allow_world_change") and {**ck["packing"], "world": WORLD} == sig and cfg["epochs"] == 1):
                raise SystemExit(f"RESUME ABORTED: signature {ck['packing']} != {sig}")
        def cursor(world, rank):                # this rank's microbatch index where step start_step+1 begins
            s = make_sampler(cfg, ds.avail, ds.neg_lens, ds.query_lens, rank=rank, world=world)
            s.set_epoch(start_epoch)
            return step_cursor(s, start_step)
        # The checkpoint holds rank 0's cursor. Ranks run different numbers of microbatches when the world size does
        # not divide pack_world, so every rank recomputes its own from the step; rank 0's must match the saved one.
        assert cursor(saved_world, 0) == start_batch, (cursor(saved_world, 0), start_batch)
        start_batch = cursor(WORLD, RANK)
        if saved_world != WORLD or start_batch != ck["batch_in_epoch"]:
            log(f"world {saved_world} -> {WORLD}: rank {RANK} microbatch cursor {ck['batch_in_epoch']} -> {start_batch}")
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        try:
            r = ck["rng"]; torch.set_rng_state(r["torch"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in r["cuda"]])
            random.setstate(r["python"]); np.random.set_state(r["numpy"])
        except Exception as e:
            log(f"(rng not restored: {type(e).__name__})")
        del ck; gc.collect()
        log(f"RESUMED {rps[-1].name}: step {start_step}, microbatch {start_batch}")

    collate = GroupCollator(tok.pad_token_id)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg["num_workers"],
                    pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    sched = dict(total=total_steps, peak=cfg["lr"], warmup=cfg["warmup_steps"])
    log(f"LR: warmup {cfg['warmup_steps']} to {cfg['lr']}, cosine to 0 at {total_steps}; "
        f"exec_chunk_tokens={cfg['exec_chunk_tokens']} exec_mode={cfg.get('exec_mode', 'gradcache')}")
    logf = open(out / f"train_{os.environ.get('SLURM_JOB_ID', 'local')}_rank{RANK}.jsonl", "a")
    LOGK = ["loss", "rank", "acc", "mrr", "rank_of_pos", "pos_score", "neg_score", "sep_mean", "sep_hardest",
            "K", "lnK1", "seqs", "seq", "tok", "chunks"]
    meter = StepMeter(LOGK, dev)
    guard = FP32Guard(model)
    gstep, tstep, last_save = start_step, time.time(), start_step
    stopped = False
    for epoch in range(start_epoch, cfg["epochs"]):
        skip = start_batch if epoch == start_epoch else 0
        ds.set_epoch(epoch); sampler.set_epoch(epoch, skip=skip)
        for bi_local, batch in enumerate(dl):
            bi = bi_local + skip
            w, last, Gstep = sampler.meta[bi]
            B, G = batch["B"], batch["G"]
            ids = batch["input_ids"].to(dev, non_blocking=True)
            am = batch["attention_mask"].to(dev, non_blocking=True)
            pos = batch["pos_idx"].to(dev, non_blocking=True)
            loss_fn = lambda sc: total_loss(sc, pos, tau=cfg["tau"])
            scores, loss, parts, nch, mode = run_microbatch(model, ids, am, B, G, loss_fn, w, cfg)
            if guard is not None:
                log(f"fp32 guard passed: {guard.remove()} module outputs checked"); guard = None
            assert loss.dtype == torch.float32
            if (gstep + 1) % cfg["log_every"] == 0:
                with torch.no_grad():
                    ps = scores.gather(1, pos[:, None]).squeeze(1)
                    rop = (scores > ps[:, None]).sum(1) + 1
                    nm = torch.ones_like(scores, dtype=torch.bool); nm.scatter_(1, pos[:, None], False)
                    ns = scores.masked_select(nm).view(B, -1)
                    meter.add(B, {"loss": loss, **{k: v.detach() for k, v in parts.items()},
                                  "acc": (scores.argmax(-1) == pos).float().mean(),
                                  "mrr": (1.0 / rop.float()).mean(), "rank_of_pos": rop.float().mean(),
                                  "pos_score": ps.mean(), "neg_score": ns.mean(),
                                  "sep_mean": (ps - ns.mean(1)).mean(),
                                  "sep_hardest": (ps - ns.max(1).values).mean(),
                                  "K": int(batch["K"]), "lnK1": math.log(int(batch["K"]) + 1),
                                  "seqs": ids.shape[0] / B, "seq": ids.shape[1], "tok": ids.numel() / B,
                                  "chunks": nch})
            if not last:
                continue
            allreduce_grads(params)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            if not torch.isfinite(gn): raise RuntimeError("non-finite gradient norm")
            lr = lr_at(gstep, **sched)
            for g in opt.param_groups: g["lr"] = lr
            opt.step(); opt.zero_grad(set_to_none=False); gstep += 1
            if gstep % cfg["log_every"] == 0:
                m = meter.reduce(WORLD > 1)
                rec = {"step": gstep, **{k: round(v, 5) for k, v in m.items()}, "G": Gstep,
                       "groups_seen": cum[gstep - 1], "lr": lr, "gnorm": float(gn), "clipped": float(gn) > cfg["clip"],
                       "mem": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                       "step_sec": round(time.time() - tstep, 2), "sec": round(time.time() - t_start, 1)}
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                log("  ".join(f"{k}={v}" for k, v in rec.items()))
            tstep = time.time()
            if gstep in ms_steps:
                kind = "final" if gstep == total_steps else "milestone"
                export_model(out / "milestones" / f"step{gstep:05d}_q{cum[gstep - 1]}.pt", model, cfg,
                             gstep, cum[gstep - 1], sig, kind)
                log(f"exported {kind} at step {gstep} ({cum[gstep - 1]:,} queries)")
            if gstep >= total_steps:
                break
            if cfg["save_every"] and gstep % cfg["save_every"] == 0:
                save_resume(out / f"resume_{gstep:05d}.pt", model, opt, gstep, epoch, bi + 1, cfg, sig)
                for old in resume_points(out)[:-cfg.get("keep_resume", 2)]:
                    if is_main(): old.unlink(missing_ok=True)
                last_save = gstep
            flag = torch.tensor([float((stop_after and time.time() - t_start > stop_after)
                                       or (stop_steps and gstep >= stop_steps))], device=dev)
            if WORLD > 1: dist.broadcast(flag, 0)
            if flag.item():
                if last_save != gstep and not os.environ.get("LCE_TEST_NOSAVE"):   # tests: timing probes
                    save_resume(out / f"resume_{gstep:05d}.pt", model, opt, gstep, epoch, bi + 1, cfg, sig)
                log(f"wall-clock stop at step {gstep}; resume point saved"); stopped = True
                break
        if stopped or gstep >= total_steps:
            break
    if not stopped and gstep >= total_steps and is_main():
        (out / "DONE").write_text(json.dumps({"step": gstep, "time": time.time() - t_start}) + "\n")
        for old in resume_points(out):
            old.unlink(missing_ok=True)
        log(f"DONE at step {gstep}")
    if WORLD > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
