"""Qwen3.5 LCE entry point; loss, sampling and training helpers come from NanoChat LCE."""
import argparse, contextlib, datetime, gc, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from shared import reference, validate_native_config, summarize_plan
from shared import StepMeter, setup_precision, assert_fp32_params, FP32Guard, lr_at
from shared import PairGroups, GroupCollator, GroupSampler, total_loss
from qwen_prompt import PairBuilder, yes_no_ids, setup_tokenizer
from modeling_qwen35_lce import Qwen35Reranker

is_main = reference.is_main
log = reference.log
make_ddp = reference.make_ddp
gradcache_backward = reference.gradcache_backward
find_latest_ckpt = reference.find_latest_ckpt
prune_ckpts = reference.prune_ckpts

def packing_signature(cfg, world):
    return {**reference.packing_signature(cfg, world),
            "model_revision": cfg.get("model_revision"),
            "objective": "lce_qwen35_tied_gap", "plan_hash": cfg["expected_plan_hash"]}

def save_ckpt(path, model, opt, gstep, epoch, bie, cfg, wid, packing):
    reference.save_ckpt(path, model, opt, gstep, epoch, bie, cfg, wid, packing)
    state = {"global_step": gstep, "epoch_index": epoch, "batch_in_epoch": bie,
             "checkpoint": str(path), "model_revision": cfg.get("model_revision"),
             "objective": "lce_qwen35_tied_gap", "plan_hash": cfg["expected_plan_hash"]}
    Path(str(path)+".json").write_text(json.dumps(state, indent=2)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    for ov in args.override:
        k, v = ov.split("=", 1); cfg[k] = yaml.safe_load(v)

    validate_native_config(cfg)
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    if (out / "ckpt_final.pt").exists() and cfg.get("resume", "auto") == "auto":
        # a crash AFTER the final save must not make a retry/requeue retrain the tail
        log(f"{out}/ckpt_final.pt exists: run already finished, nothing to do"); return
    if ddp:
        # set_device MUST come first: NCCL binds to the CURRENT device at init
        # (measured: all_reduce of [1,2] returned 1 instead of 3 otherwise).
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # 30 min, NOT 3: this timeout governs EVERY collective, including the wait
        # while rank 0 writes a checkpoint (checkpoint staging can take several minutes).
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30),
                                device_id=torch.device(f"cuda:{os.environ['LOCAL_RANK']}"))
    dev = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
    torch.manual_seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank); np.random.seed(cfg["seed"] + rank)
    dtype = setup_precision(cfg, log)
    log(json.dumps(cfg, indent=2))

    from transformers import AutoTokenizer
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision"),
        local_files_only=cfg.get("local_files_only", False)))
    PROMPT = cfg.get("prompt", "nochat")
    yes_id, no_id = yes_no_ids(tok, PROMPT, cfg.get("answer_tokens"))
    builder = PairBuilder(tok, max_doc_len=cfg["max_doc_len"],
                          max_query_len=cfg["max_query_len"], prompt=PROMPT)
    log(f"prompt={PROMPT} answers={tok.decode([yes_id])!r}/{tok.decode([no_id])!r} (ids {yes_id}/{no_id}) "
        f"scaffold={builder.overhead} tokens | example: {builder.render('QUERY', 'PASSAGE')!r}")

    ds = PairGroups(cfg["train_parquet"], builder, seed=cfg["seed"])
    log(f"dataset: {len(ds):,} rows | teacher scores: {ds.has_teacher} | teacher ranks: {ds.has_trank} "
        f"(w_rkd={cfg.get('w_rkd', 0.0)})")
    if cfg.get("w_rkd"): assert ds.has_trank, "w_rkd > 0 needs a parquet with pos_trank/neg_trank"
    # wandb id in a tiny sidecar (peeking it from a 25 GB checkpoint cost a full read)
    wid_file = out / "wandb_id.txt"
    if cfg.get("resume", "auto") != "off" and wid_file.exists() and find_latest_ckpt(out):
        cfg.setdefault("wandb_id", wid_file.read_text().strip() or None)
    wb = None
    if cfg.get("wandb_project") and is_main():
        import wandb
        wb = wandb.init(project=cfg["wandb_project"], entity=cfg.get("wandb_entity") or None,
                        name=cfg.get("wandb_run_name")
                             or f"{out.name}-j{os.environ.get('RUN_ID','local')}",
                        group=cfg.get("wandb_group") or None, tags=cfg.get("wandb_tags") or [],
                        config=cfg, resume="allow", id=cfg.get("wandb_id") or None)
        wid_file.write_text(wb.id)
        log(f"wandb: {wb.url}")

    model = Qwen35Reranker(cfg["model_name"], yes_id=yes_id, no_id=no_id, dtype=dtype,
        attn_implementation=cfg["attn_implementation"], model_revision=cfg.get("model_revision"),
        local_files_only=cfg.get("local_files_only", False)).to(dev)
    assert_fp32_params(model)
    if cfg["grad_checkpointing"]:
        model.train()
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # Execution chunk budget; independent of the fixed logical packing plan
    execution_budget = int(cfg["execution_tokens_per_batch"])
    log(f"FROZEN packing budget={cfg['max_tokens_per_batch']} | actual execution budget={execution_budget}")
    if wb is not None: wb.config.update({"max_tokens_per_batch": cfg["max_tokens_per_batch"]},
                                        allow_val_change=True)
    sampler = GroupSampler(ds.avail, ds.neg_lens, ds.query_lens,
                           max_tokens=cfg["max_tokens_per_batch"],
                           max_groups=cfg["max_groups"], max_doc_len=cfg["max_doc_len"],
                           max_query_len=cfg["max_query_len"],
                           k_choices=tuple(cfg["k_choices"]), seed=cfg["seed"],
                           scaffold=cfg["packing_scaffold"],
                           global_groups=cfg["global_batch_groups"], rank=rank, world_size=world,
                           pack_world=cfg.get("pack_world"), query_factor=builder.query_factor)
    plan_summary = summarize_plan(sampler, cfg)
    # Record the actual plan even when reference checks are disabled for new data.
    cfg["expected_plan_hash"] = plan_summary["sha256"]
    packing = packing_signature(cfg, world)
    if is_main():
        import yaml
        (out / "resolved_config.yaml").write_text(yaml.safe_dump({**cfg, "_packing": packing}, sort_keys=True))
        (out / "plan_summary.json").write_text(json.dumps(plan_summary, indent=2) + "\n")
        tok.save_pretrained(out / "tokenizer")
        (out / "qwen_text_config.json").write_text(model.config.to_json_string())
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=GroupCollator(tok.pad_token_id),
                    num_workers=cfg["num_workers"], pin_memory=True,
                    persistent_workers=cfg["num_workers"] > 0)
    _G = [g for _, last, g in sampler.meta if last]
    log(f"microbatches/epoch (this rank): {len(sampler):,}  local budget {cfg['max_tokens_per_batch']:,} tok | "
        f"global {cfg['global_batch_groups']} groups/step -> {sampler.n_steps:,} steps, "
        f"G mean {sum(_G)/max(1,len(_G)):.1f} [{min(_G)}..{max(_G)}], "
        f"{len(sampler.meta)/max(1,sampler.n_steps):.2f} microbatches/step/rank")

    ck = None; start_step = start_epoch = start_batch = 0
    if cfg.get("resume", "auto") != "off":
        cand = (Path(cfg["resume"]) if cfg.get("resume") not in (None, "auto", "off")
                else find_latest_ckpt(out))
        if cand and Path(cand).exists():
            # Page checkpoint tensors in on demand to reduce host-memory pressure.
            ck = torch.load(cand, map_location="cpu", weights_only=False, mmap=True)
            saved = ck.get("packing")
            if saved is not None and saved != packing:
                raise SystemExit(f"RESUME ABORTED: checkpoint packing {saved} != current {packing}. "
                                 f"Run on the same logical packing and world size as the original run.")
            start_step, start_epoch = ck["step"], ck.get("epoch", 0)
            start_batch = ck.get("batch_in_epoch", 0)
            model.load_state_dict(ck["model"], strict=True)   # any key mismatch raises
            log(f"RESUMING from {cand}: step={start_step} epoch={start_epoch} microbatch={start_batch}")
        else:
            log("no checkpoint found; starting fresh")

    if cfg.get("require_resume_min_step") and (ck is None or start_step < cfg["require_resume_min_step"]):
        raise RuntimeError("Required same-lineage checkpoint is missing; refusing fresh training")

    if cfg.get("unfreeze_after"):
        raise SystemExit("unfreeze_after is no longer supported: every parameter trains from step 0")
    model.train()
    r = model.trainable_report()
    log(f"backbone={r['backbone']/1e9:.2f}B head={r['head']} "
        f"trainable={r['trainable']:,} ({r['trainable']/1e6:.2f}M)")
    ddp_model = make_ddp(model) if ddp else model

    steps_per_epoch = sampler.n_steps          # exact: microbatches are packed into steps
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg.get("max_steps"): total_steps = min(total_steps, cfg["max_steps"])
    # ONE optimizer over all parameters for the whole run
    opt = torch.optim.AdamW(list(model.parameters()), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95), fused=True)
    if ck is not None:
        opt.load_state_dict(ck["optimizer"])          # strict: a failure must not be silent
        log("restored optimizer state")
        try:   # cosmetic: nothing random in training (no dropout; data order seeded per row)
            rr = ck.get("rng") or {}
            torch.set_rng_state(rr["torch"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in rr["cuda"]])
            random.setstate(rr["python"]); np.random.set_state(rr["numpy"])
        except Exception as e: log(f"(RNG state not restored: {type(e).__name__})")
        del ck; gc.collect()
    sched_kw = dict(total=total_steps, peak=cfg["lr"], warmup=cfg["warmup_steps"])
    log(f"steps/epoch={steps_per_epoch:,} total={total_steps:,} start={start_step} | LR: linear warmup "
        f"{sched_kw['warmup']} steps to {cfg['lr']}, cosine to 0 at {total_steps}")

    _jid = os.environ.get("RUN_ID", "local")
    logf = open(out / f"train_j{_jid}_rank{rank}.jsonl", "a")
    wandb_id = (wb.id if wb is not None else None) or cfg.get("wandb_id")
    gstep, t0 = start_step, time.time()

    LOGK = ["loss", "rank", "kd", "rkd", "rkd_agree", "acc", "mrr", "rank_of_pos", "pos_score", "neg_score",
            "sep_mean", "sep_hardest", "K", "lnK1", "seqs", "seq", "tok"]
    meter = StepMeter(LOGK, dev)
    n_chunked = 0                                 # oversized-group microbatches run via GradCache
    guard = FP32Guard(model)                      # armed for the first microbatch only
    for epoch in range(start_epoch, cfg["epochs"]):
        skip = start_batch if epoch == start_epoch else 0
        ds.set_epoch(epoch); sampler.set_epoch(epoch, skip=skip)
        if skip: log(f"skipping {skip:,} already-done microbatches of epoch {epoch}")
        tstep = time.time()
        for bi_local, batch in enumerate(dl):
            bi = bi_local + skip
            w, last, Gstep = sampler.meta[bi]
            B, G = batch["B"], batch["G"]
            ids = batch["input_ids"].to(dev, non_blocking=True)
            am = batch["attention_mask"].to(dev, non_blocking=True)
            pos = batch["pos_idx"].to(dev, non_blocking=True)
            teacher_t = kd_w = None
            if cfg.get("w_kd") and batch.get("teacher") is not None:
                teacher_t = batch["teacher"].to(dev, non_blocking=True)
                ov = cfg.get("kd_subset_weights") or {}
                kd_w = torch.tensor([0.0 if not ok else float(ov.get(s, 1.0))
                                     for s, ok in zip(batch["subset"],
                                                      batch["teacher_mask"].tolist())],
                                    device=dev)
            trank_t = rkd_w = None                       # listwise-teacher rank distillation (KD_PLAN §5)
            if batch.get("trank") is not None:          # w_rkd = 0: rkd/rkd_agree are logged, not trained on
                trank_t = batch["trank"].to(dev, non_blocking=True)
                ov = cfg.get("rkd_subset_weights") or {}
                rkd_w = torch.tensor([float(ov.get(s, 1.0)) for s in batch["subset"]], device=dev)
            loss_fn = lambda sc: total_loss(sc, pos, tau=cfg["tau"], teacher=teacher_t,
                                            w_kd=cfg.get("w_kd", 0.0),
                                            tau_t=cfg.get("tau_teacher", 1.0), kd_row_w=kd_w,
                                            tau_s_kd=cfg.get("tau_s_kd"),
                                            trank=trank_t, w_rkd=cfg.get("w_rkd", 0.0), rkd_row_w=rkd_w)
            if ids.numel() <= execution_budget:
                # one all-reduce per STEP: skip DDP's sync on all but the last microbatch
                sync = contextlib.nullcontext() if (not ddp or last) else ddp_model.no_sync()
                with sync:
                    scores = ddp_model(ids, am).scores.view(B, G)
                    loss, parts = loss_fn(scores)
                    # w = world*B_micro/G_step: the step's gradient is the exact mean over its groups
                    (loss * w).backward()
            else:   # one oversized group on a small GPU: split its sequences (exact gradient)
                n_chunked += 1
                scores, loss, parts = gradcache_backward(model, ddp_model, ids, am, B, G,
                                                         execution_budget, loss_fn, w, last, ddp)
            if guard is not None:
                log(f"fp32 guard passed: {guard.remove()} module outputs checked"); guard = None
            assert loss.dtype == torch.float32
            if (gstep + 1) % cfg["log_every"] == 0:              # this step will be logged
                with torch.no_grad():
                    sc = scores.detach()
                    ps = sc.gather(1, pos[:, None]).squeeze(1)
                    rop = (sc > ps[:, None]).sum(1) + 1
                    nm = torch.ones_like(sc, dtype=torch.bool); nm.scatter_(1, pos[:, None], False)
                    ns = sc.masked_select(nm).view(B, -1)
                    Kb = int(batch["K"])
                    meter.add(B, {"loss": loss.detach(), **{k: v.detach() for k, v in parts.items()},
                                  "acc": (sc.argmax(-1) == pos).float().mean(),
                                  "mrr": (1.0 / rop.float()).mean(), "rank_of_pos": rop.float().mean(),
                                  "pos_score": ps.mean(), "neg_score": ns.mean(),
                                  "sep_mean": (ps - ns.mean(1)).mean(),
                                  "sep_hardest": (ps - ns.max(1).values).mean(),
                                  "K": Kb, "lnK1": math.log(Kb + 1), "seqs": ids.shape[0] / B,
                                  "seq": ids.shape[1], "tok": ids.numel() / B})
            if not last:
                continue
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["clip"])
            lr = lr_at(gstep, **sched_kw)
            for g in opt.param_groups: g["lr"] = lr
            # set_to_none=False: with bucket-view DDP the .grad tensors ARE the buckets
            if not torch.isfinite(gn): raise RuntimeError("Non-finite gradient norm; refusing optimizer update")
            opt.step(); opt.zero_grad(set_to_none=False); gstep += 1
            if gstep % cfg["log_every"] == 0:
                m = meter.reduce(ddp)                           # every rank, every microbatch of the step
                rec = {"step": gstep, "epoch": epoch, **{k: round(v, 5) for k, v in m.items()},
                       "G": Gstep, "rank_vs_random": round(m["lnK1"] - m["rank"], 5),
                       "chunked_microbatches": n_chunked,
                       "lr": lr, "gnorm": float(gn), "clipped": float(gn) > cfg["clip"],
                       "mem": round(torch.cuda.max_memory_allocated()/2**30, 1),
                       "sec": round(time.time()-t0, 1), "step_sec": round(time.time()-tstep, 2),
                       "frac_epoch": round((bi + 1) / max(1, len(sampler.meta)), 4)}
                logf.write(json.dumps(rec)+"\n"); logf.flush()
                log("  ".join(f"{k}={v}" for k, v in rec.items()))
                if wb is not None:
                    wb.log({**{f"train/{k}": v for k, v in rec.items()
                               if k not in ("step", "epoch")}, "epoch": epoch}, step=gstep)
            tstep = time.time()
            if gstep == 1 or (cfg.get("save_first_resumed_update") and gstep == start_step + 1) or (cfg["save_every"] and gstep % cfg["save_every"] == 0):
                if is_main():
                    save_ckpt(out/f"ckpt_{gstep}.pt", model, opt, gstep, epoch,
                              bi+1, cfg, wandb_id, packing)
                    prune_ckpts(out, cfg.get("keep_ckpts", 3)); log(f"saved ckpt_{gstep}.pt")
                if ddp:
                    dist.barrier()   # hold every rank until rank 0 finishes writing
            if cfg.get("max_steps") and gstep >= cfg["max_steps"]:
                break
        if cfg.get("max_steps") and gstep >= cfg["max_steps"]:
            break
    if is_main():
        save_ckpt(out/"ckpt_final.pt", model, opt, gstep, epoch, 0, cfg, wandb_id, packing)
        log(f"saved ckpt_final.pt at step {gstep}")
    if ddp: dist.barrier()
    if wb: wb.finish()
    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
