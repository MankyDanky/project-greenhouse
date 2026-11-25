#!/usr/bin/env python3
"""Turn a run config (configs/*.yaml) into nanochat/scripts/base_train.py arguments for train.sh.

    full_args.py CONFIG.yaml --check [--ngpu N]   human-readable summary; exit 1 if the config is invalid
    full_args.py CONFIG.yaml --ngpu N             bash assignments (RUN, ATTN, NUM_ITERS, ARGS) for train.sh to eval

A config has `run` (the checkpoint directory name), `attention` (fa3 or sdpa), `gpus` (the GPU count the released
model was trained with) and `train_args`, whose keys must all be flags of the trainer. train.sh sets --model-tag,
--run and --auto-resume itself, so those are rejected here. The step count and token total are computed with the
trainer's own batch-schedule code, for the GPU count given, exactly as base_train will compute them.
"""
import argparse
import os
import re
import shlex
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NANOCHAT = os.path.join(HERE, "nanochat")
TOKENS_PER_SHARD = 44.9e6  # packed training tokens per ClimbMix shard (measured: 156 shards for 7.0B tokens)
TRAIN_SHARDS = 6542        # shard_00000 .. shard_06541; shard_06542 is the validation shard
RESERVED = {"model-tag", "run", "auto-resume"}


def schedule(ta, ngpu):
    """(num_iterations, total_tokens, batch sizes used, schedule or None) as base_train computes them on ngpu GPUs."""
    if "batch-schedule" in ta:
        sys.path.insert(0, NANOCHAT)
        from nanochat.batch_schedule import build_batch_schedule
        micro = ta["device-batch-size"] * ta.get("max-seq-len", 2048) * ngpu
        s = build_batch_schedule(str(ta["batch-schedule"]), int(ta["target-tokens"]),
                                 int(ta.get("canonical-batch", 2**20)), granularity=micro)
        return s.num_steps, s.tokens_before[-1], sorted(set(s.batch)), s
    return ta["num-iterations"], ta["num-iterations"] * ta["total-batch-size"], [ta["total-batch-size"]], None


def load(path, ngpu):
    cfg = yaml.safe_load(open(path))
    errors = [f"missing top-level key: {k}" for k in ("run", "attention", "gpus", "train_args") if k not in cfg]
    if errors:
        return cfg, errors
    flags = set(re.findall(r'add_argument\("--([a-z0-9-]+)"', open(f"{NANOCHAT}/scripts/base_train.py").read()))
    ta = cfg["train_args"]
    for k in ta:
        if k in RESERVED:
            errors.append(f"train_args.{k} is set by train.sh; remove it")
        elif k not in flags:
            errors.append(f"train_args.{k} is not a flag of nanochat/scripts/base_train.py")
    if cfg["attention"] not in ("fa3", "sdpa"):
        errors.append("attention must be fa3 or sdpa")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(cfg["run"])):
        errors.append("run must match [A-Za-z0-9._-]+ (it names the checkpoint directory)")
    if not isinstance(cfg["gpus"], int) or cfg["gpus"] <= 0:
        errors.append("gpus must be a positive integer")
    if "batch-schedule" in ta:
        for k in ("num-iterations", "total-batch-size"):
            if k in ta:
                errors.append(f"train_args.{k} conflicts with batch-schedule (the horizon is target-tokens)")
        if not isinstance(ta.get("target-tokens"), int) or ta["target-tokens"] <= 0:
            errors.append("train_args.target-tokens must be a positive integer with batch-schedule")
    else:
        for k in ("num-iterations", "total-batch-size"):
            if not isinstance(ta.get(k), int) or ta[k] <= 0:
                errors.append(f"train_args.{k} must be a positive integer")
    if not isinstance(ta.get("device-batch-size"), int) or ta["device-batch-size"] <= 0:
        errors.append("train_args.device-batch-size must be a positive integer")
    if errors:
        return cfg, errors
    try:
        _, tokens, batches, sched = schedule(ta, ngpu)
    except Exception as e:  # a malformed schedule string
        return cfg, [f"batch-schedule: {e}"]
    micro = ta["device-batch-size"] * ta.get("max-seq-len", 2048) * ngpu
    for b in batches:
        if b % micro:
            errors.append(f"batch {b:,} is not a multiple of {micro:,} (device-batch-size x seq len x {ngpu} GPUs)")
    if sched is not None and sched.overshoot:
        errors.append(f"on {ngpu} GPUs the last step overshoots target-tokens by {sched.overshoot:,} tokens; the released "
                      f"run did not (use {cfg['gpus']} GPUs or a device-batch-size that divides the last step)")
    if tokens / TOKENS_PER_SHARD > 0.99 * TRAIN_SHARDS:
        errors.append(f"needs ~{tokens / TOKENS_PER_SHARD:.0f} train shards but only {TRAIN_SHARDS} exist; the data would repeat")
    return cfg, errors


def argv(train_args):
    out = []
    for k, v in train_args.items():
        if v is True:
            out.append(f"--{k}")
        elif v is not False and v is not None:
            out += [f"--{k}", str(v)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--check", action="store_true")
    p.add_argument("--ngpu", type=int, default=None, help="GPUs of this job (default: the config's gpus)")
    a = p.parse_args()
    cfg0 = yaml.safe_load(open(a.config))
    ngpu = a.ngpu or int(cfg0.get("gpus", 1))
    cfg, errors = load(a.config, ngpu)
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        sys.exit(1)
    ta = cfg["train_args"]
    steps, tokens, batches, sched = schedule(ta, ngpu)
    if a.check:
        print(f"config     {a.config}")
        print(f"run        {cfg['run']}   (attention {cfg['attention']}, {ngpu} GPUs; released run: {cfg['gpus']})")
        print(f"tokens     {tokens:,} = {tokens / 1e9:.2f}B over {steps:,} steps (~{tokens / TOKENS_PER_SHARD:,.0f} of {TRAIN_SHARDS:,} train shards)")
        if sched is not None:
            for b, s0, s1 in sched.phase_ranges():
                print(f"  batch {b:>10,}: steps {s0:>7,}-{s1:<7,} tokens {sched.tokens_before[s0] / 1e9:7.2f}B-{sched.tokens_before[s1 + 1] / 1e9:.2f}B")
            wr = float(ta.get("warmdown-ratio", 0.65))
            start = round(sched.canonical_horizon - round(wr * sched.canonical_horizon))
            st = next(i for i, c in enumerate(sched.canonical_before) if c >= start)
            print(f"  canonical horizon {sched.canonical_horizon:,.0f}; LR warmdown from step {st:,} ({sched.tokens_before[st] / 1e9:.2f}B tokens)")
        print(f"base_train {shlex.join(argv(ta))}")
        print("OK")
        return
    print(f"RUN={shlex.quote(str(cfg['run']))}")
    print(f"ATTN={shlex.quote(cfg['attention'])}")
    print(f"NUM_ITERS={steps}")
    print(f"ARGS=({' '.join(shlex.quote(x) for x in argv(ta))})")


if __name__ == "__main__":
    main()
