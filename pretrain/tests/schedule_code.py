"""Helpers for the schedule tests: load the trainers' own schedule code without running the trainers.

Both base_train.py files are scripts, so their schedule functions are pulled out of the source with `ast` and
executed in a namespace that provides what they read (args, the horizon, the schedule position)."""
import ast
import importlib.util
import math
import os
import re
import sys
import types

import yaml

PRETRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_TRAIN = os.path.join(PRETRAIN, "nanochat", "scripts", "base_train.py")
D34 = os.path.join(PRETRAIN, "d34_original_run")
D34_PRODUCTION = os.path.join(D34, "stage3_2m_production", "production")
D34_LAUNCH = os.path.join(D34, "stage1_ramp_to_92B", "launch_ramp.sh")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_schedule_module():
    return load_module(os.path.join(PRETRAIN, "nanochat", "nanochat", "batch_schedule.py"), "unified_batch_schedule")


def functions_from(path, names, namespace):
    """exec the top-level `def`s called `names` from the script at `path` into `namespace`; return them."""
    tree = ast.parse(open(path).read())
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(n.name for n in found) == sorted(names), (path, [n.name for n in found])
    exec(compile(ast.Module(body=found, type_ignores=[]), path, "exec"), namespace)
    return [namespace[n] for n in names]


def config(name):
    return yaml.safe_load(open(os.path.join(PRETRAIN, "configs", name)))


def unified_schedule(cfg):
    """Per optimizer step of a config: (batch, tokens_before, canonical_before, lr multiplier, Muon momentum,
    Muon weight decay relative to the first phase's amplitude), from the unified trainer's own code."""
    ta = cfg["train_args"]
    granularity = ta["device-batch-size"] * ta.get("max-seq-len", 2048) * cfg["gpus"]
    s = batch_schedule_module().build_batch_schedule(ta["batch-schedule"], ta["target-tokens"],
                                                     ta.get("canonical-batch", 2**20), granularity=granularity)
    ns = {
        "math": math,
        "args": types.SimpleNamespace(warmup_steps=ta.get("warmup-steps", 40), warmdown_ratio=ta["warmdown-ratio"],
                                      final_lr_frac=ta.get("final-lr-frac", 0.05),
                                      weight_decay_batch_cap=ta.get("weight-decay-batch-cap", -1)),
        "batch_schedule": s,
        "sched_pos": lambda it: s.canonical_before[it],
        "sched_horizon": s.canonical_horizon,
        "weight_decay_scaled": 1.0,
    }
    lr, mom, wd, phase = functions_from(UNIFIED_TRAIN, ["get_lr_multiplier", "get_muon_momentum", "get_weight_decay",
                                                        "phase_wd_scale"], ns)
    rows = []
    for i, b in enumerate(s.batch):
        rows.append((b, s.tokens_before[i], s.canonical_before[i], lr(i), mom(i), wd(i) * phase(b)))
    return rows, s


def d34_run_schedule():
    """The same per-step quantities for the original d34 run, from d34_original_run/'s code: stage 1 is nanochat's
    trainer run in legs with the STAGES and STOP_RAMP_AT of launch_ramp.sh (step = canonical step, horizon = the last
    STAGES end); stages 2-3 follow production_clock.TokenClock at batch 2M from the hand-off; the LR, momentum and
    weight-decay functions are the production trainer's, which are nanochat's evaluated on the canonical step."""
    sys.path.insert(0, D34_PRODUCTION)
    clock = load_module(os.path.join(D34_PRODUCTION, "production_clock.py"), "d34_production_clock")
    launch = open(D34_LAUNCH).read()
    stages = [tuple(int(x) for x in st.split(":")[:2]) for st in re.search(r'STAGES="([^"]+)"', launch).group(1).split(",")]
    stop = int(re.search(r"STOP_RAMP_AT=(\d+)", launch).group(1))
    horizon = stages[-1][0]
    assert horizon == clock.HORIZON and stop == clock.RAMP_SOURCE_STEP
    warmdown = float(re.search(r'WARMDOWN=([0-9.]+)', launch).group(1))
    production_train = os.path.join(D34_PRODUCTION, "base_train.py")
    # stage 3 holds the 1M weight-decay amplitude; stage 1 recomputes it from each leg's batch (nanochat's sqrt rule)
    assert "weight_decay_scaled = args.weight_decay * math.sqrt(REFERENCE_BATCH / B_REF) * (D_REF / target_tokens)" \
        in open(production_train).read()
    ns = {"math": math, "num_iterations": horizon, "weight_decay_scaled": 1.0,
          "args": types.SimpleNamespace(warmup_steps=40, warmdown_ratio=warmdown, final_lr_frac=0.05)}
    lr, mom, wd = functions_from(production_train, ["get_lr_multiplier", "get_muon_momentum", "get_weight_decay"], ns)
    first = stages[0][1]
    amp = lambda b: math.sqrt(min(b, clock.REFERENCE_BATCH) / first)
    rows, tokens = [], 0
    for step in range(stop):
        b = next(batch for end, batch in stages if step < end)
        rows.append((b, tokens, step, lr(step), mom(step), wd(step) * amp(b)))
        tokens += b
    assert tokens == clock.RAMP_SOURCE_TOKENS
    tc = clock.TokenClock(stop, stop, tokens, 2 * clock.REFERENCE_BATCH)
    for step in range(stop, tc.final_actual):
        c, b = tc.canonical(step), tc.update_tokens(step)
        assert lr(c) == clock.lr_multiplier(c)
        rows.append((b, tc.total(step), c, lr(c), mom(c), wd(c) * amp(b)))
    return rows
