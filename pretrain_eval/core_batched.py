"""CORE (DCLM ICL suite) exactly as nanochat renders and scores it, batched across examples.
Prompt rendering, few-shot sampling (Random(1234+idx)), task shuffling (Random(1337)), BOS prepending,
continuation spans, left truncation and the scoring rules are nanochat's own functions/logic; only the
forward pass is batched (right-padded, causal), which changes nothing but floating-point order."""
import argparse, csv, importlib.util, json, os, random, time, yaml, torch, torch.nn.functional as F
from models import load

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True); p.add_argument("--src"); p.add_argument("--ckpt"); p.add_argument("--step", type=int)
p.add_argument("--hf"); p.add_argument("--bundle", required=True); p.add_argument("--core-eval", required=True)
p.add_argument("--out", required=True); p.add_argument("--max-per-task", type=int, default=-1)
p.add_argument("--max-seq-len", type=int, default=2048, help="left-truncate every sequence to this many tokens; 0 = none")
p.add_argument("--token-budget", type=int, default=24576); p.add_argument("--unbatched", action="store_true")
p.add_argument("--tasks", default=None)
a = p.parse_args()

spec = importlib.util.spec_from_file_location("head_core_eval", a.core_eval); ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
t0 = time.time()
model, tok, native_ctx, desc = load(a.kind, a.src, a.ckpt, a.step, a.hf)
print(f"loaded {desc} in {time.time()-t0:.0f}s native_ctx={native_ctx} truncate_at={a.max_seq_len}", flush=True)
BOS = tok.get_bos_token_id(); dev = torch.device("cuda")

def build(idx, data, meta):
    item = data[idx]; k = meta["num_fewshot"]; delim = meta["continuation_delimiter"]; tt = meta["task_type"]
    few = []
    if k > 0:
        rng = random.Random(1234 + idx)
        few = [data[i] for i in rng.sample([i for i in range(len(data)) if i != idx], k)]
    if tt == "multiple_choice":
        toks, s, e = ce.batch_sequences_mc(tok, ce.render_prompts_mc(item, delim, few))
    elif tt == "schema":
        toks, s, e = ce.batch_sequences_schema(tok, ce.render_prompts_schema(item, delim, few))
    else:
        toks, s, e = ce.batch_sequences_lm(tok, ce.render_prompts_lm(item, delim, few))
    truncated = False
    if a.max_seq_len > 0:
        nt, ns, ne = [], [], []
        for t, si, ei in zip(toks, s, e):
            if len(t) > a.max_seq_len:
                c = len(t) - a.max_seq_len; truncated = True
                assert si - c >= 0 and ei - c >= 0
                nt.append(t[-a.max_seq_len:]); ns.append(si - c); ne.append(ei - c)
            else:
                nt.append(t); ns.append(si); ne.append(ei)
        toks, s, e = nt, ns, ne
    return toks, s, e, tt, item.get("gold"), truncated

@torch.no_grad()
def score(rows):
    """rows: list of (tokens, s, e). Returns per-row (mean_loss, greedy_ok) over positions s-1..e-2."""
    T = max(len(t) for t, _, _ in rows)
    x = torch.full((len(rows), T), BOS, dtype=torch.long)
    for i, (t, _, _) in enumerate(rows):
        x[i, :len(t)] = torch.tensor(t)
    x = x.to(dev)
    logits = model(x)
    out = []
    for i, (t, s, e) in enumerate(rows):
        lg = logits[i, s - 1:e - 1].float()
        tgt = x[i, s:e]
        loss = F.cross_entropy(lg, tgt, reduction="none").mean().item()
        ok = bool((lg.argmax(-1) == tgt).all().item())
        out.append((loss, ok))
    return out

cfg = yaml.safe_load(open(os.path.join(a.bundle, "core.yaml")))
baselines = {r["Eval Task"]: float(r["Random baseline"]) for r in csv.DictReader(open(os.path.join(a.bundle, "eval_meta_data.csv")))}
results, centered, info = {}, {}, {}
for task in cfg["icl_tasks"]:
    label = task["label"]
    if a.tasks and label not in a.tasks.split(","): continue
    ts = time.time()
    meta = {"task_type": task["icl_task_type"], "num_fewshot": task["num_fewshot"][0], "continuation_delimiter": task.get("continuation_delimiter", " ")}
    data = [json.loads(l) for l in open(os.path.join(a.bundle, "eval_data", task["dataset_uri"]))]
    random.Random(1337).shuffle(data)
    if a.max_per_task > 0: data = data[:a.max_per_task]
    ex = [build(i, data, meta) for i in range(len(data))]
    ntrunc = sum(1 for x in ex if x[5])
    order = sorted(range(len(ex)), key=lambda i: max(len(t) for t in ex[i][0]))
    correct = [None] * len(ex)
    b = 0
    while b < len(order):
        batch = [order[b]]; T = max(len(t) for t in ex[order[b]][0]); rows = len(ex[order[b]][0])
        if not a.unbatched:
            while b + len(batch) < len(order):
                j = order[b + len(batch)]; Tj = max(T, max(len(t) for t in ex[j][0])); rj = rows + len(ex[j][0])
                if Tj * rj > a.token_budget: break
                batch.append(j); T, rows = Tj, rj
        flat, owner = [], []
        for j in batch:
            toks, s, e, tt, gold, _ = ex[j]
            for r in range(len(toks)): flat.append((toks[r], s[r], e[r])); owner.append(j)
        sc = score(flat)
        k = 0
        for j in batch:
            toks, s, e, tt, gold, _ = ex[j]; n = len(toks); part = sc[k:k + n]; k += n
            if tt == "language_modeling":
                correct[j] = part[0][1]
            else:
                ml = [x[0] for x in part]; correct[j] = (ml.index(min(ml)) == gold)
        b += len(batch)
    acc = sum(correct) / len(correct)
    rb = baselines[label]; cen = (acc - 0.01 * rb) / (1.0 - 0.01 * rb)
    results[label], centered[label] = acc, cen
    info[label] = {"n": len(ex), "truncated": ntrunc, "seconds": round(time.time() - ts, 1)}
    print(f"{label:36s} n={len(ex):6d} trunc={ntrunc:5d} acc={acc:.4f} centered={cen:.4f} {time.time()-ts:7.1f}s", flush=True)

core = sum(centered.values()) / len(centered)
json.dump({"model": desc, "core_metric": core, "results": results, "centered_results": centered, "info": info,
           "max_per_task": a.max_per_task, "max_seq_len": a.max_seq_len, "batched": not a.unbatched,
           "torch": torch.__version__, "seconds": time.time() - t0}, open(a.out, "w"), indent=2)
print(f"CORE {core:.4f} | {desc} | total {(time.time()-t0)/60:.1f} min", flush=True)
