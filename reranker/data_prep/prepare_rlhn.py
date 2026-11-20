"""
Stage A: offline prep of rlhn/rlhn-680K into training-ready groups.

Design note: we deliberately do NOT store token ids. Storing ~649k x ~28 passages
of int32 ids would be ~10-35GB, and HF fast tokenizers comfortably out-run the
training step anyway. Instead we store cleaned text + a precomputed token LENGTH
per passage, which is all the length-aware batching in Stage B needs.

Output schema (parquet):
    query_id   : str
    subset     : str
    query      : str
    query_len  : int32                 # token length, uncapped
    pos_texts  : list[str]             # title+text normalised
    pos_lens   : list[int32]
    neg_texts  : list[str]
    neg_lens   : list[int32]
"""
import argparse, json, math, os, random, sys
from collections import Counter, defaultdict

def norm_passage(p):
    title = (p.get("title") or "").strip()
    text = (p.get("text") or "").strip()
    if not text:
        return None
    return f"{title}\n{text}" if title else text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rlhn/rlhn-680K")
    ap.add_argument("--json-file", default=None,
                    help="load this local jsonl/jsonl.gz instead of the hub dataset, so the job needs "
                         "no network (Tevatron/msmarco-passage ships train.jsonl.gz)")
    ap.add_argument("--subset-name", default=None,
                    help="value for the 'subset' column when the source has none (Tevatron does not); "
                         "the sampler and the per-subset KD/RKD weights key off this")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="data/rlhn_prepared")
    ap.add_argument("--min-negatives", type=int, default=7,
                    help="drop rows with fewer than this many usable negatives")
    ap.add_argument("--subset-temp", type=float, default=0.5,
                    help="p_s ~ n_s^T. 1.0=natural (75%% msmarco), 0.5=balanced, 0.0=uniform")
    ap.add_argument("--total-budget", type=int, default=0,
                    help="rows to keep after optional subset rebalancing. "
                         "0 (default) = KEEP EVERY ROW, no rebalancing.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only load N rows")
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    split = f"train[:{args.limit}]" if args.limit else "train"
    if args.json_file:
        ds = load_dataset("json", data_files=args.json_file, split="train")
        print(f"loaded {len(ds):,} rows from {args.json_file}", flush=True)
    else:
        ds = load_dataset(args.dataset, split=split)
    if "subset" not in ds.column_names:
        name = args.subset_name or args.dataset.split("/")[-1]
        ds = ds.add_column("subset", [name] * len(ds))
        print(f"source has no 'subset' column: filled with {name!r}", flush=True)
    print(f"loaded {len(ds):,} rows | cols={ds.column_names}", flush=True)

    # ---- pass 1: clean, dedupe, drop unusable rows -------------------------
    def clean(batch):
        out = {k: [] for k in
               ("query_id","subset","query","pos_texts","neg_texts")}
        for qid, sub, q, pos, neg in zip(batch["query_id"], batch["subset"],
                                         batch["query"], batch["positive_passages"],
                                         batch["negative_passages"]):
            q = (q or "").strip()
            if not q:
                continue
            seen, ptexts, pdoc = set(), [], set()
            for p in pos:
                t = norm_passage(p)
                if t and t not in seen:
                    seen.add(t); ptexts.append(t); pdoc.add(p.get("docid"))
            ntexts = []
            for p in neg:
                t = norm_passage(p)
                # guard against a "negative" that is textually or id-identical to a positive
                if t and t not in seen and p.get("docid") not in pdoc:
                    seen.add(t); ntexts.append(t)
            if not ptexts or len(ntexts) < args.min_negatives:
                continue
            out["query_id"].append(qid); out["subset"].append(sub); out["query"].append(q)
            out["pos_texts"].append(ptexts); out["neg_texts"].append(ntexts)
        return out

    ds = ds.map(clean, batched=True, batch_size=1000, num_proc=args.num_proc,
                remove_columns=ds.column_names, desc="clean+dedupe")
    print(f"after clean/dedupe: {len(ds):,} rows", flush=True)

    counts = Counter(ds["subset"])
    print("subset counts (post-filter):")
    for s, n in counts.most_common():
        print(f"  {s:18s} {n:>8,}  {n/len(ds):6.2%}")

    # ---- subset rebalance: p_s ~ n_s^T, downsample only, redistribute ------
    T, budget = args.subset_temp, args.total_budget
    if budget <= 0:
        print("\n--total-budget 0: keeping ALL rows, no subset rebalancing.")
    else:
        w = {s: n ** T for s, n in counts.items()}
        Z = sum(w.values())
        quota = {s: min(counts[s], int(round(budget * w[s] / Z))) for s in counts}
        for _ in range(3):                    # hand leftover back to subsets with room
            leftover = budget - sum(quota.values())
            if leftover <= 0: break
            room = {s: counts[s] - quota[s] for s in counts if counts[s] > quota[s]}
            if not room: break
            Zr = sum(w[s] for s in room)
            for s in room:
                quota[s] = min(counts[s], quota[s] + int(leftover * w[s] / Zr))

        print(f"\nrebalanced (T={T}, budget={budget:,}):")
        tot = max(1, sum(quota.values()))
        for sub in sorted(quota, key=lambda x: -quota[x]):
            print(f"  {sub:18s} {quota[sub]:>8,}  {quota[sub]/tot:6.2%}   (from {counts[sub]:,})")

        by_subset = defaultdict(list)
        for i, sub in enumerate(ds["subset"]):
            by_subset[sub].append(i)
        keep = []
        for sub, idxs in by_subset.items():
            rng.shuffle(idxs)
            keep.extend(idxs[: quota[sub]])
        rng.shuffle(keep)
        ds = ds.select(keep)
        print(f"\nafter rebalance: {len(ds):,} rows", flush=True)

    # ---- pass 2: token lengths (batched, fast tokenizer) -------------------
    def add_lens(batch):
        qs = [q for q in batch["query"]]
        batch["query_len"] = [len(x) for x in tok(qs, add_special_tokens=False)["input_ids"]]
        for fld, out in (("pos_texts","pos_lens"), ("neg_texts","neg_lens")):
            flat, spans = [], []
            for lst in batch[fld]:
                spans.append(len(flat)); flat.extend(lst)
            enc = tok(flat, add_special_tokens=False)["input_ids"]
            lens, c = [], 0
            for i, lst in enumerate(batch[fld]):
                lens.append([len(enc[c + j]) for j in range(len(lst))]); c += len(lst)
            batch[out] = lens
        return batch

    ds = ds.map(add_lens, batched=True, batch_size=200, num_proc=args.num_proc,
                desc="token lengths")

    os.makedirs(args.out, exist_ok=True)
    ds.to_parquet(os.path.join(args.out, "train.parquet"))

    # ---- stats we actually need for Stage B budgeting ---------------------
    import numpy as np
    npos = np.array([len(x) for x in ds["pos_texts"]])
    nneg = np.array([len(x) for x in ds["neg_texts"]])
    qlen = np.array(ds["query_len"])
    dlen = np.array([l for lst in ds["neg_lens"] for l in lst])
    stats = {
        "rows": len(ds),
        "subset_counts": dict(Counter(ds["subset"])),
        "n_positives":  {"mean": float(npos.mean()), "p50": int(np.percentile(npos,50)),
                         "p90": int(np.percentile(npos,90)), "max": int(npos.max())},
        "n_negatives":  {"mean": float(nneg.mean()), "min": int(nneg.min()),
                         "p10": int(np.percentile(nneg,10)), "p50": int(np.percentile(nneg,50)),
                         "max": int(nneg.max())},
        "query_tokens": {"mean": float(qlen.mean()), "p50": int(np.percentile(qlen,50)),
                         "p95": int(np.percentile(qlen,95)), "p99": int(np.percentile(qlen,99)),
                         "max": int(qlen.max())},
        "doc_tokens":   {"mean": float(dlen.mean()), "p50": int(np.percentile(dlen,50)),
                         "p95": int(np.percentile(dlen,95)), "p99": int(np.percentile(dlen,99)),
                         "max": int(dlen.max())},
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("\n" + json.dumps(stats, indent=2))
    print(f"\nwrote {args.out}/train.parquet")

if __name__ == "__main__":
    main()
