"""
Self-filtering: use one of our own trained rerankers in place of the Gemma-4 listwise teacher.

Same rule as the Gemma-4 teacher filter of earlier runs, same output columns, so a training config can point at the
result unchanged: drop a positive when MORE than max(9, 0.375 * n_negatives) negatives outrank it; drop a
row when no positive survives; ArguAna is exempt (counter-argument retrieval inverts a relevance judge).
The only difference is who ranks the candidates -- our checkpoint instead of Gemma, which is ~2 GPU-hours
of pointwise scoring instead of 15.4 H200-hours of listwise generation.

Why this rather than distilling the checkpoint's order (measured on run G, 2026-09-26):
  * its order agrees with Gemma's on 0.875 of nDCG-weighted pairs, and on pairs where they disagree it is
    right 49.3% of the time against Gemma's 50.7% -- so its ORDER carries nothing new;
  * but it puts a true positive first far more often than Gemma (P@1 0.87-0.90 vs 0.83), and that holds on
    rows it never trained on (0.862 vs 0.844 macro P@1, seen vs unseen), which is exactly the
    top-of-list judgement the filter rule depends on.

Two modes:
  score  one shard of rows -> <work>/scores_<shard>.npz   (array job; length-sorted token-budget batches)
  merge  all shards -> <out>/train.parquet + stats.json   (ranks from our scores, filter applied)

  python data_prep/self_filter.py score --ckpt <ckpt> --shard 0 --shards 8 --work <dir>
  python data_prep/self_filter.py merge --work <dir> --out data/rlhn250k_self_filtered
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RECIPE_ROOT", os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC_DEFAULT = f"{ROOT}/data/rlhn250k_prepared_climbmix_lens/train.parquet"


def row_groups_for(pf, lo, hi):
    """Row groups covering global rows [lo, hi), with each group's global start."""
    starts, acc = [], 0
    for i in range(pf.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        if acc < hi and acc + n > lo:
            starts.append((i, acc))
        acc += n
    return starts


def do_score(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from prompt_builder_nc import PairBuilder, yes_no_ids, setup_tokenizer

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = sd["cfg"]
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"]))
    prompt = cfg.get("prompt", "chat")
    yes_id, no_id = yes_no_ids(tok, prompt, cfg.get("answer_tokens"))
    pb = PairBuilder(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"], prompt=prompt)
    if cfg.get("backbone_impl", "hf") == "upstream":
        from modeling_nanochat_upstream import UpstreamNanochatReranker
        m = UpstreamNanochatReranker(cfg["model_name"], yes_id=yes_id, no_id=no_id, dtype=dtype,
                                     nanochat_src=cfg.get("nanochat_src"), step=cfg.get("model_step"))
    else:
        from modeling_nanochat_rr import NanochatReranker
        m = NanochatReranker(cfg["model_name"], yes_id=yes_id, no_id=no_id, dtype=dtype,
                             attn_implementation=cfg.get("attn_implementation", "sdpa"))
    m.load_state_dict(sd["model"], strict=True)
    m = m.to("cuda", dtype=dtype).eval()
    m.head.data = m.head.data.float()

    pf = pq.ParquetFile(a.src)
    total = pf.metadata.num_rows
    per = (total + a.shards - 1) // a.shards
    lo, hi = a.shard * per, min(total, (a.shard + 1) * per)
    print(f"shard {a.shard}/{a.shards}: rows [{lo}, {hi}) of {total:,}", flush=True)

    @torch.no_grad()
    def score_batch(seqs):
        n = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
        return m(ids.to("cuda", non_blocking=True), att.to("cuda", non_blocking=True)).scores.float().tolist()

    all_scores, n_pos, n_neg, rows_done, t0 = [], [], [], 0, time.time()
    buf = []          # (global_slot, token_ids)
    slot = 0
    def flush():
        nonlocal buf
        if not buf: return
        buf.sort(key=lambda x: len(x[1]))          # length-sorted: far less padding
        i = 0
        while i < len(buf):
            j, tokens = i, 0
            while j < len(buf) and (j == i or tokens + len(buf[j][1]) <= a.max_tokens):
                tokens += len(buf[j][1]); j += 1
            for s, v in zip([x[0] for x in buf[i:j]], score_batch([x[1] for x in buf[i:j]])):
                all_scores[s] = v
            i = j
        buf = []

    for rg, gstart in row_groups_for(pf, lo, hi):
        t = pf.read_row_group(rg, columns=["query", "pos_texts", "neg_texts"])
        qs, ps, ns = (t.column(c).to_pylist() for c in ("query", "pos_texts", "neg_texts"))
        for k in range(len(qs)):
            g = gstart + k
            if not (lo <= g < hi):
                continue
            docs = list(ps[k]) + list(ns[k])
            n_pos.append(len(ps[k])); n_neg.append(len(ns[k]))
            all_scores.extend([0.0] * len(docs))
            for d in docs:
                buf.append((slot, pb.build(qs[k], d))); slot += 1
            rows_done += 1
            if len(buf) >= a.buffer:
                flush()
                if rows_done % 2000 < 64:
                    el = time.time() - t0
                    print(f"  {rows_done:,} rows, {slot:,} candidates, {el:.0f}s "
                          f"({slot/max(el,1):.0f} cand/s, eta {(hi-lo-rows_done)*(el/max(rows_done,1))/60:.0f} min)", flush=True)
    flush()
    os.makedirs(a.work, exist_ok=True)
    np.savez(os.path.join(a.work, f"scores_{a.shard:02d}.npz"), lo=lo, hi=hi,
             scores=np.array(all_scores, dtype=np.float32),
             n_pos=np.array(n_pos, dtype=np.int32), n_neg=np.array(n_neg, dtype=np.int32))
    print(f"shard {a.shard} done: {rows_done:,} rows, {slot:,} candidates in {(time.time()-t0)/60:.1f} min", flush=True)


def do_merge(a):
    import pyarrow as pa, pyarrow.parquet as pq
    shards = sorted(f for f in os.listdir(a.work) if f.startswith("scores_") and f.endswith(".npz"))
    assert shards, f"no scores_*.npz in {a.work}"
    scores_by_row, expect = {}, 0
    for f in shards:
        z = np.load(os.path.join(a.work, f))
        lo, hi = int(z["lo"]), int(z["hi"])
        sc, npos, nneg = z["scores"], z["n_pos"], z["n_neg"]
        assert len(npos) == hi - lo, (f, len(npos), hi - lo)
        off = 0
        for i in range(hi - lo):
            k = int(npos[i]) + int(nneg[i])
            scores_by_row[lo + i] = (sc[off:off + int(npos[i])], sc[off + int(npos[i]):off + k])
            off += k
        assert off == len(sc), (f, off, len(sc))
        expect += hi - lo
    print(f"{len(shards)} shards, scores for {len(scores_by_row):,} rows", flush=True)

    pf = pq.ParquetFile(a.src)
    assert pf.metadata.num_rows == len(scores_by_row), (pf.metadata.num_rows, len(scores_by_row))
    names = pf.schema_arrow.names
    out = {c: [] for c in names}
    for extra in ("pos_trank", "neg_trank", "pos_teacher", "neg_teacher", "orig_row",
                  "dropped_pos_texts", "dropped_pos_neg_ahead"):
        out[extra] = []
    st = {"rows_in": 0, "rows_out": 0, "pos": 0, "pos_dropped": 0, "rows_removed": 0}
    per_sub = defaultdict(lambda: {"rows": 0, "pos_dropped": 0, "rows_removed": 0})
    g = 0
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "train.parquet")
    # One explicit schema for every batch: inferring per batch gives different types when a batch
    # happens to hold only empty lists, and the writer then rejects it.
    schema = pa.schema(list(pf.schema_arrow) + [
        pa.field("pos_trank", pa.list_(pa.int32())), pa.field("neg_trank", pa.list_(pa.int32())),
        # raw scores as well as ranks: a pointwise judge gives magnitudes, which order-only
        # distillation throws away (losses_nc.kd_loss reads these)
        pa.field("pos_teacher", pa.list_(pa.float32())), pa.field("neg_teacher", pa.list_(pa.float32())),
        pa.field("orig_row", pa.int64()), pa.field("dropped_pos_texts", pa.list_(pa.string())),
        pa.field("dropped_pos_neg_ahead", pa.list_(pa.int32()))])
    writer = [None]

    def flush(final=False):
        """Write the buffered rows as one row group. Rows are written in batches, never as a single
        table: with these nested list columns a whole-file table is chunked, and pyarrow then refuses
        ("Nested data conversions not implemented for chunked array outputs"), as does HF datasets."""
        if not out["query"] and not final:
            return
        if out["query"]:
            batch = pa.table({k: v for k, v in out.items()}, schema=schema)
            if writer[0] is None:
                writer[0] = pq.ParquetWriter(dst, schema)
            writer[0].write_table(batch)
            for v in out.values():
                v.clear()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        cols = {c: t.column(c).to_pylist() for c in names}
        for i in range(t.num_rows):
            ps, ns = scores_by_row[g]; g += 1
            sub = cols["subset"][i]
            st["rows_in"] += 1; per_sub[sub]["rows"] += 1
            npos, nneg = len(ps), len(ns)
            st["pos"] += npos
            # ranks over ALL candidates of the row, 1 = best (highest score), like the teacher's ranks
            allsc = np.concatenate([ps, ns])
            order = np.argsort(-allsc, kind="stable")
            rank = np.empty(len(allsc), dtype=np.int32)
            rank[order] = np.arange(1, len(allsc) + 1, dtype=np.int32)
            pr, nr = rank[:npos].tolist(), rank[npos:].tolist()
            ahead = [int((ns > p).sum()) for p in ps]          # negatives outranking this positive
            thr = max(a.min_ahead, a.frac_ahead * nneg)
            keep = [True] * npos if sub in a.exempt else [x <= thr for x in ahead]
            st["pos_dropped"] += npos - sum(keep)
            per_sub[sub]["pos_dropped"] += npos - sum(keep)
            if not any(keep):
                st["rows_removed"] += 1; per_sub[sub]["rows_removed"] += 1
                continue
            for c in names:
                v = cols[c][i]
                if c in ("pos_texts", "pos_lens"):
                    v = [x for x, k in zip(v, keep) if k]
                out[c].append(v)
            out["pos_trank"].append([r for r, k in zip(pr, keep) if k])
            out["neg_trank"].append(nr)
            out["pos_teacher"].append([float(x) for x, k in zip(ps, keep) if k])
            out["neg_teacher"].append([float(x) for x in ns])
            out["orig_row"].append(g - 1)
            out["dropped_pos_texts"].append([x for x, k in zip(cols["pos_texts"][i], keep) if not k])
            out["dropped_pos_neg_ahead"].append([int(x) for x, k in zip(ahead, keep) if not k])
            st["rows_out"] += 1
            if len(out["query"]) >= 2000:
                flush()
    flush(final=True)
    writer[0].close()
    st["rule"] = f"drop positive if negatives_ahead > max({a.min_ahead}, {a.frac_ahead} * n_negatives); exempt {a.exempt}"
    st["judge"] = a.judge or "our own checkpoint (self-filtering)"
    st["by_subset"] = {k: dict(v) for k, v in per_sub.items()}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "by_subset"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "merge"])
    ap.add_argument("--ckpt")
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--work", default=f"{ROOT}/data/_self_scores")
    ap.add_argument("--out", default=f"{ROOT}/data/rlhn250k_self_filtered")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768, help="tokens per scoring batch")
    ap.add_argument("--buffer", type=int, default=4096, help="candidates buffered before length-sorted batching")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--min-ahead", type=float, default=9)
    ap.add_argument("--frac-ahead", type=float, default=0.375)
    ap.add_argument("--exempt", nargs="*", default=["arguana"])
    ap.add_argument("--judge", default=None)
    a = ap.parse_args()
    if a.mode == "score":
        assert a.ckpt, "--ckpt is required for scoring"
        do_score(a)
    else:
        do_merge(a)


if __name__ == "__main__":
    main()
