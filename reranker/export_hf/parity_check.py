#!/usr/bin/env python
"""Score-level parity between the training/eval wrapper and the transformers port.

Both models are loaded on the same GPU in bf16, exactly as every published number was produced,
and scored on real (query, passage) pairs from the cached BM25 top-100 runs.

Three comparisons, because they fail for different reasons:
  batch1    one pair per forward, so no padding and no explicit mask -> isolates the port's maths
  ragged    the harness's own batching (right padding, mask built by the port) -> isolates padding
  leftpad   the same pairs left-padded, which the wrapper does not support -> checks the port alone
            is padding-side invariant
"""
import argparse, json, os, sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(RECIPE, "src"))
sys.path.insert(0, os.path.join(RECIPE, "..", "eval"))


def load_reference(ckpt, dev, dtype):
    from transformers import AutoTokenizer
    from prompt_builder_nc import PairBuilder, readout_offset, setup_tokenizer, yes_no_ids
    from modeling_nanochat_upstream import UpstreamNanochatReranker
    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = dict(sd["cfg"])
    bb = cfg["model_name"]
    if not os.path.exists(bb):
        bb = os.path.join(RECIPE, "models", os.path.basename(bb.rstrip("/")))
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(bb))
    prompt = cfg.get("prompt", "chat")
    yes_id, no_id = yes_no_ids(tok, prompt, cfg.get("answer_tokens"))
    pb = PairBuilder(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"],
                     prompt=prompt)
    m = UpstreamNanochatReranker(bb, yes_id=yes_id, no_id=no_id, dtype=dtype,
                                 nanochat_src=os.environ.get("EVAL_NANOCHAT_SRC",
                                                             str(Path(__file__).resolve().parents[1] / "third_party/nanochat-92d63d4")),
                                 readout_offset=readout_offset(prompt))
    m.load_state_dict(sd["model"], strict=True)
    m = m.to(dev, dtype=dtype).eval()
    m.head.data = m.head.data.float()
    return m, tok, pb


@torch.no_grad()
def ref_scores(m, tok, pb, pairs, bs, pad_side="right"):
    out = []
    for s0 in range(0, len(pairs), bs):
        seqs = [pb.build(q, d) for q, d in pairs[s0:s0 + bs]]
        n = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            if pad_side == "right":
                ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            else:
                ids[i, n - len(x):] = torch.tensor(x); att[i, n - len(x):] = 1
        out += m(ids.to(m.head.device), att.to(m.head.device)).scores.float().tolist()
    return out


@torch.no_grad()
def port_scores(model, tok, pairs, bs, pad_side="right"):
    dev = next(model.parameters()).device
    out = []
    for s0 in range(0, len(pairs), bs):
        seqs = [model.build_input_ids(tok, q, d) for q, d in pairs[s0:s0 + bs]]
        n = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), n), model.config.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            if pad_side == "right":
                ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            else:
                ids[i, n - len(x):] = torch.tensor(x); att[i, n - len(x):] = 1
        out += model(ids.to(dev), att.to(dev)).logits[:, 0].float().tolist()
    return out


def stats(a, b):
    import statistics
    d = [x - y for x, y in zip(a, b)]
    ad = [abs(x) for x in d]
    rng = max(max(a), max(b)) - min(min(a), min(b))
    return {"n": len(a), "max_abs": max(ad), "mean_abs": sum(ad) / len(ad),
            "max_abs_rel_range": max(ad) / rng if rng else 0.0,
            "score_range": rng, "mean_ref": statistics.fmean(a), "mean_port": statistics.fmean(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="converted repo directory")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    runs = os.environ.get("EVAL_RUNS", "data/bm25_top100")
    rows = [json.loads(l) for l in
            open(os.path.join(runs, f"retrieve_results_{args.dataset}_top100.jsonl"))][:args.queries]
    sys.path.insert(0, os.path.join(RECIPE, "..", "eval"))
    from rerank_eval import doc_text
    pairs, qbounds = [], []
    for r in rows:
        start = len(pairs)
        pairs += [(r["query"]["text"], doc_text(c)) for c in r["candidates"]]
        qbounds.append((start, len(pairs)))
    print(f"{len(rows)} queries, {len(pairs)} pairs from {args.dataset}", flush=True)

    dev, dtype = torch.device("cuda"), torch.bfloat16
    ref, tok, pb = load_reference(args.ckpt, dev, dtype)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    port = AutoModelForSequenceClassification.from_pretrained(
        args.repo, trust_remote_code=True, dtype=dtype).to(dev).eval()
    ptok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    # the rows follow the load dtype, as in the wrapper; the matmul is fp32 in both
    print(f"port score row dtype: {port.score.weight.dtype} (expected {dtype})", flush=True)
    assert port.score.weight.dtype == dtype

    # the prompts themselves must be identical, before any arithmetic
    bad = [i for i, (q, d) in enumerate(pairs[:200])
           if pb.build(q, d) != port.build_input_ids(ptok, q, d)]
    print(f"prompt id mismatch on {len(bad)}/200 pairs", flush=True)
    assert not bad

    res = {}
    for label, bs, side in (("batch1", 1, "right"), ("ragged", args.batch_size, "right")):
        a = ref_scores(ref, tok, pb, pairs, bs, side)
        b = port_scores(port, ptok, pairs, bs, side)
        res[label] = stats(a, b)
        same_order = sum(
            sorted(range(s, e), key=lambda i: -a[i]) == sorted(range(s, e), key=lambda i: -b[i])
            for s, e in qbounds)
        res[label]["queries_with_identical_ranking"] = f"{same_order}/{len(qbounds)}"
        print(label, json.dumps(res[label], indent=1), flush=True)

    # the port alone: right vs left padding
    r = port_scores(port, ptok, pairs, args.batch_size, "right")
    l = port_scores(port, ptok, pairs, args.batch_size, "left")
    res["leftpad_vs_rightpad_port_only"] = stats(r, l)
    print("leftpad", json.dumps(res["leftpad_vs_rightpad_port_only"], indent=1), flush=True)

    print("### PARITY ###")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
