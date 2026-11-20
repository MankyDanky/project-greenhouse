"""
Rerank a cached first-stage run and score it with trec_eval.

Works directly off rank_llm's cached BM25 jsonl files, which are self-contained
(query text + 100 candidates WITH document text). That means no pyserini and no
lucene indexes at scoring time (eval/candidates/ builds the files once).

Backends:
  ours       a checkpoint trained by reranker/, score = logit(" true") - logit(" false")   [pointwise]
  gaggle_hf  the released transformers port of the same reranker (reranker/export_hf)     [pointwise]
  monot5     seq2seq baseline, score = logit(true) - logit(false)                          [pointwise]
  qwen3      Qwen3-Reranker baseline, score = logit(yes) - logit(no)                       [pointwise]
  jinav3     released jina-reranker-v3, model.rerank()                                     [listwise]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import torch

# Overridable (EVAL_RUNS, EVAL_QRELS_DIR) so the harness runs on machines without a shared filesystem.
HERE = os.path.dirname(os.path.abspath(__file__))
# first-stage runs: rank_llm's cached BM25 top-100 jsonl files (query text + 100 candidates WITH text)
RUNS = os.environ.get("EVAL_RUNS", os.path.join(HERE, "..", "data", "bm25_top100"))
# project copy (from anserini/tools/topics-and-qrels; SciDocs/FiQA renamed to the old pyserini-cache names):
# ~/.cache was wiped on 2026-09-21, and re-scoring saved runs with these files reproduced every stored number
QRELS_DIR = os.environ.get("EVAL_QRELS_DIR", os.path.join(HERE, "qrels"))   # eval/fetch_qrels.sh fills this
def _find_trec_eval():
    """`trec_eval` on PATH, else the binaries anserini/pyserini ship, else fail loudly later.

    Not a nicety: an eval array on 2026-09-30 reranked all 12 datasets, then died at the scoring
    step on every task because the login shell had `trec_eval` and the compute nodes did not. The
    run files survived and were rescored by hand; this avoids the round trip. Note that
    anserini/bin/trec_eval is a JVM wrapper that needs a built fatjar -- the real binary is the one
    under tools/eval.
    """
    import shutil
    found = shutil.which(os.environ.get("EVAL_TREC_EVAL", "trec_eval"))
    if found:
        return found
    for cand in (os.path.expanduser("~/anserini/tools/eval/trec_eval.9.0.4/trec_eval"),
                 os.path.expanduser("~/pyserini/tools/eval/trec_eval.9.0.4/trec_eval")):
        if os.access(cand, os.X_OK):
            print(f"trec_eval not on PATH; using {cand}", flush=True)
            return cand
    return os.environ.get("EVAL_TREC_EVAL", "trec_eval")


TREC_EVAL = _find_trec_eval()

QRELS = {
    "dl19": "qrels.dl19-passage.txt",
    "dl20": "qrels.dl20-passage.txt",
    "dl21": "qrels.dl21-passage.txt",
    "dl22": "qrels.dl22-passage.txt",
    "dl23": "qrels.dl23-passage.txt",
    "covid": "qrels.beir-v1.0.0-trec-covid.test.txt",
    "news": "qrels.beir-v1.0.0-trec-news.test.txt",
    "robust04": "qrels.beir-v1.0.0-robust04.test.txt",
    "nfc": "qrels.beir-v1.0.0-nfcorpus.test.txt",
    "scifact": "qrels.beir-v1.0.0-scifact.test.txt",
    "scidocs": "qrels.beir-v1.0.0-scidocs-test.txt",
    "fiqa": "qrels.beir-v1.0.0-fiqa-test.txt",
    "msmarco-dev": "qrels.msmarco-passage-dev-sample1000.txt",   # held-out dev, diagnostics only
}


def doc_text(cand, max_chars=None):
    d = cand.get("doc") or {}
    t = d.get("contents") or d.get("text") or ""
    title = d.get("title") or ""
    if title and not t.startswith(title):
        t = f"{title}\n{t}"
    t = " ".join(t.split())
    return t[:max_chars] if max_chars else t


# ----------------------------------------------------------------- backends
def load_monot5(name, dev, dtype):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    t_id = tok.encode("true", add_special_tokens=False)[0]
    f_id = tok.encode("false", add_special_tokens=False)[0]

    @torch.no_grad()
    def score(query, docs, bs, max_len):
        out = []
        texts = [f"Query: {query} Document: {d} Relevant: " for d in docs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [0.0] * len(texts)
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            b = tok([texts[i] for i in sel], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev)
            dec = torch.full((len(sel), 1), m.config.decoder_start_token_id, device=dev)
            lg = m(**b, decoder_input_ids=dec).logits[:, 0, :]
            for i, v in zip(sel, (lg[:, t_id].float() - lg[:, f_id].float()).tolist()):
                res[i] = v
        return res
    return score


def load_qwen3(name, dev, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the '
                     'requirements based on the Query and the Instruct provided. Note '
                     'that the answer can only be "yes" or "no".<|im_end|>\n'
                     '<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     add_special_tokens=False)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    task = "Given a web search query, retrieve relevant passages that answer the query"
    # Per-field caps (EVAL_MAX_QUERY_LEN / EVAL_MAX_DOC_LEN). Unset = the historical behaviour:
    # head-truncate the ASSEMBLED instruct+query+document string at --max-len. Set = cap the query
    # and the document separately, as our own checkpoints do (128/512), so the document always gets
    # its full budget instead of whatever the query leaves over. Fields are tokenised on their own,
    # which can differ from in-context tokenisation at the boundaries by a token.
    QCAP = int(os.environ.get("EVAL_MAX_QUERY_LEN", 0))
    DCAP = int(os.environ.get("EVAL_MAX_DOC_LEN", 0))
    ins = tok.encode(f"<Instruct>: {task}\n<Query>: ", add_special_tokens=False)
    mid = tok.encode("\n<Document>: ", add_special_tokens=False)
    if QCAP and DCAP:
        print(f"qwen3 per-field truncation: query<={QCAP} doc<={DCAP} tokens, "
              f"template={len(pre) + len(ins) + len(mid) + len(suf)} -> max "
              f"{len(pre) + len(ins) + QCAP + len(mid) + DCAP + len(suf)} tokens/pair", flush=True)

    @torch.no_grad()
    def score(query, docs, bs, max_len):
        if QCAP and DCAP:
            q_ids = tok.encode(query, add_special_tokens=False)[:QCAP]
            head = pre + ins + q_ids + mid
            ids = [head + tok.encode(d, add_special_tokens=False)[:DCAP] + suf for d in docs]
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
            res = [0.0] * len(ids)
            for s in range(0, len(order), bs):
                sel = order[s:s + bs]
                b = tok.pad({"input_ids": [ids[i] for i in sel]},
                            padding=True, return_tensors="pt").to(dev)
                lg = m(**b).logits[:, -1, :]
                for i, v in zip(sel, (lg[:, yes].float() - lg[:, no].float()).tolist()):
                    res[i] = v
            return res
        texts = [f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [0.0] * len(texts)
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            enc = tok([texts[i] for i in sel], padding=False, truncation=True,
                      return_attention_mask=False,
                      max_length=max_len - len(pre) - len(suf))
            enc["input_ids"] = [pre + e + suf for e in enc["input_ids"]]
            b = tok.pad(enc, padding=True, return_tensors="pt").to(dev)
            lg = m(**b).logits[:, -1, :]
            for i, v in zip(sel, (lg[:, yes].float() - lg[:, no].float()).tolist()):
                res[i] = v
        return res
    return score


def load_jinav3(name, dev, dtype):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, dtype=dtype, trust_remote_code=True).to(dev).eval()

    @torch.no_grad()
    def score(query, docs, bs, max_len):
        res = m.rerank(query, docs, max_doc_length=512, max_query_length=256)
        out = [0.0] * len(docs)
        for r in res:
            out[r["index"]] = float(r["relevance_score"])
        return out
    return score


def load_ours_pointwise(ckpt, dev, dtype, sd):
    """nanochat POINTWISE yes/no reranker (nanochat/src): score = logit(yes) -
    logit(no) at the last position of each (query, doc) sequence. Backbone in
    `dtype` for eval; the 2-row head stays fp32 so the logit difference is exact."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reranker/src"))
    from transformers import AutoTokenizer
    from prompt_builder_nc import readout_offset, PairBuilder, yes_no_ids, setup_tokenizer
    from modeling_nanochat_rr import NanochatReranker
    cfg = dict(sd["cfg"])
    if os.environ.get("EVAL_MAX_DOC_LEN"):              # length probe (--max-doc-len)
        cfg["max_doc_len"] = int(os.environ["EVAL_MAX_DOC_LEN"])
    # A checkpoint trained on another cluster records that cluster's backbone path (R0 came from
    # another cluster under an absolute path). Resolve to the same backbone here: an explicit
    # EVAL_BACKBONE_DIR wins, else models/<basename> in the recipe, which is a symlink to the
    # shared copy. Falling through unchanged lets the loader raise with the original name.
    def _resolve_backbone(path):
        override = os.environ.get("EVAL_BACKBONE_DIR")
        if override:
            return override
        if os.path.exists(path):
            return path
        local = os.path.normpath(os.path.join(HERE, "..", "reranker", "models", os.path.basename(str(path).rstrip("/"))))
        if os.path.exists(local):
            print(f"backbone {path} is absent here; using {local}", flush=True)
            return local
        return path
    cfg["model_name"] = _resolve_backbone(cfg["model_name"])
    if os.environ.get("EVAL_NANOCHAT_SRC"):             # an explicit nanochat checkout wins
        cfg["nanochat_src"] = os.environ["EVAL_NANOCHAT_SRC"]
    elif cfg.get("nanochat_src") and not os.path.exists(cfg["nanochat_src"]):
        fallback = os.path.join(HERE, "..", "reranker", "third_party", "nanochat-92d63d4")
        print(f"nanochat_src {cfg['nanochat_src']} is absent here; using {fallback}", flush=True)
        cfg["nanochat_src"] = fallback
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"]))
    prompt = cfg.get("prompt", "chat")                  # the checkpoint's own training prompt
    yes_id, no_id = yes_no_ids(tok, prompt, cfg.get("answer_tokens"))
    pb = PairBuilder(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"], prompt=prompt,
                     **({"bos": cfg["bos"]} if "bos" in cfg else {}))   # the training BOS (non-nanochat backbones)
    print(f"ours-pointwise: prompt={prompt} answers ids {yes_id}/{no_id} | example: {pb.render('Q', 'P')!r}",
          flush=True)
    if cfg.get("backbone_impl", "hf") == "upstream":     # upstream nanochat checkpoint (see modeling_nanochat_upstream)
        from modeling_nanochat_upstream import UpstreamNanochatReranker
        m = UpstreamNanochatReranker(cfg["model_name"], yes_id=yes_id, no_id=no_id,
                                     dtype=dtype, nanochat_src=cfg.get("nanochat_src"),
                                     bidirectional=cfg.get("bidirectional", False),
                                     readout_offset=readout_offset(prompt),
                                     score_mode=cfg.get("score_mode", "lm_rows"))
    else:
        m = NanochatReranker(cfg["model_name"], yes_id=yes_id, no_id=no_id, dtype=dtype,
                             attn_implementation=cfg.get("attn_implementation", "sdpa"))
    m.load_state_dict(sd["model"], strict=True)
    m = m.to(dev, dtype=dtype).eval()
    m.head.data = m.head.data.float()
    if getattr(m, "score_mode", "lm_rows") == "meanpool":   # the .to(dtype) above would have cast it
        m.pool_head = m.pool_head.float()

    @torch.no_grad()
    def score(query, docs, bs, max_len):
        out = []
        for s0 in range(0, len(docs), bs):
            seqs = [pb.build(query, d) for d in docs[s0:s0 + bs]]
            n = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, x in enumerate(seqs):
                ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            out += m(ids.to(dev), att.to(dev)).scores.float().tolist()
        return out
    return score


def load_gaggle_hf(path, dev, dtype):
    """The released `transformers` port of our pointwise reranker (hf_export/).

    `path` is a converted repo directory (or hub id): config.json with an auto_map, safetensors
    weights and the two code files. Scores identically to `ours`/upstream -- that equivalence is
    what hf_export/parity_check.py measures -- but needs neither a nanochat checkout nor FA3.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # trust_remote_code on the TOKENIZER too: the repo config carries an auto_map, so without
    # it transformers prompts on stdin ("Do you wish to run the custom code?") and a batch job hangs
    # or silently falls back to the base config.
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    m = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True, dtype=dtype)
    m = m.to(dev).eval()
    # the scoring rows follow the load dtype, exactly as the `ours` backend leaves them
    # (m.to(dtype) then head.float()); the matmul that reads them is fp32 inside the model
    print(f"gaggle_hf: prompt={m.config.prompt} caps {m.config.max_query_len}/{m.config.max_doc_len} "
          f"readout_offset={m.config.readout_offset}", flush=True)

    @torch.no_grad()
    def score(query, docs, bs, max_len):
        return m.compute_score([(query, d) for d in docs], tokenizer=tok, batch_size=bs)
    return score


def load_ours(ckpt, dev, dtype):
    """A checkpoint written by reranker/ (any engine whose readout is the two-row logit gap)."""
    sd0 = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    assert "projector_dim" not in sd0["cfg"], "listwise (jina-style) checkpoints are not part of this repo"
    return load_ours_pointwise(ckpt, dev, dtype, sd0)


BACKENDS = {"monot5": load_monot5, "qwen3": load_qwen3, "jinav3": load_jinav3, "ours": load_ours,
            "gaggle_hf": load_gaggle_hf}

# Native usable context per backend. monoT5 is a T5: RELATIVE position
# embeddings, so it runs past its 512 training length (BERT-family cross-encoders
# cannot -- absolute position embeddings hard-cap at 512).
DEFAULT_MAXLEN = {"monot5": 512, "qwen3": 512, "jinav3": 512, "ours": 512, "gaggle_hf": 512}   # matches the reference table setup


def windows(text, tok, budget, stride_frac=0.5):
    """Split a document into overlapping token windows.

    MaxP: robust04's longest document is 48,532 tokens and 22.5% exceed 2048,
    so no context size fixes it -- head truncation would silently discard most
    of the document. We score every window and take the max, which is the
    standard treatment for these collections and drops nothing.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return [text]
    stride = max(1, int(budget * stride_frac))
    out = []
    for s0 in range(0, len(ids), stride):
        chunk = ids[s0:s0 + budget]
        if not chunk:
            break
        out.append(tok.decode(chunk))
        if s0 + budget >= len(ids):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-doc-len", type=int, default=0,
                    help="override the checkpoint's passage cap (tokens): a length probe, since the "
                         "backbone's rotary tables cover 20,480 positions but it was pretrained at 2,048 "
                         "and its 'long' attention windows are 2,048 wide")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="model dtype for scoring (default bf16, as for every reported number); "
                         "fp32 also turns TF32 matmuls off (strict fp32)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=0,
                    help="0 = per-backend native window (see DEFAULT_MAXLEN)")
    ap.add_argument("--max-query-len", type=int, default=0,
                    help="cap the query at N tokens (qwen3: enables per-field truncation "
                         "together with --max-doc-len; ours: comes from the checkpoint cfg)")
    ap.add_argument("--max-doc-chars", type=int, default=0,
                    help="0 = no character cap; token windowing governs")
    ap.add_argument("--maxp", dest="maxp", action="store_true", default=False,
                    help="opt-in: score long docs as overlapping windows, take the "
                         "max. OFF by default -- plain head truncation at --max-len.")
    ap.add_argument("--no-maxp", dest="maxp", action="store_false")
    ap.add_argument("--maxp-cap", type=int, default=16,
                    help="max windows per document (bounds robust04's 48k-token tail)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--first-stage", default=None,
                    help="TREC run (may contain {dataset}) whose order replaces BM25's; "
                         "e.g. runs/monot5-3b-10k_{dataset}.trec for a mono->duo pipeline")
    ap.add_argument("--rerank-depth", type=int, default=0,
                    help="score only the top-N of the (first-stage) order; the rest keep "
                         "that order below them. 0 = all candidates")
    args = ap.parse_args()

    run_file = Path(RUNS) / f"retrieve_results_{args.dataset}_top100.jsonl"
    assert run_file.exists(), f"no cached first-stage run: {run_file}"
    qrels = Path(QRELS_DIR) / QRELS[args.dataset]
    assert qrels.exists(), f"missing qrels: {qrels}"

    dev = "cuda"
    if args.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"scoring dtype: {args.dtype}", flush=True)
    tag = args.tag or f"{args.backend}_{Path(args.model).name}"
    if not args.max_len:
        args.max_len = DEFAULT_MAXLEN.get(args.backend, 2048)
    if args.max_doc_len:
        os.environ["EVAL_MAX_DOC_LEN"] = str(args.max_doc_len)
        print(f"passage cap overridden: {args.max_doc_len} tokens", flush=True)
    if args.max_query_len:
        os.environ["EVAL_MAX_QUERY_LEN"] = str(args.max_query_len)
        print(f"query cap overridden: {args.max_query_len} tokens", flush=True)
    scorer = BACKENDS[args.backend](args.model, dev, dtype)
    # Only needed for MaxP windowing (off by default). trust_remote_code is
    # REQUIRED here: without it, models with custom configs (nemotron) fall back
    # to the base PreTrainedConfig and die with a misleading
    # "no attribute max_position_embeddings" during rope standardisation.
    wtok = None
    if args.maxp:
        from transformers import AutoTokenizer
        wtok = AutoTokenizer.from_pretrained(
            args.model if args.backend != "ours" else "Qwen/Qwen3-0.6B",
            trust_remote_code=True)
    # reserve room for the query + template inside the window
    budget = max(64, args.max_len - 96)
    print(f"backend={args.backend} max_len={args.max_len} window={budget} "
          f"maxp={args.maxp}", flush=True)

    rows = [json.loads(l) for l in open(run_file)]
    if args.limit: rows = rows[:args.limit]
    fs_rank = None
    if args.first_stage:
        fsp = Path(args.first_stage.format(dataset=args.dataset))
        assert fsp.exists(), f"missing first-stage run: {fsp}"
        fs_rank = {}
        for line in open(fsp):
            qq, _, did, rk, *_ = line.split()
            fs_rank.setdefault(qq, {})[did] = int(rk)
        print(f"first stage: {fsp} ({len(fs_rank)} queries), rerank depth {args.rerank_depth or 'all'}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    outp = Path(args.out) / f"{tag}_{args.dataset}.trec"

    t0 = time.time(); n_empty_q = 0
    with open(outp, "w") as f:
        for n, r in enumerate(rows):
            q, qid = r["query"]["text"], r["query"]["qid"]
            cands = r["candidates"]
            if not cands:
                n_empty_q += 1
                continue
            if fs_rank is not None:             # reorder by the first-stage run (e.g. monoT5-3B)
                rk = fs_rank.get(str(qid), {})
                cands = sorted(cands, key=lambda c: rk.get(c["docid"], 10**9))
            tail = []
            if args.rerank_depth and len(cands) > args.rerank_depth:
                cands, tail = cands[: args.rerank_depth], cands[args.rerank_depth:]
            docs = [doc_text(c, args.max_doc_chars or None) for c in cands]
            if args.maxp:
                flat, owner = [], []
                for i, d in enumerate(docs):
                    ws = windows(d, wtok, budget)[: args.maxp_cap]
                    flat.extend(ws); owner.extend([i] * len(ws))
                ws_scores = scorer(q, flat, args.batch_size, args.max_len)
                sc = [float("-inf")] * len(docs)
                for oi, v in zip(owner, ws_scores):
                    if v > sc[oi]:
                        sc[oi] = v
            else:
                sc = scorer(q, docs, args.batch_size, args.max_len)
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            for rank, i in enumerate(order, 1):
                f.write(f"{qid} Q0 {cands[i]['docid']} {rank} {sc[i]:.6f} {tag}\n")
            lo = min(sc) if sc else 0.0             # un-scored tail keeps first-stage order, below
            for k, c in enumerate(tail, 1):
                f.write(f"{qid} Q0 {c['docid']} {len(order) + k} {lo - k:.6f} {tag}\n")
            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(rows)} queries  {el:.0f}s  "
                      f"eta {el/(n+1)*(len(rows)-n-1):.0f}s", flush=True)

    if n_empty_q:
        print(f"  note: {n_empty_q} queries had 0 candidates and were skipped "
              f"(trec_eval scores them 0)", flush=True)
    ev = subprocess.run([TREC_EVAL, "-c", "-m", "ndcg_cut.10", "-m", "map",
                         "-m", "recip_rank", str(qrels), str(outp)],
                        capture_output=True, text=True)
    print(ev.stdout.strip() or ev.stderr.strip())
    metrics = {}
    for line in ev.stdout.splitlines():
        p = line.split()
        if len(p) == 3: metrics[p[0]] = float(p[2])
    res = {"dataset": args.dataset, "backend": args.backend, "model": args.model,
           "tag": tag, "n_queries": len(rows), "seconds": round(time.time() - t0, 1),
           **metrics}
    Path("results").mkdir(exist_ok=True)
    Path(HERE, "results").mkdir(parents=True, exist_ok=True)
    with open(Path(HERE, "results") / f"{tag}_{args.dataset}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
