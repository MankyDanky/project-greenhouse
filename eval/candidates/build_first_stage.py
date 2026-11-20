"""
Build a BM25 top-100 first-stage run in rank_llm's cached jsonl format, so the
new datasets slot into rerank_eval.py exactly like the pre-existing ones.

Output format (matches retrieve_results_*_top100.jsonl):
  {"query": {"text": ..., "qid": ...},
   "candidates": [{"docid":..., "score":..., "doc": {"id":..., "contents":...}}]}
"""
import argparse, json, os
from pathlib import Path

# dataset -> (pyserini prebuilt index, pyserini topics key, qrels key)
SPEC = {
    # The paper's 12 test sets. The first seven were originally taken from rank_llm's cached BM25 runs, which
    # rank_llm builds the same way (pyserini prebuilt index, BM25 k1=0.9 b=0.4); check_candidates verifies.
    "dl19":     ("msmarco-v1-passage",                "dl19-passage",                "dl19-passage"),
    "dl20":     ("msmarco-v1-passage",                "dl20",                        "dl20-passage"),
    "covid":    ("beir-v1.0.0-trec-covid.flat",       "beir-v1.0.0-trec-covid-test", "beir-v1.0.0-trec-covid-test"),
    "news":     ("beir-v1.0.0-trec-news.flat",        "beir-v1.0.0-trec-news-test",  "beir-v1.0.0-trec-news-test"),
    "robust04": ("beir-v1.0.0-robust04.flat",         "beir-v1.0.0-robust04-test",   "beir-v1.0.0-robust04-test"),
    "nfc":      ("beir-v1.0.0-nfcorpus.flat",         "beir-v1.0.0-nfcorpus-test",   "beir-v1.0.0-nfcorpus-test"),
    "scifact":  ("beir-v1.0.0-scifact.flat",          "beir-v1.0.0-scifact-test",    "beir-v1.0.0-scifact-test"),
    "fiqa":     ("beir-v1.0.0-fiqa.flat",    "beir-v1.0.0-fiqa-test",    "beir-v1.0.0-fiqa-test"),
    "scidocs":  ("beir-v1.0.0-scidocs.flat", "beir-v1.0.0-scidocs-test", "beir-v1.0.0-scidocs-test"),
    "dl21":     ("msmarco-v2-passage",       "dl21",                     "dl21-passage"),
    "dl22":     ("msmarco-v2-passage",       "dl22",                     "dl22-passage"),
    "dl23":     ("msmarco-v2-passage",       "dl23",                     "dl23-passage"),
}



# Raw-JSON schemas differ per corpus and getting this wrong fails SILENTLY:
#   BEIR            -> {"_id", "title", "text"}
#   MS MARCO v1     -> {"id", "contents"}
#   MS MARCO v2 psg -> {"pid", "passage"}          <- the one that bit us
TEXT_FIELDS = ("passage", "text", "contents", "body", "segment")
TITLE_FIELDS = ("title", "headline")


def extract_text(raw, docid):
    r = raw.raw()
    if r is None:
        r = raw.contents()
    if r is None:
        raise RuntimeError(f"docid {docid!r}: neither .raw() nor .contents()")
    r = r.strip()
    if not r.startswith("{"):
        return " ".join(r.split())
    j = json.loads(r)
    body = next((j[k] for k in TEXT_FIELDS if j.get(k)), None)
    if body is None:
        raise RuntimeError(
            f"docid {docid!r}: no known text field. keys={sorted(j)}. "
            f"Add the right one to TEXT_FIELDS.")
    title = next((j[k] for k in TITLE_FIELDS if j.get(k)), "")
    out = f"{title}\n{body}" if title else body
    return " ".join(out.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPEC))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="data/bm25_top100")
    ap.add_argument("--bm25-k1", type=float, default=0.9)
    ap.add_argument("--bm25-b", type=float, default=0.4)
    ap.add_argument("--remove-query", dest="remove_query", action="store_true", default=True,
                    help="drop the hit whose docid == qid, matching pyserini's BEIR "
                         "2CR. A no-op on our 12 datasets (verified: 0 matches), but "
                         "load-bearing for ArguAna/Quora where a query IS a document.")
    ap.add_argument("--keep-query", dest="remove_query", action="store_false")
    ap.add_argument("--max-queries", type=int, default=0, help="seeded random sample of judged queries")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    from pyserini.search.lucene import LuceneSearcher
    from pyserini.search import get_topics, get_qrels

    index, topics_key, qrels_key = SPEC[args.dataset]
    print(f"index={index} topics={topics_key}", flush=True)
    searcher = LuceneSearcher.from_prebuilt_index(index)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)

    topics = get_topics(topics_key)
    qrels = get_qrels(qrels_key)

    # pyserini's get_qrels() returns a dict; trec_eval needs a FILE, and the
    # BEIR qrels for fiqa/scidocs are not shipped as files in the cache. Write
    # them out in TREC format so rerank_eval.py can point at them.
    qdir = Path(os.path.expanduser("~/.cache/pyserini/topics-and-qrels"))
    qfile = qdir / f"qrels.{qrels_key}.txt"
    if not qfile.exists():
        with open(qfile, "w") as qf:
            for qid, docs in qrels.items():
                for docid, rel in docs.items():
                    qf.write(f"{qid} 0 {docid} {rel}\n")
        print(f"wrote qrels -> {qfile}", flush=True)
    # only judged queries -- unjudged ones cost reranking time and score nothing
    qids = [q for q in topics if str(q) in qrels or q in qrels]
    if args.max_queries and len(qids) > args.max_queries:
        import random
        qids = sorted(random.Random(args.sample_seed).sample(sorted(qids, key=str), args.max_queries), key=str)
    print(f"{len(topics)} topics, {len(qids)} with qrels (used)", flush=True)

    outp = Path(args.out) / f"retrieve_results_{args.dataset}_top100.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    n = n_docs = n_empty = 0
    with open(outp, "w") as f:
        for qid in qids:
            qtext = topics[qid]["title"]
            hits = searcher.search(qtext, k=args.k + (1 if args.remove_query else 0))
            if args.remove_query:
                hits = [h for h in hits if str(h.docid) != str(qid)][: args.k]
            else:
                hits = hits[: args.k]
            cands = []
            for h in hits:
                raw = searcher.doc(h.docid)
                if raw is None:
                    raise RuntimeError(f"no stored document for docid {h.docid!r} "
                                       f"in index {index!r}")
                contents = extract_text(raw, h.docid)
                cands.append({"docid": h.docid, "score": float(h.score),
                              "doc": {"id": h.docid, "contents": contents}})
                n_docs += 1
                if not contents.strip():
                    n_empty += 1
            f.write(json.dumps({"query": {"text": qtext, "qid": str(qid)},
                                "candidates": cands}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(qids)}", flush=True)
    frac = n_empty / max(1, n_docs)
    print(f"wrote {outp}  ({n} queries, {n_docs:,} candidates, "
          f"{n_empty:,} empty = {frac:.2%})")
    # Hard gate: the previous run of this script produced 100% empty documents
    # and still exited 0. Never let that happen silently again.
    if frac > 0.01:
        outp.unlink(missing_ok=True)
        raise SystemExit(f"ABORT: {frac:.1%} of documents had empty text "
                         f"(threshold 1%). Output deleted.")


if __name__ == "__main__":
    main()
