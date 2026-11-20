"""Fingerprint BM25 top-100 candidate files without shipping their text.

The candidate files (rank_llm's retrieve_results_<dataset>_top100.jsonl format) carry every candidate's text, and
two of the test collections (Robust04, TREC-News) may not be redistributed. So the repo ships, per dataset:
  * runs/<dataset>.bm25.trec.gz  -- the doc-ID-only BM25 ranking (qid Q0 docid rank score bm25), and
  * DIGESTS                       -- one SHA256 over what the rerankers actually read: for every query (sorted by
                                     qid), the query text, then each candidate's docid and its text exactly as
                                     eval/rerank_eval.py's doc_text() normalises it.
A rebuilt candidate file can differ byte for byte (JSON field names and order, title handling) and still be the
same input; DIGESTS is the check that matters.

  python eval/candidates/digest.py write <dir of jsonl files>     # maintainers: refresh runs/ and DIGESTS
  python eval/candidates/digest.py check <dir of jsonl files>     # everyone: verify rebuilt candidates
"""
import gzip, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rerank_eval import doc_text  # noqa: E402  (the reranker-side normalisation, so the digest covers it)

DATASETS = ["dl19", "dl20", "dl21", "dl22", "dl23", "covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa"]


def digest(path):
    """Order of queries in the file does not matter (sorted by qid); order of candidates does (it is the ranking)."""
    per_q, trec = {}, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qid, qtext = str(r["query"]["qid"]), r["query"]["text"]
            h = hashlib.sha256(f"Q\t{qid}\t{qtext}\n".encode())
            for rank, c in enumerate(r["candidates"], 1):
                h.update(f"D\t{c['docid']}\t{doc_text(c)}\n".encode())
                trec.append(f"{qid} Q0 {c['docid']} {rank} {float(c['score']):.6f} bm25\n")
            per_q[qid] = h.hexdigest()
    total = hashlib.sha256("".join(f"{q}\t{per_q[q]}\n" for q in sorted(per_q)).encode())
    return total.hexdigest(), trec


def main():
    mode, src = sys.argv[1], sys.argv[2]
    table = os.path.join(HERE, "DIGESTS")
    if mode == "write":
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        lines = []
        for d in DATASETS:
            dg, trec = digest(os.path.join(src, f"retrieve_results_{d}_top100.jsonl"))
            with gzip.open(os.path.join(HERE, "runs", f"{d}.bm25.trec.gz"), "wt") as f:
                f.writelines(trec)
            lines.append(f"{dg}  {d}\n"); print(d, dg, len(trec), "candidates")
        open(table, "w").writelines(lines)
    else:
        want = dict(reversed(l.split()) for l in open(table))
        bad = 0
        for d in DATASETS:
            p = os.path.join(src, f"retrieve_results_{d}_top100.jsonl")
            if not os.path.exists(p):
                print(f"{d}: missing"); bad += 1; continue
            ok = digest(p)[0] == want[d]; bad += not ok
            print(f"{d}: {'identical to the paper inputs' if ok else 'DIFFERENT from the paper inputs'}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
