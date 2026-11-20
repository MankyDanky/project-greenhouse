"""BM25 and oracle rows of the headline tables, from the shipped doc-ID-only BM25 runs (no document text needed).

  python eval/oracle.py            # needs eval/qrels (bash eval/fetch_qrels.sh) and trec_eval

BM25: the top-100 in BM25 order. Oracle: the same 100 candidates sorted by qrels grade, unjudged documents at
grade 0, ties kept in BM25 order -- the best nDCG@10 any reranker of these candidates can reach.
Writes eval/results/{bm25,oracle}_<dataset>.json, the same format as eval/run_eval.sh.
"""
import collections, gzip, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rerank_eval import QRELS, QRELS_DIR, TREC_EVAL  # noqa: E402

DATASETS = ["dl19", "dl20", "dl21", "dl22", "dl23", "covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa"]


def ndcg10(qrels_file, run_file):
    out = subprocess.run([TREC_EVAL, "-c", "-m", "ndcg_cut.10", qrels_file, run_file],
                         capture_output=True, text=True, check=True).stdout
    return float(next(l.split()[2] for l in out.splitlines() if l.startswith("ndcg_cut_10 ")))


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for d in DATASETS:
        qf = os.path.join(QRELS_DIR, QRELS[d])
        grade = collections.defaultdict(dict)
        for line in open(qf):
            p = line.split()
            if len(p) >= 4:
                grade[p[0]][p[2]] = int(p[3])
        ranked = collections.defaultdict(list)          # qid -> docids in BM25 order
        with gzip.open(os.path.join(HERE, "candidates", "runs", f"{d}.bm25.trec.gz"), "rt") as f:
            for line in f:
                qid, _, docid, rank, score, _ = line.split()
                ranked[qid].append((int(rank), docid, float(score)))
        with tempfile.TemporaryDirectory() as tmp:
            bm25, oracle = os.path.join(tmp, "bm25.trec"), os.path.join(tmp, "oracle.trec")
            with open(bm25, "w") as fb, open(oracle, "w") as fo:
                for qid, c in ranked.items():
                    c.sort()
                    for rank, docid, score in c:
                        fb.write(f"{qid} Q0 {docid} {rank} {score:.6f} bm25\n")
                    order = sorted(c, key=lambda x: -grade[qid].get(x[1], 0))   # stable: ties keep BM25 order
                    for rank, (_, docid, _) in enumerate(order, 1):
                        fo.write(f"{qid} Q0 {docid} {rank} {1000 - rank} oracle\n")
            for tag, run in (("bm25", bm25), ("oracle", oracle)):
                v = ndcg10(qf, run)
                json.dump({"dataset": d, "backend": tag, "tag": tag, "n_queries": len(ranked), "ndcg_cut_10": v},
                          open(os.path.join(HERE, "results", f"{tag}_{d}.json"), "w"))
                print(f"{d:9s} {tag:6s} {v:.4f}")


if __name__ == "__main__":
    main()
