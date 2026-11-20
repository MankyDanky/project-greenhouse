"""Print TREC DL and BEIR nDCG@10 tables for every tag in eval/results (as written by eval/run_eval.sh).

  python eval/report.py [tag ...]
"""
import glob, json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DL = ["dl19", "dl20", "dl21", "dl22", "dl23"]
BEIR = ["covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa"]
res = collections.defaultdict(dict)
for p in glob.glob(os.path.join(HERE, "results", "*.json")):
    r = json.load(open(p))
    res[r["tag"]][r["dataset"]] = r["ndcg_cut_10"]
tags = sys.argv[1:] or sorted(res)
for name, cols in (("TREC DL", DL), ("BEIR", BEIR)):
    print(f"\n{name}\n{'tag':36s}" + "".join(f"{c:>9s}" for c in cols) + f"{'mean':>9s}")
    for t in tags:
        row = res.get(t, {})
        vals = [row.get(c) for c in cols]
        mean = sum(vals) / len(vals) if all(v is not None for v in vals) else None
        print(f"{t:36s}" + "".join(f"{v:9.4f}" if v is not None else f"{'-':>9s}" for v in vals)
              + (f"{mean:9.4f}" if mean is not None else f"{'-':>9s}"))
