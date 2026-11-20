"""Compare reranking scores with the numbers printed in the paper.

  python paper/check_results.py                               # the scores recorded when each run was evaluated
  python paper/check_results.py --results eval/results \\      # your own runs: map row ids to your eval tags
      --tag sft.run1=my-run1 --tag sft.soup=my-soup

A row passes when every score the paper prints for it matches after rounding to 3 decimals, within 0.001 (the
paper rounds; benchmark means are compared with the printed means). Rows the paper prints per dataset
(the headline tables and Sec. 5) are checked per dataset; Sec. 6 rows are checked on their TREC DL and BEIR means.
Run-to-run noise between full retrainings is larger than this tolerance (Sec. 5: SD ~0.001 on means, ~0.003 per
dataset), so for a retrained row use --tolerance 0.005 and read the per-dataset deltas.
"""
import argparse, json, os, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DL = ["dl19", "dl20", "dl21", "dl22", "dl23"]
BEIR = ["covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa"]


def load_eval_results(results_dir, tag):
    out = {}
    for d in DL + BEIR:
        p = os.path.join(results_dir, f"{tag}_{d}.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))["ndcg_cut_10"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="eval/results directory (default: paper/results/recorded.json)")
    ap.add_argument("--tag", action="append", default=[], help="row_id=eval_tag, repeatable")
    ap.add_argument("--tolerance", type=float, default=0.001)
    args = ap.parse_args()
    man = yaml.safe_load(open(os.path.join(HERE, "manifest.yaml")))
    if args.results:
        tags = dict(t.split("=", 1) for t in args.tag)
        scores = {rid: load_eval_results(args.results, tag) for rid, tag in tags.items()}
    else:
        scores = json.load(open(os.path.join(HERE, "results", "recorded.json")))
    n_ok = n_bad = n_skip = 0
    for rid, m in man.items():
        s = scores.get(rid)
        if not s or len(s) < len(DL + BEIR):
            if args.results and rid in scores:
                print(f"{rid:34s} INCOMPLETE ({len(s or {})}/12 datasets)")
            n_skip += 1
            continue
        mean = lambda ds: sum(s[d] for d in ds) / len(ds)
        if "paper" in m:
            diffs = {d: round(s[d], 3) - m["paper"][d] for d in DL + BEIR}
        else:
            diffs = {}
        diffs["DL mean"] = round(mean(DL), 3) - m["paper_means"]["dl"]          # the printed means
        diffs["BEIR mean"] = round(mean(BEIR), 3) - m["paper_means"]["beir"]
        worst = max(diffs, key=lambda k: abs(diffs[k]))
        ok = abs(diffs[worst]) <= args.tolerance + 1e-9
        n_ok += ok; n_bad += not ok
        print(f"{rid:34s} {'ok  ' if ok else 'DIFF'}  DL {mean(DL):.4f}  BEIR {mean(BEIR):.4f}"
              + ("" if ok else f"   largest gap: {worst} {diffs[worst]:+.3f}"))
    print(f"\n{n_ok} rows match, {n_bad} differ, {n_skip} not checked")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
