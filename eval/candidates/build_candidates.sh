#!/bin/bash
# Build the BM25 top-100 candidate files the rerankers read (data/bm25_top100/), then check they are the paper's
# inputs. Run from the repo root. Needs pyserini (and its Java 21) and downloads its prebuilt indexes; DL21-23 use
# the MS MARCO v2 passage index (~100 GB).
#
#   bash eval/candidates/build_candidates.sh [dataset ...]      # default: all 12
#
# The files carry document text, including Robust04 and TREC-News text, which we may not redistribute; that is
# why the repo ships only the doc-ID rankings (eval/candidates/runs/) and a digest of the exact reranker inputs
# (eval/candidates/DIGESTS). BM25 is k1 = 0.9, b = 0.4 over pyserini's prebuilt indexes, as in rank_llm.
set -euo pipefail
OUT=${EVAL_RUNS:-data/bm25_top100}
DS=("$@"); [ ${#DS[@]} -gt 0 ] || DS=(dl19 dl20 dl21 dl22 dl23 covid news robust04 nfc scifact scidocs fiqa)
for d in "${DS[@]}"; do
  [ -s "$OUT/retrieve_results_${d}_top100.jsonl" ] || python eval/candidates/build_first_stage.py --dataset "$d" --out "$OUT"
done
python eval/candidates/digest.py check "$OUT"
