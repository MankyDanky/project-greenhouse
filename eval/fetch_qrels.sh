#!/bin/bash
# Fetch the 12 qrels files the evaluation needs into eval/qrels/ and verify their checksums.
# They are the public anserini-tools qrels; two are renamed to the names rerank_eval.py uses
# (pyserini's download names for SciDocs and FiQA).
set -e -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SRC=${ANSERINI_TOOLS:-}
if [ -z "$SRC" ]; then
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 https://github.com/castorini/anserini-tools.git "$TMP/tools" >/dev/null
  SRC=$TMP/tools/topics-and-qrels
fi
mkdir -p "$HERE/qrels"
for f in qrels.dl19-passage.txt qrels.dl20-passage.txt qrels.dl21-passage.txt qrels.dl22-passage.txt \
         qrels.dl23-passage.txt qrels.beir-v1.0.0-trec-covid.test.txt qrels.beir-v1.0.0-trec-news.test.txt \
         qrels.beir-v1.0.0-robust04.test.txt qrels.beir-v1.0.0-nfcorpus.test.txt qrels.beir-v1.0.0-scifact.test.txt; do
  cp "$SRC/$f" "$HERE/qrels/$f"
done
cp "$SRC/qrels.beir-v1.0.0-scidocs.test.txt" "$HERE/qrels/qrels.beir-v1.0.0-scidocs-test.txt"
cp "$SRC/qrels.beir-v1.0.0-fiqa.test.txt"    "$HERE/qrels/qrels.beir-v1.0.0-fiqa-test.txt"
cd "$HERE/qrels" && sha256sum -c "$HERE/qrels.sha256" | grep -v ": OK$" || true
echo "qrels ready in $HERE/qrels ($(ls "$HERE/qrels" | wc -l) files, checksums verified)"
