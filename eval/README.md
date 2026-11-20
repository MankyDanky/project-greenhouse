# eval/

Reranking evaluation: BM25 top-100 candidates, nDCG@10 with `trec_eval -c`, unweighted means over TREC DL 2019–23 and
seven BEIR collections. Run everything from the repo root.

| Path | Contents |
|---|---|
| `fetch_qrels.sh`, `qrels.sha256` | the 12 qrels files, from anserini-tools, checksum-verified |
| `candidates/` | the BM25 builder (pyserini prebuilt indexes, k1 0.9, b 0.4), doc-ID-only runs of the paper's candidates (`runs/`), and `DIGESTS` with `digest.py`, which checks that rebuilt candidates are the paper's exact reranker inputs |
| `rerank_eval.py` | the evaluator: `ours` (our checkpoints), `gaggle_hf` (released models), `monot5`, `qwen3`, `jinav3` |
| `backends/` | engine-specific scorers that plug into the same evaluator: `multi`, `gemma4`, `qwen35`, `ce` |
| `run_eval.sh` | one model over the 12 sets |
| `oracle.py` | BM25 and oracle rows |
| `baselines/` | MonoT5-3B, Qwen3-Reranker-4B/8B, Jina-Reranker-v3 (`run_baselines.sh`); Gemma-4 and Qwen3.8 listwise with RankLLM (`listwise/`) |
| `report.py` | DL / BEIR tables from `results/` |

The candidate files hold document text, and two collections (Robust04, TREC-News) may not be redistributed. They live
under the git-ignored `data/bm25_top100/`; the repository ships only doc IDs and digests.
