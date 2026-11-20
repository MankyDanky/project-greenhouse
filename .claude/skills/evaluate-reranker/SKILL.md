---
name: evaluate-reranker
description: Score a reranker (a Greenhouse checkpoint, a released Hugging Face model, or a baseline such as MonoT5, Qwen3-Reranker, Jina-Reranker-v3 or a listwise LLM) on the paper's 12 test sets with BM25 top-100 candidates and nDCG@10. Use for any evaluation, re-scoring, baseline or BM25/oracle request.
---

# Evaluate a reranker

All commands run from the repo root.

## 1. Inputs (once)

```bash
bash eval/fetch_qrels.sh                         # eval/qrels, 12 files, checksums verified
bash eval/candidates/build_candidates.sh         # data/bm25_top100, then digest check
```

- `build_candidates.sh` needs pyserini with Java 21. It downloads prebuilt indexes; DL21–23 use the ~100 GB
  MS MARCO v2 passage index.
- Pass dataset names to build a subset.
- The final `digest.py check` must say "identical to the paper inputs" for each set. If it does not, scores are
  not comparable to the paper. Report it; never edit `eval/candidates/DIGESTS`.
- The candidate files contain Robust04 and TREC-News text. Keep them under `data/` and never commit or share them.

## 2. Score

```bash
bash eval/run_eval.sh <scorer> <model> <tag> [datasets...]      # default: all 12
```

| Scorer | Model argument |
|---|---|
| `ours` | a checkpoint from the nanochat engine, or run 4 |
| `multi` | a multi-engine milestone (MiniCPM rows, Gemma at 1e-5) |
| `gemma4`, `qwen35` | the native Gemma / Qwen rows |
| `ce` | the pointwise-CE row |
| `gaggle_hf` | a released or exported `transformers` model, e.g. `castorini/gaggle-reranker-20261005` |

For a paper row, the scorer is in `paper/manifest.yaml`.

**Defaults** (as in the paper): BF16 backbone with FP32 scoring rows; queries capped at 128 tokens, passages at 512.
`DTYPE=fp32` switches to strict FP32, which moves single datasets by up to ~0.003.

**Baselines:**
- `bash eval/baselines/run_baselines.sh`: MonoT5-3B, Qwen3-Reranker-4B/8B (query/doc capped 128/512) and
  Jina-Reranker-v3.
- `RANK_LLM=<rank_llm checkout at 443adf3> bash eval/baselines/listwise/run_listwise.sh gemma4|qwen38`: the
  listwise LLMs. Needs vLLM.
- `python eval/oracle.py`: BM25 and the oracle, from the doc-ID runs. Needs no candidate text.

## 3. Report

```bash
python eval/report.py <tag> [<tag> ...]          # DL / BEIR tables with means
python paper/check_results.py --results eval/results --tag <row id>=<tag>
```

Results are in `eval/results/<tag>_<dataset>.json`; run files are in `eval/runs/`. Means are unweighted over DL19–23
and over the seven BEIR sets. Differences under ~0.001 on means or ~0.003 per dataset are run-to-run noise.
