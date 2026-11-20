---
name: prepare-data
description: Download the Greenhouse backbones and build the training parquets (RLHN-250K, RLHN-680K, Tevatron MS MARCO, self-filtered sets) and evaluation inputs, verified against the checksums of the files the paper used. Use before any training or evaluation, or when a data or model checksum fails.
---

# Prepare models and data

## Models (from `reranker/`)

```bash
bash data_prep/fetch_models.sh gaggle [gaggle-bidir] [gemma4] [qwen35] [minicpm5] [nanochat-d34]
```

- Every download is pinned. `gaggle` and `gaggle-bidir` are checked against their repos' `SHA256SUMS`.
- `gaggle-bidir` copies the causal model's HF tokenizer files, because the two models share one tokenizer.
- `nanochat-d34` converts Karpathy's checkpoint to an FP32 HF model. It needs ~40 GB of RAM, and its reference
  check needs nanochat at `f37d45c`, which the script clones.

## Training data (from `reranker/`, CPU)

```bash
bash data_prep/prepare_data.sh rlhn250k          # most rows
bash data_prep/prepare_data.sh rlhn680k tevatron # Sec. 6.2
bash data_prep/prepare_data.sh rlhn250k-qwen     # the nanochat-d34 row (Qwen3-0.6B length columns)
```

- **Stages:** `prepare_rlhn.py` cleans the public dataset into one row per query. Each row holds positive and
  negative passages plus a token length per passage. `relength_nanochat.py` re-measures the lengths with our
  ClimbMix tokenizer.
- **Why lengths matter:** they decide packing, which decides the training plan. So each output is checked against
  `data_prep/SHA256SUMS`.
- **On a mismatch:** report it, along with the upstream dataset revision if you can find it. Do not edit
  `SHA256SUMS`, and do not substitute another file silently.

Self-filtered sets (Sec. 6.4) need run 1's checkpoint:

```bash
bash data_prep/self_score.sh checkpoints/sft_run1_seed17/ckpt_final.pt <n_gpus>
```

Expected: 14,887 positives and 1,484 queries dropped, 246,050 queries kept.

## Evaluation inputs (from the repo root)

```bash
bash eval/fetch_qrels.sh
bash eval/candidates/build_candidates.sh [datasets...]
```

The candidate files carry Robust04 and TREC-News document text, which NIST licenses and which may not be
redistributed. They stay under the git-ignored `data/`. Only doc-ID runs (`eval/candidates/runs/`) and digests
(`eval/candidates/DIGESTS`) are in the repository. Never add text-bearing files to git, and never paste such text
into issues or logs you share.
