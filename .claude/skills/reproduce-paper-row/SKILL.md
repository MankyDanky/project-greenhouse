---
name: reproduce-paper-row
description: Retrain and evaluate one row of the Greenhouse paper (a fine-tuning run, a Sec. 6 ablation, or the soup) and check it against the printed scores. Use when asked to reproduce, rerun or verify a specific table row, run, backbone, ablation or the released reranker.
---

# Reproduce a paper row

## 1. Find the row

Open `paper/manifest.yaml`. Each entry gives:
- the row id, e.g. `attention.causal_qpq` or `sft.run1`;
- `config` (under `reranker/`) and `scorer`;
- `original_run`;
- `paper` (per-dataset scores) and `paper_means`.

Map the user's wording to a row id:
- "run 1", "baseline", "G" → `sft.run1`
- "QPQ", "repeated query" → `attention.causal_qpq`
- "MiniCPM at 1e-5" → `backbones.minicpm5_2b_lr1e-5`
- "the soup", "the released reranker" → `sft.soup` (see the `model-soup` skill)

Baseline rows (`baseline.*`) are evaluation only: use the `evaluate-reranker` skill.

Read the config header. It states the expected means, the original hardware, the update count and the engine.

## 2. Check prerequisites (from `reranker/`)

| Config reads | Prepare with |
|---|---|
| `models/gaggle-nanochat-pretrained-climbmix-20260924` | `bash data_prep/fetch_models.sh gaggle` |
| `models/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929` | `bash data_prep/fetch_models.sh gaggle-bidir` |
| `models/gemma-4-e2b`, `models/minicpm5-2b-base`, `models/nanochat-d34-base-hf` | `fetch_models.sh gemma4` / `minicpm5` / `nanochat-d34` |
| `data/rlhn250k_prepared_climbmix_lens` | `bash data_prep/prepare_data.sh rlhn250k` |
| `data/rlhn250k_prepared` (nanochat-d34 row) | `prepare_data.sh rlhn250k-qwen` |
| `data/rlhn680k_prepared_climbmix_lens`, `data/msmarco_tevatron_prepared_climbmix_lens` | `prepare_data.sh rlhn680k` / `tevatron` |
| `data/rlhn250k_self_filtered*` | train `sft.run1` first, then `bash data_prep/self_score.sh checkpoints/sft_run1_seed17/ckpt_final.pt 2` |

`prepare_data.sh` reports whether each parquet matches the paper's SHA256. A mismatch means the upstream dataset
changed. Report it to the user; do not "fix" `SHA256SUMS`.

## 3. Check the hardware

Configs pin the per-GPU token budget (`max_tokens_per_batch`) and `pack_world` the row ran with, and set
`require_exact_budget: true`. The trainer stops if the GPU cannot afford the budget. Do not remove that line to
make a run start: the budget defines which queries share an optimizer step.
- **39,312 tokens:** 96 GB GPUs.
- **60,000 tokens:** 141 GB H200s.
- **Multi-engine rows** (`exec_chunk_tokens`, `exec_mode: gradcache`): any GPU size. Execution chunking does not
  change the gradient.

If the user only has smaller GPUs, say so and offer a multi-engine row instead.

## 4. Train

```bash
cd reranker
bash launch/train.sh <config> <n_gpus>          # the engine comes from the config header
```

- It logs to `logs/<config>_<time>.log` and writes `checkpoints/<section>_<name>/`.
- Re-running the same command resumes from the newest checkpoint.
- Confirm the update count against the header: 1,931 for most 250K rows, 1,918 for self-filtered, 5,060 for
  RLHN-680K, 3,112 for Tevatron.

## 5. Evaluate and check

```bash
cd ..
bash eval/run_eval.sh <scorer from manifest> <final checkpoint> <tag>
python eval/report.py <tag>
python paper/check_results.py --results eval/results --tag <row id>=<tag> --tolerance 0.005
```

Final checkpoint by engine:

| Engine | Final checkpoint |
|---|---|
| nanochat | `checkpoints/<name>/ckpt_final.pt` |
| multi | `checkpoints/<name>/milestones/step01931_*.pt` |
| ce | `checkpoints/loss_pointwise_ce/final` |
| gemma4 / qwen35 | the final checkpoint directory the trainer prints |

A retrained row should agree within the paper's noise: about 0.001 on means and 0.003 per dataset. Report the
per-dataset deltas, not just pass or fail. Then strip the optimizer state:
`python reranker/tools/strip_optimizer.py <ckpt>`.
