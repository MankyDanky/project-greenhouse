# Reproducing the paper

Every number below is in [`paper/manifest.yaml`](paper/manifest.yaml):
- the config each row trains with;
- the scorer that evaluates it;
- the scores the paper prints.

[`paper/results/recorded.json`](paper/results/recorded.json) holds the per-dataset nDCG@10 recorded when each
original run was evaluated, and `python paper/check_results.py` confirms they match the paper (29 of 29 rows).
"Recorded" in the tables is that file; "Paper" is the printed mean.

Set-up (environment, models, training data, candidates, qrels) is in [ONBOARDING.md](ONBOARDING.md). In short:

```bash
pip install -r reranker/requirements.txt
bash eval/fetch_qrels.sh && bash eval/candidates/build_candidates.sh
cd reranker && bash data_prep/fetch_models.sh gaggle && bash data_prep/prepare_data.sh rlhn250k && cd ..
```

All fine-tuning commands run from `reranker/`; all evaluation commands from the repo root.

## What has been verified

Before release (2026-11-19/20), everything below was checked against the original runs:

| Check | Result |
|---|---|
| Recorded scores vs the paper (`paper/check_results.py`) | 29 / 29 rows match the printed numbers |
| Training plans (`tests/check_plans.py`) | 17 / 17 configs rebuild their original run's step count; run 1's plan hash `04c557c3…` for every row that shares it |
| Training data (`reranker/data_prep/prepare_data.sh` on the public RLHN / Tevatron releases) | every parquet a config reads matches its original SHA256 byte for byte |
| Training code: 10 updates, original code vs this repo, same GPUs and data, for each of the five engines | identical batches and first step; later steps differ only as much as the original code differs from itself (GPU nondeterminism) |
| Evaluation: original checkpoints re-scored with `eval/run_eval.sh` (`paper/results/public_rerun.json`) | 24 rows; 21 match the paper's means; causal QPQ (DL .645 vs .647) and Qwen3-Reranker-4B / 8B (within .004) are close; single datasets move by up to .004 (bf16 across GPU types) |
| Base-model scores (`pretrain_eval/`) | ours: CORE .4578, MMLU .4927, MMLU-Pro .1776; MNTP: MMLU .4534, MMLU-Pro .1871; OPEN-1B CORE .2392, all matching the paper |

Not checked: the pointwise-CE row's evaluation (its checkpoint was unreadable), the MiniCPM 5e-5 row (checkpoint not on
our cluster), the listwise LLM baselines (not re-run), and full retraining of any row.

## How close a reproduction gets

The paper measures run-to-run variation (Sec. 5): across four trainings of the same recipe, the sample SD is
about 0.001 on benchmark means and about 0.003 on single collections.

| What you reproduce | Expected agreement |
|---|---|
| Re-scoring a released checkpoint | identical to 4 decimals |
| Retraining a row on the paper's GPU type and microbatch budget | within that noise |
| Retraining on other hardware | the plan stays fixed (configs pin `max_tokens_per_batch` and the plan SHA256); kernels and reduction order differ, so expect noise-level differences |

Use `python paper/check_results.py --results eval/results --tag <row>=<your tag> --tolerance 0.005` for retrained rows.

## Sec. 4: pre-training

| Item | Where | Notes |
|---|---|---|
| Causal d34 backbone (`gaggle-nanochat-pretrained-climbmix-20260924`) | [`pretrain/README.md`](pretrain/README.md), `configs/d34_3.3b.yaml` | one pass over ClimbMix, 287B tokens, ~1,600 H100-hours on one 8×H100 node; released step 227,158. The config follows the original run's schedule step for step; the run as it ran, stage by stage, is [`pretrain/d34_original_run/`](pretrain/d34_original_run/) |
| Bidirectional MNTP variant | [`pretrain/mntp/README.md`](pretrain/mntp/README.md) | continues causal step 209,158 for 41.75B tokens, ~340 H100-hours; that source checkpoint is not released, so keep it from your own stage-3 run (the README says how) |
| Validation-loss curve (Fig. 1) | the pre-training logs (`metrics.jsonl` per stage; see `pretrain/d34_original_run/README.md`) | the batch-size doublings are the discontinuities |
| Base-model comparison (CORE / MMLU / MMLU-Pro) | [`pretrain_eval/`](pretrain_eval/) | `bash pretrain_eval/run_eval.sh nanochat\|mntp\|hf …`; reproduces our two rows (0.458 / 0.493 / 0.178; MNTP 0.453 / 0.187) and OPEN-1B's CORE (0.239); the MNTP row's CORE and the other external models are not included yet |

## Sec. 3 and 5: headline results, run-to-run variability, the soup

```bash
cd reranker
bash launch/train.sh configs/sft/run1_seed17.yaml 2                # runs 1-3: nanochat engine, 1,931 updates
bash launch/train.sh configs/sft/run2_seed18.yaml 2
bash launch/train.sh configs/sft/run3_seed17_exact_budget.yaml 2
bash launch/train.sh configs/sft/run4_seed17_gradcache.yaml 2      # run 4: multi engine, GradCache chunks
bash launch/soup.sh                                                # equal-weight average -> checkpoints/sft_soup
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/sft_soup/ckpt_final.pt soup
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `sft.run1` | `configs/sft/run1_seed17.yaml` | `ours` | 2× RTX PRO 6000 | 0.641 | 0.544 | 0.6414 | 0.5438 |
| `sft.run2` | `configs/sft/run2_seed18.yaml` | `ours` | 2× RTX PRO 6000 | 0.640 | 0.542 | 0.6396 | 0.5422 |
| `sft.run3` | `configs/sft/run3_seed17_exact_budget.yaml` | `ours` | 2× RTX PRO 6000 | 0.641 | 0.545 | 0.6407 | 0.5454 |
| `sft.run4` | `configs/sft/run4_seed17_gradcache.yaml` | `ours` | 2× H100 | 0.638 | 0.545 | 0.6383 | 0.5449 |
| `sft.soup` | `launch/soup.sh` | `ours` | CPU | 0.641 | 0.547 | 0.6412 | 0.5466 |

Runs 1 and 2 pin the microbatch budgets they ran with: 39,312 and 39,347 tokens per GPU. Run 3 sets
`require_exact_budget`, so it stops rather than repack on a GPU that cannot give that budget. Run 4 executes run 1's
frozen plan (SHA256 `04c557c3…`) in GradCache chunks of 16,384 tokens; the gradient is the same, only the execution
differs.

**Checkpoint trajectory** ("training queries" table, run 4): the multi engine writes a milestone every 50,000
training queries (`checkpoints/sft_run4_seed17_gradcache/milestones/step*_q*.pt`). Score each with
`bash eval/run_eval.sh multi <milestone> run4-<n>k`.

**Baselines (headline tables):**

| Row | Command |
|---|---|
| BM25, Oracle | `python eval/oracle.py` |
| MonoT5-3B, Qwen3-Reranker-4B/8B, Jina-Reranker-v3 | `bash eval/baselines/run_baselines.sh` |
| Gemma-4-26B-A4B, Qwen3.8-27B listwise | `RANK_LLM=<rank_llm@443adf3> bash eval/baselines/listwise/run_listwise.sh gemma4` (and `qwen38`) |
| GPT-6.1 Sol, Jev, RLHN Qwen2.5-3B | not included |
| RankVicuna, RankZephyr, FirstMistral, Rank0, Rank1 | copied from prior work, as the paper marks |

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `baseline.bm25` | – | `bm25` | – | 0.393 | 0.398 | 0.3926 | 0.3975 |
| `baseline.oracle` | – | `oracle` | – | 0.789 | 0.735 | 0.7890 | 0.7346 |
| `baseline.gemma4_26b_listwise` | – | `listwise` | – | 0.637 | 0.548 | 0.6367 | 0.5484 |
| `baseline.qwen38_27b_listwise` | – | `listwise` | – | 0.633 | 0.554 | 0.6328 | 0.5543 |
| `baseline.jina_v3` | – | `jinav3` | – | 0.611 | 0.511 | 0.6112 | 0.5108 |
| `baseline.monot5_3b` | – | `monot5` | – | 0.604 | 0.520 | 0.6038 | 0.5195 |
| `baseline.qwen3_rr_4b` | – | `qwen3` | – | 0.615 | 0.541 | 0.6151 | 0.5410 |
| `baseline.qwen3_rr_8b` | – | `qwen3` | – | 0.605 | 0.541 | 0.6052 | 0.5406 |

## Sec. 6.1: backbones

```bash
bash data_prep/fetch_models.sh gemma4 minicpm5 nanochat-d34
bash data_prep/prepare_data.sh rlhn250k-qwen                      # the nanochat-d34 row reads Qwen3-0.6B lengths
bash launch/train.sh configs/backbones/minicpm5_2b_lr1e-5.yaml 2  # ... one per row below
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `backbones.nanochat_d34_lr5e-5` | `configs/backbones/nanochat_d34_lr5e-5.yaml` | `ours` | 4× RTX A6000 | 0.621 | 0.521 | 0.6208 | 0.5207 |
| `backbones.gemma4_e2b_lr5e-5` | `configs/backbones/gemma4_e2b_lr5e-5.json` | `gemma4` | 2× RTX PRO 6000 | 0.627 | 0.537 | 0.6267 | 0.5366 |
| `backbones.qwen35_4b_lr5e-5` | `configs/backbones/qwen35_4b_lr5e-5.json` | `qwen35` | 2× H200 NVL | 0.626 | 0.530 | 0.6261 | 0.5304 |
| `backbones.minicpm5_2b_lr5e-5` | `configs/backbones/minicpm5_2b_lr5e-5.yaml` | `multi` | 2× H100 | 0.627 | 0.530 | 0.6273 | 0.5303 |
| `backbones.gemma4_e2b_lr1e-5` | `configs/backbones/gemma4_e2b_lr1e-5.yaml` | `multi` | 2× H200 NVL | 0.640 | 0.546 | 0.6397 | 0.5456 |
| `backbones.minicpm5_2b_lr1e-5` | `configs/backbones/minicpm5_2b_lr1e-5.yaml` | `multi` | 2× RTX PRO 6000 → 2× H200 | 0.642 | 0.551 | 0.6415 | 0.5511 |

The two `.json` rows are native trainers for Gemma-4-E2B and Qwen3.5-4B-Base: tied, soft-capped readout for
Gemma, its own BOS conventions for Qwen. They load their backbone from the Hub at the pinned revision. The multi
engine reads the raw two-row logit gap with no soft cap; that is why its Gemma row differs from the native one at
the same learning rate. `diagnostics/weight_scale.py` measures the median matrix RMS behind the learning-rate
discussion: 0.136 for our backbone against 0.013–0.026 for the AdamW-trained ones.

## Sec. 6.2: training data

```bash
bash data_prep/prepare_data.sh rlhn680k tevatron
bash launch/train.sh configs/data/rlhn680k.yaml 2                 # 5,060 updates
bash launch/train.sh configs/data/tevatron_msmarco.yaml 2         # 3,112 updates
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `data.rlhn680k` | `configs/data/rlhn680k.yaml` | `ours` | 2× H200 NVL | 0.646 | 0.545 | 0.6459 | 0.5447 |
| `data.tevatron_msmarco` | `configs/data/tevatron_msmarco.yaml` | `ours` | 2× H200 NVL | 0.618 | 0.470 | 0.6181 | 0.4700 |

## Sec. 6.3: causal versus bidirectional

```bash
bash data_prep/fetch_models.sh gaggle-bidir
bash launch/train.sh configs/attention/mntp_standard.yaml 2       # ... and the other four
cd .. && bash diagnostics/attention_blocking.sh reranker/checkpoints/attention_mntp_standard/ckpt_final.pt mntp_standard
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `attention.causal_qpq` | `configs/attention/causal_qpq.yaml` | `ours` | 2× RTX PRO 6000 | 0.647 | 0.551 | 0.6465 | 0.5513 |
| `attention.causal_meanpool` | `configs/attention/causal_meanpool.yaml` | `ours` | 2× H200 NVL | 0.642 | 0.543 | 0.6420 | 0.5433 |
| `attention.mntp_standard` | `configs/attention/mntp_standard.yaml` | `ours` | 2× RTX PRO 6000 | 0.643 | 0.544 | 0.6428 | 0.5443 |
| `attention.mntp_mask` | `configs/attention/mntp_mask.yaml` | `ours` | 2× H200 NVL | 0.643 | 0.543 | 0.6426 | 0.5431 |
| `attention.mntp_meanpool` | `configs/attention/mntp_meanpool.yaml` | `ours` | 2× H200 NVL | 0.638 | 0.538 | 0.6381 | 0.5377 |

The attention-blocking test re-scores DL19, DL20 and SciFact with the same weights in three ways: the trained mask,
later tokens blocked, and a size-matched random block. The paper reports 0.753 → 0.700 (standard prompt) and
0.750 → 0.688 (appended mask) for the blocked variant.

## Sec. 6.4: self-filtering and self-distillation

```bash
bash data_prep/self_score.sh checkpoints/sft_run1_seed17/ckpt_final.pt 2   # scores RLHN-250K with run 1, filters
bash launch/train.sh configs/curation/self_filter.yaml 2                  # 1,918 updates
bash launch/train.sh configs/curation/self_filter_self_kd.yaml 2
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `curation.self_filter` | `configs/curation/self_filter.yaml` | `ours` | 2× RTX PRO 6000 | 0.641 | 0.548 | 0.6408 | 0.5475 |
| `curation.self_filter_self_kd` | `configs/curation/self_filter_self_kd.yaml` | `ours` | 2× RTX PRO 6000 | 0.640 | 0.547 | 0.6398 | 0.5473 |

The filter drops 14,887 positives (2.6%) and 1,484 queries (0.6%), leaving 246,050 queries. ArguAna is exempt.
Self-distillation adds 0.3·τ²·KL over the negatives with τ_KD = 2 for both teacher and student; LCE keeps τ = 1.

## Sec. 6.5: loss functions

```bash
bash launch/train.sh configs/loss/pointwise_ce.yaml 2
cd .. && bash eval/run_eval.sh ce reranker/checkpoints/loss_pointwise_ce/final ce
```

| Row | Config (in `reranker/`) | Scorer | Paper's GPUs | Paper DL | Paper BEIR | Recorded DL | Recorded BEIR |
|---|---|---|---|---:|---:|---:|---:|
| `loss.pointwise_ce` | `configs/loss/pointwise_ce.yaml` | `ce` | 2× RTX PRO 6000 | 0.628 | 0.528 | 0.6282 | 0.5279 |

## Appendix C: shared fine-tuning settings

Every config spells its settings out in full (no inheritance). The defaults behind the appendix table:
- full fine-tuning of all retained parameters, with the two pre-trained LM-head rows for " true" / " false" as the
  scorer;
- FP32 weights, gradients and AdamW states, with TF32 matmuls;
- query / passage caps of 128 / 512 tokens;
- K ∈ {7, 15, 23} negatives, and 128 query groups per update;
- AdamW (0.9, 0.95), learning rate 5e-5, weight decay 0.01, clipping at 12;
- 150 warm-up updates, then cosine decay to zero;
- one epoch, seed 17;
- evaluation with a BF16 backbone and FP32 scoring rows.

## After the paper: the 0.9B and 0.6B models, their rerankers, and the chat models

Released in November 2026 with the paper's data and recipes. Set-up is the same as above; the smaller backbones come
from `bash reranker/data_prep/fetch_models.sh gaggle-0.9b gaggle-0.6b`.

| Model | How | Hardware and time | Expected |
|---|---|---|---|
| 0.9B base (`gaggle-nanochat-0.9b-pretrained-climbmix-20261105`) | [`pretrain/`](pretrain/README.md), `configs/d32x1152_0.9b.yaml` | 4 × H200, ~560 H200-hours | val bpb 0.658; CORE 0.370 (500 per task, in training), 0.355 (every example); MMLU 0.357; MMLU-Pro 0.118 |
| 0.6B base (`gaggle-nanochat-0.6b-pretrained-climbmix-20261121`) | [`pretrain/`](pretrain/README.md), `configs/d28x1024_0.6b.yaml` | 8 × H100, ~450 H100-hours | val bpb 0.678; CORE 0.342 / 0.333; MMLU 0.353; MMLU-Pro 0.124 |
| 0.9B reranker (`gaggle-reranker-0.9b-20261108`) | `bash launch/train.sh configs/experiments/gaggle09b_lce.yaml 1` | 1 × H200, ~14 h | DL 0.632, BEIR 0.533 |
| 0.6B reranker (`gaggle-reranker-0.6b-20261122`) | `bash launch/train.sh configs/experiments/gaggle06b_lce.yaml 1` | 1 × H200, ~10 h | DL 0.634, BEIR 0.533 |
| 3.3B chat model (`gaggle-nanochat-ift-climbmix-20261031`) | [`posttrain/`](posttrain/README.md), `POST_MODEL=3b` | 1 × H100, ~7 h | ChatCORE 0.412 (`chat_eval -i sft`) |
| 0.9B chat model (`gaggle-nanochat-0.9b-ift-climbmix-20261108`) | [`posttrain/`](posttrain/README.md), `POST_MODEL=0.9b` | 2 × H200, ~35 min | ChatCORE 0.259; MMLU 38.14, GSM8K 19.41 |

Reranker means are over the same 12 test sets and BM25 top-100 candidates as the paper (`eval/run_eval.sh ours
<ckpt_final.pt> <tag>`, then `eval/report.py <tag>`).

### What has been verified for these

Checked in November 2026 against the original runs, as for the paper's rows:

| Check | Result |
|---|---|
| Pre-training schedule (`pretrain/tests/`, CPU) | the 3.3B config's batch, tokens, LR, Muon momentum and weight decay equal the original staged run's at all 229,067 steps; the 0.9B and 0.6B configs rebuild their runs' schedules |
| Pre-training: this trainer vs the code each release ran, from scratch on 4 × H200 (150 steps for the 0.9B and 0.6B, 100 for the 3.3B) | the same LR and weight-decay scaling and step-0 validation loss; identical for the first 3 or 4 steps, then the usual nondeterministic drift: largest loss difference over 60 steps 3.7e-4 (0.9B) and 3.0e-4 (0.6B), within the 5.8e-4 between two runs of identical code, and 1.5e-3 (3.3B, whose own run-to-run spread was not measured) |
| Pre-training stop and resume | a run stopped at a given step and one stopped by its job's time limit both resumed from their checkpoint with the saved validation loss; as in nanochat, the loader resumes at the next row group |
| Reranker training plans (`tests/check_plans.py`) | both configs rebuild their original run's plan: 1,928 steps each (18,246 and 17,941 microbatches) |
| Reranker training: original code vs this repo, same GPU and data | identical batches and first step; later steps differ only as much as the original code differs from itself |
| Reranker export (`reranker/export_hf/`) on the original checkpoints | weights, config and model code byte-identical to the released Hugging Face repos |
| Released rerankers re-scored with `eval/run_eval.sh` (DL19, NFCorpus, SciFact) | within 0.0027 of the original scores (bf16 across GPU types) |
| Base models re-scored with `pretrain_eval/`, loaded through `pretrain/nanochat` | MMLU / MMLU-Pro 0.3568 / 0.1182 (0.9B) and 0.3531 / 0.1243 (0.6B), as recorded. CORE 0.3701 / 0.3397 with 500 examples per task (0.3705 / 0.3415 in training) and 0.3545 / 0.3346 on every example (0.3549 / 0.3326 recorded), within 0.002 |
| 0.9B chat model: midtraining (240 steps) and SFT (701 steps), this repo vs the original run | first steps identical; midtraining validation bpb 0.3723 vs 0.3717 at step 150; SFT validation loss within 0.000125 at every evaluation; final MMLU identical |
