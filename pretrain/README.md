# Pre-training the Gaggle backbones on ClimbMix (3.3B, 0.9B, 0.6B)

One recipe, one trainer, one config per model. Every Gaggle base model is nanochat trained from scratch on one pass
over ClimbMix with the same data, tokenizer, optimizer and schedule; only the shape and the GPU layout differ.

| Model | Shape | Parameters | Config | Trained on | Val bpb | CORE (500/task) | MMLU | MMLU-Pro |
|---|---|---:|---|---|---:|---:|---:|---:|
| [`castorini/gaggle-nanochat-pretrained-climbmix-20260924`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924) | depth 34 × width 2,176, value embeddings on alternate layers | 3.29B | [`configs/d34_3.3b.yaml`](configs/d34_3.3b.yaml) | 8 × H100, ~1,600 H100-hours | 0.596 | 0.458 | 0.493 | 0.178 |
| [`castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105`](https://huggingface.co/castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105) | depth 32 × width 1,152, value embeddings on every 4th layer | 0.89B | [`configs/d32x1152_0.9b.yaml`](configs/d32x1152_0.9b.yaml) | 4 × H200, 562 H200-hours | 0.658 | 0.370 | 0.357 | 0.118 |
| [`castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121`](https://huggingface.co/castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121) | depth 28 × width 1,024, value embeddings on every 7th layer | 0.55B | [`configs/d28x1024_0.6b.yaml`](configs/d28x1024_0.6b.yaml) | 8 × H100, 446 H100-hours | 0.678 | 0.342 | 0.353 | 0.124 |

CORE is nanochat's average centered accuracy over 22 in-context tasks; MMLU and MMLU-Pro are 5-shot, scored by the
log-likelihood of the answer letter ([`../pretrain_eval/`](../pretrain_eval/)). On every example instead of 500 per
task, CORE is 0.355 (0.9B) and 0.333 (0.6B). The 3.3B release is its best-CORE checkpoint (step 227,158 of 229,067);
the smaller two are their final checkpoints, which were also their best.

## The recipe

| | All three models |
|---|---|
| Data | ClimbMix ([`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)), one pass: 287,049,515,008 tokens in 229,067 optimizer steps; the last of its 6,543 shards is the validation shard |
| Tokenizer | the released 32,768-token byte-level BPE (`tokenizer.pkl` SHA256 `ae73c5f7…`; [`d34_original_run/stage0_tokenizer/`](d34_original_run/stage0_tokenizer/) rebuilds it byte for byte) |
| Sequences | 2,048 tokens, BOS-aligned best-fit packing (nanochat's) |
| Optimizer | nanochat's: Muon for the transformer matrices, AdamW for embeddings and scalars; no gradient clipping |
| Batch | 262,144 tokens → 524,288 at 10.0B tokens → 1,048,576 at 30.4B → 2,097,152 from 92.54B (`--batch-schedule`) |
| Learning rates | computed once for the 262,144 batch (nanochat's √(B/524,288) rule: × 0.7071) and then held as the batch grows; Muon 0.014142 |
| Schedule clock | the LR, Muon momentum and weight-decay schedules run on a canonical clock of 321,817 steps of 2^20 tokens (a 2M step counts 2), so a batch change does not move them |
| LR schedule | 40-step warmup, constant, then linear decay to 0.05 over the last 55.28% of the clock (from canonical step 143,917 = 100.5B tokens) |
| Muon momentum | 0.85 → 0.97 over 400 steps, 0.97, then linear to 0.90 over the LR decay |
| Muon weight decay | cautious; nanochat's amplitude rule, following √batch up to the 1M batch and held at 2M (`--weight-decay-batch-cap`); cosine to 0 over the clock |
| Precision, attention | FP8 matrix multiplications (tensorwise), BF16 elsewhere; FlashAttention-3 |

Every value with its derivation: [`d34_original_run/HYPERPARAMETERS.md`](d34_original_run/HYPERPARAMETERS.md) (written
for the 3.3B model; the same rules give the smaller models' values, e.g. their weight-decay amplitudes, which the trainer
prints at start).

**One difference between the runs.** The 0.9B and 0.6B runs switched to the 2M batch one step later than the 3.3B run
(step 136,317 instead of 136,316), so they need no half-batch last step; their LR, momentum and weight-decay schedules
differ from the 3.3B run's by at most 5.3e-6, 3.9e-7 and 1e-5 (relative). Each config keeps the schedule its released
run used.

## The trainer

[`nanochat/`](nanochat/) is karpathy/nanochat at `92d63d4` with these changes (every one is opt-in or keeps nanochat's
behaviour when unused):

| Change | Where |
|---|---|
| value embeddings on every k-th layer (`--ve-every`, default 2 = nanochat's alternating layers; written to the checkpoint meta only when it is not 2) | `nanochat/gpt.py`, `scripts/base_train.py`, `scripts/chat_sft.py` |
| the batch schedule in one process (`--batch-schedule`, `--target-tokens`, `--canonical-batch`), with an exact-horizon last step; `--weight-decay-batch-cap` | `nanochat/batch_schedule.py`, `scripts/base_train.py` |
| the weight-decay reference model is nanochat's d12 at aspect ratio 64 for every shape (nanochat built it at the run's aspect ratio, which silently changed weight decay across shapes) | `scripts/base_train.py` |
| `--auto-resume` from the newest *complete* checkpoint (each save writes `complete_<step>` last), `--keep-checkpoints N` | `scripts/base_train.py` |
| saves and exits 15 minutes before a Slurm job's time limit (`--deadline-margin-s`), or at `--stop-at-step`; re-applies the run's learning rates after an optimizer resume | `scripts/base_train.py` (from the 3.3B run, marked `(d34cm)`) |
| FlashAttention-3 only on Ampere, Ada and Hopper GPUs (its kernels cannot run on Blackwell); `NANOCHAT_ATTN=fa3\|sdpa` forces a kernel | `nanochat/flash_attention.py` |
| an opt-in index of all 6,543 shards (`D34_SHARD_MANIFEST`), used by the original 3.3B run's stage 3 | `nanochat/dataset.py` |

## Run

Slurm and one node per job; ~0.9 TB on a filesystem shared by login and compute nodes; [`uv`](https://docs.astral.sh/uv/);
internet on the login node only.

```bash
cd pretrain
cp config.example.sh config.sh             # edit PT_ROOT and PT_GPU_SBATCH (and PT_CPU_SBATCH, PT_ENV_SETUP)
export PT_CONFIG=$PWD/config.sh
bash setup.sh                              # login node: Python env, FA3 kernels, eval bundle, tokenizer, 558 GiB corpus
python -m unittest discover -s tests -t .  # CPU, seconds: the schedule checks below
```

Then, on a GPU node with the config's GPU count (`gpus`):

```bash
SMOKE=1 bash train.sh configs/d28x1024_0.6b.yaml           # 12 steps, no checkpoint: FA3, memory, throughput
bash submit_chain.sh configs/d28x1024_0.6b.yaml 4          # 4 chained 24-hour jobs (the 0.6B needs ~2.3 days)
```

Every job resumes from the newest complete checkpoint and saves before its time limit; one that finds the final
checkpoint exits at once. A run refuses to start on a partial corpus (nanochat's loader lists the shards once and
then cycles) and on a config that differs from the one it started with (`run_config.yaml` in its checkpoint
directory). Checkpoints go to `$PT_ROOT/base/base_checkpoints/<run>/`: 24 GB (3.3B), 6.2 GB (0.9B) or 4.4 GB (0.6B)
each with optimizer state; the configs keep the newest two or three.

**Other GPU counts.** A config runs on any count whose micro-batch (device batch × 2,048 × GPUs) divides every batch
size; `full_args.py <config> --check --ngpu N` says. The optimizer sees the same batches, but which documents share a
micro-batch changes, so the losses are not identical to the released run's. The 3.3B config also needs its half-batch
last step to be whole micro-batches (8 GPUs × 8 sequences: yes).

**Checks against the released runs** (the trainer prints them at start):
- `Batch schedule ... 229,067 steps, 321,817 canonical steps`, the four phases, and `LR warmdown starts at canonical
  step 143,917 = step 140,117`.
- Muon weight decay per phase: 0.010882 / 0.015389 / 0.021764 / 0.021764 (3.3B), 0.039826 / 0.056322 / 0.079651 /
  0.079651 (0.9B), 0.056492 / 0.079892 / 0.112984 / 0.112984 (0.6B).
- Throughput: ~395k tokens/s (3.3B, 8 × H100), 556–583k (0.9B, 4 × H200), 1.36–1.42M (0.6B, 8 × H100).
- Validation bits per byte (every 1,000 steps) and CORE (every 5,000) of the two smaller runs:
  [`expected_curves_0.9b_0.6b.csv`](expected_curves_0.9b_0.6b.csv).

nanochat's printed "Estimated FLOPs per token" (and so MFU) leaves out the FP8 layers, which hold almost all matrix
FLOPs: with `--fp8` it reads 3.96e8 for the 0.9B model, whose true training cost is 3.68e9 FLOPs per token.

## Tests

`python -m unittest discover -s tests -t .` (CPU, seconds):
- `test_schedule_matches_d34.py`: with the 3.3B config, the trainer's batch, token count, canonical step, LR
  multiplier, Muon momentum and Muon weight decay equal the original 3.3B run's at all 229,067 steps. The run's side
  comes from [`d34_original_run/`](d34_original_run/): stage 1's batch stages and hand-off step from `launch_ramp.sh`,
  stages 2–3 from `production_clock.TokenClock`, and the schedule functions from the production trainer. It also pins
  the smaller configs' one-step-later hand-off.
- `test_batch_schedule.py`, `test_configs.py`: the schedule builder, and the three configs (one ClimbMix pass each;
  refused on a GPU layout that does not divide the batches).

## How the shapes were chosen, and distillation

**Shapes.** For each size, a sweep of shapes trained for the same 7B tokens (one GPU each, batch 2^20, 6,676 steps,
SDPA attention, the same data order; CORE on every example for the 0.9B sweep, 500 per task for the 0.6B sweep). At
an equal parameter budget, deeper and narrower won, with fewer value-embedding tables to pay for the depth:
- **0.9B:** d32 × 1,152 with tables on every 4th layer reached val bpb 0.7219 and CORE 0.258, against 0.7296 and 0.246
  for nanochat's stock depth 20 (897M); two other shapes and two learning rates were in between.
- **0.6B:** d28 × 1,024 with tables on every 7th layer reached 0.7373 and 0.242; d32 × 896 (VE/4) 0.7403 and 0.240,
  d24 × 1,024 (VE/4) 0.7413 and 0.239, nanochat's stock depth 16 (VE/2) 0.7562 and 0.227.

**Distillation was tried and is not used.** The same 0.6B shape was also trained over the full pass with logit
distillation from the 3.3B model for the first 65% of tokens (2.1× the compute). It started well ahead of the
undistilled run, 0.056 val bpb at 0.3B tokens and 0.028 at 92B, but the lead shrank at each batch doubling and while
the teacher was phased out, and ended at 0.002, within the ~0.003 spread between repeated runs. It finished behind on
CORE (0.326 vs 0.333 on every example) and MMLU (0.299 vs 0.353). A 7B-token study on the 0.9B sweep's stock shape
agreed: distillation lowers loss per token, but the same compute spent on more plain data did better. The distillation
code is not part of this release.

## Layout

| Path | What |
|---|---|
| `config.example.sh` | the only file with site paths and Slurm arguments |
| `setup.sh` | one-time setup: Python env, FA3 kernels, eval bundle, tokenizer, corpus |
| `train.sh`, `full_args.py` | one training job on this node's GPUs; a config → trainer flags (with the step count the trainer will use) |
| `submit_chain.sh`, `train.sbatch` | a chain of Slurm jobs for one run |
| `configs/` | one file per released model |
| `expected_curves_0.9b_0.6b.csv` | the two smaller runs' validation loss and CORE during training |
| `nanochat/` | the trainer |
| `tests/` | the CPU checks above |
| `data/` | the 6,543-shard index and its loader hook (used by the original 3.3B run's stage 3) |
| `d34_original_run/` | the 3.3B model's run as it happened, in three stages with their own trainers: the provenance of the released checkpoint, and what [`mntp/`](mntp/) continues from |
| `export_hf/` | conversion to `transformers` and parity checks (all three models) |
| `mntp/` | the bidirectional MNTP continuation of the 3.3B model |
| `continue/` | continued pre-training from a released checkpoint |

## Validation of this version

- **CPU tests.** The tests above pass, and so do the original 3.3B run's own unit tests (`data/`, and
  `d34_original_run/`'s stage 2 and stage-3 production code) after the move. `d34_original_run/setup.sh` rebuilds the
  run's nanochat from upstream `92d63d4` and its two patches, and the result matches all 53 files of
  `d34_original_run/records/original_nanochat.sha256`; its stage-1 `DRY_RUN` resolves the first leg as before.
- **Parity with the code each release ran**, on 4 × H200, from scratch, with the same flags: this trainer against the
  0.9B's and 0.6B's training code (150 steps each) and against the 3.3B run's stage-1 trainer (100 steps).
  - All three print the same learning-rate and weight-decay scaling, give the same validation loss at step 0, and train
    bit for bit identically for the first 3 or 4 steps. After that the runs drift apart, as two runs of identical code do
    (the bf16 and FP8 kernels are not deterministic): two runs of the same 0.6B code, identical for 4 or 5 steps, then
    differed by up to 1.9e-4 in loss over the first 30 steps and 5.8e-4 over 60.
  - The largest loss differences over the first 60 steps: 3.7e-4 (0.9B) and 3.0e-4 (0.6B), within that spread; 1.5e-3
    (3.3B), whose own run-to-run spread was not measured. Validation bits per byte at the last step differ by 1.2e-4
    (0.9B, step 150), 9.3e-4 (0.6B, step 150) and 9.3e-4 (3.3B, step 100).
- **Stop and resume.** A run stopped at step 30 (`--stop-at-step`) and a run stopped by its job's time limit (at step
  152) both saved, marked the checkpoint complete, and continued from it with `--auto-resume`; the reloaded model's
  validation loss equals the saved one's. As in nanochat, the loader resumes at the next Parquet row group, so a resumed
  run skips the unread rest of one row group (about half a million tokens here) and does not replay the uninterrupted
  run bit for bit.
- **Exact horizon.** A small run whose schedule ends with a half batch trained that last step as one (65,536 tokens)
  and landed exactly on its token target.
- **Evaluation.** The released 0.9B and 0.6B checkpoints load through this trainer's model code in
  [`../pretrain_eval/`](../pretrain_eval/): CORE 0.370 and 0.340 (500 per task), 0.355 and 0.335 (every example);
  MMLU 0.357 and 0.353, MMLU-Pro 0.118 and 0.124, as recorded.
