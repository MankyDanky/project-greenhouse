# The original run of the d34 (3.3B) backbone, stage by stage

**This folder is the provenance of the released 3.3B model, not the recipe to use.** To train any of the Gaggle
backbones (3.3B, 0.9B, 0.6B), use the single-process trainer of [`../README.md`](../README.md): its d34 config
reproduces this run's batch, learning-rate, momentum and weight-decay schedule at every one of its 229,067 steps
(checked by `../tests/test_schedule_matches_d34.py`). This folder keeps the run as it happened: three stages with
their own trainers, because the 2M-token batch and the pinned learning rate were decided during the run.


The recipe that produced **[`castorini/gaggle-nanochat-pretrained-climbmix-20260924`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924)**,
the 3.29B-parameter causal base model behind the Gaggle reranker ([`../../reranker`](../../reranker),
[`castorini/gaggle-reranker-20261005`](https://huggingface.co/castorini/gaggle-reranker-20261005)). Only the released
model's path is here, stage by stage, with the settings it ran with; site paths and Slurm accounts live in one config
file. Contact: Lingwei Gu.

## Result

| | |
|---|---|
| Model | karpathy/nanochat @ `92d63d4`, depth 34: 34 layers, width 2,176, 17 heads x 128, 2,048 context, value embeddings on alternate layers, 3 of 4 layers with a 512-token window, 32,768-token BPE |
| Data | ClimbMix ([`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)), one pass = 287.05B tokens; the last of 6,543 shards is the held-out validation shard |
| Compute | one 8xH100 node at a time in 24-hour self-chaining Slurm legs; ~395k tokens/s, ~8.4 node-days (~1,600 H100-hours) of training plus evaluation |
| Released checkpoint | step 227,158 = 283.05B tokens, the best CORE measured during training |
| Scores | CORE 0.458, MMLU 5-shot 0.493, MMLU-Pro 0.178, validation 0.596 bits/byte |

## The schedule

nanochat's optimizer and schedules are unchanged (Muon for matrices, AdamW for embeddings/scalars, FP8 tensorwise matmuls,
BF16 elsewhere, device batch 8, warmup 40 steps, warmdown ratio 0.5528 of the one-pass horizon, final LR fraction 0.05).
Two things differ from nanochat's defaults: the batch grows in steps, and the learning rates stay at their batch-262,144
values (Muon 0.0141) instead of growing with the square root of the batch. All values: [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md).

| Stage | Steps | Tokens | Batch | Code |
|---|---|---|---:|---|
| 1a | 0 – 38,147 | 0 – 10.0B | 262,144 | `stage1_ramp_to_92B/` |
| 1b | 38,147 – 77,057 | 10.0B – 30.4B | 524,288 | `stage1_ramp_to_92B/` |
| 1c | 77,057 – 136,316 | 30.4B – 92.54B | 1,048,576 | `stage1_ramp_to_92B/` |
| 2 | 136,316 – 137,340 | 92.54B – 94.69B | 2,097,152 | `stage2_2m_continuation/` (1,024 updates) |
| 3 | 137,340 – 229,067 | 94.69B – 287.05B | 2,097,152 | `stage3_2m_production/` (the last update is a half batch, to land exactly on the horizon) |

The LR, weight-decay and momentum schedules are defined on a "canonical" clock of 321,817 steps at batch 1,048,576
(= one pass), so changing the batch does not move them: the LR multiplier is 1 until canonical step 143,917 and then
decays linearly to 0.05. Stage 3's trainer keeps this clock explicitly (`production/production_clock.py`).

## Requirements

Slurm; for each GPU job one node with 8 x 80 GB Hopper GPUs (H100 or H200); ~1.5 TB on a filesystem shared by login and
compute nodes; [`uv`](https://docs.astral.sh/uv/); internet on the login node only (compute nodes run offline).

## Run

```bash
cd pretrain/d34_original_run
cp config.example.sh config.sh             # edit D34_ROOT, D34_GPU_SBATCH, D34_CPU_SBATCH (and D34_ENV_SETUP)
export D34_CONFIG=$PWD/config.sh
bash setup.sh                              # login node: frozen code, Python env, FA3 kernels, eval bundle, 558 GiB corpus
```

Then, one step after another (each command returns at once; wait for the "done" file before the next GPU step):

| # | Command | Done when | Time |
|---|---|---|---|
| 0 | `bash submit.sh cpu stage0_tokenizer/tok_train.sbatch` | `$D34_ROOT/base/tokenizer/tokenizer.pkl`; its SHA256 should equal the released one in `records/` | <2 h, CPU |
| 0' | `bash submit.sh cpu stage3_2m_production/prepare_corpus.sbatch` | `$D34_ROOT/receipts/full_corpus_receipt.json` | hours, CPU; any time after `setup.sh` |
| 1 | `bash stage1_ramp_to_92B/launch_ramp.sh` | `$D34_ROOT/stage1/state/d34veramp/BASE_DONE` | ~3 days in 24-h legs |
| 2 | `bash submit.sh gpu stage2_2m_continuation/run_branch.sbatch` | `$D34_ROOT/stage2/d34abl136316_b2m_lr014142/COMPLETE.json` | ~2 h |
| 3a | `bash submit.sh cpu stage3_2m_production/prepare_release.sbatch` | `$D34_ROOT/receipts/release.json` | <1 h, CPU |
| 3 | `bash stage3_2m_production/launch_production.sh` | `$D34_ROOT/production/d34prod_arm3_b2m_lr014142/BASE_DONE` | ~6 days in 24-h legs |
| 3' | `bash submit.sh cpu stage3_2m_production/best_core/watch_job.sbatch` (with step 3) | `$D34_ROOT/production/best_core/best.json` | runs alongside |
| 4 | [`../export_hf/`](../export_hf/README.md): convert the best-CORE checkpoint to `transformers` safetensors, check parity | | |

Where things land: checkpoints in `$D34_ROOT/base/base_checkpoints/<tag>/` (24 GB each with optimizer state; stages 1
and 3 keep their newest three, stage 1 also the stage boundaries and model-only milestones every 96,000 steps), Slurm logs in
`$D34_ROOT/logs/`, stage-1 leg history in `stage1/state/d34veramp/legs.log`, stage-2 metrics in
`stage2/<tag>/metrics.jsonl`, stage-3 `chain.log` / `metrics.jsonl` / `legs.tsv` in `production/<tag>/`. A chain stops
itself after three consecutive failed legs and leaves a `CHAIN_STOP` file. To restart after fixing the cause: stage 1,
delete `CHAIN_STOP` and `fail_count` in `stage1/state/d34veramp/` and run `RESUME=1 bash stage1_ramp_to_92B/launch_ramp.sh`;
stage 3, delete `CHAIN_STOP`, set `consecutive_failures` to 0 in `chain_state.json` and rerun `launch_production.sh`.
`DRY_RUN=1 SLURM_JOB_ID=0 SLURM_JOB_NAME=d34veramp-base TAG=d34veramp STAGES=... bash stage1_ramp_to_92B/head_base.sbatch`
prints what the next stage-1 leg would run.

Checks against the original run: stage 1 logs "Scaling weight decay from 0.280000 to 0.010882" (then 0.015389 and
0.021764 at the 524K and 1M stages) and ~370-400k tok/s on 8xH100; stage 2 ends near 0.669 validation bits/byte;
stage 3 keeps the 1M weight-decay amplitude (0.021764) and reports CORE every 1,200 canonical steps.

## Layout

| Path | What |
|---|---|
| `config.example.sh` | the only file with site paths and Slurm arguments; everything else derives from it |
| `setup.sh`, `submit.sh` | one-time setup (freezes the code into `$D34_ROOT/exp`, which stages 2-3 run from and stage 3 hashes); sbatch with the configured placement |
| (rebuilt) `$D34_ROOT/nanochat_original` | the run's nanochat, which `setup.sh` rebuilds and checks against `records/original_nanochat.sha256` (the repository's `../nanochat/` is the unified trainer): `92d63d4` with two changes: `scripts/base_train.py` (marked `(d34cm)`; `stage1_ramp_to_92B/d34cm_head.patch`) stops and checkpoints at a stage boundary or before the Slurm limit and re-applies the run's learning rates after an optimizer resume; `nanochat/dataset.py` has an opt-in shard-index hook (`../data/dataset_indexed_manifest.patch`) |
| `../data/` | the 6,543-shard index the stage-3 loader uses, so every checkpoint's `pq_idx` cursor keeps its meaning |
| `stage0_tokenizer/` | nanochat's `tok_train` / `tok_eval` on the first 2B characters |
| `stage1_ramp_to_92B/` | self-chaining legs (`head_base.sbatch`) with the batch ramp; `launch_ramp.sh` pins the LRs (`PIN_LR_BATCH=262144`) and ends at step 136,316 (`STOP_RAMP_AT`); checkpoint retention (`prune_ckpts.py`) |
| `stage2_2m_continuation/` | trainer (`base_train.py` + `ablation_runtime.py`, built on nanochat's) that resumes step 136,316 at batch 2,097,152 for exactly 1,024 updates and records every rank's loader cursor in `COMPLETE.json` |
| `stage3_2m_production/` | `production/`: the production trainer and its leg wrapper ([README](stage3_2m_production/production/README.md): clock, checkpoint and resume contract); the 8-rank NCCL/runtime probe and asset hashing it runs before each leg; `prepare_*.sbatch` and `launch_production.sh`; `best_core/` keeps a hard-linked copy of the best-CORE checkpoint |
| `continue/` | continued pre-training from any checkpoint on another data mix: nanochat `92d63d4` + a patch (config file, start from a checkpoint, data and tokenizer dirs, literal learning rates), built by `setup.sh`; used by `benchmaxxing/cpt` ([README](continue/README.md)) |
| `../export_hf/` | conversion to `transformers`, the model code, the vLLM plugin, parity checks; `MODEL_CARD.md` is the published card |
| `records/` | SHA256 pins of the original run and its production code manifest |

## Things to know

- **Why the learning rates are pinned.** nanochat scales LRs by sqrt(batch / 524,288). In the original run an upstream
  resume bug (`optimizer.load_state_dict` restores the checkpoint's `lr`/`initial_lr`) kept stage 1's rates after every
  batch change, so the released model trained at Muon 0.0141 throughout. Our patch fixes the bug, so the recipe sets those
  rates explicitly: `PIN_LR_BATCH=262144` makes `head_base.sbatch` pass `--matrix-lr 0.02 / 0.01414 / 0.01` (and the AdamW
  rates scaled the same way) for the 262K / 524K / 1M stages, an effective Muon LR of 0.014142 in each. Weight decay is
  recomputed every step from the current batch, as in the original run. Stage 2 keeps the checkpoint's AdamW rates and
  sets Muon's to 0.014142 (the value it already has); stage 3 restores every rate from the checkpoint and only asserts
  Muon's.
- **Data order.** Every checkpoint stores all eight ranks' loader cursors; stages 2-3 resume from them, so the token stream
  continues where the previous stage stopped (job-boundary resumes are row-group-exact, not packing-exact). The world size
  must stay 8 (1x8 or, in stage 3, 2x4 GPUs).
- **Stage 3 refuses to start on unverified inputs.** Before each leg it checks the frozen code, the stage-2 checkpoint and
  `COMPLETE.json`, the tokenizer and every corpus shard against the SHA256 manifests that `prepare_*.sbatch` wrote, then
  runs an 8-rank Hopper/FA3/NCCL probe. The release JSON carries the original run's literal gate string
  `all_four_arms_complete_and_comparable`: there, step 2 was a 2x2 screen of batch (1M, 2M) x Muon LR (0.0141, 0.02), and
  arm 3 (2M, 0.0141) won. Only arm 3 is in this recipe; its `d34abl136316_*` names and tags are kept because stage 3
  checks them.
- **Stage 2 cannot resume.** It runs one 3-hour job; if it dies, move `stage2/<tag>/` and its checkpoint directory aside
  and resubmit. In the original run its 1,024 updates ran on two H200s through an adapter that replays the eight-rank data
  streams and gradient accumulation exactly; `run_branch.sbatch` is the equivalent 8-GPU launcher.
- **Kernels.** Training used FA3 (Hopper, bf16). All reported evaluations used PyTorch SDPA; across kernels bf16
  log-probabilities differ in the third decimal.
- **Bit-exact replay is not possible** (nondeterministic kernels and job boundaries); the recipe reproduces the run's
  configuration and data order.
- **What changed from the code that ran.** The training code is unchanged: `nanochat/`, the stage-2 trainer and the
  stage-3 `production/` code are byte-identical to the run's (`records/` lists the hashes). The launchers changed: paths,
  account and GPU type come from `config.sh`; `#SBATCH --mail-type=NONE` everywhere (stage 3 passes it on the command line,
  because its leg wrapper is frozen code); stage 2 also accepts H200s; the best-CORE watcher takes its paths as arguments;
  our cluster-specific staging, relay and job scripts are gone.
- **Validation of this version.** The unit tests pass (`python -m unittest` in `../data/`, `stage2_2m_continuation/` and
  `stage3_2m_production/production/`; one test that needs the ablation records is skipped). `setup.sh` ran on Linux (all
  steps but the full download), and stage 0 there reproduced the released `tokenizer.pkl` byte for byte. Stage-1 leg
  resolution was checked with `DRY_RUN` at every stage boundary, and the stage-3 preparation and leg gates end to end on
  placeholder files (CPU only). This portable version has not yet trained on a cluster.
