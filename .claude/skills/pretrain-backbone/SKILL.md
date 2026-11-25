---
name: pretrain-backbone
description: Train or resume a Gaggle base model (3.3B, 0.9B or 0.6B nanochat) on ClimbMix with the pre-training recipe, check its schedule, export it, or work with the original 3.3B run and its bidirectional MNTP continuation. Use for anything about pre-training, the backbone checkpoints, CORE, or the bidirectional model.
---

# Pre-train a backbone

The recipe, with every setting and the checks, is `pretrain/README.md`; the original 3.3B run is
`pretrain/d34_original_run/README.md`; MNTP is `pretrain/mntp/README.md`. This is the working checklist.

## Scale

- One pass over ClimbMix (287.05B tokens, 229,067 steps): ~1,600 H100-hours for the 3.3B model (8 GPUs), 562
  H200-hours for the 0.9B (4 GPUs), 446 H100-hours for the 0.6B (8 GPUs).
- The data is 558 GiB, on a filesystem shared by login and compute nodes.
- Confirm the budget with the user before submitting anything.

## Steps (from `pretrain/`)

1. **Configure.** `cp config.example.sh config.sh`, set `PT_ROOT` and `PT_GPU_SBATCH`, `export PT_CONFIG=$PWD/config.sh`,
   and run `bash setup.sh` on a login node with internet: Python env, FA3 kernels, eval bundle, tokenizer, corpus.
2. **Check.** `python -m unittest discover -s tests -t .` (CPU). `python full_args.py configs/<model>.yaml --check --ngpu N`
   says whether a GPU count works with that config.
3. **Smoke.** `SMOKE=1 bash train.sh configs/<model>.yaml` on a GPU node: 12 steps, nothing saved.
4. **Train.** `bash submit_chain.sh configs/<model>.yaml <jobs>`: chained jobs that resume from the newest complete
   checkpoint and save before their time limit. Compare the start-up lines and the curves with the README's checks
   (`expected_curves_0.9b_0.6b.csv` for the smaller models).
5. **Export.** `export_hf/` converts a checkpoint to `transformers` and checks parity.

Never delete checkpoints to restart; a run refuses a config that differs from the one it started with.

## The original 3.3B run

`d34_original_run/` keeps the run as it happened (three stages, three trainers). Use it only to reproduce that run
exactly or for MNTP, which continues from its stage-3 checkpoint at step 209,158 (not released). The recipe above
reproduces its schedule at every step (`tests/test_schedule_matches_d34.py`).

## Bidirectional MNTP variant

- **Source checkpoint:** causal step 209,158 from the original run's stage 3, which must have kept it.
- **Run:**
  ```bash
  SRC_DIR=... NANOCHAT_BASE_DIR=... OUT_DIR=... bash pretrain/mntp/launch_mntp.sh
  ```
  19,909 updates, ~340 H100-hours.
- **Smoke test first:** `pretrain/mntp/smoke/smoke_bidir.py` checks the attention patch and the masking before a
  long run.
