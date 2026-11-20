---
name: pretrain-backbone
description: Run or resume the Greenhouse pre-training of the nanochat d34 backbone on ClimbMix (tokenizer, batch-ramp stages, 2M-token production stage, best-CORE export) and its bidirectional MNTP continuation. Use for anything about pre-training, the backbone checkpoints, CORE, or the bidirectional model.
---

# Pre-train the backbone

The full procedure, with every setting, is `pretrain/README.md`; MNTP is `pretrain/mntp/README.md`. This is the
working checklist.

## Scale

- One pass over ClimbMix (287.05B tokens) takes ~1,600 H100-hours on one 8×H100/H200 node, in 24-hour
  self-chaining Slurm legs.
- The data is 558 GiB, on a filesystem shared by login and compute nodes.
- Confirm the budget with the user before submitting anything.

## Steps (from `pretrain/`)

1. **Configure.** `cp config.example.sh config.sh`, then set `D34_ROOT`, `D34_GPU_SBATCH` and `D34_CPU_SBATCH`.
   Then `export D34_CONFIG=$PWD/config.sh` and run `bash setup.sh` on a login node with internet. It sets up the
   frozen code, the Python environment, FA3 kernels, the eval bundle and the corpus.
2. **Tokenizer.** `bash submit.sh cpu stage0_tokenizer/tok_train.sbatch`. Its SHA256 must equal the one in
   `records/README.md`.
3. **Stages 1–3.** Follow the README table in order, waiting for each "done" file:
   - stage 1: batch 262K → 524K → 1M tokens up to 92.54B tokens;
   - stage 2: 1,024 updates at 2M;
   - stage 3: 2M-token production to 287.05B.
   Learning rates stay at their 262K-batch values (Muon 0.0141); the schedule runs on a fixed canonical clock.
4. **Best-CORE.** `stage3_2m_production/best_core/` keeps the highest-CORE checkpoint. The released model is step
   227,158 (283.05B tokens, CORE 0.458); the final step is 0.004 lower.
5. **Export.** `export_hf/` converts the checkpoint to `transformers` and checks parity.

**Checks:**
- Stage 1 must log the weight-decay rescaling lines quoted in the README.
- On a failure, a chain stops after three consecutive failed legs and leaves `CHAIN_STOP`. The README says how to
  restart; never delete checkpoints to restart.

## Bidirectional MNTP variant

- **Source checkpoint:** MNTP needs causal step 209,158, which stage 3 must have kept. It is not a released file.
- **Run:**
  ```bash
  SRC_DIR=... NANOCHAT_BASE_DIR=... OUT_DIR=... bash pretrain/mntp/launch_mntp.sh
  ```
  19,909 updates, ~340 H100-hours.
- **Smoke test first:** `pretrain/mntp/smoke/smoke_bidir.py` checks the attention patch and the masking before a
  long run.
