# d34 ClimbMix: optimizer hyperparameters

What the released model (`castorini/gaggle-nanochat-pretrained-climbmix-20260924`, step 227,158, 283B tokens) trained
with. Taken from the training code in this folder (nanochat `92d63d4`: `nanochat/scripts/base_train.py`, `nanochat/nanochat/gpt.py`,
`nanochat/nanochat/optim.py`), the production trainer, and the run logs. Everything is nanochat's default unless marked **ours**.

## Run shape

| | |
|---|---|
| Model | depth 34, width 2,176, 17 heads x 128, context 2,048, vocab 32,768 |
| Batch (**ours**, grows in steps) | 262,144 → 524,288 at 10.0B tokens → 1,048,576 at 30.4B → 2,097,152 at 92.5B; one pass = 287.05B |
| Micro-batch | 8 sequences x 2,048 tokens per GPU, 8 GPUs; gradient accumulation 2 / 4 / 8 / 16 |
| Precision | FP8 (tensorwise) matmuls, BF16 elsewhere; no gradient clipping |
| Clock | schedules run on a canonical step count T = 321,817 (one pass at batch 1,048,576), so batch changes don't move them |

## Learning rates (fixed at the batch-262,144 values for the whole run — **ours**)

nanochat scales LRs by √(B / 524,288) and AdamW LRs also by (2,176 / 768)^-0.5 = 0.594. We kept the batch-262,144
values (scale 0.7071) at every stage instead of raising them with the batch. In the original run this came from an
upstream resume bug that kept stage 1's rates; the recipe pins them explicitly (`PIN_LR_BATCH=262144`).

| Parameters | Optimizer | Base arg | Effective LR | betas | eps |
|---|---|---:|---:|---|---|
| Transformer matrices | Muon (ns_steps 5, beta2 0.9) | 0.02 | **0.014142** | momentum: see below | — |
| Output layer (lm_head) | AdamW | 0.008 | 0.0033607 | (0.8, 0.96) | 1e-10 |
| Token embedding | AdamW | 0.3 | 0.126025 | (0.8, 0.995) | 1e-10 |
| Value embeddings | AdamW | 0.3 x 0.5 | 0.063013 | (0.8, 0.995) | 1e-10 |
| Residual scalars (resid_lambdas) | AdamW | 0.5 x 0.01 | 0.0035355 | (0.8, 0.95) | 1e-10 |
| x0 scalars (x0_lambdas) | AdamW | 0.5 | 0.353553 | (0.96, 0.95) | 1e-10 |
| Smear gate/lambda, backout lambda | AdamW | — | 0.2 (fixed, not scaled) | (0.8, 0.95) | 1e-10 |

**LR schedule** (multiplier on all groups): linear warmup over 40 steps → 1.0 → linear warmdown over the last 55.28%
of the horizon (warmdown ratio 0.5528, **ours**; nanochat's default is 0.65) to 0.05 at T. Warmdown starts at canonical
step 143,917 = 100.5B tokens. At the released checkpoint (canonical step 318,000) the multiplier is 0.070.

**Muon momentum:** 0.85 → 0.97 linearly over the first 400 steps, 0.97 until the warmdown starts, then linear to 0.90 at
T (0.9015 at the released checkpoint).

## Weight decay

**Matrices (Muon):** decoupled and "cautious" (applied only where update and weight have the same sign):
each step `W -= lr · λ(t) · W · mask`.

- Amplitude: λ = 0.28 · √(B / 524,288) · (D_ref / D) with D_ref / D = 0.05496 (d12 reference horizon / d34 horizon,
  tokens-to-parameters ratio 12). Logged: "Scaling weight decay from 0.280000 to 0.021764" at batch 1,048,576.
- Over time: cosine decay to zero over the horizon, λ(t) = λ · ½ (1 + cos(π t / T)); about 0.62 at 92.5B tokens and
  0.0004 at the released checkpoint. The LR multiplier above also scales the per-step decay.
- The amplitude follows the batch size (unlike the LR):

| Batch | Tokens | λ before cosine |
|---|---|---:|
| 262,144 | 0 – 10.0B | 0.010882 |
| 524,288 | 10.0 – 30.4B | 0.015389 |
| 1,048,576 | 30.4 – 92.5B | 0.021764 |
| 2,097,152 | 92.5 – 287B | 0.021764 (**ours**: kept at the 1M value, not 0.0308) |

**Other parameters (AdamW):** decoupled (`W *= 1 − lr · wd`), constant, no schedule: output layer 0.01, value embeddings
0.01, token embedding 0.001, residual scalars 0.05, x0 scalars and smear/backout parameters 0.

## Where to check

- Formulas: `nanochat/scripts/base_train.py` (scaling ~L285–320, schedules ~L370–396, per-step update ~L572–578);
  groups: `nanochat/nanochat/gpt.py` `setup_optimizer`; update rules: `nanochat/nanochat/optim.py`.
- 2M stages: `stage3_2m_production/production/README.md` (canonical clock, 1M weight-decay amplitude);
  stage 2's optimizer restore: `stage2_2m_continuation/ablation_runtime.py`.
- Stage-1 learning-rate pin: `stage1_ramp_to_92B/head_base.sbatch` (`PIN_LR_BATCH`).
