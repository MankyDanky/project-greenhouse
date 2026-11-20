# Base-model comparison (CORE, MMLU, MMLU-Pro)

The paper's pre-training table scores base models with one harness:

| Benchmark | How it is scored | Script |
|---|---|---|
| **CORE** | nanochat's 22-task suite, 500 examples per task. Each tokenizer's own default prefix token, left truncation at 2,048 tokens, forward passes batched across examples. | `core_batched.py`, with the rendering and scoring of `core_eval_head.py` |
| **MMLU** | LM Evaluation Harness 0.4.9 `mmlu`, 5-shot, the answer letter the model rates most likely | `mmlu_any.py` |
| **MMLU-Pro** | LM Evaluation Harness 0.4.9 `leaderboard_mmlu_pro`, scored the same way over 10 options | `mmlu_any.py` |

`models.py` loads nanochat checkpoints natively and any `transformers` base model through one interface. The adapter
(`models.py`, `mmlu_any.py`) is Lingwei Gu's, byte for byte (SHA256 `b3ff5514…` and `f442bace…`); `core_batched.py`
and `core_eval_head.py` are from the same harness.

```bash
EVAL_BUNDLE=<nanochat eval_bundle> PY=<python with lm-eval 0.4.9> \
  bash pretrain_eval/run_eval.sh nanochat <dir with model_227158.pt, meta, tokenizer/> 227158 gaggle
```

## Verified

The public copy reproduces the paper's printed scores. Scores from 2026-11-19, on an H200; JSONs in
`paper/results/pretraining/`.

| Model | CORE | MMLU | MMLU-Pro |
|---|---|---|---|
| `gaggle-nanochat-pretrained-climbmix-20260924` (step 227,158) | 0.4578 (paper 0.458) | 0.4927 (0.493) | 0.1776 (0.178) |
| `…-bidirectional-20260929` (step 229,067, masked answers) | not scored, see below | 0.4534 (0.453) | 0.1871 (0.187) |
| OPEN-1B (`Gensyn/open-1b-base`) | 0.2392 (0.239) | † | † |

† The paper copies these from the developer.

OPEN-1B defines no BOS token. Its match confirms that the prefix-token rule here is the paper's final one.

## Not included yet

- **CORE of the bidirectional model with masked prediction** (paper 0.339). It needs a masked path in the CORE
  scorer that this copy does not have.
- **The other eleven comparison models.** Their exact Hub ids and revisions are not pinned here. Known ids:
  `allenai/Olmo-3-1025-7B`, `allenai/Olmo-Hybrid-7B`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-2B-Base`,
  `google/gemma-4-E2B`, `ibm-granite/granite-4.1-3b-base`, `openbmb/MiniCPM5-2B-Base`, and `karpathy/nanochat-d34`
  (scored as a nanochat checkpoint at nanochat `f37d45c` with `EVAL_AUTOCAST_BF16=1`). Puro 2B, Apertus v1.1 4B,
  K2-Horizon (tag `pretrain_1100000`) and LFM2.5-2.6B still need theirs.
- **The developer-reported numbers** in the table (marked † in the paper) are copied, not scored.
