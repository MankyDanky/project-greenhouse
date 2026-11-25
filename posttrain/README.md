# Gaggle midtraining + SFT (chat models)

The recipe that produced **[`castorini/gaggle-nanochat-ift-climbmix-20261031`](https://huggingface.co/castorini/gaggle-nanochat-ift-climbmix-20261031)**:
the chat model made from the pre-trained Gaggle base
([`castorini/gaggle-nanochat-pretrained-climbmix-20260924`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924),
step 227,158; recipe in [`../pretrain`](../pretrain)) with nanochat's two-stage chat recipe: midtraining, then
supervised fine-tuning (SFT). There is no RL. The run takes one GPU and about 7 hours. Site paths and Slurm arguments
live in one config file. Contact: Lingwei Gu.

The same recipe, unchanged, also made
**[`castorini/gaggle-nanochat-0.9b-ift-climbmix-20261108`](https://huggingface.co/castorini/gaggle-nanochat-0.9b-ift-climbmix-20261108)**
from the 0.9B Gaggle base ([`castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105`](https://huggingface.co/castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105),
step 229,067) on two GPUs: set `POST_MODEL=0.9b` in `config.sh`, and see [The 0.9B model](#the-09b-model) below.
Contact for the 0.9B: Sahel Sharifymoghaddam.

## Result

| | |
|---|---|
| Starting point | Gaggle base, step 227,158 (`model_227158.pt`, 32,768-token tokenizer) |
| Recipe | nanochat commit `348fbb3` (`scripts/mid_train.py`, then `scripts/chat_sft.py`), data and hyperparameters unchanged, ported onto nanochat `92d63d4` (the code the base was trained with) |
| Compute | one H100 80 GB: midtraining 3 h 10 min, SFT 25 min, chat evaluation about 2.5 h |
| Released checkpoints | SFT step 700 (`chatsft_checkpoints/gaggle227158/model_000700.pt`) and midtraining step 816 (`mid_checkpoints/gaggle227158/model_000816.pt`) |
| Chat scores (SFT, zero-shot) | ARC-Easy 81.57, ARC-Challenge 68.00, MMLU 44.93, MMLU-Pro 19.97, GSM8K 30.63, HumanEval 15.85, ChatCORE 0.412 |

nanochat removed midtraining on 2026-01-31 (commit `1ddaad1`); `348fbb3` is the last version with it. Its scripts were
ported with API changes only:
- no autocast, since `92d63d4` handles the compute dtype inside the model;
- the full model config is saved in each checkpoint's metadata;
- no `report.md`.

The model card compares this model with nanochat d34 before and after the same recipe.

## The recipe

| | Midtraining | SFT |
|---|---|---|
| Script | `scripts/mid_train.py` | `scripts/chat_sft_348.py` (`348fbb3`'s `chat_sft.py`) |
| Data (rows) | smol-smoltalk train 460,341; MMLU `auxiliary_train` 99,842; GSM8K train 7,473; nanochat identity conversations x2 2,000; SimpleSpelling 200,000; SpellingBee 80,000 = **849,656** | ARC-Easy train 2,251; ARC-Challenge train 1,119; GSM8K train 7,473; smol-smoltalk first 10,000 (seed-42 shuffle); identity 1,000; SimpleSpelling 300; SpellingBee 300 = **22,443** |
| Batch | 524,288 tokens per step, 2,048-token BOS-aligned best-fit packing | 32 conversations per step |
| Loss | all tokens | assistant tokens only |
| Learning rate | embeddings 0.2, unembedding 0.004, Muon 0.02, no weight decay; constant for 80% of the steps, then linear to 0; Muon momentum 0.85 to 0.95 over the first 300 steps | the same learning rates x 0.02, linear to 0 |
| Length | one epoch = 816 steps | one epoch = 701 steps; the checkpoint is saved at step 700 |

Both stages start from a fresh optimizer, so the pretraining optimizer state is not needed. Neither stage replays
pretraining (ClimbMix) text. On one GPU the batches are reached by gradient accumulation:
- midtraining accumulates 64 micro-batches of 4 x 2,048 tokens per step;
- SFT accumulates 8 micro-batches of 4 conversations per step.

## Requirements

- Slurm, with one GPU of 80 GB (H100 or H200) for the job (the 0.9B: two).
- About 60 GB on a filesystem shared by the login and compute nodes.
- [`uv`](https://docs.astral.sh/uv/).
- Internet on the login node only; the compute node runs offline.
- `../pretrain/nanochat` (nanochat `92d63d4`) from this repository. The 0.9B needs its value-embedding spacing
  (`ve_every`), which that copy has.

## Run

```bash
cd posttrain
cp config.example.sh config.sh             # edit POST_ROOT, POST_GPU_SBATCH, POST_CPU_SBATCH (and POST_ENV_SETUP);
                                           # POST_MODEL=0.9b for the 0.9B (and two GPUs in POST_GPU_SBATCH)
export POST_CONFIG=$PWD/config.sh
bash setup.sh                              # login node: nanochat copy + overlay, Python env, FA3, inputs, task data
bash submit.sh gpu run.sbatch --export=ALL,SMOKE=1   # ~15 min: 4 midtraining + 4 SFT steps + chat_eval on 8 problems
bash submit.sh gpu run.sbatch              # the full run: midtraining -> SFT -> chat_eval (~7 h)
```

`setup.sh` checks its inputs:
- the SHA256 of the base checkpoint, its metadata and the tokenizer;
- the identity file's 1,000 rows;
- the row count of every dataset (the numbers above).

The smoke job ends with `SMOKE_OK`. The full job ends with `ALL_DONE`. A stage whose checkpoint already exists is
skipped, so resubmitting `run.sbatch` after a failure continues from the last finished stage.

Where things land: everything goes under `$POST_ROOT`.
- **Checkpoints:** `base/mid_checkpoints/gaggle227158/` (model + optimizer, 24 GB) and
  `base/chatsft_checkpoints/gaggle227158/` (10.6 GB).
- **Logs:** `logs/run-<job>/` (one per stage) and `logs/gaggle-posttrain-<job>.out`.

## Checks against the original run

The original run used one H100. On other GPUs or attention kernels, bf16 rounding moves the
numbers slightly.

| Stage | What to compare |
|---|---|
| midtraining | 816 steps (the last one reports 100.03%: it crosses into epoch 2, as upstream does); ~13.96 s/step, 37.6k tokens/s, MFU 48.8%; peak memory 59 GB |
| midtraining validation bpb | step 0: 0.3902; 150: 0.3122; 300: 0.3133; 450: 0.3117; 600: 0.3097; 750: 0.2951; 816: 0.2880 |
| SFT | 701 steps; step 700 validation loss 0.7122; in-training MMLU 0.4541 and ARC-Easy 0.8154 (1,024-question subsets) |
| `chat_eval -i sft` | ARC-E 81.57, ARC-C 68.00, MMLU 44.93, GSM8K 30.63, HumanEval 15.85, ChatCORE 0.4116 |
| `chat_eval -i mid` | ARC-E 80.01, ARC-C 65.70, MMLU 44.32, GSM8K 13.95, HumanEval 16.46, ChatCORE 0.3676 |
| `chat_eval -i sft -a "MMLU\|MMLU-Pro"` | MMLU 44.93, MMLU-Pro 19.97 |

## Pitfalls

- **Use `chat_sft_348.py`, not `chat_sft.py`.** `92d63d4`'s own `chat_sft.py` is a different, single-stage recipe
  that takes its batch size and learning rates from the checkpoint's metadata.
- **Never pass `--num-iterations` to a real midtraining run.** With `--num-iterations > 0` the old `mid_train.py`
  counts LR progress in micro-batches, and the multiplier goes negative. The smoke run uses it with `--dry-run` only.
  The full run uses the default -1, one epoch.
- **Micro-batch size.** For the 3B keep it at 4 on an 80 GB GPU; the midtraining script's own default (32) does not
  fit d34. `config.sh` sets the GPU count and the per-GPU micro-batches (`POST_NPROC`, `POST_MID_DBS`, `POST_SFT_DBS`);
  the scripts accumulate up to the same total batch on any number of GPUs. The 3B ran on one GPU and the 0.9B on two;
  pankajmathur's nanochat d34 run used 4 per GPU on 8 A100s.
- **Bad GPUs.** A GPU in a bad state ("GPU requires reset") can be listed by NVML while CUDA fails to start, and torch
  then runs silently on the CPU. `run.sbatch` checks `torch.cuda.init()` first and stops if it fails; exclude that node.
- **Job length.** Midtraining checkpoints only at its end, so the job must allow about 3.5 h for it. `run.sbatch` asks
  for 12 h.
- **GSM8K on a pre-trained model is slow.** `chat_eval` generates the full 512 tokens for every problem, because the
  model never ends its reply. To split it across jobs, set `CHAT_EVAL_SLICE=start:stop` (overlay `chat_eval.py`) and
  sum the `Final: x/n` lines.

## Layout

| Path | What |
|---|---|
| `config.example.sh` | the only file with site paths and Slurm arguments; `POST_MODEL` picks the base (3b or 0.9b) and derives its checkpoint, checksums, GPU count and micro-batches |
| `setup.sh`, `submit.sh` | one-time login-node setup; sbatch with the configured placement |
| `run.sbatch` | smoke (`SMOKE=1`) or full run: midtraining -> SFT -> chat_eval |
| `nanochat_overlay/` | files copied over `../pretrain/nanochat` by `setup.sh`; each change against `92d63d4` is marked `(gaggle)` |
| `nanochat_overlay/scripts/mid_train.py`, `chat_sft_348.py` | the `348fbb3` scripts, ported (`mid_train.py` also records `ve_every` in its checkpoint, which the 0.9B needs) |
| `nanochat_overlay/tasks/customjson.py`, `spellingbee.py` | from `348fbb3` (not in `92d63d4`) |
| `nanochat_overlay/tasks/mmlu_pro.py` | MMLU-Pro in nanochat's chat format (nanochat has no MMLU-Pro task) |
| `nanochat_overlay/scripts/chat_eval.py` | adds `-a MMLU-Pro` and the opt-in `CHAT_EVAL_SLICE` |
| `nanochat_overlay/nanochat/checkpoint_manager.py` | adds the `mid` checkpoint source (`-i mid`, `--source=mid`) |

## Beyond this folder

The Hugging Face repo holds everything else, with code and logs:
- **5-shot MMLU / MMLU-Pro** (LM Evaluation Harness through our nanochat adapter): `code/eval/` and
  `logs/mmlu_5shot/`.
- **CORE and the four-model comparison** with nanochat d34 (run through nanochat `f37d45c`): `code/comparison/` and
  `logs/comparison/`.
- **The `transformers` / vLLM release:**
  - `convert_ift.py` runs the base repo's converter unchanged and adds nanochat's chat format as `chat_template`.
  - The checks are in `code/hf/` and `logs/hf/`: logits bitwise identical to nanochat in bf16; chat template
    identical to nanochat's rendering on 5,700 conversations; vLLM fp32 log-probabilities within about 0.001 nats.
  - The template accepts message content as a string or as a list of text parts, because vLLM's chat API passes the
    latter.

## The 0.9B model

[`castorini/gaggle-nanochat-0.9b-ift-climbmix-20261108`](https://huggingface.co/castorini/gaggle-nanochat-0.9b-ift-climbmix-20261108)
is this recipe on the 0.9B Gaggle base, with every setting unchanged (data, learning rates, schedules, batch sizes,
one epoch of each stage). `POST_MODEL=0.9b` sets what differs:

| | 3B (above) | 0.9B |
|---|---|---|
| Base | `castorini/gaggle-nanochat-pretrained-climbmix-20260924`, step 227,158 | `castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105`, step 229,067 (d32 x 1152, value embeddings on every 4th layer) |
| GPUs | one H100 | two H200s |
| Midtraining micro-batch | 4 rows x 2,048 tokens, 64 accumulation steps | 16 rows per GPU x 2 GPUs, 8 accumulation steps (the same 524,288 tokens per step) |
| SFT micro-batch | 4 conversations, 8 accumulation steps | 4 per GPU x 2 GPUs, 4 accumulation steps (the same 32 conversations per step) |
| Code | `../pretrain/nanochat` + `nanochat_overlay/` | the same, with one line in `mid_train.py`: it records `ve_every` in the midtraining checkpoint, without which SFT would rebuild the model with nanochat's default of 2 and fail to load it |
| Time | midtraining 3 h 10 min, SFT 25 min | midtraining 29 min, SFT about 6 min |

The learning rates are nanochat's defaults, including its width scaling of the AdamW rates (1/sqrt(1152/768) = 0.816
for this model), as in the 3B run.

Checks against the original run (two H200s; bf16 rounding moves the numbers slightly on other GPUs or kernels):

| Stage | What to compare |
|---|---|
| midtraining | 816 steps, ~2.17 s/step, ~241k tokens/s, peak memory 58 GB per GPU |
| midtraining validation bpb | step 0: 0.4509; 150: 0.3717; 300: 0.3739; 450: 0.3704; 600: 0.3652; 750: 0.3453; 816: 0.3360 |
| SFT | 701 steps; validation loss 0.8279 at step 0 and 0.8349 at step 700; in-training MMLU 0.3867 and ARC-Easy 0.6543 (1,024-question subsets) |
| `chat_eval -i sft` | ARC-E 65.95, ARC-C 49.32, MMLU 38.14, GSM8K 19.41, HumanEval 5.49, ChatCORE 0.2589 |
| `chat_eval -i mid` | ARC-E 63.22, ARC-C 48.21, MMLU 38.33, GSM8K 7.43, HumanEval 6.71, ChatCORE 0.2276 |
| `chat_eval -i sft -a "MMLU\|MMLU-Pro"` | MMLU 38.14, MMLU-Pro 16.69 |
