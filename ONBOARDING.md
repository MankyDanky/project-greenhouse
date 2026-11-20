# Onboarding

This guide takes you from a fresh clone to:
1. an evaluated released model;
2. a fine-tuning run you trained yourself;
3. a new experiment of your own.

Keep [REPRODUCE.md](REPRODUCE.md) open next to it for the per-row commands.

## 1. The project in five minutes

Gaggle is a pointwise reranker. It reads `Query: {q}\nPassage: {p}\nRelevant:` and scores the pair as
logit(" true") − logit(" false") at the last token. Those are the two LM-head rows of our own pre-trained model,
fine-tuned with everything else; no new head.

| Stage | What happens | Where |
|---|---|---|
| Pre-training | nanochat d34 (3.29B parameters) on ClimbMix, one pass of 287B tokens, batch ramped 262K → 2M tokens at fixed learning rates | `pretrain/` |
| Fine-tuning | one epoch of RLHN-250K with LCE: a softmax over one positive and K ∈ {7, 15, 23} hard negatives per query, 128 queries per update | `reranker/` |
| Evaluation | rerank BM25 top-100 on TREC DL 2019–23 and seven BEIR sets; nDCG@10, unweighted means per group | `eval/` |

Three ideas run through the code:
- **One config per paper row.** Configs spell out every setting, with no inheritance, and carry a header naming
  the paper row, its expected scores, the original run and its hardware.
- **The training plan is part of the experiment.** Which queries share an optimizer step depends on the per-GPU
  token budget, so configs pin the budget the row ran with and stop rather than repack on a GPU that cannot
  afford it. The multi engine separates the plan from execution: it can split a microbatch into chunks or use
  GradCache without changing the gradient.
- **Noise first.** Four runs of the same recipe differ by about 0.001 on benchmark means and 0.003 on single
  collections (Sec. 5). Treat smaller differences as noise.

## 2. Environment

You need Python 3.11+, a CUDA GPU with at least 80 GB for fine-tuning (evaluation runs on 24 GB), and about 60 GB of
disk for one model plus data.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r reranker/requirements.txt
pip install pyserini                 # only to build BM25 candidates; needs Java 21 (JAVA_HOME)
```

`trec_eval` must be on `PATH` or in `EVAL_TREC_EVAL`. Pyserini and Anserini ship one under `tools/eval/`, or build
it from https://github.com/usnistgov/trec_eval.

## 3. Evaluate the released reranker (about 30 minutes on one GPU)

```bash
bash eval/fetch_qrels.sh                                  # qrels for the 12 test sets, checksum-verified
bash eval/candidates/build_candidates.sh dl19 dl20        # BM25 top-100 from pyserini's prebuilt indexes
bash eval/run_eval.sh gaggle_hf castorini/gaggle-reranker-20261005 gaggle dl19 dl20
python eval/report.py gaggle                              # DL19 ≈ 0.757, DL20 ≈ 0.723
```

`build_candidates.sh` ends with `digest.py check`, which confirms you rebuilt exactly the inputs the paper scored.
It compares query texts, candidate doc IDs, their order, and the normalised text the reranker reads.

**Robust04 and TREC-News.** Their documents may not be redistributed: TREC Disks 4&5 and the Washington Post
collection are licensed by NIST. The candidate files contain that text, so they live only under `data/`, which is
git-ignored. Never commit them, or anything derived that contains document text. Doc IDs, qrels, run files and
scores are fine. DL21–23 need Pyserini's MS MARCO v2 passage index (~100 GB).

## 4. Train a row (run 1)

From `reranker/`:

```bash
bash data_prep/fetch_models.sh gaggle            # backbone + tokenizer, SHA256-verified
bash data_prep/prepare_data.sh rlhn250k          # RLHN-250K -> data/rlhn250k_prepared_climbmix_lens (~1 h CPU)
bash launch/train.sh configs/sft/run1_seed17.yaml 2
```

**Training data:**
- `prepare_data.sh` rebuilds the training set from the public `rlhn/rlhn-250K` and checks the result against the
  SHA256 of the file the paper's runs read (`data_prep/SHA256SUMS`).
- If the upstream dataset has changed since September 2026, the hash will differ. The run will still train, but on
  a different plan.

**The run:**
- `launch/train.sh` reads the engine from the config header and starts `torchrun` on the given number of GPUs.
- It writes `checkpoints/sft_run1_seed17/`:
  - full checkpoints, about 36 GB with AdamW state;
  - `resolved_config.yaml`;
  - per-rank JSONL metrics.
- `resume: auto` continues from the newest checkpoint after an interruption.
- Run 1 pins a 39,312-token budget per GPU on two ranks, which needs 96 GB GPUs. On anything smaller the trainer
  stops by design.

**After training:**

```bash
python tools/strip_optimizer.py checkpoints/sft_run1_seed17/ckpt_final.pt   # 36 GB -> 12 GB; still evaluable
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/sft_run1_seed17/ckpt_final.pt my-run1
python paper/check_results.py --results eval/results --tag sft.run1=my-run1 --tolerance 0.005
```

## 5. Where things are

| You want to… | Look at |
|---|---|
| change the loss, the prompt or the readout | `reranker/src/losses_nc.py`, `prompt_builder_nc.py`, `modeling_nanochat_upstream.py` |
| understand packing and the plan | `reranker/src/dataset_nc.py`, `stepping.py`; `reranker/engines/multi/sampling.py` |
| use a different backbone | `reranker/engines/multi/backbones.py` (nanochat, Qwen, Gemma, MiniCPM) |
| add an evaluation set | `eval/candidates/build_first_stage.py` (`SPEC`) and `eval/rerank_eval.py` (`QRELS`) |
| see what every paper row used | `paper/manifest.yaml` and the header of each config |

The four engines share `reranker/src` for data, plan, loss, LR schedule and the FP32 guard:
- **nanochat**: most rows;
- **multi**: any backbone on run 1's frozen plan;
- **gemma4** and **qwen35**: the native trainers of two Sec. 6.1 rows;
- **ce**: pointwise cross-entropy.

They differ only where the paper's rows were produced by different execution code. Keeping each row on the
code that produced it keeps every row reproducible.

## 6. Adding an experiment

1. Copy the closest paper config into `reranker/configs/experiments/<name>.yaml`. Never edit a config under
   `sft/`, `backbones/`, `data/`, `attention/`, `curation/` or `loss/`: those are the paper.
2. Change one thing, and say what and why in the header.
3. Train, evaluate with a new tag, and compare against run 1 per dataset (`python eval/report.py`). Judge it with the
   noise guideline above.

## 7. Agents

`AGENTS.md` has the repository rules for coding agents. `.claude/skills/` has step-by-step guides that Claude Code
loads on demand:
- `reproduce-paper-row`
- `evaluate-reranker`
- `prepare-data`
- `pretrain-backbone`
- `model-soup`

Each one names its inputs, commands, expected outputs and checks.
