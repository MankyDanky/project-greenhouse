# Continued pretraining on the evaluation corpora

Continue a Gaggle base model on the raw documents of the evaluation collections plus replayed pre-training data,
then fine-tune and evaluate the reranker exactly as the paper does. **[PLAN.md](PLAN.md)** says what to run and why:
when and what to mix, how much replay, the learning rate, the run order and the decision rules.

Everything is set by YAML: the data mix in `configs/data/`, the training run in `configs/train/`. Paths in the
configs are relative to this folder. Run every command from here unless it says otherwise.

**The 0.6B Gaggle model is not released yet.** Its configs carry placeholders for its repository, step and
original weight decay: `configs/train/gaggle06b_ir_r25.yaml`, `configs/rerank/gaggle06b_lce.yaml`, and
`REPO_06B` / `STEP_06B` in `fetch_models.sh`. `train.py` stops until they are filled in. Everything else for it is
ready.

## Files

| Path | What it is |
|---|---|
| `PLAN.md` | the experiment plan and the literature behind it |
| `setup.sh` | builds the trainer: karpathy/nanochat at `92d63d4` + `nanochat_cpt.patch` in `third_party/nanochat` (`--env` also creates the training environment) |
| `requirements-data.txt` | the data-preparation environment (plus Java 21 for pyserini) |
| `nanochat_cpt.patch` | the changes to nanochat, each commented in the code (listed in `setup.sh`) |
| `fetch_models.sh` | downloads the starting checkpoints from Hugging Face and verifies their SHA256 |
| `prepare_data.py` | builds a mix: corpus documents + replay, as nanochat shards, plus three evaluation sets |
| `train.py` | turns a train YAML into nanochat's `scripts/base_train.py` flags and runs it |
| `evaluate.py` | bpb on the evaluation sets (and optionally CORE) for a run or its starting checkpoint |
| `export_for_reranker.py` | copies a CPT checkpoint into `reranker/models/` and prints the fine-tuning and evaluation commands |
| `configs/data/` | `ir_r25` (the main mix: BEIR, MS MARCO v1 and v2 at 1 : 1 : 1, 25% replay), `ir_r00/05/50` (replay sweep), `control_replay_only`, `ir_r25_minus_scifact`; options no run uses: `ir_r25_no_ood` (without the out-of-domain collections), `ir_all_r25` (every corpus in full) |
| `configs/train/` | the 3B arms (main, control) and the 0.6B arms (main, replay sweep, LR probe, control, leave-one-out) |
| `configs/rerank/gaggle06b_lce.yaml` | the reranker recipe for the 0.6B model (the 3B model uses the paper's `reranker/configs/sft/run1_seed17.yaml`) |
| `tests/` | CPU smoke tests: `smoke_data.yaml`, `smoke_minus.yaml`, `smoke_no_ood.yaml`, `smoke_control.yaml` (data), `smoke_train.sh` (train → evaluate → export) |

Generated, not committed: `third_party/`, `models/`, `work/` (caches, checkpoints), `data/` (mixes), `results/`.

## 1. Set up

Two Python environments: one for data preparation, one for training (nanochat's pinned GPU stack).

```bash
python -m venv .venv-data && .venv-data/bin/pip install -r requirements-data.txt   # + Java 21 on PATH (pyserini)
bash setup.sh --env                     # nanochat @ 92d63d4 + patch; .venv with uv (needs uv and CUDA 12.8 drivers)
bash fetch_models.sh                    # the starting models (3B: 10.6 GB); or: 3b | 0.6b | --tokenizer
```

Check the pipeline on CPU in a few minutes:

```bash
.venv-data/bin/python prepare_data.py tests/smoke_data.yaml --limit 3000      # groups, a partial pass
.venv-data/bin/python prepare_data.py tests/smoke_minus.yaml --limit 3000     # exclude_sources
.venv-data/bin/python prepare_data.py tests/smoke_no_ood.yaml --limit 3000    # exclude_ood
.venv-data/bin/python prepare_data.py tests/smoke_control.yaml --limit 3000   # replay only
PYTHON=third_party/nanochat/.venv/bin/python bash tests/smoke_train.sh
```

## 2. Build a mix

```bash
.venv-data/bin/python prepare_data.py configs/data/ir_r25.yaml
```

Resources: a few hours on one CPU node for the corpus caches (about 11,000 passages a second in our test; MS MARCO
v2 has 138M, and its whole cache is built even when the mix uses part of it), a few GB of memory whatever the corpus
size, up to about 100 GB of work space, and about 2.7 bytes per token for each built mix (about 12 GB for
`ir_r25`).

Three stages, all controlled by the YAML:
1. **corpus**: reads each enabled source, formats `title\n\ntext`, splits long documents, drops duplicates, sets
   aside held-out pieces by hash, and counts tokens. Cached per source in `work/ir/corpus/`, in a fixed random order,
   so the other arms reuse it. MS MARCO v2 comes through `ir_datasets`, which downloads the full collection once
   (about 20 GB); Robust04 and TREC-News through Pyserini's prebuilt BEIR indexes.
2. **replay**: reads ClimbMix shards in a seeded order up to the token target. Cached in `work/ir/replay/`; an arm
   with less replay reads a prefix of the same stream.
3. **mix**: sizes each source from the group ratios (below), interleaves the sources in proportion, then corpus and
   replay following `replay.schedule`, and writes `data/<name>/`:
   - `train/`: the shards. The last one is the held-out mix nanochat reports as validation bpb.
   - `heldout_corpus/`: corpus pieces never trained on.
   - `seen_corpus/`: the same number of trained-on pieces, per source.
   - `heldout_replay/`: documents from the pre-training validation shard.
   - `manifest.json`: sources, token counts, the realised replay share per shard, and the SHA256 of every shard.

**How much of each corpus.** Every source has a `group` and `passes`, the most passes it may get (also its weight
within the group). `groups:` gives each group a `ratio`. The group that runs out first at its sources' passes sets the
corpus size; every other group contributes its ratio of that, all its sources scaled together. Below one pass a
source contributes a fixed random subset of its documents. `ratio: all` takes a group whole. The `plan:` lines of the
log and `manifest.json` (`groups`, `sources`) show the result. In `ir_r25`:

| Group | Sources (most passes) | Ratio | Result |
|---|---|---|---|
| `beir` | TREC-COVID, NFCorpus, FiQA, SciDocs, SciFact (4); Robust04, TREC-News (1) | 1 | all of it, ~1.09B tokens: it sets the size |
| `msmarco-v1` | MS MARCO v1 (2) | 1 | ~1.09B: about 1.5 passes |
| `msmarco-v2` | MS MARCO v2 (1) | 1 | ~1.09B: about 13% of the passages |

The replay options, all under `replay:` in the YAML:

| Key | Meaning |
|---|---|
| `fraction` | replay tokens / all training tokens, added on top of the corpus (0 = none) |
| `tokens` | an absolute replay budget instead of `fraction` (the replay-only control) |
| `schedule`, `skew` | `constant`; or `front_loaded` / `back_loaded`, where the share runs from f·(1+skew) to f·(1−skew) or the reverse, with the same total |
| `source` | Hugging Face dataset repo, file pattern and shard range to draw from |
| `heldout_shard`, `heldout_docs` | the pre-training validation shard and how many of its documents to use for the forgetting measure |

**Out-of-domain collections.** TREC-COVID, NFCorpus, Robust04 and TREC-News, which the paper counts as out of
domain for the fine-tuning data, are marked `ood: true`. `exclude_ood: true` keeps them out of the mix entirely
(they are not even read) and sizes the mix without them: `configs/data/ir_r25_no_ood.yaml`, with its own ratios
(1 : 4 : 4, because in-domain BEIR is only ~18M tokens), about 0.65B corpus tokens.

Other keys:
- `exclude_sources`: leave sources out after the mix is sized, so the others contribute exactly what they do with
  them (the leave-one-out arm);
- `enable_sources`: turn on sources marked `enabled: false`;
- `corpus[].sample`: keep a fixed fraction of a source's documents (by id hash) when its cache is built;
- `heldout.fraction` / `max_docs_per_source`, `format.max_chars`, `output.tokens_per_shard`.

A config can `extends:` another and override only what differs (see `ir_r05.yaml`).

**Licensed corpora.** Robust04 and TREC-News are always in the mix. Their text is licensed by NIST and may not be
redistributed: never publish a built mix or the caches (`data/`, `work/` are git-ignored), and check the licence
before releasing a model trained on them.

## 3. Train

```bash
python train.py configs/train/gaggle3b_ir_r25.yaml --check           # derived flags, validation; no training
python train.py configs/train/gaggle3b_ir_r25.yaml --ngpu 8
```

`train.py` reads the starting checkpoint's `meta_<step>.json` and derives:
- the architecture flags;
- the peak learning rates: the original run's peak times `lr.scale`;
- the number of steps: one pass over the mix;
- the warmup and decay.

It then runs nanochat's trainer, with `--init-from-dir` (the released weights, fresh optimizer) and
`--literal-hparams` (no batch-size or horizon rescaling). Checkpoints go to
`work/nanochat_base/base_checkpoints/<run>/`. A rerun resumes from the newest complete checkpoint, so a preempted job
can just be resubmitted.

The YAML keys: `init`, `data`, `tokens`, `batch_tokens`, `device_batch_size`, `fp8`, `lr.{scale, warmup_frac,
decay_frac, final_frac}`, `weight_decay`, `eval_every`, `save_every`, `keep_checkpoints`, `wandb`, and `extra_args`
(any other `base_train.py` flag, passed through), and `rerank_config` (the reranker recipe for the export). Under
Slurm, wrap the same command in your job script.

## 4. Evaluate the language model

```bash
python evaluate.py configs/train/gaggle3b_ir_r25.yaml --init --core    # the starting checkpoint
python evaluate.py configs/train/gaggle3b_ir_r25.yaml --core           # the run's final checkpoint
```

Writes `results/<run>/{init,step_N}.json` with bpb on `heldout_corpus` (gain), `seen_corpus` (its difference from
held-out is memorisation), `heldout_replay` (forgetting) and CORE. Every arm reads the sets of `eval_data` (default
`data/ir_r25`), so the numbers compare across arms.

## 5. Fine-tune and evaluate the reranker

```bash
python export_for_reranker.py configs/train/gaggle3b_ir_r25.yaml      # -> reranker/models/cpt-gaggle3b_ir_r25
```

It prints the exact commands. For the 3B model that is the paper's run-1 recipe with the backbone swapped:

```bash
cd ../../reranker
bash launch/train.sh configs/sft/run1_seed17.yaml 2 --override model_name=models/cpt-gaggle3b_ir_r25 \
  output_dir=checkpoints/cpt_gaggle3b_ir_r25 wandb_run_name=cpt_gaggle3b_ir_r25
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/cpt_gaggle3b_ir_r25/ckpt_final.pt cpt_gaggle3b_ir_r25
python eval/report.py cpt_gaggle3b_ir_r25
```

The reranker needs its fine-tuning data (`reranker/data_prep/`) and the BM25 candidates (`eval/candidates/`); see
the repository's [ONBOARDING.md](../../ONBOARDING.md). The 0.6B model uses
`configs/rerank/gaggle06b_lce.yaml` (each train config names its recipe in `rerank_config`).
