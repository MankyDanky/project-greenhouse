# Continued pretraining on the evaluation corpora

Continue a Gaggle base model on the raw documents of the evaluation collections plus replayed pre-training data,
then fine-tune and evaluate the reranker exactly as the paper does. **[PLAN.md](PLAN.md)** says what is run and why:
when and what to mix, how much replay, the learning rate, and the literature behind them.

Two YAML files hold every setting: `configs/data.yaml` for the data mix and `configs/train.yaml` for the training
run. Their values are the defaults the plan runs; any key can be changed in the file or on the command line
(`--set KEY=VALUE`, dotted keys, YAML values). Paths in the configs are relative to this folder. Run every command
from here unless it says otherwise.

**The 0.6B Gaggle model is not released yet.** Its entry carries placeholders for its repository, step and
original weight decay: `models.gaggle06b` in `configs/train.yaml`, `configs/rerank/gaggle06b_lce.yaml`, and
`REPO_06B` / `STEP_06B` in `fetch_models.sh`. `train.py` stops until they are filled in.

## Files

| Path | What it is |
|---|---|
| `PLAN.md` | the plan and the literature behind it |
| `setup.sh` | builds the trainer: karpathy/nanochat at `92d63d4` + `nanochat_cpt.patch` in `third_party/nanochat` (`--env` also creates the training environment) |
| `requirements-data.txt` | the data-preparation environment (plus Java 21 for pyserini) |
| `nanochat_cpt.patch` | the changes to nanochat, each commented in the code (listed in `setup.sh`) |
| `fetch_models.sh` | downloads the starting checkpoints from Hugging Face and verifies their SHA256 |
| `prepare_data.py` | builds the mix: corpus documents + replay, as nanochat shards, plus three evaluation sets |
| `train.py` | turns the train config into nanochat's `scripts/base_train.py` flags and runs it |
| `evaluate.py` | bpb on the evaluation sets (and optionally CORE) for the run or its starting checkpoint |
| `export_for_reranker.py` | copies a CPT checkpoint into `reranker/models/` and prints the fine-tuning and evaluation commands |
| `configs/data.yaml` | the data config |
| `configs/train.yaml` | the training config, with the two starting models (`models`) |
| `configs/rerank/gaggle06b_lce.yaml` | the reranker recipe for the 0.6B model (the 3B model uses the paper's `reranker/configs/sft/run1_seed17.yaml`) |
| `tests/` | CPU smoke tests: `smoke_data.sh` (data preparation) and `smoke_train.sh` (train → evaluate → export) |

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
PYTHON=.venv-data/bin/python bash tests/smoke_data.sh
PYTHON=third_party/nanochat/.venv/bin/python bash tests/smoke_train.sh
```

## 2. Build the mix

```bash
.venv-data/bin/python prepare_data.py configs/data.yaml          # -> data/ir_mix
```

Resources: a few hours on one CPU node for the corpus caches (about 11,000 passages a second in our test; MS MARCO
v2 has 138M, and its whole cache is built even though the mix uses part of it), a few GB of memory whatever the
corpus size, up to about 100 GB of work space, and about 12 GB for the built mix.

Three stages:
1. **corpus**: reads each source, formats `title\n\ntext`, splits long documents, drops duplicates, sets aside
   held-out pieces by hash, and counts tokens. Cached per source in `work/ir/corpus/`, in a fixed random order. MS
   MARCO v2 comes through `ir_datasets`, which downloads the full collection once (about 20 GB); Robust04 and
   TREC-News through Pyserini's prebuilt BEIR indexes.
2. **replay**: reads ClimbMix shards in a seeded order up to the token target. Cached in `work/ir/replay/`.
3. **mix**: sizes each source from the group ratios (below), interleaves the sources in proportion, then corpus and
   replay following `replay.schedule`, and writes `data/<name>/`:
   - `train/`: the shards. The last one is the held-out mix nanochat reports as validation bpb.
   - `heldout_corpus/`: corpus pieces never trained on.
   - `seen_corpus/`: the same number of trained-on pieces, per source.
   - `heldout_replay/`: documents from the pre-training validation shard.
   - `manifest.json`: the configuration, sources, token counts, the realised replay share per shard, and the SHA256
     of every shard.

**How much of each corpus.** Every source has a `group` and `passes`, the most passes it may get (also its weight
within the group). `groups:` gives each group a `ratio`. The group that runs out first at its sources' passes sets the
corpus size; every other group contributes its ratio of that, all its sources scaled together. Below one pass a
source contributes a fixed random subset of its documents. `ratio: all` takes a group whole. The `plan:` lines of the
log and `manifest.json` (`groups`, `sources`) show the result:

| Group | Sources (most passes) | Ratio | Result |
|---|---|---|---|
| `beir` | TREC-COVID, NFCorpus, FiQA, SciDocs, SciFact (4); Robust04, TREC-News (1) | 1 | all of it, ~1.09B tokens: it sets the size |
| `msmarco-v1` | MS MARCO v1 (2) | 1 | ~1.09B: about 1.5 passes |
| `msmarco-v2` | MS MARCO v2 (1) | 1 | ~1.09B: about 13% of the passages |

**Replay** (`replay:`): `fraction` (0.25: replay tokens / all training tokens, added on top of the corpus);
`schedule` and `skew` (constant; or front- / back-loaded with the same total); `source` (the ClimbMix shards to draw
from); `heldout_shard` and `heldout_docs` (the forgetting measure); `tokens` (an absolute budget instead of
`fraction`).

**Out-of-domain collections.** TREC-COVID, NFCorpus, Robust04 and TREC-News, which the paper counts as out of
domain for the fine-tuning data, are marked `ood: true`. `exclude_ood: true` keeps them out of the mix entirely (they
are not even read) and sizes it with `groups_exclude_ood` (1 : 4 : 4, because in-domain BEIR is only ~18M tokens):
about 0.65B corpus tokens.

Other keys: `exclude_sources` (leave sources out after the mix is sized), `enable_sources`, `corpus[].sample` (keep a
fixed fraction of a source's documents), `heldout`, `format.max_chars`, `output.tokens_per_shard`.

**Licensed corpora.** Robust04 and TREC-News are always in the mix. Their text is licensed by NIST and may not be
redistributed: never publish a built mix or the caches (`data/`, `work/` are git-ignored), and check the licence
before releasing a model trained on them.

## 3. Train

```bash
python train.py configs/train.yaml --check            # derived flags, validation; no training
python train.py configs/train.yaml --ngpu 8           # the 3B model; --set model=gaggle06b for the 0.6B model
```

`model` picks an entry of `models`: its checkpoint, original weight decay, device batch and reranker recipe.
`train.py` reads the checkpoint's `meta_<step>.json` and derives:
- the architecture flags;
- the peak learning rates: the original run's peak times `lr.scale`;
- the number of steps: one pass over the mix;
- the warmup and decay.

It then runs nanochat's trainer, with `--init-from-dir` (the released weights, fresh optimizer) and
`--literal-hparams` (no batch-size or horizon rescaling). Checkpoints go to
`work/nanochat_base/base_checkpoints/<run>/`, where the run is named `<model>_<mix name>` (e.g. `gaggle3b_ir_mix`).

- **Resume:** a rerun resumes from the newest complete checkpoint, so a preempted job can just be resubmitted.
- **Saved config:** the run saves its resolved config there as `cpt_config.yaml`, and a rerun with different
  settings is refused rather than resuming with another recipe.
- **Overrides:** an override that changes the recipe also needs `run=<new name>`.

Under Slurm, wrap the same command in your job script.

## 4. Evaluate the language model

```bash
python evaluate.py configs/train.yaml --init --core   # the starting checkpoint
python evaluate.py configs/train.yaml --core          # the run's final checkpoint (pass the run's --set, if any)
```

Writes `results/<run>/{init,step_N}.json` with bpb on `heldout_corpus` (gain), `seen_corpus` (its difference from
held-out is memorisation), `heldout_replay` (forgetting) and CORE.

## 5. Fine-tune and evaluate the reranker

```bash
python export_for_reranker.py configs/train.yaml      # -> reranker/models/cpt-gaggle3b_ir_mix
```

It prints the exact commands. For the 3B model that is the paper's run-1 recipe with the backbone swapped:

```bash
cd ../../reranker
bash launch/train.sh configs/sft/run1_seed17.yaml 2 --override model_name=models/cpt-gaggle3b_ir_mix \
  output_dir=checkpoints/cpt_gaggle3b_ir_mix wandb_run_name=cpt_gaggle3b_ir_mix
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/cpt_gaggle3b_ir_mix/ckpt_final.pt cpt_gaggle3b_ir_mix
python eval/report.py cpt_gaggle3b_ir_mix
```

The reranker needs its fine-tuning data (`reranker/data_prep/`) and the BM25 candidates (`eval/candidates/`); see
the repository's [ONBOARDING.md](../../ONBOARDING.md). The 0.6B model uses `configs/rerank/gaggle06b_lce.yaml` (each
entry of `models` names its recipe in `rerank_config`).
