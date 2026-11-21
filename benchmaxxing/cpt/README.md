# Continued pretraining on the evaluation corpora

Continue a Gaggle base model on the raw documents of the evaluation collections plus replayed pre-training data,
then fine-tune and evaluate the reranker exactly as the paper does. **[PLAN.md](PLAN.md)** says what is run and why.

What is specific to this experiment is its data: `prepare_data.py` and `configs/data.yaml`. Everything else uses the
repository's shared tools:
- training: the continued-pretraining trainer in [`pretrain/continue/`](../../pretrain/continue/), with this run's
  settings in `configs/train.yaml`;
- language-model evaluation: nanochat's `scripts/base_eval.py`, from the same place;
- the reranker: [`reranker/`](../../reranker/) and [`eval/`](../../eval/), which take the run's checkpoint directory
  as a backbone directly.

Run every command from this folder unless it says otherwise; paths in the configs are relative to it.

| Path | What it is |
|---|---|
| `PLAN.md` | the plan and the literature behind it |
| `prepare_data.py` | builds the mix: corpus documents + replay, as nanochat shards, plus three evaluation sets |
| `configs/data.yaml` | the data config |
| `configs/train.yaml` | the run's trainer flags (3B model) |
| `requirements-data.txt` | the data-preparation environment (plus Java 21 for pyserini) |
| `tests/smoke_data.sh` | CPU smoke test of data preparation |

Generated, not committed: `work/` (caches, checkpoints) and `data/` (mixes).

## 1. Set up

```bash
python -m venv .venv-data && .venv-data/bin/pip install -r requirements-data.txt   # + Java 21 on PATH (pyserini)
bash ../../pretrain/continue/setup.sh --env                    # the trainer and its environment
(cd ../../reranker && bash data_prep/fetch_models.sh gaggle)   # the 3B base model (10.6 GB)
PYTHON=.venv-data/bin/python bash tests/smoke_data.sh          # CPU check of the data preparation
```

## 2. Build the mix

```bash
.venv-data/bin/python prepare_data.py configs/data.yaml          # -> data/ir_mix
```

Any key of the config can also be set on the command line (`--set KEY=VALUE`, dotted keys, YAML values). At the
end, the log prints `num-iterations` for one pass over the mix; `configs/train.yaml` has the value for the default mix.

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
source ../../pretrain/continue/third_party/nanochat/.venv/bin/activate
NANOCHAT_BASE_DIR=work/nanochat_base PYTHONPATH=../../pretrain/continue/third_party/nanochat \
  torchrun --standalone --nproc_per_node=8 -m scripts.base_train --config configs/train.yaml
```

The run starts from the released 3B checkpoint with a fresh optimizer and trains one pass over the mix (settings in
PLAN.md, section 5). Its checkpoints, with the model's tokenizer, go to
`work/nanochat_base/base_checkpoints/gaggle3b_ir_mix/`. Rerunning the command resumes the run.

## 4. Evaluate the language model

Before and after, on the mix's evaluation sets, with `base_eval.py --checkpoint-dir`:

```bash
export PYTHONPATH=../../pretrain/continue/third_party/nanochat
for ckpt in ../../reranker/models/gaggle-nanochat-pretrained-climbmix-20260924 work/nanochat_base/base_checkpoints/gaggle3b_ir_mix; do
  for set in heldout_corpus seen_corpus heldout_replay; do
    NANOCHAT_DATA_DIR=data/ir_mix/$set python -m scripts.base_eval --eval bpb --checkpoint-dir $ckpt \
      --device-batch-size 8 --split-tokens 524288
  done
  NANOCHAT_BASE_DIR=work/nanochat_base python -m scripts.base_eval --eval core --checkpoint-dir $ckpt
done
```

`base_eval` reports bpb for the two shards of each set; take their mean.
- **Gain:** a lower `heldout_corpus` bpb after CPT.
- **Memorisation:** how much more `seen_corpus` falls than `heldout_corpus`.
- **Forgetting:** a rise in `heldout_replay` bpb.

## 5. Fine-tune and evaluate the reranker

The run's directory is the backbone; the newest step (the final one) is used. From the repository root:

```bash
cd reranker
bash launch/train.sh configs/sft/run1_seed17.yaml 2 \
  --override model_name=../benchmaxxing/cpt/work/nanochat_base/base_checkpoints/gaggle3b_ir_mix \
  output_dir=checkpoints/cpt_gaggle3b_ir_mix wandb_run_name=cpt_gaggle3b_ir_mix
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/cpt_gaggle3b_ir_mix/ckpt_final.pt cpt_gaggle3b_ir_mix
python eval/report.py cpt_gaggle3b_ir_mix
```

The reranker needs its fine-tuning data (`reranker/data_prep/`) and the BM25 candidates (`eval/candidates/`); see
the repository's [ONBOARDING.md](../../ONBOARDING.md). The 0.6B model, once released, uses
`reranker/configs/experiments/gaggle06b_lce.yaml`.
