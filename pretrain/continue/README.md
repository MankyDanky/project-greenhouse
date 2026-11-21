# Continued pre-training

Continue pre-training a nanochat checkpoint (e.g. a released Gaggle base model) on another data mix, and use the
result like any released checkpoint: evaluate it, or fine-tune a reranker on it. Any experiment that trains further
from a checkpoint uses this trainer; it brings only its own data and config (e.g.
[`benchmaxxing/cpt/`](../../benchmaxxing/cpt/)).

The trainer is nanochat's own `scripts/base_train.py` at `92d63d4`, the commit the Gaggle models were pre-trained
from, plus `nanochat_continue.patch`. `setup.sh` builds it in `third_party/nanochat` and lists what the patch
changes. The paper's pre-training code in `pretrain/nanochat` is not touched.

## Set up

```bash
bash setup.sh --env        # karpathy/nanochat @ 92d63d4 + patch; .venv with uv (needs uv and CUDA 12.8 drivers)
source third_party/nanochat/.venv/bin/activate
```

The starting checkpoint is a directory with `model_<step>.pt`, `meta_<step>.json` and `tokenizer/`. A released
Gaggle model fetched for the reranker is one: from `reranker/`, `bash data_prep/fetch_models.sh gaggle`.

## Train

All of a run's settings go in one YAML file whose keys are the trainer's flags. A flag on the command line overrides
the file.

```bash
NANOCHAT_BASE_DIR=<work dir> PYTHONPATH=<this folder>/third_party/nanochat \
  torchrun --standalone --nproc_per_node=8 -m scripts.base_train --config <run>.yaml
```

The flags this patch adds:

| Flag | Meaning |
|---|---|
| `--config FILE.yaml` | flag values from a file; an unknown key is an error |
| `--init-from-dir`, `--init-from-step` | start from that checkpoint's weights with a fresh optimizer. The architecture flags (`depth`, `aspect-ratio`, `head-dim`, `window-pattern`, `max-seq-len`, `ve-every`) must match its meta, which is checked at load. |
| `--data-dir` | the directory of training shards (parquet, column `text`; the last shard is the validation split) |
| `--tokenizer-dir` | default: the starting checkpoint's `tokenizer/` |
| `--literal-hparams` | use the given learning rates and weight decay as they are: no rescaling for batch size or training horizon |

The upstream flags still apply: `--num-iterations`, `--total-batch-size`, `--warmup-steps`, `--warmdown-ratio`,
`--final-lr-frac`, `--keep-checkpoints`, `--auto-resume` and so on.

Checkpoints go to `$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/`. A run started from a checkpoint also gets that
checkpoint's tokenizer files there, so its directory is laid out like a released one. Rerunning the same command
resumes from the newest complete checkpoint.

## Evaluate

`scripts/base_eval.py --checkpoint-dir <dir>` scores any checkpoint directory, released or a run (newest step, or
`--step`):
- `--eval bpb` gives bits per byte on the two shards of `$NANOCHAT_DATA_DIR`;
- `--eval core` runs CORE.

```bash
PYTHONPATH=third_party/nanochat NANOCHAT_DATA_DIR=<eval shards> python -m scripts.base_eval \
  --eval bpb --checkpoint-dir <checkpoint dir> --device-batch-size 8 --split-tokens 524288
```

## Fine-tune a reranker on it

A run's checkpoint directory is a reranker backbone as it is. Pass it as `model_name`; the newest step is used, or
the one named by `model_step`:

```bash
cd reranker
bash launch/train.sh configs/sft/run1_seed17.yaml 2 --override model_name=<run dir> \
  output_dir=checkpoints/<name> wandb_run_name=<name>
```

A model whose value embeddings are not on alternate layers (`ve_every` other than 2) also needs
`nanochat_src=../pretrain/continue/third_party/nanochat`. The reranker's pinned snapshot does not know `ve_every`;
the trainer writes it to the meta only when it is not 2.

## Test

```bash
PYTHON=third_party/nanochat/.venv/bin/python bash tests/smoke.sh
```

A CPU test, a few minutes long, with the Gaggle tokenizer and a synthetic corpus. It checks:
- the config file, the command-line override and the start from a checkpoint (tokenizer included);
- `base_eval --checkpoint-dir` on a released-style directory and on the run;
- the run's directory as a reranker backbone: the newest step or `model_step`, and scoring through
  `eval/rerank_eval.py`.
