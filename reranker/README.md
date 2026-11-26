# reranker/

Supervised fine-tuning of the Gaggle rerankers (paper Sec. 5 and 6). Run everything from this folder.

```bash
bash data_prep/fetch_models.sh gaggle          # backbones -> models/
bash data_prep/prepare_data.sh rlhn250k        # training data -> data/, SHA256-checked
bash launch/train.sh configs/sft/run1_seed17.yaml 2
```

| Path | Contents |
|---|---|
| `configs/` | one file per paper row: `sft/` (Sec. 5), `backbones/` (6.1), `data/` (6.2), `attention/` (6.3), `curation/` (6.4), `loss/` (6.5); each header names the row, its expected scores, the original run and its hardware |
| `configs/experiments/` | recipes beyond the paper's rows: `gaggle06b_lce.yaml`, the released 0.6B reranker (see below) |
| `src/` | the shared core and the nanochat engine: data and packing (`dataset_nc.py`, `stepping.py`), LCE / self-KD (`losses_nc.py`), prompts (`prompt_builder_nc.py`), upstream nanochat model and readouts (`modeling_nanochat_upstream.py`, `bidirectional.py`), trainer (`train_nc.py`) |
| `engines/multi/` | any backbone (nanochat, Qwen, Gemma, MiniCPM) on run 1's frozen plan, with chunked / GradCache execution: run 4 and the MiniCPM / low-LR Gemma rows |
| `engines/gemma4/`, `engines/qwen35/` | the native Gemma-4-E2B and Qwen3.5-4B-Base trainers of Sec. 6.1 |
| `engines/ce/` | pointwise cross-entropy over the full vocabulary (Sec. 6.5) |
| `data_prep/` | model download, data preparation and checksums, self-filtering |
| `plans/` | records of the multi-engine runs' frozen plans and resolved configs |
| `soup.py`, `launch/soup.sh` | the model soup of runs 1–4 |
| `export_hf/` | conversion to a `transformers` repo, parity check, end-to-end check |
| `third_party/nanochat-92d63d4/` | the exact upstream model code every run loaded, plus `nanochat-92d63d4-ve_every.patch` (value embeddings on every k-th layer, which the 0.6B backbone needs; with the default k = 2 the model is upstream's, so the paper's rows are unaffected) |

`launch/train.sh` reads the engine from the config's `# Engine:` line. All engines share `src/` for data, the
packing plan, the LCE loss, the LR schedule and the FP32 guard. They differ only where the paper's rows were
produced by different execution code.

**Backbones.** A nanochat backbone (`backbone_impl: upstream`) is a checkpoint directory: `model_<step>.pt`,
`meta_<step>.json` and the tokenizer. That is a released model in `models/`, or a training run's directory as
`pretrain/continue/` writes it. A run's directory holds its last few steps; the newest is used unless the config sets
`model_step`. Relative paths in a config are relative to this folder, and `eval/rerank_eval.py` resolves them the
same way.

**The 0.6B reranker.** Run 1's recipe on the 0.6B Gaggle backbone, on one H200. The config pins the original run's
per-GPU token budget and packing, so `../tests/check_plans.py` rebuilds its training plan exactly (1,928 steps; the
plan hash is recorded there).

```bash
bash data_prep/fetch_models.sh gaggle-0.6b                  # backbone -> models/ (step 229,067), checksummed
bash data_prep/prepare_data.sh rlhn250k                     # the same training file as run 1
python ../tests/check_plans.py configs/experiments/gaggle06b_lce.yaml
bash launch/train.sh configs/experiments/gaggle06b_lce.yaml 1   # 1x H200, ~10 h
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/gaggle06b_lce/ckpt_final.pt gaggle06b_lce
```

| Config | Released as | Backbone | TREC DL | BEIR |
|---|---|---|---:|---:|
| `gaggle06b_lce.yaml` | [`castorini/gaggle-reranker-0.6b-20261122`](https://huggingface.co/castorini/gaggle-reranker-0.6b-20261122) | [`gaggle-nanochat-0.6b-pretrained-climbmix-20261121`](https://huggingface.co/castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121) | 0.634 | 0.533 |

The numbers are mean nDCG@10 over TREC DL19–23 and the seven BEIR sets, BM25 top-100, as in the paper. On another
GPU, or with another backbone of the same architecture (e.g. a continued-pretraining run's dir from
`../pretrain/continue/`), drop the pinning with `--override pack_world=<n_gpus> require_exact_budget=false`: that is a
new run, and its plan differs.

A [Modal reproduction](docs/onboarding-gaggle06b-sft.md) trains the 0.6B model on RLHN-250K and compares its final checkpoint with the released reranker on DL19.
