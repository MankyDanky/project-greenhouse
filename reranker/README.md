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
| `src/` | the shared core and the nanochat engine: data and packing (`dataset_nc.py`, `stepping.py`), LCE / self-KD (`losses_nc.py`), prompts (`prompt_builder_nc.py`), upstream nanochat model and readouts (`modeling_nanochat_upstream.py`, `bidirectional.py`), trainer (`train_nc.py`) |
| `engines/multi/` | any backbone (nanochat, Qwen, Gemma, MiniCPM) on run 1's frozen plan, with chunked / GradCache execution: run 4 and the MiniCPM / low-LR Gemma rows |
| `engines/gemma4/`, `engines/qwen35/` | the native Gemma-4-E2B and Qwen3.5-4B-Base trainers of Sec. 6.1 |
| `engines/ce/` | pointwise cross-entropy over the full vocabulary (Sec. 6.5) |
| `data_prep/` | model download, data preparation and checksums, self-filtering |
| `plans/` | records of the multi-engine runs' frozen plans and resolved configs |
| `soup.py`, `launch/soup.sh` | the model soup of runs 1–4 |
| `export_hf/` | conversion to a `transformers` repo, parity check, end-to-end check |
| `third_party/nanochat-92d63d4/` | the exact upstream model code every run loaded |

`launch/train.sh` reads the engine from the config's `# Engine:` line. All engines share `src/` for data, the
packing plan, the LCE loss, the LR schedule and the FP32 guard. They differ only where the paper's rows were
produced by different execution code.
