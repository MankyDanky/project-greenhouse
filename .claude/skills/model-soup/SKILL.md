---
name: model-soup
description: Build the released Gaggle reranker (gaggle-reranker-20261005) as an equal-weight parameter average of fine-tuning runs 1-4, export it to Hugging Face transformers format and verify it. Use when asked about the soup, the released checkpoint, averaging runs, or exporting a reranker.
---

# Build and export the soup

The released reranker averages the parameters of the four Sec. 5 runs, 0.25 each. It is not an average of their
scores.

## 1. The four runs (from `reranker/`)

| Run | Config | Engine | Paper GPUs |
|---|---|---|---|
| 1 | `configs/sft/run1_seed17.yaml` | nanochat | 2× 96 GB |
| 2 | `configs/sft/run2_seed18.yaml` | nanochat | 2× 96 GB |
| 3 | `configs/sft/run3_seed17_exact_budget.yaml` | nanochat | 2× 96 GB |
| 4 | `configs/sft/run4_seed17_gradcache.yaml` | multi | 2× H100, GradCache |

Train each with `bash launch/train.sh <config> 2` (see the `reproduce-paper-row` skill). Runs 1, 3 and 4 share
seed 17 and the same plan; run 2 is seed 18.

## 2. Average

```bash
bash launch/soup.sh                  # -> checkpoints/sft_soup/ckpt_final.pt
```

`soup.py` refuses checkpoints that differ in keys, shapes, prompt, readout or backbone. If it refuses, a run used
a different config. Find out why; do not force it.

## 3. Evaluate and compare

```bash
cd .. && bash eval/run_eval.sh ours reranker/checkpoints/sft_soup/ckpt_final.pt my-soup
python paper/check_results.py --results eval/results --tag sft.soup=my-soup --tolerance 0.005
```

Paper: DL 0.641, BEIR 0.547. The soup's BEIR mean beats every single run; its DL mean matches the best run.

## 4. Export to transformers

```bash
bash reranker/export_hf/export.sh reranker/checkpoints/sft_soup/ckpt_final.pt exports/my-gaggle-soup
```

The script:
1. writes sharded safetensors, the config, the tokenizer, `modeling_gaggle.py` and `SHA256SUMS`;
2. checks the HF model scores like the checkpoint (`parity_check.py`);
3. evaluates DL19, NFCorpus and SciFact end to end with the `gaggle_hf` scorer.

Uploading to the Hub is a separate, deliberate step. Confirm the repository name and visibility with the user first.
