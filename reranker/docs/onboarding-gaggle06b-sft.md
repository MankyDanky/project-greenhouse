# Reproducing Gaggle 0.6B SFT on Modal
## Setup

Clone the repo and configure a [Modal](https://modal.com/docs/guide/getting-started) profile for your own workspace:

```bash
git clone https://github.com/castorini/project-greenhouse.git
cd project-greenhouse
python -m pip install modal
modal setup
modal profile list
export MODAL_PROFILE=your-modal-profile
```

Replace `your-modal-profile` with the name shown by `profile list`. Run the commands below from the repo root. The scripts pin the training code to commit `9c878928`, the [0.6B backbone](https://huggingface.co/castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121) to revision `c3c0d9ff`, and the [released reranker](https://huggingface.co/castorini/gaggle-reranker-0.6b-20261122) to revision `1b53dcf2` for comparison. The backbone and prepared RLHN-250K file are SHA256-checked before training.

## Train the reranker

Stage the backbone and RLHN-250K data, check the training inputs, then deploy the trainer:

```bash
modal run reranker/modal_gaggle06b_sft.py --stage stage
modal run reranker/modal_gaggle06b_sft.py --stage preflight
modal deploy reranker/modal_gaggle06b_sft.py
python -c 'import modal; call = modal.Function.from_name("greenhouse-gaggle06b-rlhn-sft", "train").spawn(); print(call.object_id)'
```

The last command submits a detached job. It runs `configs/experiments/gaggle06b_lce.yaml` from the pinned commit, changing only the released backbone path and output directory. The run uses one H200, the 120,000-token budget, and 1,928 updates. It saves a resumable checkpoint every 200 updates in the `greenhouse-gaggle06b-sft` Modal Volume. If the job is interrupted, submit the detached call again; `resume=auto` picks up the latest checkpoint. The completed checkpoint is `gaggle06b_lce_rlhn250k_20261125/ckpt_final.pt`. To start another run in the same workspace, change `RUN_NAME` in both Modal scripts first.

## Compare the rerankers

Build the repo's BM25 top-100 candidates, check their published digest and the DL19 qrels hash, then score the same 43 queries with both models:

```bash
modal run reranker/modal_gaggle06b_compare.py --stage prepare
modal run reranker/modal_gaggle06b_compare.py --stage compare
modal volume get greenhouse-gaggle06b-sft \
  gaggle06b_lce_rlhn250k_20261125/comparison/full/summary.json \
  /tmp/gaggle06b-comparison.json
```

The comparison uses the final step-1,928 checkpoint and the published Hugging Face weights, with the same candidates, BF16 scoring, and `eval/rerank_eval.py` for both. The candidate text stays in the Modal Volume. The JSON summary and TREC run files are saved under `comparison/full/`.

## Results

On the [model card's anemia example](https://huggingface.co/castorini/gaggle-reranker-0.6b-20261122), both models rank the three passages in the same order. These are the scores measured on the Modal H200:

| Passage | Published score | Reproduced score |
|---|---:|---:|
| Anemia symptoms | 15.761 | 17.654 |
| Dietary iron | 7.888 | 9.614 |
| Iron as a metal | 6.110 | 8.017 |

The DL19 scores also agree closely:

| Collection | Queries | Published nDCG@10 | Reproduced nDCG@10 | Same top result |
|---|---:|---:|---:|---:|
| DL19 | 43 | 0.7602 | 0.7581 | 42/43 |

Mean top-10 overlap was 0.9628 and mean Spearman rank correlation was 0.9964. The raw scores and full rankings do not match exactly. This checks DL19, not the published five-set TREC DL or seven-set BEIR means.

+ Results reproduced by [@MankyDanky](https://github.com/MankyDanky) on 2026-11-26 using a Modal H200. The training job resumed from its step-600 checkpoint after an interruption and finished at step 1,928. Modal reported $54.94 across all listed apps in the personal workspace on November 25–26, below the $100 budget.
