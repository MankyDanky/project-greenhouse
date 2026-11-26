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

Replace `your-modal-profile` with the name shown by `profile list`. Run the commands below from the repo root.

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

Fetch the BM25 top-100 candidates, check their digests and qrels hashes against the repo, then score all five TREC DL sets with both models:

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

The TREC DL scores are:

| Collection | Queries | Published nDCG@10 | Reproduced nDCG@10 | Same top result |
|---|---:|---:|---:|---:|
| DL19 | 43 | 0.7601 | 0.7581 | 41/43 |
| DL20 | 54 | 0.7065 | 0.7083 | 49/54 |
| DL21 | 53 | 0.7059 | 0.7020 | 44/53 |
| DL22 | 76 | 0.5233 | 0.5232 | 70/76 |
| DL23 | 82 | 0.4706 | 0.4688 | 75/82 |
| Five-set mean | | 0.6333 | 0.6321 | 279/308 |

The mean is unweighted across the five sets. The 0.0012 gap is within the paper's four-run DL mean standard deviation of 0.0014, measured with the larger 3.3B backbone. The raw scores and full rankings do not match exactly. BEIR was not evaluated.

+ Results reproduced by [@MankyDanky](https://github.com/MankyDanky) on 2026-11-26 using a Modal H200.
