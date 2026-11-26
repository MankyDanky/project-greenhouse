# Project Greenhouse

Project Greenhouse builds **Gaggle**, a family of fully open models for search and agentic search, trained from
scratch. Every stage is public:
- the pre-training data;
- the pre-training and fine-tuning code;
- the configurations and checkpoints.

No third-party open-weight backbone is involved. This repository holds the code, configurations and checks behind
each release.

The project website source, release notes, and publishing instructions are in [website/](website/).

## Papers

**[Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search](https://arxiv.org/abs/2610.11922)**
(Lin, Sharifymoghaddam, Gu, Jedidi; 2026).

The first milestone is the Gaggle base reranker, a 3B pointwise reranker built end to end from public data:
1. a decoder pre-trained from scratch on ClimbMix with [nanochat](https://github.com/karpathy/nanochat);
2. one epoch of supervised fine-tuning on RLHN-250K with localized contrastive estimation (LCE).

Reranking BM25 top-100, it reaches nDCG@10 0.641 on TREC DL 2019–2023 and 0.547 on seven BEIR collections. Everything
the paper reports from our own runs is reproducible from this repository; [REPRODUCE.md](REPRODUCE.md) maps every
table to its configs, commands and expected scores.

```bibtex
@article{lin2026greenhouse,
  title   = {Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search},
  author  = {Jimmy Lin and Sahel Sharifymoghaddam and Lingwei Gu and Nour Jedidi},
  journal = {arXiv preprint arXiv:2610.11922},
  year    = {2026}
}
```

## Models

| Model | What it is | Paper section |
|---|---|---|
| [`castorini/gaggle-nanochat-pretrained-climbmix-20260924`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924) | 3.29B causal LM, 283B tokens of ClimbMix | Sec. 4 |
| [`castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929) | the same model adapted to bidirectional attention (MNTP) | Sec. 4, 6.3 |
| [`castorini/gaggle-reranker-20261005`](https://huggingface.co/castorini/gaggle-reranker-20261005) | **Gaggle**: the soup of runs 1–4, used for the headline results | Sec. 3, 5 |
| [`castorini/gaggle-reranker-run1-20260926`](https://huggingface.co/castorini/gaggle-reranker-run1-20260926) | baseline run 1 (seed 17), the reference row of every Sec. 6 table | Sec. 5, 6 |
| [`castorini/gaggle-reranker-run2-20261004`](https://huggingface.co/castorini/gaggle-reranker-run2-20261004) | baseline run 2 (seed 18) | Sec. 5 |
| [`castorini/gaggle-reranker-run3-20261002`](https://huggingface.co/castorini/gaggle-reranker-run3-20261002) | baseline run 3 (seed 17, independent reproduction) | Sec. 5 |
| [`castorini/gaggle-reranker-run4-20261005`](https://huggingface.co/castorini/gaggle-reranker-run4-20261005) | baseline run 4 (seed 17, H100s with GradCache) | Sec. 5 |

Scoring with the released reranker needs only `transformers`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name = "castorini/gaggle-reranker-20261005"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True).eval()
print(model.compute_score([("what is a greenhouse gas", "Carbon dioxide traps heat in the atmosphere."),
                           ("what is a greenhouse gas", "Greenhouses are buildings for growing plants.")],
                          tokenizer=tok))
```

## What is where

| Path | Paper | Contents |
|---|---|---|
| [`pretrain/`](pretrain/) | Sec. 4 | ClimbMix pre-training of the d34 backbone, stage by stage (tokenizer, batch ramp, 2M-token production) |
| [`pretrain/mntp/`](pretrain/mntp/) | Sec. 4 | the bidirectional MNTP continuation |
| [`pretrain/continue/`](pretrain/continue/) | – | continued pre-training from a released checkpoint on another data mix (used by `benchmaxxing/`) |
| [`pretrain_eval/`](pretrain_eval/) | Sec. 4 | CORE / MMLU / MMLU-Pro harness for the base-model comparison |
| [`reranker/`](reranker/) | Sec. 5, 6 | supervised fine-tuning: one package, one config per paper row, data preparation, model soup, HF export |
| [`eval/`](eval/) | Sec. 3, 5, 6 | BM25 candidates, the reranking evaluator, BM25 / oracle rows, the fine-tuned and listwise baselines |
| [`diagnostics/`](diagnostics/) | Sec. 6.1, 6.3 | attention-blocking test of the MNTP rerankers, backbone weight scale |
| [`paper/`](paper/) | all | `manifest.yaml` (every reproduced row, its config and its printed scores) and `check_results.py` |
| [`benchmaxxing/`](benchmaxxing/) | – | after the paper: adapting Gaggle models to the evaluation collections, e.g. continued pretraining on their corpora ([plan](benchmaxxing/cpt/PLAN.md)) |
| [`.claude/skills/`](.claude/skills/) | – | task guides for coding agents (reproduce a row, evaluate, prepare data, pre-train, build the soup) |

## Getting started

- **[ONBOARDING.md](ONBOARDING.md)**: set up, evaluate the released reranker, train your first row, add an experiment.
- **[REPRODUCE.md](REPRODUCE.md)**: every table and figure of the paper, with the command, hardware and expected
  scores for each row.

The fastest end-to-end check, on one GPU, after `pip install -r reranker/requirements.txt`:

```bash
bash eval/fetch_qrels.sh                                    # 12 qrels files, checksum-verified
bash eval/candidates/build_candidates.sh dl19               # BM25 top-100 (pyserini), verified against the paper's inputs
bash eval/run_eval.sh gaggle_hf castorini/gaggle-reranker-20261005 gaggle dl19
python eval/report.py gaggle                                # DL19 nDCG@10 ≈ 0.757 (±0.002 across GPUs)
```

## What the first release covers

**Included** (arXiv:2610.11922):
- every row of the paper that we trained or evaluated ourselves:
  - pre-training and the MNTP variant;
  - the four fine-tuning runs and the soup;
  - all contrastive conditions of Sec. 6;
  - BM25, the oracle, Gemma-4 and Qwen3.8 listwise, Jina-Reranker-v3, MonoT5 and Qwen3-Reranker.

**Not included:**
- the GPT-6.1 Sol, Jev and RLHN Qwen2.5-3B rows;
- rows the paper copies from prior work (RankVicuna, RankZephyr, FirstMistral, Rank0, Rank1);
- in the base-model comparison: the MNTP model's masked CORE and the external models other than OPEN-1B (see
  `pretrain_eval/README.md`).

**Data licences:** Robust04 and TREC-News documents may not be redistributed, so the repository ships only doc-ID
rankings and fingerprints of the reranker inputs. You rebuild the text from Pyserini's prebuilt indexes; see
[`eval/candidates/`](eval/candidates/).

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE)); vendored third-party code keeps its own licence ([THIRD_PARTY.md](THIRD_PARTY.md)).
