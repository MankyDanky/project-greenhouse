Title: Project Greenhouse and the first Gaggle reranker
Summary: The first paper traces a pointwise reranker from public pretraining data to released weights.
Model: 3.3b

The [first Project Greenhouse paper](https://arxiv.org/abs/2610.11922) describes how we built Gaggle, a pointwise reranker trained end to end from public data. The 3.29B causal backbone was pretrained from scratch with nanochat on 283B tokens of ClimbMix.

## Supervised fine-tuning

Four reranking runs fine-tuned the backbone on RLHN-250K using localized contrastive estimation. The headline model averages those runs’ weights into a single checkpoint. Reranking BM25’s top 100 candidates, it scored **0.641** mean nDCG@10 on TREC DL 2019–2023 and **0.547** across seven BEIR collections.

The [reproduction guide](https://github.com/castorini/project-greenhouse/blob/main/REPRODUCE.md) maps the paper’s results to training configurations, evaluation commands, and expected scores.

[Compare all three released backbones and rerankers](../gaggle-06b/#pretrained-model-comparison).
