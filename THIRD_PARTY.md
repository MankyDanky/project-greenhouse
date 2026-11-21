# Third-party code

| Path | Source | Licence |
|---|---|---|
| `reranker/third_party/nanochat-92d63d4/` | [karpathy/nanochat](https://github.com/karpathy/nanochat) at `92d63d4`, unmodified (model code the rerankers load) | MIT (`LICENSE` inside) |
| `pretrain/nanochat/` | nanochat `92d63d4` with the stage-boundary checkpoint patch and a shard-index hook, described in `pretrain/README.md` | MIT (`LICENSE` inside) |
| `pretrain/mntp/upstream_snapshot_bidir/` | nanochat `92d63d4` with bidirectional sliding-window attention | MIT (`LICENSE` inside) |
| `pretrain/continue/nanochat_continue.patch` | a patch to nanochat `92d63d4`, which `pretrain/continue/setup.sh` clones (not vendored) | MIT |
| `eval/baselines/listwise/gemma_non_thinking_template.yaml` | RankLLM's `qwen_non_thinking_template.yaml` at `443adf3`, without its Qwen-only `/no_think` line | Apache-2.0 |

Not vendored, fetched at run time: Pyserini and its prebuilt indexes, RankLLM, vLLM, `trec_eval`, the Hugging Face
models and datasets named in the configs. Each comes under its own licence. The Robust04 and TREC-News collections
are licensed by NIST and are never stored in this repository.
