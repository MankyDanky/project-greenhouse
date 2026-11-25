# Third-party code

| Path | Source | Licence |
|---|---|---|
| `reranker/third_party/nanochat-92d63d4/` | [karpathy/nanochat](https://github.com/karpathy/nanochat) at `92d63d4` (model code the rerankers load), with the 15-line value-embedding spacing patch next to it (`nanochat-92d63d4-ve_every.patch`; no change for the 3.3B model) | MIT (`LICENSE` inside) |
| `pretrain/nanochat/` | nanochat `92d63d4` with the changes listed in `pretrain/README.md` (batch schedule, value-embedding spacing, auto-resume, the 3.3B run's stage-boundary patch, a shard-index hook) | MIT (`LICENSE` inside) |
| `pretrain/d34_original_run/` | patches to nanochat `92d63d4`, which its `setup.sh` clones and patches to rebuild the 3.3B run's exact code (not vendored) | MIT |
| `posttrain/nanochat_overlay/` | midtraining and SFT scripts and tasks from nanochat `348fbb3`, ported to `92d63d4`; our changes are marked `(gaggle)` | MIT |
| `pretrain/mntp/upstream_snapshot_bidir/` | nanochat `92d63d4` with bidirectional sliding-window attention | MIT (`LICENSE` inside) |
| `pretrain/continue/nanochat_continue.patch` | a patch to nanochat `92d63d4`, which `pretrain/continue/setup.sh` clones (not vendored) | MIT |
| `eval/baselines/listwise/gemma_non_thinking_template.yaml` | RankLLM's `qwen_non_thinking_template.yaml` at `443adf3`, without its Qwen-only `/no_think` line | Apache-2.0 |

Not vendored, fetched at run time: Pyserini and its prebuilt indexes, RankLLM, vLLM, `trec_eval`, the Hugging Face
models and datasets named in the configs. Each comes under its own licence. The Robust04 and TREC-News collections
are licensed by NIST and are never stored in this repository.
