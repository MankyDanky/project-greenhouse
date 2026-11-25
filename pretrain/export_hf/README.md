# Release: nanochat checkpoint -> `transformers` + vLLM

How the best-CORE stage-3 checkpoint became
[`castorini/gaggle-nanochat-pretrained-climbmix-20260924`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924).
The repo keeps nanochat's own files (`model_227158.pt`, `meta_227158.json`, `tokenizer/`) unchanged; this folder adds a
`transformers` port and a vLLM plugin that read the same weights. [`MODEL_CARD.md`](MODEL_CARD.md) is the published card.

| File | What |
|---|---|
| `configuration_gaggle.py`, `modeling_gaggle.py` | self-contained `trust_remote_code` model (torch + transformers only; SDPA attention) |
| `convert_nanochat_to_hf.py` | renames the tensors (`k` -> `model.k`; values and dtypes unchanged) into safetensors, writes `config.json` and `tokenizer.json` |
| `parity_check.py` | transformers port vs nanochat `92d63d4` on the original `.pt`: logits at 2-2,048 tokens, loss, greedy generations |
| `vllm_plugin/`, `vllm_check.py` | the vLLM model class (`pip install ./vllm_plugin`) and its check against the transformers port |
| `hf_baselines.py`, `readme_examples.py` | yardsticks for the vLLM comparison; the model card's usage snippets |
| `upload_to_hub.py` | adds the port's files to the existing repo in one commit; it pins the released `model_227158.pt` SHA256 |

The commands use the variables of `../d34_original_run/config.sh`, the run that produced the 3.3B checkpoint.

```bash
B=$D34_ROOT/production/best_core/step_227158            # your best-CORE step; best.json names it
cp -r $D34_BASE_DIR/tokenizer $B/                        # the converter reads <ckpt>/tokenizer/tokenizer.pkl
python convert_nanochat_to_hf.py --ckpt $B --step 227158 --out hf_227158
# tokenizer.json needs transformers' tiktoken converter: transformers==4.57.6 tiktoken blobfile tokenizers
python convert_nanochat_to_hf.py --ckpt $B --step 227158 --out hf_227158 --tokenizer-only
E=$D34_BASE_DIR/eval_bundle/eval_data
TX="$E/reading_comprehension/coqa.jsonl $E/reading_comprehension/squad.jsonl $E/reading_comprehension/narrative_qa.jsonl $E/world_knowledge/arc_easy.jsonl $E/programming/human_eval.jsonl"
python parity_check.py --nanochat-src $D34_EXP_DIR/upstream_snapshot --ckpt $B --step 227158 --hf hf_227158 --texts $TX --out parity.json
pip install ./vllm_plugin && python vllm_check.py --hf hf_227158 --parity parity.json --texts $TX --out vllm.json
```

Results for the released checkpoint: transformers and nanochat give bitwise-identical logits (bf16 on H200, fp32 on CPU)
and the same CORE / MMLU / MMLU-Pro (0.458 / 0.493 / 0.178); vLLM's prompt log-probabilities differ from transformers'
by 0.0009 on average in fp32 (0.016 in bf16, the size of bf16 batching noise).

## The 0.9B and 0.6B models

The same files converted
[`castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105`](https://huggingface.co/castorini/gaggle-nanochat-0.9b-pretrained-climbmix-20261105)
and
[`castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121`](https://huggingface.co/castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121)
(final step 229,067 of runs trained with [`../README.md`](../README.md)). Their value embeddings are on every 4th and
every 7th layer instead of alternate layers: the converter reads that spacing (`ve_every`) from the checkpoint's meta and
writes it to `config.json`, where a missing value means alternate layers, so the 3.3B model's config is unchanged. It
also refuses `model_config` fields the port does not model and checks the `<|bos|>` and `<|assistant_end|>` ids against
the tokenizer. For the parity check of these two, pass `--nanochat-src ../nanochat`, the trainer, which reads
`ve_every`. Both released repos ship these `configuration_gaggle.py`, `modeling_gaggle.py` and
`convert_nanochat_to_hf.py` byte for byte. Their logits are bitwise identical to nanochat's in bf16 on GPU, and in
fp32 at every tested length for the 0.6B model and at 5 of 8 for the 0.9B (each model card has the table).
