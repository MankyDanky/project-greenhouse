---
license: mit
language: [en]
library_name: transformers
pipeline_tag: text-generation
datasets: [karpathy/climbmix-400b-shuffle]
tags: [nanochat, pretraining, climbmix, d34, base-model, custom_code, vllm, gaggle]
---

# gaggle-nanochat-pretrained-climbmix-20260924

A 3.3B-parameter decoder-only base language model, pre-trained from scratch with
[nanochat](https://github.com/karpathy/nanochat) on one pass over ClimbMix (287B tokens).
This is the causal model described in the pre-training section of
[Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search](https://arxiv.org/abs/2610.11922)
([PDF](https://arxiv.org/pdf/2610.11922)). It has had no
post-training (no instruction tuning or RLHF), so it continues text rather than following instructions.

The same weights load three ways: with `transformers` (`trust_remote_code=True`), with vLLM (through a small
plugin), and with nanochat's own code from the original checkpoint files, which are unchanged.

## Model

| | |
|---|---|
| Architecture | nanochat depth-34 decoder-only transformer, unmodified: 34 layers, width 2,176, 17 attention heads of dimension 128, 2,048-token context |
| Components | rotary position embeddings, QK normalization, parameter-free RMSNorm, squared-ReLU MLP, untied input/output embeddings, logit soft-capping; a value embedding on every other layer; 3 of every 4 layers use a 512-token sliding window |
| Parameters | 3.29B: 1.93B in the transformer blocks, 1.21B in the value-embedding tables, 0.14B in the input and output embeddings. Value embeddings are lookups, so per-token compute is that of a dense model of about 2B parameters. |
| Tokenizer | byte-level BPE, 32,768 entries, trained on the first 2B characters of the training split |
| Code | [`karpathy/nanochat`](https://github.com/karpathy/nanochat) at commit `92d63d4` |

## Training

- **Data:** ClimbMix ([`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)),
  one pass of 287B tokens. The last of its 6,543 shards is held out for validation. Documents are packed into
  2,048-token sequences with nanochat's BOS-aligned best-fit packing.
- **Optimizer and schedule:** nanochat's, unchanged. Muon updates the weight matrices of the transformer blocks and
  AdamW the embeddings and scalar parameters; the learning rate warms up briefly, stays constant, and then decays linearly.
- **Batch size:** our only change. The batch grows from 262,144 to 2,097,152 tokens over training. Unlike nanochat,
  which scales its learning rates with the square root of the batch size, we keep their peak values unchanged as
  the batch grows.
- **Compute:** one node of eight NVIDIA H100 GPUs with FP8 matrix multiplications, about 1,600 H100-hours.
- **This checkpoint:** step 227,158, after 283B tokens. It is the checkpoint with the highest CORE measured during
  training; the final checkpoint (287B tokens) scored 0.004 lower.

## Evaluation

Base-model benchmarks that need no instruction following:
CORE (nanochat's average centered accuracy over 22 in-context learning tasks, 500 examples each), and 5-shot
MMLU and MMLU-Pro scored by the log-likelihood of the answer letter with the LM Evaluation Harness.

| Model | Tokens | CORE | MMLU | MMLU-Pro |
|---|---:|---:|---:|---:|
| **This model** | 283B | **0.458** | **0.493** | **0.178** |
| [nanochat d34](https://huggingface.co/karpathy/nanochat-d34) | 89B | 0.338 | 0.257 | 0.115 |

All numbers are from our evaluation except nanochat d34's CORE, which is the published number.

Validation loss on the held-out ClimbMix shard is 0.596 bits per byte.

## Usage

Sequences must start with the BOS token `<|bos|>`, as in training. The `transformers` tokenizer adds it for you
(`add_special_tokens=True`, the default), and so does vLLM.

### transformers

```bash
pip install "transformers>=4.57" "torch>=2.5"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "castorini/gaggle-nanochat-pretrained-climbmix-20260924"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")   # starts with <|bos|>
out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Output on one H200 GPU:

```text
The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France
```

Log-likelihood of a continuation, as in the nanochat example below:

```python
@torch.inference_mode()
def continuation_logprob(context, continuation):
    ctx = tokenizer(context)["input_ids"]                                    # starts with <|bos|>
    cont = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ctx + cont], device="cuda")
    logp = torch.log_softmax(model(ids).logits[0, len(ctx) - 1:-1], -1)
    return logp[torch.arange(len(cont)), torch.tensor(cont, device="cuda")].sum().item()

print(continuation_logprob("The capital of France is", " Paris"))    # -0.9538
print(continuation_logprob("The capital of France is", " Berlin"))   # -7.7336
```

These can differ from the nanochat example's printout in the last digits: nanochat runs attention with Flash
Attention 3 on Hopper GPUs when that kernel is available, while this port always uses PyTorch's
`scaled_dot_product_attention`, which is nanochat's own fallback and the kernel all our evaluations ran with. With the
same kernel the two produce bit-identical logits (see below); across kernels, bf16 rounding moves log-probabilities
slightly.

`trust_remote_code=True` is needed because the architecture is nanochat's rather than one `transformers` ships;
`configuration_gaggle.py` and `modeling_gaggle.py` in this repo define it. They are plain PyTorch, import nothing
beyond `torch` and `transformers`, and are short enough to read first. No nanochat checkout and no Flash Attention
build are involved. `model(input_ids).logits` are fp32 and already soft-capped, as nanochat returns them; passing
`labels=input_ids` gives the usual next-token loss. Batches may be padded on either side.

With the LM Evaluation Harness, pass `add_bos_token=True` (it does not add special tokens by default):

```bash
lm_eval --model hf --model_args pretrained=castorini/gaggle-nanochat-pretrained-climbmix-20260924,trust_remote_code=True,dtype=bfloat16,add_bos_token=True --tasks mmlu --num_fewshot 5
```

### vLLM

vLLM has no built-in class for this architecture, so install the small plugin that provides one; vLLM then finds
it automatically:

```bash
pip install vllm https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-20260924/resolve/main/vllm_gaggle-0.1.0-py3-none-any.whl
vllm serve castorini/gaggle-nanochat-pretrained-climbmix-20260924 --trust-remote-code
```

```python
from vllm import LLM, SamplingParams

llm = LLM("castorini/gaggle-nanochat-pretrained-climbmix-20260924", trust_remote_code=True)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=20, temperature=0))
print(out[0].outputs[0].text)
```

Output on one H200 GPU:

```text
 Paris. The capital of France is Paris. The capital of France is Paris. The capital of France
```

The plugin (`vllm_plugin/` in this repo; one model file of about 230 lines) registers the architecture through
vLLM's `vllm.general_plugins` entry point, so every vLLM process finds it; nothing in vLLM itself is changed. It
reads the same `config.json` and safetensors as `transformers`. Every nanochat component maps onto ordinary tensor
operations except the embedding *smear*, which mixes the previous token's embedding into each position: while
decoding, vLLM gives the model only the newest token, so the port recovers the previous embedding from the cache
with one extra attention layer whose two-token window is set up to select the previous position exactly. It runs
on one GPU (no tensor parallelism) with the model's 2,048-token context, with or without CUDA graphs and
`torch.compile`. On one H200, 256 prompts x 256 new tokens run at about 13,000 tokens/s, three times
`transformers` `generate()` on the same batch.

### nanochat

The original training checkpoint is still here, unchanged (`model_227158.pt`, `meta_227158.json`, `tokenizer/`),
so nanochat's own code loads it exactly as before. Set up nanochat at the commit the model was trained with:

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat && git checkout 92d63d4
uv sync --extra gpu        # use --extra cpu on a machine without a CUDA GPU
```

Save the following as `example.py` in the nanochat repository root (nanochat is imported from there, not
installed as a package) and run `uv run --extra gpu --with huggingface_hub python example.py`.

```python
import os, torch
from huggingface_hub import snapshot_download

# only the nanochat files; the safetensors are for transformers and vLLM
path = snapshot_download("castorini/gaggle-nanochat-pretrained-climbmix-20260924",
                         allow_patterns=["model_227158.pt", "meta_227158.json", "tokenizer/*"])
os.environ["NANOCHAT_BASE_DIR"] = path   # nanochat loads the tokenizer from $NANOCHAT_BASE_DIR/tokenizer

from nanochat.checkpoint_manager import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, meta = build_model(path, step=227158, device=device, phase="eval")

# Greedy continuation
prompt = "The capital of France is"
ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
new = list(model.generate(ids, max_tokens=20, temperature=0.0))
print(prompt + tokenizer.decode(new))

# Log-likelihood of a continuation given a context
@torch.inference_mode()
def continuation_logprob(context, continuation):
    ctx = tokenizer.encode(context, prepend=tokenizer.get_bos_token_id())
    cont = tokenizer.encode(continuation)
    x = torch.tensor([ctx + cont[:-1]], device=device)
    y = torch.tensor([[-1] * (len(ctx) - 1) + cont], device=device)   # -1 = ignore context positions
    return -model(x, targets=y, loss_reduction="sum").item()

print(continuation_logprob("The capital of France is", " Paris"))
print(continuation_logprob("The capital of France is", " Berlin"))
```

Output on one H200 GPU:

```text
The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France
-0.9807320833206177
-7.733334541320801
```

Notes:

- On GPUs with bfloat16 support the model computes in bfloat16; on CPU and MPS it uses float32. Override with
  `NANOCHAT_DTYPE=bfloat16|float16|float32`.
- For further pre-training or fine-tuning, use nanochat's training scripts (`scripts/base_train.py`,
  `scripts/chat_sft.py`) with this checkpoint as the starting point.

## The three implementations agree

All three load the same tensors. We checked them against nanochat's own forward pass on the original checkpoint,
which is what every number in the Evaluation section was computed with.

| Check | Result |
|---|---|
| `transformers` vs nanochat, bf16 on an H200: logits of real text at 2 to 2,048 tokens | **bitwise identical** |
| same, a right-padded batch vs nanochat on the same batch | **bitwise identical** |
| same, fp32 on CPU | **bitwise identical** |
| loss on a 2,048-token sequence | identical (2.2183) |
| greedy generation, 5 prompts x 48 tokens, with and without the KV cache | identical token for token |
| tokenizer vs nanochat's, 400 documents | identical ids, identical decoding |
| `transformers` 4.57 vs 5.17 | bitwise identical logits |
| CORE / MMLU / MMLU-Pro run through `transformers` | 0.458 / 0.493 / 0.178, as in the table above |
| MMLU with the stock LM Evaluation Harness (no project code) | 0.493 |
| vLLM vs `transformers`, fp32: per-token log-probabilities | mean difference 0.0009 nats; all 5 greedy generations identical |
| vLLM vs `transformers`, bf16 | mean difference 0.016 nats, the same size as `transformers`' own bf16 rounding between a single sequence and a padded batch (0.013-0.026) |

In bf16, any change of kernel shape — batching, padding, a KV cache, a different attention kernel — moves logits by
rounding, which can flip near-tied greedy choices after a few dozen tokens; nanochat itself behaves the same way.
[`convert_nanochat_to_hf.py`](convert_nanochat_to_hf.py) is the converter that produced the safetensors from
`model_227158.pt`.

## Files

- `model-0000{1,2,3}-of-00003.safetensors`, `model.safetensors.index.json`: the weights for `transformers` and vLLM.
  These are the tensors of `model_227158.pt`, unchanged in value and dtype (fp32 matrices, bf16 embedding tables),
  renamed to `model.<nanochat name>` (`lm_head.weight` keeps its name).
- `config.json`, `generation_config.json`, `configuration_gaggle.py`, `modeling_gaggle.py`: the architecture for
  `transformers` (and the configuration vLLM reads).
- `tokenizer.json`, `tokenizer_config.json`: the tokenizer, converted from `tokenizer/tokenizer.pkl`; it encodes
  every text exactly as nanochat does and prepends `<|bos|>`.
- `model_227158.pt`: model weights (nanochat state dict), for nanochat.
- `meta_227158.json`: model configuration, training configuration, and validation loss.
- `tokenizer/`: the 32,768-entry tokenizer in nanochat's format (`tokenizer.pkl`, `token_bytes.pt`).
- `SHA256SUMS`: checksums of the weight and metadata files.

Optimizer state is not included.

## Related

- [`castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929):
  the bidirectional variant, converted from this run's causal checkpoint at 245B tokens and trained on the remaining
  42B tokens with bidirectional attention and masked next-token prediction, following LLM2Vec.
- [`castorini/gaggle-reranker-20261005`](https://huggingface.co/castorini/gaggle-reranker-20261005): the pointwise
  reranker fine-tuned from this checkpoint.

## Citation

If you use this checkpoint, please cite our paper:

```bibtex
@misc{lin2026projectgreenhouseprogressfully,
  title={{Project Greenhouse}: Progress Toward Fully Open and Sovereign Agentic Search},
  author={Jimmy Lin and Sahel Sharifymoghaddam and Lingwei Gu and Nour Jedidi},
  year={2026},
  eprint={2610.11922},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2610.11922}
}
```
