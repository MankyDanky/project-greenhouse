#!/usr/bin/env python
"""Convert the nanochat checkpoint (model_<step>.pt + meta_<step>.json + tokenizer/) into a `transformers` repo.

The tensors are copied unchanged -- same values, same dtypes (fp32 matrices, bf16 embeddings), only renamed:
nanochat key `k` becomes `model.k`, and `lm_head.weight` stays `lm_head.weight`. So `to_nanochat` below is an
exact inverse, and the original .pt can be rebuilt bit for bit.

    python convert_nanochat_to_hf.py --ckpt DIR --step 227158 --out OUT
Every shape field of the checkpoint's model_config (including ve_every, the value-embedding spacing; absent = 2,
upstream's alternating layers) goes into config.json, and the conversion refuses any key/shape mismatch.
"""
import argparse, hashlib, json, os, pickle, shutil, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from configuration_gaggle import GaggleConfig  # noqa: E402

BOS = "<|bos|>"


def rename(k):
    k = k.removeprefix("_orig_mod.")
    return k if k == "lm_head.weight" else "model." + k


def build_tokenizer(tok_pkl, out_dir, model_max_length=2048):
    """tokenizer.pkl (a pickled tiktoken Encoding) -> tokenizer.json, plus a post-processor that prepends <|bos|>
    whenever special tokens are requested, which is how nanochat encodes a document."""
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast
    enc = pickle.load(open(tok_pkl, "rb"))
    tmp = os.path.join(out_dir, "_tok_tmp")
    convert_tiktoken_to_fast(enc, tmp)
    tj = json.load(open(os.path.join(tmp, "tokenizer.json")))
    shutil.rmtree(tmp)
    bos_id = enc.encode_single_token(BOS)
    tj["post_processor"] = {"type": "Sequence", "processors": [
        tj["post_processor"],
        {"type": "TemplateProcessing",
         "single": [{"SpecialToken": {"id": BOS, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
         "pair": [{"SpecialToken": {"id": BOS, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                  {"SpecialToken": {"id": BOS, "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
         "special_tokens": {BOS: {"id": BOS, "ids": [bos_id], "tokens": [BOS]}}},
    ]}
    json.dump(tj, open(os.path.join(out_dir, "tokenizer.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "bos_token": BOS,
        "eos_token": BOS,
        "pad_token": "<|assistant_end|>",
        "add_bos_token": True,
        "padding_side": "left",
        "model_input_names": ["input_ids", "attention_mask"],   # no token_type_ids: generate() would reject them
        "clean_up_tokenization_spaces": False,
    }, open(os.path.join(out_dir, "tokenizer_config.json"), "w"), indent=1)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir with model_<step>.pt, meta_<step>.json, tokenizer/")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-size", default="5GB")
    ap.add_argument("--tokenizer-only", action="store_true", help="only (re)build tokenizer.json / tokenizer_config.json")
    a = ap.parse_args()
    if a.tokenizer_only:
        build_tokenizer(os.path.join(a.ckpt, "tokenizer", "tokenizer.pkl"), a.out)
        print("tokenizer written to", a.out)
        return
    os.makedirs(a.out, exist_ok=True)

    meta = json.load(open(os.path.join(a.ckpt, f"meta_{a.step}.json")))
    mc = meta["model_config"]
    sd = torch.load(os.path.join(a.ckpt, f"model_{a.step}.pt"), map_location="cpu", weights_only=True)
    sd = {rename(k): v.contiguous() for k, v in sd.items()}

    known = {"vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len", "window_pattern", "ve_every"}
    assert set(mc) <= known, f"model_config has fields this port does not model: {set(mc) - known}"
    cfg = GaggleConfig(vocab_size=mc["vocab_size"], n_layer=mc["n_layer"], n_head=mc["n_head"],
                       n_kv_head=mc["n_kv_head"], n_embd=mc["n_embd"], sequence_len=mc["sequence_len"],
                       window_pattern=mc.get("window_pattern", "SSSL"), ve_every=mc.get("ve_every", 2))
    # the special-token ids in the config must be the tokenizer's own
    enc = pickle.load(open(os.path.join(a.ckpt, "tokenizer", "tokenizer.pkl"), "rb"))
    for name, tid in (("<|bos|>", cfg.bos_token_id), ("<|assistant_end|>", cfg.pad_token_id)):
        assert enc.encode_single_token(name) == tid, f"{name} is id {enc.encode_single_token(name)}, config says {tid}"
    cfg.architectures = ["GaggleForCausalLM"]
    cfg.auto_map = {"AutoConfig": "configuration_gaggle.GaggleConfig",
                    "AutoModel": "modeling_gaggle.GaggleModel",
                    "AutoModelForCausalLM": "modeling_gaggle.GaggleForCausalLM"}
    cfg.torch_dtype = "bfloat16"       # what every published number was computed in; the files keep fp32 matrices
    cfg.nanochat = {"commit": "92d63d4", "step": a.step, "source_file": f"model_{a.step}.pt",
                    "val_bpb": meta.get("val_bpb")}

    # the HF model must accept exactly these tensors
    sys.path.insert(0, HERE)
    from modeling_gaggle import GaggleForCausalLM
    with torch.device("meta"):
        ref = GaggleForCausalLM(cfg)
    want = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    have = {k: tuple(v.shape) for k, v in sd.items()}
    assert want == have, f"key/shape mismatch: missing {set(want) - set(have)}, unexpected {set(have) - set(want)}"

    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    split = split_torch_state_dict_into_shards(sd, filename_pattern="model{suffix}.safetensors",
                                               max_shard_size=a.max_shard_size)
    for fname, keys in split.filename_to_tensors.items():
        save_file({k: sd[k] for k in keys}, os.path.join(a.out, fname), metadata={"format": "pt"})
    if split.is_sharded:
        json.dump({"metadata": {"total_size": sum(v.numel() * v.element_size() for v in sd.values())},
                   "weight_map": split.tensor_to_filename},
                  open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    cfg.save_pretrained(a.out)
    json.dump({"bos_token_id": cfg.bos_token_id, "eos_token_id": cfg.eos_token_id,
               "pad_token_id": cfg.pad_token_id, "do_sample": False},
              open(os.path.join(a.out, "generation_config.json"), "w"), indent=1)
    for f in ("configuration_gaggle.py", "modeling_gaggle.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(a.out, f))
    build_tokenizer(os.path.join(a.ckpt, "tokenizer", "tokenizer.pkl"), a.out, model_max_length=cfg.sequence_len)

    dtypes = {}
    for v in sd.values():
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + v.numel()
    print(json.dumps({"out": a.out, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "ve_every": cfg.ve_every,
                      "value_embedding_layers": [i for i in range(cfg.n_layer) if cfg.has_value_embedding(i)],
                      "tensors": len(sd), "params": sum(v.numel() for v in sd.values()),
                      "params_by_dtype": dtypes, "shards": list(split.filename_to_tensors)}, indent=1))


if __name__ == "__main__":
    main()
