"""
Convert karpathy/nanochat-d34 (raw nanochat checkpoint: model_169150.pt + tokenizer.pkl, BASE
pretraining only) to a HuggingFace NanoChatForCausalLM folder, keeping FP32 weights.

Everything comes from Karpathy's release; nothing from a third-party repo. It follows transformers' own
convert_nanochat_checkpoints.py (main branch; not in the 5.16.1 release), except that script casts every
weight to bf16, and our trainers run fp32, so we keep fp32 (the one bf16 tensor is upcast exactly):
  * config.json    -- from meta_169150.json, with the same field mapping as transformers' converter
                      (34 layers, 2176 wide, 17 heads, vocab 65536, context 2048); dtype=float32.
                      bos/eos/pad ids are NanoChatConfig's defaults (0/1/1), as in transformers' converter.
                      They are not nanochat's special tokens (<|bos|> is 65527); our code never uses them,
                      but pad_token_id=1 makes embedding row 1 (the byte 0x01) a padding row with no gradient.
                      Runs A-D trained with exactly these values, so they are kept.
  * tokenizer.json -- from tokenizer.pkl with transformers' convert_tiktoken_to_fast (needs tiktoken and
                      blobfile). Byte-identical to pankajmathur/nanochat-d34-sft-hf's tokenizer.json, which
                      the first conversion (2026-09-14) copied instead.
  * generation_config.json -- written by save_pretrained from the config.

Checks (all must pass):
  1. every checkpoint tensor mapped, strict load;
  2. tokenizer.json == tokenizer.pkl on sample texts (special tokens included);
  3. HF forward == Karpathy's own nanochat gpt.py forward (commit f37d45c, 2025-11-20, the week the
     checkpoint was released), fp32 on CPU, logits max-abs-diff reported.
"""
import argparse, json, os, pickle, shutil, sys
import torch


def convert_layer(i):
    o, n = f"transformer.h.{i}", f"model.layers.{i}"
    return {f"{o}.attn.c_q.weight": f"{n}.self_attn.q_proj.weight", f"{o}.attn.c_k.weight": f"{n}.self_attn.k_proj.weight",
            f"{o}.attn.c_v.weight": f"{n}.self_attn.v_proj.weight", f"{o}.attn.c_proj.weight": f"{n}.self_attn.o_proj.weight",
            f"{o}.mlp.c_fc.weight": f"{n}.mlp.fc1.weight", f"{o}.mlp.c_proj.weight": f"{n}.mlp.fc2.weight"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/karpathy-nanochat-d34")
    ap.add_argument("--dst", default="models/nanochat-d34-base-hf")
    ap.add_argument("--nanochat-src", default=None, help="dir containing the nanochat/ package (reference check)")
    a = ap.parse_args()
    from transformers import NanoChatConfig, NanoChatForCausalLM, PreTrainedTokenizerFast
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast

    # config: the field mapping of transformers' convert_nanochat_checkpoints.load_config_from_checkpoint
    meta = json.load(open(os.path.join(a.src, "meta_169150.json")))["model_config"]
    cfg = NanoChatConfig(vocab_size=meta["vocab_size"], hidden_size=meta["n_embd"], num_hidden_layers=meta["n_layer"],
                         num_attention_heads=meta["n_head"], num_key_value_heads=meta["n_kv_head"],
                         max_position_embeddings=meta["sequence_len"],
                         intermediate_size=meta.get("intermediate_size", meta["n_embd"] * 4))
    cfg.tie_word_embeddings = False
    cfg.dtype = "float32"

    old = torch.load(os.path.join(a.src, "model_169150.pt"), map_location="cpu", weights_only=True)
    bf = [k for k, v in old.items() if v.dtype != torch.float32]
    mapping = {"transformer.wte.weight": "model.embed_tokens.weight", "lm_head.weight": "lm_head.weight"}
    for i in range(cfg.num_hidden_layers): mapping.update(convert_layer(i))
    unmapped = [k for k in old if k not in mapping]
    assert not unmapped, f"unmapped checkpoint tensors: {unmapped}"
    sd = {mapping[k]: v.float().contiguous() for k, v in old.items()}
    with torch.device("meta"):
        model = NanoChatForCausalLM(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.float()
    print(f"[1] mapped {len(sd)} tensors, strict load OK; {sum(v.numel() for v in sd.values()):,} params; upcast from bf16: {bf}")
    os.makedirs(a.dst, exist_ok=True)
    model.save_pretrained(a.dst, safe_serialization=True)
    # tokenizer: Karpathy's tiktoken encoding -> tokenizer.json (transformers' converter; its tiktoken/ side file is dropped)
    enc = pickle.load(open(os.path.join(a.src, "tokenizer.pkl"), "rb"))
    tmp = os.path.join(a.dst, "_tok_tmp")
    convert_tiktoken_to_fast(enc, tmp)
    shutil.move(os.path.join(tmp, "tokenizer.json"), os.path.join(a.dst, "tokenizer.json"))
    shutil.rmtree(tmp)

    # [2] tokenizer equivalence: the converted tokenizer.json vs Karpathy's tokenizer.pkl
    hf_tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.dst, "tokenizer.json"))
    samples = ["Query: what are intranets\nPassage: An intranet is a private network.\nRelevant:",
               "<|bos|>Hello, world! 1234567 éè 中文 emoji \U0001F600 tabs\tand  double  spaces",
               "<|bos|><|user_start|>Judge whether the Document meets the requirements<|user_end|><|assistant_start|>",
               " true false  Relevant: true", "def f(x):\n    return x**2  # code"]
    for s in samples:
        a_ids = enc.encode(s, allowed_special="all"); b_ids = hf_tok.encode(s, add_special_tokens=False)
        assert a_ids == b_ids, f"tokenizer mismatch on {s!r}:\n pkl {a_ids}\n hf  {b_ids}"
    print(f"[2] tokenizer.json == tokenizer.pkl on {len(samples)} samples (incl. special tokens)")

    # [3] forward equivalence vs Karpathy's own implementation, using the model RELOADED from disk
    #     (also checks the saved files; the in-memory copy was built on the meta device)
    del model
    model = NanoChatForCausalLM.from_pretrained(a.dst, dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "reloaded model is not fp32"
    if a.nanochat_src:
        # Karpathy's gpt.py (f37d45c) computes its rotary tables in fp32, rounds them to bf16, and asserts they are
        # bf16. To compare the two implementations in fp32, load it with exactly that one assertion removed (nothing
        # else changed; fail if the line is not found) and give it tables recomputed without the bf16 rounding.
        import types
        sys.path.insert(0, a.nanochat_src)
        gpt_path = os.path.join(a.nanochat_src, "nanochat", "gpt.py")
        code = open(gpt_path).read()
        rope_assert = 'assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"'
        assert code.count(rope_assert) == 1, f"expected the bf16 rotary assertion once in {gpt_path}"
        import nanochat                                            # the package, for gpt.py's own imports
        mod = types.ModuleType("nanochat.gpt"); mod.__file__ = gpt_path; sys.modules["nanochat.gpt"] = mod
        exec(compile(code.replace(rope_assert, "pass  # bf16 rotary assertion removed for the fp32 comparison"),
                     gpt_path, "exec"), mod.__dict__)
        ref = mod.GPT(mod.GPTConfig(sequence_len=meta["sequence_len"], vocab_size=meta["vocab_size"], n_layer=meta["n_layer"],
                                    n_head=meta["n_head"], n_kv_head=meta["n_kv_head"], n_embd=meta["n_embd"]))
        ref.load_state_dict({k: v.float() for k, v in old.items()}, strict=True)
        cos_bf, sin_bf = ref.cos.clone(), ref.sin.clone()          # the reference's own bf16 rotary tables,
        assert cos_bf.dtype == torch.bfloat16                      # captured before .float() upcasts every buffer
        ref = ref.float().eval()
        # GPT._precompute_rotary_embeddings (base 10000) without its `cos.bfloat16()` rounding
        hd = meta["n_embd"] // meta["n_head"]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        freqs = torch.outer(torch.arange(cos_bf.size(1), dtype=torch.float32), inv_freq)
        cos32, sin32 = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        assert cos32.shape == cos_bf.shape and (cos32.bfloat16() == cos_bf).all() and (sin32.bfloat16() == sin_bf).all()
        ids = torch.tensor([enc.encode(samples[0], allowed_special="all")])
        with torch.no_grad():
            lh = model.eval()(input_ids=ids).logits.float()
            ref.cos, ref.sin = cos32, sin32                        # strict: both implementations fully fp32
            lr = ref(ids).float()                                  # softcapped logits
            ref.cos, ref.sin = cos_bf, sin_bf                      # as released: bf16 rotary tables (reported only)
            lr_bf = ref(ids).float()
        d = (lr - lh).abs().max().item()
        d_bf = (lr_bf - lh).abs().max().item()
        same_argmax = bool((lr.argmax(-1) == lh.argmax(-1)).all())
        nxt = lh[0, -1]; t_true, t_false = enc.encode(" true"), enc.encode(" false")
        print(f"[3] HF vs Karpathy gpt.py logits, all fp32: max|diff| = {d:.2e} over {ids.shape[1]} tokens; argmax identical: {same_argmax}")
        print(f"    with the reference's bf16 rotary tables (as released): max|diff| = {d_bf:.2e} (reported, not asserted)")
        print(f"    after 'Relevant:' base model logit(' true')={nxt[t_true[0]]:.3f} logit(' false')={nxt[t_false[0]]:.3f}")
        assert d < 1e-3 and same_argmax, "HF forward does not match the reference implementation"
    json.dump({"source": "karpathy/nanochat-d34 model_169150.pt (base pretraining, step 169150)",
               "dtype": "float32", "config_from": "meta_169150.json (transformers convert_nanochat_checkpoints mapping)",
               "tokenizer_from": "tokenizer.pkl via transformers convert_tiktoken_to_fast (verified == tokenizer.pkl)",
               "reference_check": "nanochat gpt.py @ f37d45c, all fp32 (rotary tables recomputed without bf16 rounding; bf16 assertion removed), max-abs-diff < 1e-3" if a.nanochat_src else None},
              open(os.path.join(a.dst, "CONVERSION.json"), "w"), indent=2)
    print(f"saved {a.dst}")


if __name__ == "__main__":
    main()
