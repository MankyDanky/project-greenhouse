#!/usr/bin/env python
"""Check that the `transformers` port reproduces nanochat's own model, on one GPU, in bf16.

Reference = nanochat @ 92d63d4 `build_model(...)` on the original .pt (the code path every published number used:
SDPA attention, bf16 activations). Candidate = AutoModelForCausalLM.from_pretrained(HF_DIR, trust_remote_code=True).

    python parity_check.py --nanochat-src SRC --ckpt CKPT_DIR --step 227158 --hf HF_DIR --texts FILE.jsonl --out out.json
"""
import argparse, glob, json, os, random, sys, time

import torch


def load_texts(paths, n):
    out = []
    for p in paths:
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:          # one malformed line in the bundle's narrative_qa.jsonl
                continue
            for key in ("context", "text", "query", "passage", "story", "question"):
                if isinstance(d.get(key), str) and len(d[key]) > 200:
                    out.append(d[key])
                    break
    random.Random(0).shuffle(out)
    return out[:n]


def stat(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {"bitwise_equal": bool(torch.equal(a, b)), "max_abs_diff": d.max().item(),
            "mean_abs_diff": d.mean().item(), "argmax_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanochat-src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-reference", action="store_true", help="only run the HF side and save its logits")
    ap.add_argument("--ref-logits", default=None, help="compare against logits saved by another run")
    ap.add_argument("--device", default="cuda", help="cuda (bf16, as evaluated) or cpu (fp32, as nanochat runs on CPU)")
    ap.add_argument("--lengths", default="2,17,300,512,513,514,1000,2048")
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--dtype", default=None, help="override: float32 or bfloat16")
    a = ap.parse_args()
    dev = torch.device(a.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    if a.dtype:
        dtype = getattr(torch, a.dtype)
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    R = {"device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "dtype": str(dtype),
         "torch": torch.__version__}
    import transformers
    R["transformers"] = transformers.__version__

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.hf, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(a.hf, trust_remote_code=True, dtype=dtype).to(dev).eval()
    R["hf_load_s"] = round(time.time() - t0, 1)
    R["hf_param_dtypes"] = sorted({str(p.dtype) for p in hf.parameters()})
    R["hf_fp32_params"] = [n for n, p in hf.named_parameters() if p.dtype == torch.float32]
    R["hf_params"] = sum(p.numel() for p in hf.parameters())
    bos = tok.bos_token_id
    texts = load_texts(a.texts, 400)

    # ---------------- sequences: real text, BOS-prefixed, lengths covering both window regimes ----------------
    stream = []
    for t in texts:
        stream += tok(t, add_special_tokens=False)["input_ids"]
    lengths = [int(x) for x in a.lengths.split(",")]
    seqs = []
    off = 0
    for L in lengths:
        seqs.append([bos] + stream[off:off + L - 1]); off += L
    prompts = ["The capital of France is", "def fibonacci(n):", "In 1905, Albert Einstein",
               "The three primary colors are", "Water boils at"]

    @torch.inference_mode()
    def hf_logits(ids):
        return hf(torch.tensor([ids], device=dev)).logits[0]

    out_hf = {"single": [hf_logits(s).cpu() for s in seqs]}
    # right-padded batch with a mask, and left-padded batch with a mask
    bidx = [i for i in (1, 2, 3, 6) if i < len(seqs)]
    batch = [seqs[i] for i in bidx]
    n = max(len(s) for s in batch)
    pad = tok.pad_token_id
    ids_r = torch.full((len(batch), n), pad); am_r = torch.zeros_like(ids_r)
    ids_l = torch.full((len(batch), n), pad); am_l = torch.zeros_like(ids_l)
    for i, s in enumerate(batch):
        ids_r[i, :len(s)] = torch.tensor(s); am_r[i, :len(s)] = 1
        ids_l[i, n - len(s):] = torch.tensor(s); am_l[i, n - len(s):] = 1
    with torch.inference_mode():
        lr = hf(ids_r.to(dev), attention_mask=am_r.to(dev)).logits.cpu()
        ll = hf(ids_l.to(dev), attention_mask=am_l.to(dev)).logits.cpu()
    out_hf["right_padded"] = [lr[i, :len(s)] for i, s in enumerate(batch)]
    out_hf["left_padded"] = [ll[i, n - len(s):] for i, s in enumerate(batch)]

    # padding invariance inside the port itself
    R["hf_right_padded_vs_single"] = [stat(out_hf["right_padded"][i], out_hf["single"][bidx[i]])
                                      for i in range(len(batch))]
    R["hf_left_padded_vs_single"] = [stat(out_hf["left_padded"][i], out_hf["single"][bidx[i]])
                                     for i in range(len(batch))]

    # KV-cache decoding vs one full forward, inside the port
    s = max(seqs, key=len)                                 # crosses the 512 window when >= 700 tokens
    with torch.inference_mode():
        full = hf(torch.tensor([s], device=dev)).logits[0]
        cut = min(600, len(s) - 101)
        o = hf(torch.tensor([s[:cut]], device=dev), use_cache=True)
        pkv, steps = o.past_key_values, [o.logits[0, -1]]
        for j in range(cut, cut + 100):
            o = hf(torch.tensor([[s[j]]], device=dev), past_key_values=pkv, use_cache=True,
                   prev_input_ids=torch.tensor([[s[j - 1]]], device=dev))
            pkv = o.past_key_values; steps.append(o.logits[0, -1])
    R["hf_cache_decode_vs_full"] = stat(torch.stack(steps), full[cut - 1:cut + 100])

    # generation
    gens = {}
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(dev)
        assert ids["input_ids"][0, 0].item() == bos
        with torch.inference_mode():
            g_cache = hf.generate(**ids, max_new_tokens=a.gen_tokens, do_sample=False)[0, ids["input_ids"].size(1):].tolist()
            g_nocache = hf.generate(**ids, max_new_tokens=a.gen_tokens, do_sample=False,
                                    use_cache=False)[0, ids["input_ids"].size(1):].tolist()
        gens[p] = {"hf_cache": g_cache, "hf_nocache": g_nocache}
    # batched left-padded generation
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    with torch.inference_mode():
        gb = hf.generate(**enc, max_new_tokens=a.gen_tokens, do_sample=False)[:, enc["input_ids"].size(1):].tolist()
    for p, g in zip(prompts, gb):
        gens[p]["hf_batched_left"] = g

    # loss
    with torch.inference_mode():
        x = torch.tensor([max(seqs, key=len)], device=dev)
        R["hf_loss_longest"] = hf(x, labels=x).loss.item()

    torch.save(out_hf, a.out + ".hf_logits.pt")
    if a.ref_logits:
        ref = torch.load(a.ref_logits)
        R["vs_saved_hf_logits"] = [stat(x, y) for x, y in zip(out_hf["single"], ref["single"])]

    if not a.skip_reference:
        # ---------------- nanochat reference ----------------
        os.environ["NANOCHAT_BASE_DIR"] = a.ckpt
        sys.path.insert(0, a.nanochat_src)
        try:                                   # nanochat imports rustbpe at module level but only trains with it
            import rustbpe  # noqa: F401
        except ImportError:
            import types
            sys.modules["rustbpe"] = types.ModuleType("rustbpe")
        import nanochat.flash_attention as fa
        R["nanochat_uses_fa3"] = bool(fa.USE_FA3)
        from nanochat.checkpoint_manager import build_model
        ref, ntok, meta = build_model(a.ckpt, a.step, dev, phase="eval")

        # tokenizer parity on the raw texts
        mism = 0
        for t in texts:
            if ntok.encode(t) != tok(t, add_special_tokens=False)["input_ids"]:
                mism += 1
        special = "<|bos|>hello<|assistant_end|> world <|user_start|>x"
        R["tokenizer"] = {"texts": len(texts), "mismatches": mism,
                          "bos_prepended": tok("hi")["input_ids"][0] == ntok.get_bos_token_id(),
                          "decode_roundtrip": all(tok.decode(tok(t, add_special_tokens=False)["input_ids"]) == t
                                                  for t in texts[:100]),
                          "special_tokens_equal": ntok.encode(special, allowed_special="all") ==
                          tok(special, add_special_tokens=False)["input_ids"]
                          if "allowed_special" in ntok.encode.__code__.co_varnames else "n/a"}

        @torch.inference_mode()
        def nc_logits(ids):
            return ref(torch.tensor([ids], device=dev))[0]

        R["single"] = []
        for L, s, h in zip(lengths, seqs, out_hf["single"]):
            r = nc_logits(s).cpu()
            R["single"].append({"len": L, **stat(h, r)})
        with torch.inference_mode():
            nr = ref(ids_r.to(dev)).cpu()                 # nanochat on the same right-padded batch (no mask)
        R["right_padded_vs_nanochat_batch"] = [stat(out_hf["right_padded"][i], nr[i, :len(s)])
                                               for i, s in enumerate(batch)]
        with torch.inference_mode():
            longest = max(seqs, key=len)
            x = torch.tensor([longest], device=dev)
            y = torch.tensor([longest[1:] + [-1]], device=dev)
            R["nanochat_loss_longest"] = ref(x, targets=y).item()
        for p in prompts:
            ids = ntok.encode(p, prepend=ntok.get_bos_token_id())
            gens[p]["nanochat"] = list(ref.generate(ids, max_tokens=a.gen_tokens, temperature=0.0))
            gens[p]["text_nanochat"] = ntok.decode(gens[p]["nanochat"])
            gens[p]["text_hf"] = tok.decode(gens[p]["hf_cache"])
            gens[p]["agree"] = {k: gens[p][k] == gens[p]["nanochat"] for k in ("hf_cache", "hf_nocache", "hf_batched_left")}
    R["generation"] = gens
    json.dump(R, open(a.out, "w"), indent=1)
    summary = {k: v for k, v in R.items() if k not in ("generation", "hf_fp32_params")}
    print(json.dumps(summary, indent=1)[:6000])
    if "single" in R:
        print("generation agree:", {p: g["agree"] for p, g in gens.items()})


if __name__ == "__main__":
    main()
