"""Weight scale of a base checkpoint by parameter group, to pick the fine-tuning LR from the relative AdamW step lr/rms.

AdamW moves every entry by about lr per step whatever its size, so the same lr is a larger relative change for a model
with small weights. Muon-trained nanochat has large matrices (median RMS 0.136); AdamW-trained models are 5-11x smaller.

For each group (attention, MLP, routed/shared experts, router, embeddings, LM head, other 2-D) prints: matrix count,
parameters, median per-matrix RMS, parameter-weighted RMS, median per-matrix mean |w|, and the largest |w|. The headline
number is the median RMS over all 2-D matrices except the input embedding (tools/weight_drift.py's metric). Reads
safetensors in either transformers' layout or nanochat's (`transformer.h.N.attn.c_q`, `value_embeds`, `wte`), and also
prints the tokenizer's ids for the answer words and whether it prepends a BOS.

    python tools/weight_scale.py models/marin-8b-base [more model dirs ...]
"""
import json, re, statistics as st, sys
from pathlib import Path

from safetensors import safe_open


def group(k):
    if "embed_tokens" in k or ".wte." in k or "value_embeds" in k: return "embed"
    if k.startswith("lm_head") or ".lm_head." in k: return "lm_head"
    if ".self_attn." in k or ".attn." in k or ".attention." in k: return "attention"
    if ".mlp.shared_experts." in k: return "shared_experts"
    if re.search(r"\.mlp\.experts\.", k): return "routed_experts"
    if k.endswith(".mlp.gate.weight"): return "router"
    if ".mlp." in k or ".feed_forward." in k: return "mlp"
    return "other"


def scan(d):
    acc = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if len(s.get_slice(k).get_shape()) != 2: continue
                w = s.get_tensor(k).float()
                a = acc.setdefault(group(k), {"rms": [], "mabs": [], "ss": 0.0, "n": 0, "max": 0.0, "dtype": set()})
                a["dtype"].add(str(s.get_slice(k).get_dtype()))
                ss = float(w.pow(2).sum()); n = w.numel()
                a["rms"].append((ss / n) ** 0.5); a["mabs"].append(float(w.abs().mean())); a["ss"] += ss; a["n"] += n
                a["max"] = max(a["max"], float(w.abs().max()))
    out = {g: {"matrices": len(a["rms"]), "params_B": round(a["n"] / 1e9, 3), "median_rms": round(st.median(a["rms"]), 5),
               "weighted_rms": round((a["ss"] / a["n"]) ** 0.5, 5), "median_mean_abs": round(st.median(a["mabs"]), 5),
               "max_abs": round(a["max"], 3), "dtype": sorted(a["dtype"])} for g, a in sorted(acc.items())}
    allr = [r for g, a in acc.items() if g != "embed" for r in a["rms"]]
    alla = [r for g, a in acc.items() if g != "embed" for r in a["mabs"]]
    out["ALL_NON_EMBED"] = {"matrices": len(allr), "median_rms": round(st.median(allr), 5), "median_mean_abs": round(st.median(alla), 5)}
    return out


for d in sys.argv[1:]:
    print(json.dumps({"model": d, "groups": scan(d)}, indent=1), flush=True)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        print({w: tok.encode(w, add_special_tokens=False) for w in (" true", " false", " yes", " no")},
              "| encode('hello world'):", tok("hello world").input_ids, "| bos", tok.bos_token, tok.bos_token_id,
              "| eos", tok.eos_token, tok.eos_token_id, "| pad", tok.pad_token, flush=True)
    except Exception as e:                                   # nanochat's HF tokenizer is fine; others may lack one
        print("tokenizer:", type(e).__name__, str(e)[:200], flush=True)
