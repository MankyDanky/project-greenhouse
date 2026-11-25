#!/usr/bin/env python
"""Convert a nanochat_lce reranker checkpoint into a `transformers` (trust_remote_code) repo.

    python reranker/export_hf/convert_to_hf.py --ckpt reranker/checkpoints/sft_soup/ckpt_final.pt \
                            --out  exports/gaggle-reranker-20261005

Writes model.safetensors (sharded), config.json, tokenizer files, the two code files, and
SHA256SUMS. Tensors are streamed one at a time from the mmap'd checkpoint, so peak memory is one
shard rather than the whole model.

The prompt scaffold ids and the " true"/" false" ids are derived from the SAME prompt_builder_nc
the checkpoint was trained and evaluated with, never hand-copied, and the two scoring rows are
checked against the order the training wrapper used (row 0 = true, row 1 = false).
"""
import argparse, glob, hashlib, json, os, shutil, sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(RECIPE, "src"))
sys.path.insert(0, HERE)

from prompt_builder_nc import ASSISTANT_END, PairBuilder, readout_offset, setup_tokenizer, yes_no_ids  # noqa: E402
from configuration_gaggle import GaggleRerankerConfig                                   # noqa: E402

SHARD_BYTES = 4 * 1000 ** 3


def resolve_backbone(path):
    if os.path.exists(path):
        return path
    local = os.path.normpath(os.path.join(RECIPE, "models", os.path.basename(str(path).rstrip("/"))))
    assert os.path.exists(local), f"backbone {path} not found (tried {local})"
    return local


def rename(key):
    """nanochat_lce reranker state dict -> this port's module tree."""
    if key == "head":
        return "score.weight"                       # (2, n_embd): row 0 " true", row 1 " false"
    assert key.startswith("backbone."), key
    return "model." + key[len("backbone."):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="repo id to record as _name_or_path")
    args = ap.parse_args()

    sd0 = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg, weights = dict(sd0["cfg"]), sd0["model"]
    backbone = resolve_backbone(cfg["model_name"])
    meta = json.load(open(glob.glob(os.path.join(backbone, "meta_*.json"))[0]))["model_config"]
    known = {"vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len", "window_pattern", "ve_every"}
    assert set(meta) <= known, f"model_config has fields this port does not model: {set(meta) - known}"

    from transformers import AutoTokenizer
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(backbone))
    prompt = cfg.get("prompt", "chat")
    assert prompt == "nochat", f"only the nochat prompt is exported so far, got {prompt!r}"
    true_id, false_id = yes_no_ids(tok, prompt, cfg.get("answer_tokens"))
    pb = PairBuilder(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"],
                     prompt=prompt)
    assert not pb._mid2, "the qdq prompt repeats the query; not exported"

    conf = GaggleRerankerConfig(
        vocab_size=meta["vocab_size"], n_layer=meta["n_layer"], n_head=meta["n_head"],
        n_kv_head=meta["n_kv_head"], n_embd=meta["n_embd"], sequence_len=meta["sequence_len"],
        window_pattern=meta.get("window_pattern", "SSSL"), ve_every=meta.get("ve_every", 2),
        max_query_len=cfg["max_query_len"], max_doc_len=cfg["max_doc_len"], prompt=prompt,
        prompt_template=pb.render("{query}", "{passage}").replace("<|bos|>", "<|bos|>"),
        prompt_prefix_ids=pb._pre, prompt_mid_ids=pb._mid, prompt_suffix_ids=pb._suf,
        true_token_id=true_id, false_token_id=false_id, readout_offset=readout_offset(prompt),
        pad_token_id=tok.pad_token_id,
    )
    conf.auto_map = {
        "AutoConfig": "configuration_gaggle.GaggleRerankerConfig",
        "AutoModel": "modeling_gaggle.GaggleModel",
        "AutoModelForSequenceClassification": "modeling_gaggle.GaggleRerankerForSequenceClassification",
    }
    conf.architectures = ["GaggleRerankerForSequenceClassification"]
    if args.name:
        conf._name_or_path = args.name

    # the two scoring rows must be the LM-head rows for " true"/" false", in that order
    bb_sd = torch.load(glob.glob(os.path.join(backbone, "model_*.pt"))[0],
                       map_location="cpu", weights_only=True, mmap=True)
    lm = bb_sd["lm_head.weight"]
    head = weights["head"]
    d_true = (head[0].float() - lm[true_id].float()).abs().max().item()
    d_false = (head[1].float() - lm[false_id].float()).abs().max().item()
    print(f"scoring rows vs pretrained lm_head: |row0-true|={d_true:.4g} |row1-false|={d_false:.4g} "
          f"(both are fine-tuned, so these are distances, not zeros)")
    del bb_sd, lm

    os.makedirs(args.out, exist_ok=True)
    # ---- shard and write ------------------------------------------------------------------------
    keys = list(weights.keys())
    plan, cur, cur_bytes = [], [], 0
    for k in keys:
        nbytes = weights[k].numel() * weights[k].element_size()
        if cur and cur_bytes + nbytes > SHARD_BYTES:
            plan.append(cur); cur, cur_bytes = [], 0
        cur.append(k); cur_bytes += nbytes
    if cur:
        plan.append(cur)
    n = len(plan)
    index, total = {}, 0
    for si, shard in enumerate(plan, 1):
        fname = "model.safetensors" if n == 1 else f"model-{si:05d}-of-{n:05d}.safetensors"
        tensors = {}
        for k in shard:
            t = weights[k].detach().clone().contiguous()     # materialise one tensor at a time
            tensors[rename(k)] = t
            total += t.numel() * t.element_size()
            index[rename(k)] = fname
        save_file(tensors, os.path.join(args.out, fname), metadata={"format": "pt"})
        print(f"wrote {fname}: {len(tensors)} tensors", flush=True)
        del tensors
    if n > 1:
        json.dump({"metadata": {"total_size": total}, "weight_map": index},
                  open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)

    conf.save_pretrained(args.out)
    # ---- tokenizer: ship it with pad/eos pinned, so no setup helper is needed -------------------
    shutil.copy(os.path.join(backbone, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    # Written in full rather than copied from the backbone: the released base repos' tokenizer_config.json is set up for
    # generation (eos <|bos|>, left padding, add_bos_token), while every released reranker ships exactly these keys.
    tc = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": meta["sequence_len"],
          "bos_token": tok.bos_token, "eos_token": ASSISTANT_END, "pad_token": ASSISTANT_END, "padding_side": "right",
          "clean_up_tokenization_spaces": False}
    json.dump(tc, open(os.path.join(args.out, "tokenizer_config.json"), "w"), indent=1)
    for f in ("configuration_gaggle.py", "modeling_gaggle.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(args.out, f))

    with open(os.path.join(args.out, "SHA256SUMS"), "w") as fh:
        for f in sorted(os.listdir(args.out)):
            p = os.path.join(args.out, f)
            if not os.path.isfile(p) or f == "SHA256SUMS":
                continue
            h = hashlib.sha256()
            with open(p, "rb") as g:
                for chunk in iter(lambda: g.read(1 << 22), b""):
                    h.update(chunk)
            fh.write(f"{h.hexdigest()}  {f}\n")
    print(json.dumps({"out": args.out, "shards": n, "bytes": total,
                      "step": sd0.get("step"), "prompt": prompt,
                      "true_id": true_id, "false_id": false_id}, indent=1))


if __name__ == "__main__":
    main()
