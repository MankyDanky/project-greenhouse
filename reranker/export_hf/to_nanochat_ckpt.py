#!/usr/bin/env python
"""Rebuild a nanochat_lce `ckpt_final.pt` from a released transformers repo.

The repo ships safetensors only. The paper's evaluation harness
(eval/rerank_eval.py --backend ours) wants the training checkpoint layout, so
this inverts convert_to_hf.py:

    python to_nanochat_ckpt.py --repo castorini/gaggle-reranker-20261005 \
                              --backbone models/nanochat-4b-climbmix-0924 \
                              --out ckpt_final.pt

The result is score-identical to the checkpoint we trained; it carries no optimizer state, so it
evaluates but does not resume.
"""
import argparse, json, os

import torch
from safetensors.torch import load_file


def unrename(key):
    if key == "score.weight":
        return "head"
    assert key.startswith("model."), key
    return "backbone." + key[len("model."):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local directory or hub id")
    ap.add_argument("--backbone", required=True,
                    help="the nanochat backbone dir the harness should load (model_*.pt + meta_*.json)")
    ap.add_argument("--nanochat-src", default="nanochat",
                    help="path to a karpathy/nanochat checkout at 92d63d4, which the harness "
                         "(not this repo's transformers model) needs for the backbone")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.repo
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.repo, allow_patterns=["*.json", "*.safetensors"])
    conf = json.load(open(os.path.join(path, "config.json")))
    index = os.path.join(path, "model.safetensors.index.json")
    files = (sorted(set(json.load(open(index))["weight_map"].values()))
             if os.path.exists(index) else ["model.safetensors"])
    sd = {}
    for f in files:
        for k, v in load_file(os.path.join(path, f)).items():
            sd[unrename(k)] = v
    cfg = {"model_name": args.backbone, "backbone_impl": "upstream",
           "nanochat_src": args.nanochat_src, "prompt": conf["prompt"],
           "max_doc_len": conf["max_doc_len"], "max_query_len": conf["max_query_len"]}
    torch.save({"model": sd, "cfg": cfg, "step": conf.get("step")}, args.out)
    print(json.dumps({"out": args.out, "tensors": len(sd), "cfg": cfg}, indent=1))


if __name__ == "__main__":
    main()
