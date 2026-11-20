"""Evaluate an exported checkpoint with greenhouse rerank_eval.py (harness unchanged) plus one backend.

  --backend ours   greenhouse's own loader: how run G was scored. Used for R0 (cfg carries
                   backbone_impl/prompt/nanochat_src for it).
  --backend lce    the comparison runs: backbone in bf16, the trained answer rows in exact fp32, raw gap,
                   the checkpoint's own PairBuilder (2,048 total budget), candidates scored in
                   length-sorted, token-budgeted batches.

    EVAL_RUNS=... EVAL_QRELS_DIR=... python src/rerank.py --dataset dl19 --backend lce \
        --model runs/C_qwen/milestones/step01931_q247452.pt --tag C_qwen-final --out eval/runs
"""
import os, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent                 # eval/backends
EVAL = HERE.parent
REPO = EVAL.parent
ROOT = REPO / "reranker"                               # models/ and third_party/ live here
os.environ.setdefault("EVAL_RUNS", str(REPO / "data/bm25_top100"))
os.environ.setdefault("EVAL_QRELS_DIR", str(EVAL / "qrels"))
sys.path.insert(0, str(ROOT / "engines/multi"))
sys.path.insert(0, str(EVAL))


def resolve_model_dir(cfg):
    p = Path(cfg["model_name"])
    return str(p) if p.exists() else str(ROOT / "models" / p.name)   # checkpoint moved between clusters


def load_scorer(ckpt, dev="cuda", dtype=torch.bfloat16, tokens_per_batch=65536, max_batch=64):
    """Returns score(query, docs) -> list[float] for an lce checkpoint."""
    from transformers import AutoTokenizer
    import prompts
    from backbones import build_model
    from train import make_builder
    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = dict(sd["cfg"]); cfg["model_name"] = resolve_model_dir(cfg)
    if cfg["backbone"] == "nanochat":
        os.environ["NANOCHAT_DTYPE"] = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype]
    tok_dir = Path(ckpt).parent.parent / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_dir if tok_dir.exists() else cfg["model_name"]))
    pb = make_builder(cfg, tok)
    yes_id, no_id = prompts.answer_ids(tok)
    m = build_model(cfg, yes_id, no_id, dtype)
    m.load_state_dict(sd["model"], strict=True)          # bf16 params take the cast; fp32 head stays exact
    if getattr(m, "tied", False):                        # tied rows: keep the trained fp32 values
        key = [k for k in sd["model"] if k.endswith("embed_tokens.weight")]
        assert len(key) == 1, key
        rows = sd["model"][key[0]][[yes_id, no_id]].float().clone()
        m.rows = lambda r=rows.to(dev): r
    m = m.to(dev).eval()
    if hasattr(m, "head") and m.head is not None:
        assert m.head.dtype == torch.float32
    pad = tok.pad_token_id
    print(f"lce scorer: backbone={cfg['backbone']} rule={cfg['length_rule']} answers {yes_id}/{no_id} "
          f"dtype={dtype} step={sd.get('step')} queries={sd.get('groups')}", flush=True)

    @torch.no_grad()
    def score(query, docs, bs=None, max_len=None):
        seqs = [pb.build(query, d) for d in docs]
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out = [0.0] * len(seqs)
        i = 0
        while i < len(order):
            L = len(seqs[order[i]])
            n = max(1, min(max_batch, tokens_per_batch // L, len(order) - i))
            grp = order[i:i + n]; i += n
            ids = torch.full((n, L), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, j in enumerate(grp):
                ids[r, :len(seqs[j])] = torch.tensor(seqs[j]); att[r, :len(seqs[j])] = 1
            sc = m(ids.to(dev), att.to(dev)).scores.float().tolist()
            for j, s in zip(grp, sc):
                out[j] = s
        return out
    return score


def load_base_scorer(config, dev="cuda", dtype=torch.bfloat16, tokens_per_batch=65536, max_batch=64):
    """Zero-shot: the UNTRAINED base model with the run's prompt and its pretrained " true"/" false" LM-head rows."""
    from transformers import AutoTokenizer
    import prompts
    from backbones import build_model
    from train import make_builder, load_cfg
    cfg = load_cfg(config, [])
    if cfg["backbone"] == "nanochat":
        os.environ["NANOCHAT_DTYPE"] = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    pb = make_builder(cfg, tok)
    y, n = prompts.answer_ids(tok)
    m = build_model(cfg, y, n, dtype).to(dev).eval()
    pad = tok.pad_token_id
    print(f"ZERO-SHOT base scorer: {cfg['backbone']} {cfg['model_name']} answers {y}/{n} dtype={dtype}", flush=True)

    @torch.no_grad()
    def score(query, docs, bs=None, max_len=None):
        seqs = [pb.build(query, d) for d in docs]
        out = []
        for i in range(0, len(seqs), max_batch):
            grp = seqs[i:i + max_batch]; L = max(map(len, grp))
            ids = torch.full((len(grp), L), pad, dtype=torch.long); att = torch.zeros_like(ids)
            for r, x in enumerate(grp):
                ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
            out += m(ids.to(dev), att.to(dev)).scores.float().tolist()
        return out
    return score


def remap_paths(sd):
    """A checkpoint trained on another cluster records that cluster's absolute model / nanochat-source paths; point
    them at this checkout's copies (same name under models/ and third_party/). Paths only, never weights."""
    cfg = sd.get("cfg") if isinstance(sd, dict) else None
    if cfg:
        for key, sub in (("model_name", "models"), ("nanochat_src", "third_party")):
            p = cfg.get(key)
            if p and str(p).startswith("/") and not Path(p).exists() and (ROOT / sub / Path(p).name).exists():
                cfg[key] = str(ROOT / sub / Path(p).name)
    return sd


def main():
    import rerank_eval as RE
    _load = torch.load
    torch.load = lambda *a, **k: remap_paths(_load(*a, **k))      # also covers greenhouse's own `ours` loader
    RE.BACKENDS["base"] = lambda config, dev, dtype: load_base_scorer(config, dev, dtype)
    RE.DEFAULT_MAXLEN["base"] = 512
    RE.HERE = str(EVAL)                                  # results -> eval/results
    RE.BACKENDS["lce"] = lambda ckpt, dev, dtype: load_scorer(ckpt, dev, dtype)
    RE.DEFAULT_MAXLEN["lce"] = 2048
    RE.main()


if __name__ == "__main__":
    main()
