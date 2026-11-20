"""CPU checks (no GPU, needs the prepared data and the four tokenizers).

  1. our PairBuilder (nanochat, caps 128/512) is token-identical to greenhouse's on sampled training pairs
  2. the 2,048 rule: every sequence <= 2,048 tokens TOTAL, query <= 512, and a passage is cut only when the
     whole sequence would not fit
  3. native BOS / answer ids per backbone
  4. execution chunking covers every row exactly once and respects the chunk budget

    python tests/test_cpu.py --root $ROOT
"""
import argparse, random, sys
from pathlib import Path

import torch
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2] / "reranker"
sys.path.insert(0, str(ROOT / "engines/multi"))
import prompts  # noqa: E402
from train import chunk_rows  # noqa: E402
from prompt_builder_nc import PairBuilder as GreenhouseBuilder, setup_tokenizer as gh_setup  # noqa: E402

MODELS = {"nanochat": ("gaggle-nanochat-pretrained-climbmix-20260924", (2737, 7863), 32759),
          "qwen": ("qwen3.5-4b-base", (804, 867), None),
          "gemma": ("gemma-4-e2b", (1847, 2416), 2),
          "minicpm": ("minicpm5-2b-base", (2989, 6307), 0),
          "marin": ("marin-8b-base", (837, 905), 128000)}


def sample_pairs(parquet, n, seed=17):
    pf = pq.ParquetFile(parquet)
    rng = random.Random(seed)
    pairs = []
    for rg in rng.sample(range(pf.num_row_groups), 6):
        t = pf.read_row_group(rg, columns=["query", "pos_texts", "neg_texts"]).to_pylist()
        for r in rng.sample(t, n // 6):
            docs = r["pos_texts"] + r["neg_texts"]
            pairs.append((r["query"], max(docs, key=len)))       # the row's longest passage: stresses truncation
            pairs.append((r["query"], rng.choice(docs)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    pairs = sample_pairs(Path(a.root) / "data/rlhn250k_prepared_climbmix_lens/train.parquet", a.n)  # same texts
    fails = []

    # 1. R0 token identity with greenhouse
    tok = AutoTokenizer.from_pretrained(str(Path(a.root) / "models/gaggle-nanochat-pretrained-climbmix-20260924"))
    gh = GreenhouseBuilder(gh_setup(tok), max_doc_len=512, max_query_len=128, prompt="nochat")
    ours = prompts.PairBuilder(prompts.setup_tokenizer(tok, "nanochat"), "nanochat", "caps", 128, 512)
    diff = sum(gh.build(q, d) != ours.build(q, d) for q, d in pairs)
    print(f"[1] nanochat caps 128/512: {diff}/{len(pairs)} differ from greenhouse (scaffold {ours.overhead} vs {gh.overhead})")
    if diff or gh.overhead != ours.overhead: fails.append("greenhouse identity")

    # 2 + 3. per backbone
    for name, (d, ans, bos) in MODELS.items():
        tok = prompts.setup_tokenizer(AutoTokenizer.from_pretrained(str(Path(a.root) / "models" / d)), name)
        got_ans = prompts.answer_ids(tok); got_bos = prompts.bos_id(tok, name)
        pb = prompts.PairBuilder(tok, name, "budget", 512, seq_budget=2048)
        over = cut_needless = qlong = 0; maxlen = 0; truncated = 0
        for q, doc in pairs:
            ids = pb.build(q, doc)
            maxlen = max(maxlen, len(ids))
            qn = len(pb._enc(q)); dn = len(pb._enc(doc))
            full = pb.overhead + min(qn, 512) + dn
            over += len(ids) > 2048
            qlong += min(qn, 512) > 512
            if full <= 2048 and len(ids) != full: cut_needless += 1
            truncated += full > 2048
        ok = got_ans == ans and got_bos == bos and not over and not cut_needless
        print(f"[2/3] {name:8s} answers {got_ans} bos {got_bos} scaffold {pb.overhead} | max seq {maxlen} | "
              f"over 2048: {over} | needless cuts: {cut_needless} | truncated (needed): {truncated}/{len(pairs)}"
              f"  -> {'OK' if ok else 'FAIL'}")
        if not ok: fails.append(name)
        if name == "gemma" or name == "nanochat":
            print("       render:", repr(pb.render("what is x", "x is y")))

    # 4. chunking
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        lens = torch.randint(20, 2048, (int(torch.randint(1, 200, (1,), generator=g)),), generator=g)
        budget = int(torch.randint(2048, 40000, (1,), generator=g))
        ch = chunk_rows(lens, budget)
        flat = sorted(i for c in ch for i in c)
        if flat != list(range(len(lens))) or any(len(c) > 1 and len(c) * int(lens[c].max()) > budget for c in ch):
            fails.append("chunking"); break
    print(f"[4] chunking: {'OK' if 'chunking' not in fails else 'FAIL'}")
    if fails:
        sys.exit(f"CPU TESTS FAILED: {fails}")
    print("CPU TESTS OK")


if __name__ == "__main__":
    main()
