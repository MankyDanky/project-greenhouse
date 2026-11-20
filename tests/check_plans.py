"""CPU check that each paper config rebuilds the training plan its original run used, before spending GPU hours.

  cd reranker && python ../tests/check_plans.py [config ...]       # default: every config with a recorded plan

For each config this replays the sampler the trainer builds: groups, K, the per-GPU token budget, pack_world and
seed over the prepared parquet's length columns. It then compares the number of optimizer steps with the original
run's, and the full-plan SHA256 where one was recorded (run 1's plan, shared by runs 3 and 4 and the multi-engine
rows). It needs the training parquets under data/ and the backbone tokenizers under models/.
"""
import hashlib, json, os, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.join(os.path.dirname(HERE), "reranker")
sys.path.insert(0, os.path.join(RR, "src"))
G_SHA = "04c557c35b37662326d3d96873aa13c7137cab4b614d5f9ebe7ecb272e86977e"
# optimizer steps of the original runs (from their final checkpoints) and plan hashes where recorded
EXPECTED = {
    "configs/sft/run1_seed17.yaml": (1931, G_SHA), "configs/sft/run2_seed18.yaml": (1933, None),
    "configs/sft/run3_seed17_exact_budget.yaml": (1931, G_SHA), "configs/sft/run4_seed17_gradcache.yaml": (1931, G_SHA),
    "configs/backbones/nanochat_d34_lr5e-5.yaml": (1929, None),
    "configs/backbones/minicpm5_2b_lr5e-5.yaml": (1931, G_SHA), "configs/backbones/gemma4_e2b_lr1e-5.yaml": (1931, G_SHA),
    "configs/backbones/minicpm5_2b_lr1e-5.yaml": (1931, G_SHA),
    "configs/data/rlhn680k.yaml": (5060, None), "configs/data/tevatron_msmarco.yaml": (3112, None),
    "configs/attention/causal_qpq.yaml": (1930, None), "configs/attention/causal_meanpool.yaml": (1930, None),
    "configs/attention/mntp_standard.yaml": (1931, None), "configs/attention/mntp_mask.yaml": (1928, None),
    "configs/attention/mntp_meanpool.yaml": (1930, None),
    "configs/curation/self_filter.yaml": (1918, None), "configs/curation/self_filter_self_kd.yaml": (1918, None),
}


def lengths(parquet):
    import pyarrow.parquet as pq
    t = pq.ParquetFile(parquet).read(columns=["query_len", "neg_lens"])
    n = t.column("neg_lens").to_pylist()
    return [len(r) for r in n], n, t.column("query_len").to_pylist()


def plan_sha(steps):
    flat = [[item for mb in st for item in mb] for st in steps]
    return hashlib.sha256(json.dumps(flat, separators=(",", ":")).encode()).hexdigest()


def check(path, cache):
    from dataset_nc import GroupSampler
    from stepping import pack_steps
    cfg = yaml.safe_load(open(path))
    want_steps, want_sha = EXPECTED[path]
    if "packing_scaffold" in cfg:                      # multi engine: scaffold fixed in the config, query once
        scaffold, qf = int(cfg["packing_scaffold"]), 1
    else:                                              # nanochat engine: from the prompt builder, as train_nc does
        from transformers import AutoTokenizer
        from prompt_builder_nc import PairBuilder, setup_tokenizer
        tok = setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"]))
        b = PairBuilder(tok, max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"],
                        prompt=cfg.get("prompt", "chat"), **({"bos": cfg["bos"]} if "bos" in cfg else {}))
        scaffold, qf = b.overhead, b.query_factor
    pq = cfg["train_parquet"]
    if pq not in cache:
        cache[pq] = lengths(pq)
    avail, neg, ql = cache[pq]
    world = int(cfg.get("pack_world") or 2)
    s = GroupSampler(avail, neg, ql, max_tokens=int(cfg["max_tokens_per_batch"]), max_groups=cfg["max_groups"],
                     max_doc_len=cfg["max_doc_len"], max_query_len=cfg["max_query_len"],
                     k_choices=tuple(cfg["k_choices"]), seed=cfg["seed"], scaffold=scaffold, query_factor=qf,
                     global_groups=cfg["global_batch_groups"], rank=0, world_size=world, pack_world=world)
    steps = pack_steps(s._build(0), int(cfg["global_batch_groups"]), world)
    sha = plan_sha(steps)
    ok = len(steps) == want_steps and (want_sha is None or sha == want_sha)
    print(f"{'ok  ' if ok else 'FAIL'} {path:48s} steps {len(steps)} (want {want_steps})"
          + (f"  sha {sha[:12]} (want {want_sha[:12]})" if want_sha else ""))
    return ok


def main():
    paths = sys.argv[1:] or list(EXPECTED)
    cache, ok = {}, True
    for p in paths:
        ok &= check(p, cache)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
