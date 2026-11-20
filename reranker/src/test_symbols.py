"""Assert every expected symbol resolves. A string-slice edit once silently
deleted total_loss from jina's losses.py: ast.parse passed, the unit test only
imported one function, and the failure surfaced only on a compute node as
`ImportError` after the allocation started. This catches that in seconds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EXPECTED = {
    "stepping":          ["pack_steps", "shard_steps", "StepMeter", "setup_precision",
                          "assert_fp32_params", "FP32Guard", "lr_at"],
    "prompt_builder_nc":    ["PairBuilder", "yes_no_ids", "setup_tokenizer", "INSTRUCTION"],
    "modeling_nanochat_rr": ["NanochatReranker", "RerankOutput"],
    "dataset_nc":           ["PairGroups", "GroupCollator", "GroupSampler"],
    "losses_nc":            ["infonce", "kd_loss", "total_loss"],
    "train_nc":             ["main", "save_ckpt",
                             "find_latest_ckpt", "prune_ckpts",
                             "make_ddp", "token_budget", "packing_signature",
                             "gradcache_backward"],
}
bad = []
for mod, names in EXPECTED.items():
    try:
        m = __import__(mod)
    except Exception as e:
        bad.append(f"{mod}: import failed ({type(e).__name__}: {e})"); continue
    miss = [n for n in names if not hasattr(m, n)]
    if miss: bad.append(f"{mod}: missing {miss}")
    else: print(f"  {mod:22s} OK ({len(names)} symbols)")
if bad:
    print("\nFAILURES:"); [print("  " + b) for b in bad]; sys.exit(1)
print("  all module symbols resolve")
