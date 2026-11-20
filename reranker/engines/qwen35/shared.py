"""Direct imports from the existing NanoChat LCE implementation (no vendored copies)."""
import sys
from pathlib import Path
SOURCE = Path(__file__).resolve().parents[2] / "src"   # reranker/src: the shared LCE core
sys.path.insert(0, str(SOURCE))
import train_nc as reference
from stepping import StepMeter, setup_precision, assert_fp32_params, FP32Guard, lr_at, pack_steps
from dataset_nc import PairGroups, GroupCollator, GroupSampler
from losses_nc import total_loss


def validate_native_config(cfg):
    """Reject incompatible architecture switches before loading data or weights."""
    if cfg.get("backbone_impl", "hf") != "hf" or cfg.get("lora", False):
        raise ValueError("This recipe requires native full-text Qwen3.5 fine-tuning.")
    if cfg.get("prompt", "nochat") != "nochat" or cfg.get("query_factor", 1) != 1:
        raise ValueError("Qwen3.5 uses the single-query nochat prompt.")
    if cfg.get("bidirectional", False) or cfg.get("unfreeze_after"):
        raise ValueError("Use causal attention and train every text parameter from step zero.")
    if cfg.get("optimizer_state_sharding", False):
        raise ValueError("This recipe uses ordinary unsharded FP32 AdamW.")
    if cfg.get("precision", "tf32") not in ("fp32", "tf32"):
        raise ValueError("Training requires FP32 parameters and activations (optional TF32).")
    for key in ("execution_tokens_per_batch", "max_tokens_per_batch", "global_batch_groups", "epochs"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def summarize_plan(sampler, cfg):
    """Use the sibling LCE planner; reference checks are optional for new data."""
    import hashlib
    import json
    plan = [[item for mb in step for item in mb] for step in pack_steps(
        sampler._build(0), cfg["global_batch_groups"], cfg.get("pack_world") or sampler.world_size)]
    digest = hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()
    if not plan:
        raise ValueError("No trainable groups remain after candidate filtering")
    if cfg.get("expected_plan_hash") and digest != cfg["expected_plan_hash"]:
        raise ValueError(f"Packing plan mismatch: {digest} != {cfg['expected_plan_hash']}")
    if cfg.get("expected_steps") and len(plan) != cfg["expected_steps"]:
        raise ValueError(f"Step count mismatch: {len(plan)} != {cfg['expected_steps']}")
    return {"steps": len(plan), "groups": sum(map(len, plan)),
            "pairs": sum(k + 1 for step in plan for i, k, epoch in step), "sha256": digest,
            "packing_length_tokenizer": "prepared input length columns; execution uses Qwen tokens"}
