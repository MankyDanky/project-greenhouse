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
    """Reject NanoChat-only switches rather than silently ignoring them."""
    if cfg.get("backbone_impl", "hf") != "hf" or cfg.get("lora", False):
        raise ValueError("This entry point requires native full-text Gemma fine-tuning.")
    if cfg.get("prompt", "nochat") != "nochat" or cfg.get("query_factor", 1) != 1:
        raise ValueError("Gemma uses its native single-query nochat prompt.")
    if cfg.get("bidirectional", False):
        raise ValueError("NanoChat bidirectional attention is not implemented for native Gemma.")
