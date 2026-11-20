"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager
from completion_evidence import released_initial_evidence
from production_runtime import (
    ProductionRuntime, REFERENCE_BATCH, add_arguments, validate_arguments,
    restore_optimizer_for_production, preserve_rng_state, resume_loader_state, FINAL_TOKENS,
)

import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--deadline-margin-s", type=int, default=900, help="(d34cm) save a checkpoint and exit this many seconds before SLURM_JOB_END_TIME (<=0 disables)")
parser.add_argument("--stop-at-step", type=int, default=-1, help="(d34cm) batch-ramp stage boundary: save a checkpoint at this global step and exit 0 (-1 disables)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
add_arguments(parser)
# (bidir) Bidirectional MNTP continuation (LLM2Vec-style masked next-token prediction).
parser.add_argument("--bidir-origin-step", type=int, required=True, help="source causal checkpoint step the bidirectional run starts from")
parser.add_argument("--bidir-mask-prob", type=float, default=0.2, help="fraction of input positions replaced by the mask token")
parser.add_argument("--bidir-mask-token", type=str, default="<|output_end|>", help="special token used as [MASK] (never seen in pretraining)")
parser.add_argument("--bidir-rewarmup-steps", type=int, default=200, help="linear LR re-warmup after the fresh-optimizer start")
parser.add_argument("--bidir-seed", type=int, default=20260927, help="seed for the per-rank, per-step mask generator")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
import nanochat.gpt as _bidir_gpt
assert os.environ.get("NANOCHAT_BIDIRECTIONAL") == "1" and _bidir_gpt.BIDIRECTIONAL, "bidir trainer requires NANOCHAT_BIDIRECTIONAL=1 and the patched snapshot"
assert args.core_metric_every == -1 and args.sample_every == -1, "CORE/sampling are causal evaluations; invalid under bidirectional attention"
assert 0.0 < args.bidir_mask_prob < 1.0
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
validate_arguments(args, ddp_world_size)
user_config = vars(args).copy()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    # (bidir) Fail fast if the FA3 build mishandles non-causal (left,right) windows: compare with the SDPA band reference.
    import nanochat.flash_attention as _fa_mod
    _g = torch.Generator(device=device); _g.manual_seed(0)
    _q, _k, _v = (torch.randn(2, 1024, 4, 128, device=device, dtype=torch.bfloat16, generator=_g) for _ in range(3))
    for _w in ((512, 512), (2048, 2048)):
        _y1 = _fa_mod._fa3.flash_attn_func(_q, _k, _v, causal=False, window_size=_w)
        _y1 = _y1[0] if isinstance(_y1, tuple) else _y1
        _y2 = _fa_mod._sdpa_attention(_q.transpose(1, 2), _k.transpose(1, 2), _v.transpose(1, 2), _w, False, causal=False).transpose(1, 2)
        _err = (_y1.float() - _y2.float()).abs().max().item()
        print0(f"(bidir) FA3 vs SDPA non-causal window {_w}: max abs diff {_err:.2e}")
        assert _err < 3e-2, f"FA3 non-causal window {_w} disagrees with the SDPA reference ({_err})"
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    _optim_path = os.path.join(args.production_checkpoint_source_dir, f"optim_{args.resume_from_step:06d}_rank{ddp_rank:d}.pt")
    fresh_optimizer = not os.path.exists(_optim_path)
    if fresh_optimizer:
        assert args.resume_from_step == args.bidir_origin_step, "Only the causal origin checkpoint may start without optimizer state"
        print0(f"(bidir) Source has no optimizer state: starting a fresh optimizer at step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(args.production_checkpoint_source_dir, args.resume_from_step, device, load_optimizer=not fresh_optimizer, rank=ddp_rank)
    assert meta_data["step"] == args.resume_from_step
    assert meta_data["model_config"] == model_config_kwargs, "Architecture differs from the protected source"
    assert meta_data["total_batch_size"] == args.total_batch_size
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")
# (bidir) The causal run kept every learning rate at the value derived for its initial 262,144-token batch.
batch_lr_scale = (262144 / B_REF) ** 0.5
print0(f"(bidir) Using the run's fixed learning-rate scale {batch_lr_scale:.6f} (262K-token derivation)")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
# Production keeps the original 1M reference weight-decay amplitude.
assert args.weight_decay == meta_data["user_config"]["weight_decay"]
weight_decay_scaled = args.weight_decay * math.sqrt(REFERENCE_BATCH / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming and not fresh_optimizer:
    # Preserve the selected winner's optimizer state and initial learning rates.
    loaded_optimizer_settings = restore_optimizer_for_production(
        optimizer, optimizer_data, args.production_matrix_lr)
    del optimizer_data
    print0("Production resume: optimizer buffers and all base learning rates preserved from winner")
elif resuming:
    from production_runtime import group_settings
    loaded_optimizer_settings = group_settings(optimizer)
    _muon = {g["initial_lr"] for g in loaded_optimizer_settings if g["kind"] == "muon"}
    assert len(_muon) == 1 and math.isclose(_muon.pop(), meta_data["production"]["actual_matrix_lr"], rel_tol=1e-12)
    print0("(bidir) Fresh optimizer with the causal run's learning rates; buffers start at zero")

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
initial_completion_evidence = None
if "production" not in meta_data:
    initial_completion_evidence = released_initial_evidence(
        meta_data, args.production_checkpoint_source_dir, args.production_winner_checkpoint_dir,
        args.production_release, args.production_winner_completion,
        args.production_winner_completion_sha256)
dataloader_resume_state_dict = resume_loader_state(meta_data, ddp_rank, initial_completion_evidence)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = FINAL_TOKENS # Original ramp tokens plus canonical 1M continuation tokens
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_tokens / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# (bidir) MNTP masking. Positions (never BOS, never position 0) are replaced by the mask token with probability p;
# the output at position t predicts the ORIGINAL token at t+1 only when t+1 was masked (LLM2Vec MNTP). Every
# selected position is replaced (no 80/10/10 keep), so the target token is never visible to its predictor.
BIDIR_MASK_ID = tokenizer.encode_special(args.bidir_mask_token)
BIDIR_BOS_ID = tokenizer.get_bos_token_id()
print0(f"(bidir) MNTP mask token {args.bidir_mask_token!r} id {BIDIR_MASK_ID}, mask prob {args.bidir_mask_prob}")
from bidir_mntp import make_mntp
bidir_mntp = make_mntp(BIDIR_MASK_ID, BIDIR_BOS_ID, args.bidir_mask_prob, args.bidir_seed, ddp_rank)

@torch.no_grad()
def evaluate_mntp(eval_model, loader, steps):
    """Held-out masked-token loss (nats) and accuracy with fixed, step-independent masks."""
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(steps):
        vx, vy = next(loader)
        x_in, tgt = bidir_mntp(vx, vy, 10**9, i)
        logits = eval_model(x_in)
        sel = tgt != -1
        lg = logits[sel].float(); tt = tgt[sel]
        loss_sum += F.cross_entropy(lg, tt, reduction="sum").double()
        correct += (lg.argmax(-1) == tt).sum().double()
        count += sel.sum().double()
    if is_ddp_initialized():
        for t_ in (loss_sum, correct, count):
            dist.all_reduce(t_, op=dist.ReduceOp.SUM)
    return (loss_sum / count).item(), (correct / count).item(), int(count.item())

bidir_meta = {"objective": "mntp", "attention": "bidirectional; sliding windows (w,0)->(w,w)",
              "mask_prob": args.bidir_mask_prob, "mask_token": args.bidir_mask_token, "mask_token_id": BIDIR_MASK_ID,
              "origin_step": args.bidir_origin_step, "rewarmup_steps": args.bidir_rewarmup_steps,
              "val_metric": "val_bpb field holds held-out MNTP masked-token cross-entropy (nats)"}

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

production = ProductionRuntime(args, ddp_rank, ddp_world_size, checkpoint_dir, meta_data,
                           optimizer, loaded_optimizer_settings, dataloader_state_dict, initial_completion_evidence)
print0(f"Production: {production.clock.remaining:,} remaining tokens; "
       f"{production.clock.updates_to_finish:,} updates; "
       f"canonical horizon {num_iterations:,}")

# Go!
while True:
    added_tokens = production.added_tokens(step)
    canonical_step = production.canonical_step(step)
    last_step = production.clock.finished(step)
    flops_so_far = num_flops_per_token * production.clock.total(step)

    # (d34cm) graceful stop before the Slurm wall-time limit OR at a batch-ramp stage boundary (--stop-at-step):
    # rank 0 decides, all ranks follow; save a checkpoint at this step (unless one was just written) and exit 0 so
    # the next job in the chain resumes here (possibly with a different --total-batch-size). Training math is untouched.
    if args.deadline_margin_s > 0:
        stop_flag = torch.zeros(1, device=device)
        if master_process:
            _end_time = os.environ.get("SLURM_JOB_END_TIME")
            if args.deadline_margin_s > 0 and _end_time is not None and time.time() > float(_end_time) - args.deadline_margin_s:
                stop_flag.fill_(1.0)
            if args.stop_at_step > 0 and step >= args.stop_at_step:
                stop_flag.fill_(2.0)
        if is_ddp_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() >= 1.0:
            saved_this_step = production.last_saved_step == step
            if not saved_this_step:
                production.save_checkpoint(
                    save_checkpoint,
                    step,
                    orig_model.state_dict(),
                    optimizer.state_dict(),
                    {
                        "step": step,
                        "val_bpb": val_bpb,
                        "model_config": model_config_kwargs,
                        "user_config": user_config,
                        "device_batch_size": args.device_batch_size,
                        "max_seq_len": args.max_seq_len,
                        "total_batch_size": total_batch_size,
                        "dataloader_state_dict": dataloader_state_dict,
                        "bidir": bidir_meta,
                        "loop_state": {
                            "min_val_bpb": min_val_bpb,
                            "smooth_train_loss": smooth_train_loss,
                            "total_training_time": total_training_time,
                        },
                    },
                )
            production.emit("partial", reason="deadline_stop", step=step,
                            canonical_step=production.canonical_step(step),
                            total_training_tokens=production.clock.total(step))
            print0(f"PRODUCTION DEADLINE: checkpoint at actual step {step} saved")
            break

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step == args.resume_from_step or canonical_step % args.eval_every == 0):
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with preserve_rng_state():
            val_loader = build_val_loader()
            with disable_fp8(model):
                val_bpb, val_acc, val_n = evaluate_mntp(model, val_loader, eval_steps)
        production.record_eval(step, val_bpb, optimizer, dataloader_state_dict)
        production.emit("mntp_eval", step=step, canonical_step=production.canonical_step(step),
                        val_mntp_loss=val_bpb, val_mntp_acc=val_acc, val_masked_tokens=val_n)
        print0(f"Step {step:05d} | Validation MNTP loss: {val_bpb:.6f} | acc: {val_acc:.4f} | masked: {val_n:,}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (canonical_step > 0 and canonical_step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (canonical_step > 0 and canonical_step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and canonical_step % args.save_every == 0):
        production.save_checkpoint(
            save_checkpoint,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "bidir": bidir_meta,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        production.finish(step, optimizer)
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    update_tokens = production.clock.update_tokens(step)
    assert update_tokens % world_tokens_per_fwdbwd == 0
    update_grad_accum_steps = update_tokens // world_tokens_per_fwdbwd
    for micro_step in range(update_grad_accum_steps):
        production.consumed_batch(x, y)
        x_in, tgt = bidir_mntp(x, y, step, micro_step)
        loss = model(x_in, tgt)
        train_loss_sum.add_(loss.detach().double())
        loss = loss / update_grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # Aggregate the unscaled loss over every microbatch and every rank before
    # any update, so a nonfinite loss cannot corrupt the checkpoint trajectory.
    if is_ddp_initialized():
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
    train_loss_f = (train_loss_sum / (update_grad_accum_steps * ddp_world_size)).item()
    if not math.isfinite(train_loss_f):
        production.emit("failed", reason="nonfinite_train_loss", step=step)
        raise FloatingPointError("Nonfinite training loss; optimizer update blocked")
    # Advance every LR, Muon momentum and weight decay on the canonical token clock.
    schedule_step = production.canonical_step(step)
    lrm = get_lr_multiplier(schedule_step)
    lrm = lrm * min(1.0, (step - args.bidir_origin_step + 1) / args.bidir_rewarmup_steps)
    muon_momentum = get_muon_momentum(schedule_step)
    muon_weight_decay = get_weight_decay(schedule_step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    production.assert_static_optimizer_settings(optimizer)
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    production.record_step(step, train_loss_f, dt, optimizer, lrm)
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * production.canonical_step(step + 1) / num_iterations
    tok_per_sec = int(update_tokens / dt)
    flops_per_sec = num_flops_per_token * update_tokens / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - args.resume_from_step + 1
    if steps_done > 0:
        avg_time_per_step = (time.time() - production.started_at) / steps_done
        remaining_steps = production.clock.final_actual - step - 1
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d} canonical {canonical_step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

if not production.complete_written:
    print0("Production leg ended before the token horizon; BASE_DONE was not written")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
