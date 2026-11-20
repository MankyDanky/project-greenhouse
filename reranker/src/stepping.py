"""
Global batch decoupled from local batch (shared by jina_v2 and nanochat trainers).

LOCAL batch  = one microbatch on one GPU, sized by a token budget for memory.
GLOBAL batch = the groups that make up ONE optimizer step, fixed by config
               (`global_batch_groups`), independent of GPU memory and GPU count.

Round 1 tied the two together (global = world x grad_accum x whatever fit), so
the step size floated with the token budget: jina averaged ~61 groups/step
(ranging 3x with the per-microbatch group count) and nanochat ~16-20.

  pack_steps  groups consecutive microbatches into steps of ~global_batch_groups.
              The microbatch count per step is a multiple of world, so every rank
              runs the same number of backward passes (unequal counts deadlock DDP).
  shard_steps deals each step's microbatches round-robin to the ranks and gives
              each one a loss weight world*B_micro/G_step. The gradient is then
              the exact mean over every group in the step, whatever the
              microbatch sizes. A plain mean of microbatch means would overweight
              groups that land in small microbatches.
  StepMeter   sums metrics over all microbatches of a step and all ranks, so one
              logged point describes one optimizer step, not one ~7-group
              microbatch. 68% of round-1 jina's logged acc variance was binomial
              noise from exactly that.
"""
import math

import torch
import torch.distributed as dist


def pack_steps(mbs, global_groups, world):
    """mbs: list of microbatches (each a list of items). Returns a list of steps.
    Each step's microbatch count is a multiple of world, and its total group
    count is the closest such total to global_groups. The tail is dropped."""
    steps, cur, g = [], [], 0
    prev_g = None                         # total at the previous world-multiple boundary
    for mb in mbs:
        cur.append(mb); g += len(mb)
        if len(cur) % world:
            continue
        if g >= global_groups:
            if prev_g and (global_groups - prev_g) < (g - global_groups):
                steps.append(cur[:-world])          # the previous boundary was closer
                cur = cur[-world:]; g = sum(len(x) for x in cur)
                prev_g = g if g < global_groups else None
                if g >= global_groups:              # a single round already fills a step
                    steps.append(cur); cur, g, prev_g = [], 0, None
            else:
                steps.append(cur); cur, g, prev_g = [], 0, None
        else:
            prev_g = g
    return steps


def shard_steps(steps, rank, world):
    """-> (this rank's microbatches, meta[(loss_weight, is_last_of_step, G_step)])."""
    batches, meta = [], []
    for st in steps:
        G = sum(len(mb) for mb in st)
        mine = st[rank::world]
        for j, mb in enumerate(mine):
            batches.append(mb)
            meta.append((world * len(mb) / G, j == len(mine) - 1, G))
    return batches, meta


class StepMeter:
    """Group-weighted sums over one optimizer step, all-reduced across ranks.
    Each key keeps its own group count, so a key missing from some microbatches
    (e.g. kd when a microbatch has no teacher rows) is averaged over the groups
    that actually reported it instead of being diluted by zeros."""
    def __init__(self, keys, device):
        self.keys, self.dev = list(keys), device
        self.reset()

    def reset(self):
        k = len(self.keys)
        self.s = torch.zeros(2 * k + 1, device=self.dev, dtype=torch.float64)

    @torch.no_grad()
    def add(self, n, vals):
        k = len(self.keys)
        v = torch.zeros(2 * k + 1, device=self.dev, dtype=torch.float64)
        for j, key in enumerate(self.keys):
            if key in vals:
                v[j] = torch.as_tensor(vals[key], device=self.dev, dtype=torch.float64).reshape(()) * n
                v[k + j] = n
        v[-1] = n
        self.s += v

    def reduce(self, ddp):
        s = self.s.clone()
        if ddp: dist.all_reduce(s)
        k = len(self.keys)
        out = {key: float(s[j]) / float(s[k + j]) for j, key in enumerate(self.keys) if float(s[k + j]) > 0}
        out["groups"] = int(round(float(s[-1])))
        self.reset()
        return out


# ---------------------------------------------------------------- precision
# ONE global switch for numerics. Every precision bug so far came from a piece of
# the model silently running in bf16 (round-1 cosine, bf16 AdamW moments, the
# nanochat yes/no head recast by autocast, the jina projector). There is no
# autocast anywhere any more: parameters, activations, and every op run in fp32.
#   precision: fp32  -- IEEE fp32 everywhere, TF32 tensor cores OFF
#   precision: tf32  -- fp32 storage and accumulation; matmul/conv INPUTS rounded to
#                       TF32 (10-bit mantissa, 8x finer than bf16). Stored values
#                       never lose precision, so errors do not accumulate across
#                       steps. ~2-8x faster than strict fp32 on tensor-core GPUs.
PRECISIONS = ("fp32", "tf32")


def setup_precision(cfg, log=print):
    p = cfg.get("precision", "fp32")
    if p not in PRECISIONS:
        raise SystemExit(f"precision must be one of {PRECISIONS}, got {p!r} (bf16 was removed)")
    tf32 = p == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.set_default_dtype(torch.float32)
    log(f"precision={p}: fp32 params/activations/optimizer, no autocast, "
        f"TF32 matmul {'ON' if tf32 else 'OFF'}")
    return torch.float32


def assert_fp32_params(model):
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.is_floating_point() and t.dtype != torch.float32]
    if bad:
        raise SystemExit(f"non-fp32 parameters/buffers ({len(bad)}): {bad[:5]}")


def _tensors(out):
    """Every tensor inside a module output: tensor, tuple/list, dict/ModelOutput, dataclass."""
    if torch.is_tensor(out): return [out]
    if isinstance(out, (tuple, list)): return [t for o in out for t in _tensors(o)]
    if isinstance(out, dict): return [t for o in out.values() for t in _tensors(o)]
    if hasattr(out, "__dataclass_fields__"): return [t for o in vars(out).values() for t in _tensors(o)]
    return []


class FP32Guard:
    """Forward hooks on EVERY module; any floating output that is not fp32, or any
    forward running under autocast, raises. Armed for the first microbatch only,
    so a regression of this bug class fails in the first minute of a run."""
    def __init__(self, model):
        self.handles = [m.register_forward_hook(self._check) for m in model.modules()]
        self.seen = 0

    def _check(self, mod, inp, out):
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise RuntimeError(f"autocast is enabled inside {type(mod).__name__}")
        for t in _tensors(out):
            if t.is_floating_point() and t.dtype != torch.float32:
                raise RuntimeError(f"{type(mod).__name__} produced {t.dtype}, expected float32")
        self.seen += 1

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []
        return self.seen


# ---------------------------------------------------------------- LR schedule
def lr_at(step, total, peak, warmup=150):
    """Linear warmup to `peak` over `warmup` steps, then cosine decay to 0 at `total`.
    Every parameter trains from step 0 (no frozen phase); a pure function of the step,
    so a resume needs no scheduler state."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))
