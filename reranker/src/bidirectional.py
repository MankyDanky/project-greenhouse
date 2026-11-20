"""
Bidirectional attention for upstream nanochat, without touching upstream's code.

Upstream's attention calls `flash_attn.flash_attn_func(q, k, v, causal=True, window_size=w)` and takes
NO mask: causality plus right padding made one unnecessary. Bidirectional attention needs one, because
every token would otherwise see padding, and packed documents would see each other.

This module swaps that one function for an SDPA implementation that honours an explicit mask built from
three things:
  * padding      -- a real token never attends to a pad, and pad rows are made self-attending so SDPA
                    cannot produce NaNs from an all-masked row;
  * documents    -- with packing, a token attends only inside its own document (block diagonal);
  * the window   -- upstream's per-layer window (L=2048, S=512 at sequence_len 2048) is one-sided; here
                    it is mirrored, so |i - j| <= window. At reranker lengths (<= 659 tokens) this is
                    nearly inert, since even S spans 512.

Usage:
    enable(nanochat_src)                  # once, after upstream is imported
    with context(pad_mask, doc_ids):      # per forward
        ...model(...)

`causal=True` from upstream is deliberately ignored inside the context; outside it, the original
function runs unchanged, so the causal path is untouched.
"""
import contextlib
import os
import threading

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_state = threading.local()
_orig = None
# EFFICIENT_ATTENTION handles fp32 with a boolean mask and keeps memory linear in T; MATH would
# materialise [B, H, T, T] scores per layer.
_BACKEND = SDPBackend.EFFICIENT_ATTENTION


def _ctx():
    return getattr(_state, "ctx", None)


def set_context(pad_mask, doc_ids=None):
    """Install the mask state for the next forward AND its backward.

    Deliberately NOT a context manager: gradient checkpointing recomputes each block during BACKWARD,
    long after a `with` block would have exited. If the state were gone by then, the patched attention
    would silently fall back to upstream's causal path -- wrong maths, and a CheckpointError from the
    mismatched tensor count. The state simply persists until the next forward replaces it; the masks
    are a few MB of bool.
    """
    _state.ctx = {"pad": pad_mask.bool(), "doc": doc_ids, "cache": {}, "calls": 0}


def clear_context():
    _state.ctx = None


@contextlib.contextmanager
def context(pad_mask, doc_ids=None):
    """Scoped variant, for tests. Training uses set_context (see above)."""
    prev = _ctx()
    set_context(pad_mask, doc_ids)
    try:
        yield
    finally:
        _state.ctx = prev


_VARIANT = os.environ.get("BIDIR_MASK_VARIANT", "full")
_VARIANTS = ("full", "causal", "random")


def _parse_layers(spec):
    """"" -> every layer (the variant applies throughout);  "0-8,17,20-22" -> just those."""
    out = set()
    for part in filter(None, (x.strip() for x in spec.split(","))):
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# Which layers the variant applies to; the rest keep the trained mask. EVAL ONLY: the layer index is
# the attention call count within a forward, which gradient checkpointing would advance a second time
# during recomputation. Empty (the default) leaves the call counter unused entirely.
_ABLATE_LAYERS = _parse_layers(os.environ.get("BIDIR_ABLATE_LAYERS", ""))


def set_ablate_layers(spec):
    global _ABLATE_LAYERS
    _ABLATE_LAYERS = _parse_layers(spec) if isinstance(spec, str) else set(spec)
    c = _ctx()
    if c is not None:
        c["cache"].clear()


def set_variant(v):
    """Restrict which (query, key) pairs the mask allows, for the forward-attention ablation.

      full    (default) the mask the model was trained with: |i - j| <= window
      causal  drop every forward pair (j > i), keeping the one-sided window the backbone pretrained
              with. This is the intervention: same weights, no access to later tokens.
      random  drop a uniformly random subset of IN-WINDOW pairs of the same size as the forward set,
              diagonal always kept. This is the control: it removes a comparable number of
              connections without targeting direction, so "nDCG fell" cannot be explained by
              perturbation size alone. Deterministic given (B, T, window).

    Not offered: an anti-causal mask. It would leave the last real token attending to itself only --
    and that token IS the readout -- so it destroys the score by construction rather than measuring
    reliance on backward context.
    """
    global _VARIANT
    assert v in _VARIANTS, f"variant must be one of {_VARIANTS}, got {v!r}"
    _VARIANT = v
    c = _ctx()                       # _state.ctx exists only inside a context
    if c is not None:
        c["cache"].clear()


def _allowed(window, variant=None):
    """[B, 1, T, T] bool: which (query, key) pairs may attend, for this layer's window and variant."""
    c = _ctx()
    variant = _VARIANT if variant is None else variant
    key = (window, variant)
    if key in c["cache"]:
        return c["cache"][key]
    pad = c["pad"]                                    # [B, T]
    B, T = pad.shape
    dev = pad.device
    m = pad[:, None, :] & pad[:, :, None]             # both tokens real
    if c["doc"] is not None:
        d = c["doc"]
        m &= d[:, None, :] == d[:, :, None]           # same document
    idx = torch.arange(T, device=dev)
    delta = idx[None, :] - idx[:, None]                        # key j - query i
    in_window = delta.abs() <= window if (window is not None and 0 <= window < T) \
        else torch.ones_like(delta, dtype=torch.bool)
    m &= in_window
    if variant == "causal":
        m &= delta <= 0                                        # no forward pairs
    elif variant == "random":
        n_fwd = int((in_window & (delta > 0)).sum())           # how many the causal arm removes
        n_in = int(in_window.sum())
        g = torch.Generator(device="cpu").manual_seed(0x5EED ^ (T * 1000003) ^ int(window))
        keep = torch.rand(T, T, generator=g).to(dev) >= (n_fwd / max(n_in, 1))
        keep |= torch.eye(T, dtype=torch.bool, device=dev)     # a query always sees itself
        m &= keep
    m = m.unsqueeze(1)                                # [B, 1, T, T]
    # a fully-masked row (a pad token) would make softmax produce NaN: let pads attend to themselves
    eye = torch.eye(T, dtype=torch.bool, device=dev)[None, None]
    m = m | (~pad[:, None, None, :].transpose(-1, -2) & eye)
    c["cache"][key] = m
    return m


def prime(windows):
    """Build every layer's mask BEFORE the blocks run.

    The masks are cached per window. If that cache were filled lazily inside a gradient-checkpointed
    block, the original forward would create tensors that the recomputation pass finds already cached,
    and torch.utils.checkpoint fails with "A different number of tensors was saved during the original
    forward and recomputation". Priming makes both passes pure lookups."""
    c = _ctx()
    if c is None:
        return
    wanted = {_VARIANT, "full"} if _ABLATE_LAYERS else {_VARIANT}
    for w in windows:
        for v in wanted:
            _allowed(int(w) if w is not None else -1, v)


def _bidirectional_attn(q, k, v, causal=False, window_size=(-1, -1)):
    """Drop-in for upstream's flash_attn_func. q, k, v: [B, T, H, D]."""
    c = _ctx()
    if c is None:                                     # outside a context: upstream's own behaviour
        return _orig(q, k, v, causal=causal, window_size=window_size)
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))            # -> [B, H, T, D]
    window = window_size[0] if isinstance(window_size, (tuple, list)) else window_size
    variant = _VARIANT
    if _ABLATE_LAYERS:                       # blocks run in order, so the call index is the layer
        layer = c["calls"]
        c["calls"] = layer + 1
        variant = _VARIANT if layer in _ABLATE_LAYERS else "full"
    mask = _allowed(int(window) if window is not None else -1, variant)
    # Pin the backend. Left to choose, SDPA picks a different kernel in a no-grad forward than in the
    # grad-enabled recomputation of gradient checkpointing, which saves a different number of tensors
    # (43 vs 42) and makes torch.utils.checkpoint raise -- determinism_check="none" does not suppress
    # that count check in torch 2.14. One backend for both passes removes the variance.
    with sdpa_kernel(_BACKEND):
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask,
                                           enable_gqa=qh.size(1) != kh.size(1))
    return y.transpose(1, 2).contiguous()                           # back to [B, T, H, D]


def enable(nanochat_src=None):
    """Patch upstream's attention entry point once; a no-op if already patched."""
    global _orig
    import sys
    if nanochat_src and nanochat_src not in sys.path:
        sys.path.insert(0, nanochat_src)
    import nanochat.flash_attention as fa
    if _orig is None:
        _orig = fa.flash_attn.flash_attn_func
        fa.flash_attn.flash_attn_func = _bidirectional_attn
        fa.USE_FA3 = False        # the FA3 path would bypass the patch entirely
    return True


def reference_mask(pad_mask, doc_ids, window, T=None):
    """Brute-force mask, for tests: loops instead of broadcasting."""
    pad = pad_mask.bool()
    B, T = pad.shape
    out = torch.zeros(B, 1, T, T, dtype=torch.bool)
    for b in range(B):
        for i in range(T):
            if not pad[b, i]:
                out[b, 0, i, i] = True           # pad rows self-attend (NaN guard)
                continue
            for j in range(T):
                ok = bool(pad[b, j])
                if ok and doc_ids is not None:
                    ok = int(doc_ids[b, i]) == int(doc_ids[b, j])
                if ok and window is not None and 0 <= window < T:
                    ok = abs(i - j) <= window
                out[b, 0, i, j] = ok
    return out
