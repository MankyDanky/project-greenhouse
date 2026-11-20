"""
Pointwise yes/no reranker on an UPSTREAM nanochat checkpoint (karpathy/nanochat @ 92d63d4).

modeling_nanochat_rr.py wraps the HuggingFace `NanoChat` port, which models Karpathy's d34 release
(@ f37d45c). Newer upstream checkpoints -- e.g. castorini/nanochat-4b-climbmix -- add architecture the
HF port does not have (value embeddings + gates, per-layer resid/x0 mixing, embedding smearing, a
mid-layer backout, sliding-window attention), so they cannot be converted to it. This wrapper runs
upstream's own `GPT` class instead, and scores exactly as the HF wrapper does:

    score = logit(yes) - logit(no) at the last non-pad token

The trunk below mirrors `GPT.forward` up to the final norm; only the full-vocabulary `lm_head` is
replaced by the two rows we score with (kept in fp32, so the difference is exact). The compute dtype is chosen per use: fp32 for training (our policy), bf16 for evaluation (what every
published number used), with the two scoring rows always in fp32 so the logit difference is exact.
Upstream reads it from NANOCHAT_DTYPE at import, so it is set before the first import and cannot be
changed afterwards in the same process. Attention is pinned to upstream's SDPA path in both cases:
its FA3 kernels are bf16-only, so otherwise the dtype would silently change the kernel too.

Padding: sequences are RIGHT-padded and attention is causal, so a real token never attends to a pad,
and the score is read at `attention_mask.sum(1) - 1`. No attention mask is needed, which is what
upstream's attention supports.
"""
import glob
import json
import os
import types
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class RerankOutput:
    scores: torch.Tensor          # [B]  logit(yes) - logit(no), or the mean-pool head's scalar
    yes_no: torch.Tensor = None   # [B, 2] fp32; None when score_mode == "meanpool"


_DTYPE_ENV = {torch.float32: "float32", torch.bfloat16: "bfloat16"}


def _load_upstream(nanochat_src, dtype):
    """Import upstream's gpt.py with COMPUTE_DTYPE pinned to `dtype`, and force its SDPA path."""
    import sys
    want = _DTYPE_ENV[dtype]
    if "nanochat.common" in sys.modules:          # already imported: the dtype is baked in
        from nanochat.common import COMPUTE_DTYPE
        assert COMPUTE_DTYPE == dtype, (
            f"upstream was imported with COMPUTE_DTYPE={COMPUTE_DTYPE}, cannot switch to {dtype} in this process")
    else:
        os.environ["NANOCHAT_DTYPE"] = want
    if nanochat_src not in sys.path:
        sys.path.insert(0, nanochat_src)
    from nanochat.common import COMPUTE_DTYPE
    assert COMPUTE_DTYPE == dtype, (
        f"upstream COMPUTE_DTYPE is {COMPUTE_DTYPE}; set NANOCHAT_DTYPE={want} before the first import")
    import nanochat.flash_attention as fa
    fa.USE_FA3 = False              # same kernel in fp32 and bf16 (FA3 is bf16-only, so it would differ)
    from nanochat.gpt import GPT, GPTConfig, norm
    return GPT, GPTConfig, norm


class UpstreamNanochatReranker(nn.Module):
    """model_dir holds an upstream checkpoint: model_<step>.pt + meta_<step>.json (+ tokenizer.json)."""

    def __init__(self, model_dir, yes_id, no_id, dtype=torch.float32, nanochat_src=None,
                 attn_implementation=None, bidirectional=False, keep_lm_head=False,
                 upstream_precision=False, readout_offset=0, score_mode="lm_rows"):
        super().__init__()
        assert dtype in _DTYPE_ENV, f"dtype must be fp32 (training) or bf16 (evaluation), got {dtype}"
        nanochat_src = nanochat_src or os.environ.get("NANOCHAT_SRC")
        assert nanochat_src, "set nanochat_src (or NANOCHAT_SRC) to a nanochat checkout @ 92d63d4"
        GPT, GPTConfig, self._norm = _load_upstream(nanochat_src, dtype)

        meta_path = glob.glob(os.path.join(model_dir, "meta_*.json"))
        ckpt_path = glob.glob(os.path.join(model_dir, "model_*.pt"))
        assert len(meta_path) == 1 and len(ckpt_path) == 1, (meta_path, ckpt_path)
        mc = json.load(open(meta_path[0]))["model_config"]
        self.meta = mc
        # ve_every (value embeddings on every ve_every-th layer, counted back from the last) exists only in
        # checkpoints from our sweep/full-run code (e.g. the d32 x 1152 backbone, ve_every 4); absent means
        # upstream's alternating layers. It needs a nanochat_src whose GPTConfig has the field (the strict load
        # below fails loudly otherwise).
        cfg = GPTConfig(**{k: mc[k] for k in
                           ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd")
                           if k in mc},
                        **({"window_pattern": mc["window_pattern"]} if "window_pattern" in mc else {}),
                        **({"ve_every": mc["ve_every"]} if "ve_every" in mc else {}))
        with torch.device("meta"):          # upstream's __init__ is meta-device only (shapes/dtypes)
            self.backbone = GPT(cfg)
        sd = torch.load(ckpt_path[0], map_location="cpu", weights_only=True, mmap=True)
        if upstream_precision:
            # Upstream's OWN mixed policy, used for continued pretraining (adaptation):
            #   matrices          fp32 master weights; Linear.forward casts the weight to the input
            #                     dtype, so the matmuls themselves run in bf16
            #   wte, value_embeds bf16 (that is how the checkpoint stores them)
            #   activations/rope  bf16 via COMPUTE_DTYPE
            #   logits and loss   fp32 (softcap and cross-entropy)
            # Keeping the checkpoint's own dtypes is both faithful and much cheaper than upcasting.
            sd = {k: v.contiguous() for k, v in sd.items()}
            self.backbone.load_state_dict(sd, strict=True, assign=True)
        else:
            # everything fp32, as the reranker runs (its policy after four bf16 bugs)
            sd = {k: v.float().contiguous() for k, v in sd.items()}
            self.backbone.load_state_dict(sd, strict=True, assign=True)
            self.backbone = self.backbone.to(dtype)
        head = self.backbone.lm_head.weight.data[[yes_id, no_id], :].clone().float()  # always fp32   # [2, n_embd], always fp32
        self.head = nn.Parameter(head)
        self.yes_id, self.no_id = yes_id, no_id
        # readout_offset > 0 scores a position BEFORE the last real token. The MNTP prompt appends
        # [MASK] after "Relevant:", and an MNTP-adapted model reads its prediction from the position
        # before the masked one, so the score comes from ":" (offset 1), not from [MASK] itself.
        self.readout_offset = int(readout_offset)
        # score_mode:
        #   "lm_rows"  (default) the pretrained LM-head rows for " true"/" false" read at ONE position.
        #              Under causal attention that position already sees the whole sequence, so this
        #              readout cannot use bidirectionality -- runs L and M (2026/10/02) found exactly
        #              that: no change against the causal baseline.
        #   "meanpool" masked mean over every real token, scored by a fresh Linear(n_embd -> 1). This
        #              is the readout bidirectionality can actually pay for, because every token's
        #              representation enters the score. It gives up the LM-head prior, so it starts
        #              at chance instead of ~30% accuracy by step 20.
        #   "meanpool_lmrows" masked mean, scored by the SAME two LM rows. This isolates pooling
        #              from re-initialisation: the score stays (w_true - w_false)^T x, so the only
        #              change against "lm_rows" is which x is read. Note the rows were trained on a
        #              single post-norm state, and a mean over positions has a smaller norm, so the
        #              prior transfers only partially.
        assert score_mode in ("lm_rows", "meanpool", "meanpool_lmrows"), score_mode
        self.score_mode = score_mode
        if score_mode == "meanpool":
            self.pool_head = nn.Linear(mc["n_embd"], 1).float()
            nn.init.normal_(self.pool_head.weight, std=0.02)   # ~0.9 logit scale on RMS-normed states
            nn.init.zeros_(self.pool_head.bias)
            # the two LM rows are unused here; DDP only all-reduces params that require grad, and an
            # unused one that does would abort the step
            self.head.requires_grad_(False)
        # MNTP adaptation predicts over the whole vocabulary, so it keeps the LM head; a reranker drops
        # it (the other ~32k rows never score anything) and saves the memory
        if not keep_lm_head:
            del self.backbone.lm_head
            self.backbone.lm_head = None
        self.bidirectional = bidirectional
        if bidirectional:                   # patch upstream's attention entry point (src/bidirectional.py)
            import bidirectional as _bd
            _bd.enable(nanochat_src)
        # rotary tables are buffers (persistent=False), built on meta: rebuild them for real
        head_dim = mc["n_embd"] // mc["n_head"]
        cos, sin = self.backbone._precompute_rotary_embeddings(self.backbone.rotary_seq_len, head_dim,
                                                               device=torch.device("cpu"))
        self.backbone.cos, self.backbone.sin = cos, sin
        self.backbone._ckpt = False
        # train_nc.py turns checkpointing on through the HF API; keep that call working
        self.backbone.gradient_checkpointing_enable = types.MethodType(
            lambda m, gradient_checkpointing_kwargs=None: setattr(m, "_ckpt", True), self.backbone)

    # -- upstream GPT.forward, minus the lm_head ------------------------------------------------
    def _hidden(self, idx, pad_mask=None, doc_ids=None):
        """pad_mask/doc_ids are used only in bidirectional mode, where attention needs an explicit mask."""
        import contextlib
        if self.bidirectional:
            import bidirectional as _bd
            assert pad_mask is not None, "bidirectional attention needs the padding mask"
            # persists through backward, where checkpointing recomputes the blocks
            _bd.set_context(pad_mask, doc_ids)
            # build every window's mask up front: filling the cache lazily inside a checkpointed block
            # makes recomputation see a different tensor count and raises CheckpointError
            _bd.prime({w[0] if isinstance(w, (tuple, list)) else w for w in self.backbone.window_sizes})
            return self._trunk(idx)
        return self._trunk(idx)

    def _trunk(self, idx):
        m, norm = self.backbone, self._norm
        B, T = idx.shape
        assert T > 1 and T <= m.cos.size(1), (T, m.cos.size(1))
        cos_sin = m.cos[:, :T], m.sin[:, :T]
        x = norm(m.transformer.wte(idx).to(m.cos.dtype))
        # smear: mix the previous token's embedding into each position
        gate = m.smear_lambda.to(x.dtype) * torch.sigmoid(m.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x0 = x
        backout_layer = m.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(m.transformer.h):
            x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0
            ve = m.value_embeds[str(i)](idx).to(x.dtype) if str(i) in m.value_embeds else None
            if m._ckpt and self.training:
                # determinism_check="none": SDPA may pick a different backend in the no-grad forward
                # than in the grad-enabled recomputation, so the tensor COUNT differs (43 vs 42) even
                # though the maths is identical -- verified by comparing checkpointed and
                # non-checkpointed hidden states and gradients. Without this, checkpointing raises.
                # use_reentrant=True: the non-reentrant implementation compares how many tensors each
                # pass saved (43 vs 42 here) and raises, and determinism_check="none" does not suppress
                # that check in torch 2.14. The reentrant path has no such check; the equivalence of the
                # two paths is verified numerically instead (hidden states and gradients).
                x = checkpoint(block, x, ve, cos_sin, m.window_sizes[i], None, use_reentrant=True)
            else:
                x = block(x, ve, cos_sin, m.window_sizes[i], None)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - m.backout_lambda.to(x.dtype) * x_backout
        return norm(x)

    def forward(self, input_ids, attention_mask) -> RerankOutput:
        h = self._hidden(input_ids, pad_mask=attention_mask)           # [B, T, H]
        if self.score_mode in ("meanpool", "meanpool_lmrows"):
            m = attention_mask.unsqueeze(-1).to(h.dtype)               # [B, T, 1] right padding
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)          # [B, H] masked mean
            if self.score_mode == "meanpool":
                return RerankOutput(scores=self.pool_head(pooled.float()).squeeze(-1))
            yn = pooled.float() @ self.head.t()                        # [B, 2] the same two LM rows
            return RerankOutput(scores=yn[:, 0] - yn[:, 1], yes_no=yn)
        last = attention_mask.sum(1) - 1 - self.readout_offset         # right padding
        h_last = h[torch.arange(h.size(0), device=h.device), last]     # [B, H]
        yn = h_last.float() @ self.head.t()                            # [B, 2] fp32
        return RerankOutput(scores=yn[:, 0] - yn[:, 1], yes_no=yn)

    def lm_logits(self, idx, pad_mask=None, doc_ids=None):
        """Full-vocabulary logits, for MNTP adaptation (needs keep_lm_head=True). Softcap matches
        upstream's forward, so the loss sees the same scale pretraining used."""
        assert self.backbone.lm_head is not None, "build with keep_lm_head=True"
        h = self._hidden(idx, pad_mask=pad_mask, doc_ids=doc_ids)
        logits = self.backbone.lm_head(h)[..., :self.meta["vocab_size"]].float()
        return 15.0 * torch.tanh(logits / 15.0)

    def trainable_report(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = self.pool_head.weight.numel() + self.pool_head.bias.numel() \
            if self.score_mode == "meanpool" else self.head.numel()   # lm rows in the other modes
        return {"backbone": sum(p.numel() for p in self.backbone.parameters()),
                "head": head, "score_mode": self.score_mode, "trainable": tr}
