"""Backbone adapters. Every one returns score = logit(" true") - logit(" false") at the last real token,
computed from the backbone's own LM-head rows for those two tokens, in fp32, with no softcap.

  nanochat  greenhouse UpstreamNanochatReranker, unmodified (run G's wrapper; untied rows cloned to fp32)
  qwen      Qwen3ForCausalLM or Qwen3_5ForConditionalGeneration -> text decoder; tied rows read from the input embedding
  gemma     Gemma4ForConditionalGeneration  -> text decoder; tied rows read from the input embedding
  minicpm   LlamaForCausalLM                -> decoder; untied rows cloned to an fp32 parameter
  marin     LlamaForCausalLM (Marin 8B)     -> decoder; untied rows cloned to an fp32 parameter
  gravity   GravityMoEForCausalLM (remote code = transformers' DeepseekV3 + its config) -> decoder; untied rows
            cloned to an fp32 parameter. 29.6B MoE: trained with src/train_fsdp.py, where non-zero ranks build the
            same module tree on the meta device (meta=True) and receive rank 0's weights.

Vision/audio towers are dropped; every text parameter trains. Padding is on the right and attention is
causal in all six, so a real token never attends to a pad and the read position is mask.sum(1) - 1.
"""
import os, sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]   # reranker/
LCE_SRC = ROOT / "src"
NANOCHAT_SRC = ROOT / "third_party/nanochat-92d63d4"
for p in (LCE_SRC,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@dataclass
class RerankOutput:
    scores: torch.Tensor          # [B] fp32
    yes_no: torch.Tensor          # [B, 2] fp32


REMOTE_CODE = {"gravity"}        # checkpoints whose modeling code ships in the repo (trust_remote_code)


class HFReranker(nn.Module):
    def __init__(self, model_dir, backbone, yes_id, no_id, dtype=torch.float32, attn_implementation="sdpa",
                 experts_implementation=None, meta=False):
        super().__init__()
        from transformers import AutoConfig
        remote = backbone in REMOTE_CODE
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=remote)
        mt = config.model_type
        if backbone == "qwen":
            if mt == "qwen3":
                from transformers import Qwen3ForCausalLM as Cls
            elif mt == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration as Cls
            else:
                raise ValueError(f"unsupported Qwen model type: {mt}")
        elif backbone == "gemma":
            from transformers import Gemma4ForConditionalGeneration as Cls
        elif backbone in ("minicpm", "marin"):
            from transformers import LlamaForCausalLM as Cls
        elif backbone == "gravity":
            assert mt == "gravity_moe", mt
            from transformers import AutoModelForCausalLM as Cls
        else:
            raise ValueError(backbone)
        kw = dict(dtype=dtype, attn_implementation=attn_implementation)
        if remote: kw["trust_remote_code"] = True
        if meta:                                # same module tree, no storage: weights arrive from rank 0
            with torch.device("meta"):
                full = Cls.from_config(config, **kw)
            info = {}
        else:
            full, info = Cls.from_pretrained(model_dir, output_loading_info=True, **kw)
        missing = [k for k in info.get("missing_keys", []) if not k.endswith("lm_head.weight")]
        if missing or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"incomplete pretrained load of {model_dir}: {info}")
        if experts_implementation:
            # transformers only accepts experts_implementation= for classes whose own source carries the decorator,
            # which the remote GravityMoE wrapper does not; the experts read config._experts_implementation at call
            # time, so set it on the (shared) config and check every experts module sees it.
            full.config._experts_implementation = experts_implementation
            ex = [m for m in full.modules() if hasattr(m, "gate_up_proj") and hasattr(m, "config")]
            assert ex and all(m.config._experts_implementation == experts_implementation for m in ex), \
                (experts_implementation, len(ex))
        self.load_info = {k: list(v)[:8] if hasattr(v, "__iter__") else v for k, v in info.items()}
        self.model_type, self.backbone_name = mt, backbone
        self.backbone = getattr(full.model, "language_model", full.model)
        self.config = self.backbone.config
        emb = self.backbone.embed_tokens.weight
        self.tied = full.lm_head.weight is emb
        self.yes_id, self.no_id = yes_id, no_id
        self.register_buffer("answer_ids", torch.tensor([yes_id, no_id]), persistent=False)
        if self.tied:
            self.head = None                    # rows stay part of the (trainable) input embedding
        else:
            self.head = nn.Parameter(full.lm_head.weight.data[[yes_id, no_id], :].clone().float())
        del full

    def rows(self):
        return self.backbone.embed_tokens.weight.index_select(0, self.answer_ids) if self.tied else self.head

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1
        h = h[torch.arange(h.shape[0], device=h.device), last]
        yn = F.linear(h.float(), self.rows().float())
        return RerankOutput(yn[:, 0] - yn[:, 1], yn)

    def trainable_report(self):
        n = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"backbone": sum(p.numel() for p in self.backbone.parameters()),
                "head": 0 if self.head is None else self.head.numel(), "trainable": tr, "total": n}


def build_model(cfg, yes_id, no_id, dtype, meta=False):
    if cfg["backbone"] == "nanochat":
        assert not meta
        from modeling_nanochat_upstream import UpstreamNanochatReranker
        return UpstreamNanochatReranker(cfg["model_name"], yes_id=yes_id, no_id=no_id, dtype=dtype,
                                        nanochat_src=cfg.get("nanochat_src") or str(NANOCHAT_SRC),
                                        step=cfg.get("model_step"))
    return HFReranker(cfg["model_name"], cfg["backbone"], yes_id, no_id, dtype=dtype,
                      attn_implementation=cfg.get("attn_implementation", "sdpa"),
                      experts_implementation=cfg.get("experts_implementation"), meta=meta)


def enable_checkpointing(model):
    model.train()
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
