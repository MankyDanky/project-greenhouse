"""Native Gemma 4 text backbone; two tied answer rows and native logit softcap."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class RerankOutput:
    scores: torch.Tensor
    yes_no: torch.Tensor

class Gemma4Reranker(nn.Module):
    def __init__(self, model_name=None, yes_id=1847, no_id=2416,
                 dtype=torch.float32, attn_implementation="sdpa", full=None, model_revision=None):
        super().__init__()
        if full is None:
            from transformers import Gemma4ForConditionalGeneration
            full, info = Gemma4ForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, attn_implementation=attn_implementation,
                revision=model_revision, output_loading_info=True)
            missing = [k for k in info.get("missing_keys", []) if k != "lm_head.weight"]
            if missing or info.get("unexpected_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
                raise RuntimeError(f"Incomplete native Gemma load: {info}")
            print("GEMMA_PRETRAINED_LOAD", info, flush=True)
        self.backbone = getattr(full.model, "language_model", full.model)
        self.config = self.backbone.config
        assert full.lm_head.weight is self.backbone.embed_tokens.weight, "Keep native tied weights"
        self.yes_id, self.no_id = yes_id, no_id
        self.register_buffer("answer_ids", torch.tensor([yes_id, no_id]), persistent=False)
        del full  # Unused vision/audio modules; every text parameter remains trainable.

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state
        h = h[torch.arange(h.shape[0], device=h.device), attention_mask.sum(1)-1]
        rows = self.backbone.embed_tokens.weight.index_select(0, self.answer_ids)
        yn = F.linear(h.float(), rows.float())
        cap = self.config.final_logit_softcapping
        if cap is not None:
            yn = torch.tanh(yn / cap) * cap
        return RerankOutput(yn[:, 0]-yn[:, 1], yn)

    def trainable_report(self):
        n = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert n == tr, "Every text parameter, including PLE, must train"
        return {"backbone": n, "head": 0, "trainable": tr}
