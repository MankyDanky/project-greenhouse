"""Qwen3.5 text reranker with native tied output embeddings and LCE scores."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class RerankOutput:
    scores: torch.Tensor
    yes_no: torch.Tensor

class Qwen35Reranker(nn.Module):
    def __init__(self, model_name=None, yes_id=804, no_id=867,
                 dtype=torch.float32, attn_implementation="sdpa", full=None, model_revision=None,
                 local_files_only=False):
        super().__init__()
        if full is None:
            from transformers import Qwen3_5ForConditionalGeneration
            full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, attn_implementation=attn_implementation,
                revision=model_revision, local_files_only=local_files_only, output_loading_info=True)
            missing = [k for k in info.get("missing_keys", []) if k != "lm_head.weight"]
            if missing or info.get("mismatched_keys") or info.get("error_msgs"):
                raise RuntimeError(f"Incomplete pretrained load: {info}")
            print("QWEN_PRETRAINED_LOAD", info, flush=True)
        self.backbone = getattr(full.model, "language_model", full.model)
        self.config = self.backbone.config
        assert full.lm_head.weight is self.backbone.embed_tokens.weight, "Preserve native tied weights"
        self.yes_id, self.no_id = yes_id, no_id
        self.register_buffer("answer_ids", torch.tensor([yes_id, no_id]), persistent=False)
        # Only the text decoder participates in this text reranking task.
        # The output rows remain tied to the trainable input embedding, never cloned.
        del full

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1
        h = h[torch.arange(h.shape[0], device=h.device), last]
        rows = self.backbone.embed_tokens.weight.index_select(0, self.answer_ids)
        yn = F.linear(h.float(), rows.float())
        return RerankOutput(yn[:, 0] - yn[:, 1], yn)

    def trainable_report(self):
        n = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert n == tr, "Full text-backbone fine-tuning is required"
        return {"backbone": n, "head": 0, "trainable": tr}
