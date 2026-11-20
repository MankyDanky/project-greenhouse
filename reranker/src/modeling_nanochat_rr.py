"""
nanochat pointwise reranker: score = logit("yes") - logit("no") at the final
position of a (query, document) sequence.

Two deliberate choices:

1. THE LM HEAD IS REDUCED TO TWO ROWS. nanochat's head is Linear(2176, 65536)
   with UNTIED weights. A normal forward produces [B, T, 65536] -- for B=16,
   T=600 that is 1.25 GB materialised and ~41 TFLOP of matmul, essentially all
   of it thrown away. We keep only the `yes` and `no` rows as a [2, 2176]
   parameter, so the head costs B x 2 x 2176 and the vocab tensor never exists.
   We also gather only the final position first, which removes the T factor
   (minor next to the 32,768x from the row slice, but free).

2. THE HEAD RUNS IN FP32. The score is a DIFFERENCE of two logits whose
   magnitudes are typically 5-25, where bf16's ULP is 0.125 -- so
   16.30 - 16.24 rounds to exactly 0.0 and the signal disappears. Review
   2026-09-10: under bf16 autocast this matmul was silently RECAST to bf16
   (measured score error up to 0.12). Training now runs entirely in fp32 with no
   autocast (stepping.setup_precision), and FP32Guard checks this module's output.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RerankOutput:
    scores: torch.Tensor          # [B] fp32, logit(yes) - logit(no)
    yes_no: torch.Tensor          # [B, 2] fp32, raw logits


class NanochatReranker(nn.Module):
    def __init__(self, model_name: str = "pankajmathur/nanochat-d34-sft-hf",
                 yes_id: int = 10113, no_id: int = 6672,
                 dtype=torch.bfloat16, attn_implementation: str = "sdpa"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoConfig
        self.config = AutoConfig.from_pretrained(model_name)
        full = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation=attn_implementation)
        self.backbone = full.model                      # bare decoder stack
        w = full.lm_head.weight.data                    # [65536, 2176]
        # keep ONLY the two rows we score with, in fp32; drop the other 65,534
        self.head = nn.Parameter(w[[yes_id, no_id], :].clone().float())
        self.yes_id, self.no_id = yes_id, no_id
        del full.lm_head, full
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def forward(self, input_ids, attention_mask) -> RerankOutput:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state          # [B,T,H]
        # right padding -> last real token is at (sum(mask) - 1)
        last = attention_mask.sum(1) - 1
        h_last = h[torch.arange(h.size(0), device=h.device), last]     # [B,H]
        yn = h_last.float() @ self.head.t()                            # [B,2] fp32
        return RerankOutput(scores=yn[:, 0] - yn[:, 1], yes_no=yn)

    def trainable_report(self):
        bb = sum(p.numel() for p in self.backbone.parameters())
        hd = self.head.numel()
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"backbone": bb, "head": hd, "trainable": tr}
