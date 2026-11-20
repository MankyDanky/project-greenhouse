"""Native Gemma scoring plugged into the unchanged NanoChat LCE evaluation harness."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig

ROOT = Path(__file__).resolve().parents[1]          # eval/: results and qrels live here
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "reranker/engines/gemma4"))
from gemma_prompt import PairBuilder, setup_tokenizer, yes_no_ids
from modeling_gemma4_lce import Gemma4Reranker


class GemmaScorer:
    def __init__(self, checkpoint, device, dtype):
        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, mmap=True, weights_only=False, map_location="cpu")
        cfg = state["cfg"]
        assert state["packing"]["objective"] == "lce_gemma4_native_tied_softcap_gap"
        assert state["packing"]["plan_hash"] == cfg["expected_plan_hash"]
        assert cfg["model_revision"] == "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
        assert cfg["prompt"] == "nochat" and not cfg["lora"]
        self.step = state["step"]
        directory = checkpoint.parent
        config = Gemma4TextConfig.from_dict(json.loads((directory / "gemma_text_config.json").read_text()))
        config._attn_implementation = "sdpa"
        model = Gemma4Reranker(full=Gemma4ForCausalLM(config).float())
        model.load_state_dict(state["model"], strict=True, assign=True)
        assert all(not p.is_meta for p in model.parameters())
        tok = setup_tokenizer(AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True))
        self.answer_ids = yes_no_ids(tok, "nochat", cfg["answer_tokens"])
        self.builder = PairBuilder(tok, max_query_len=cfg["max_query_len"],
            max_doc_len=int(os.environ.get("EVAL_MAX_DOC_LEN") or cfg["max_doc_len"]))
        self.embedding = state["model"]["backbone.embed_tokens.weight"]
        self.answer_rows = self.embedding[list(self.answer_ids)].clone().float().to(device)
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = device
        self.cap = self.model.config.final_logit_softcapping
        assert self.cap == 30.0
        self.validated = False
        # Preserve the established native Gemma evaluator's FP32 answer-row arithmetic.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    @torch.inference_mode()
    def __call__(self, query, docs, batch_size, max_len):
        scores = []
        for start in range(0, len(docs), batch_size):
            seqs = [self.builder.build(query, doc) for doc in docs[start:start + batch_size]]
            ids = torch.full((len(seqs), max(map(len, seqs))), self.builder.tok.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
                mask[i, :len(seq)] = 1
            hidden = self.model.backbone(input_ids=ids, attention_mask=mask,
                                         use_cache=False).last_hidden_state
            last = hidden[torch.arange(len(seqs), device=self.device), mask.sum(1)-1].float()
            logits = torch.nn.functional.linear(last, self.answer_rows)
            logits = torch.tanh(logits / self.cap) * self.cap
            if not self.validated:
                native = torch.nn.functional.linear(last[:1].cpu(), self.embedding.float())
                native = torch.tanh(native / self.cap) * self.cap
                torch.testing.assert_close(logits[:1].cpu(), native[:, list(self.answer_ids)],
                                           atol=1e-4, rtol=1e-4)
                print("GEMMA_LOAD_VALIDATED", json.dumps({"step": self.step, "strict_state_load": True,
                    "full_vocab_answer_row_parity": True, "answer_ids": self.answer_ids,
                    "native_softcap": self.cap}), flush=True)
                self.embedding = None
                self.validated = True
            scores.extend((logits[:, 0] - logits[:, 1]).cpu().tolist())
        return scores


def load_harness():
    path = ROOT / "rerank_eval.py"
    spec = importlib.util.spec_from_file_location("nanochat_lce_eval", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    # The CLI, document processing, ranking, TREC formatting and metrics are LCE's own code.
    harness.BACKENDS = {"ours": GemmaScorer}
    harness.HERE = str(ROOT)
    harness.RUNS = os.environ.get("EVAL_RUNS", str(REPO / "data/bm25_top100"))
    harness.QRELS_DIR = os.environ.get("EVAL_QRELS_DIR", str(ROOT / "qrels"))
    return harness


if __name__ == "__main__":
    load_harness().main()
