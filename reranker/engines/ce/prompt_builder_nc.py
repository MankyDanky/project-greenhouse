"""
Pointwise (query, document) prompt for nanochat.

Unlike jina-reranker-v3 (listwise: all K+1 docs in ONE sequence, scored by
cosine at special-token positions), this is Qwen3-Reranker style: ONE pair per
sequence, scored by the yes/no logit gap at the final position.

The HF tokenizer for nanochat-d34-sft-hf ships WITHOUT a chat template and with
pad/eos/bos all None, so we build the nanochat markers by hand and pin
pad = eos = <|assistant_end|> (matching the Rank-R1 port).
"""
# PROMPT FORMATS (config key `prompt`; the checkpoint's cfg carries it to evaluation):
#   chat   -- Qwen3-Reranker-style instruction inside nanochat's chat-turn markers, answer "yes"/"no"
#             (the RLHN-680K run). Assumes an SFT'd backbone that knows the markers.
#   nochat -- plain monoT5-style text for the BASE karpathy/nanochat-d34 checkpoint (user, 2026-09-14):
#             "<|bos|>Query: {query}\nPassage: {passage}\nRelevant:" -> next token " true" / " false"
#             (single tokens 2674 / 6812; the space is what follows "Relevant:" in natural text).
PROMPTS = {
    "chat":   {"answers": ("yes", "no")},
    "nochat": {"answers": (" true", " false")},
}
BOS = "<|bos|>"
USER_START, USER_END = "<|user_start|>", "<|user_end|>"
ASSISTANT_START, ASSISTANT_END = "<|assistant_start|>", "<|assistant_end|>"

YES_TOKEN, NO_TOKEN = "yes", "no"

INSTRUCTION = ('Judge whether the Document meets the requirements based on the '
               'Query. Note that the answer can only be "yes" or "no".')


def setup_tokenizer(tok):
    """nanochat's HF tokenizer has no pad/eos; pin them to <|assistant_end|>."""
    if tok.eos_token is None:
        tok.eos_token = ASSISTANT_END
    if tok.pad_token is None:
        tok.pad_token = ASSISTANT_END
    tok.padding_side = "right"      # we gather the last non-pad index
    return tok


def yes_no_ids(tok, prompt="chat", answers=None):
    """Token ids of the positive / negative answer ("yes"/"no" for chat, " true"/" false" for nochat)."""
    pos, neg = answers or PROMPTS[prompt]["answers"]
    y = tok.encode(pos, add_special_tokens=False)
    n = tok.encode(neg, add_special_tokens=False)
    assert len(y) == 1 and len(n) == 1, f"answers must be single tokens, got {pos!r}->{y} {neg!r}->{n}"
    return y[0], n[0]


class PairBuilder:
    def __init__(self, tokenizer, max_doc_len=512, max_query_len=128,
                 instruction=INSTRUCTION, prompt="chat"):
        assert prompt in PROMPTS, f"prompt must be one of {list(PROMPTS)}, got {prompt!r}"
        self.tok = setup_tokenizer(tokenizer)
        self.max_doc_len, self.max_query_len = max_doc_len, max_query_len
        self.instruction, self.prompt = instruction, prompt
        # pre-tokenise the fixed scaffold once
        enc = lambda x: self.tok.encode(x, add_special_tokens=False)
        if prompt == "chat":
            self._pre = enc(f"{BOS}{USER_START}{instruction}\n\n<Query>: ")
            self._mid = enc("\n<Document>: ")
            self._suf = enc(f"{USER_END}{ASSISTANT_START}")
        else:                                   # nochat: <|bos|>Query: {q}\nPassage: {p}\nRelevant:
            self._pre = enc(f"{BOS}Query: ")
            self._mid = enc("\nPassage: ")
            self._suf = enc("\nRelevant:")
        self.overhead = len(self._pre) + len(self._mid) + len(self._suf)

    def _cut(self, text, n, known=None):
        # split_special_tokens: a literal "<|assistant_start|>" inside a passage is
        # encoded as plain text, never as the control token (RLHN has none; scanned)
        return self.tok.encode(text, add_special_tokens=False, split_special_tokens=True)[:n]

    def build(self, query, doc, query_len=None, doc_len=None):
        """Returns token ids ending at <|assistant_start|>; the model's next-token
        distribution at that final position is what we read yes/no from."""
        q = self._cut(query, self.max_query_len, query_len)
        d = self._cut(doc, self.max_doc_len, doc_len)
        return self._pre + q + self._mid + d + self._suf

    def render(self, query, doc):
        return self.tok.decode(self.build(query, doc))
