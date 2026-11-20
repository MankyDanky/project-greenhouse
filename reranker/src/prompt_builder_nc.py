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
#   nochat_mntp -- nochat with the MNTP MASK TOKEN appended (2026/10/01), for backbones adapted to
#             bidirectional attention with MNTP (castorini/nanochat-4b-climbmix-bidir-20260930):
#             "<|bos|>Query: {q}\nPassage: {p}\nRelevant:<|output_end|>", scored at the ":" -- the
#             position BEFORE the mask, which is where MNTP reads its prediction from.
#             Why: MNTP trains "the state at t predicts the token at t+1, where t+1 holds [MASK] and t
#             sees both sides". Our plain prompt reads the LAST token, which has no right context and
#             no mask marking the slot -- the one regime the adaptation degraded (measured: plain
#             next-token accuracy 0.4016 for the bidirectional backbone vs 0.4787 for the causal one
#             it came from, same 35,840 positions). Appending the mask restores the training layout.
#   nochat_qdq -- nochat with the QUERY REPEATED after the passage (2026/09/28):
#             "<|bos|>Query: {q}\nPassage: {p}\nQuery: {q}\nRelevant:"
#             Causal attention means query tokens never see the passage. Repeating the query after it
#             gives query-side positions that attend to the whole passage -- the half of bidirectional
#             attention that matters here -- while keeping the mask, the last-token read and the
#             LM-head scoring rows untouched. Costs one query length per sequence.
# readout_offset: how many tokens back from the last real token the score is read. The MNTP prompt
# appends [MASK] after "Relevant:", and MNTP reads its prediction from the position BEFORE the masked
# one, so the score comes from the ":" -- one token back.
MNTP_MASK_ID = 32767        # "<|output_end|>" in the 32k vocabulary; the id the released
                            # bidirectional checkpoint was adapted with. Our tokenizer.json omits the
                            # 32759-32767 special range, so it is appended by id, not by text.
PROMPTS = {
    "chat":        {"answers": ("yes", "no")},
    "nochat":      {"answers": (" true", " false")},
    "nochat_qdq":  {"answers": (" true", " false")},
    "nochat_mntp": {"answers": (" true", " false"), "readout_offset": 1, "append_ids": [MNTP_MASK_ID]},
}


def readout_offset(prompt):
    """Tokens back from the last real token at which the yes/no logits are read.

    This is len(append_ids) by construction: whatever the prompt appends after "Relevant:" sits
    between that colon and the end of the sequence. Deriving it instead of trusting the declared
    value means the two cannot drift -- a prompt that appended two tokens while declaring an offset
    of 1 would read every score one position off the colon, silently. The declared value is kept as
    a cross-check and must agree.
    """
    spec = PROMPTS[prompt]
    derived = len(spec.get("append_ids", []))
    declared = spec.get("readout_offset")
    assert declared is None or declared == derived, (
        f"prompt {prompt!r}: readout_offset {declared} but it appends {derived} token(s)")
    return derived
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
                 instruction=INSTRUCTION, prompt="chat", bos=BOS):
        """bos: the text the prompt starts with, parsed by the tokenizer as its BOS token. nanochat's
        "<|bos|>" by default (every nanochat run); another backbone passes its own ("<s>" for
        MiniCPM5) or "" for one that uses none (Qwen). Config key `bos`, absent for nanochat."""
        assert prompt in PROMPTS, f"prompt must be one of {list(PROMPTS)}, got {prompt!r}"
        self.tok = setup_tokenizer(tokenizer)
        self.max_doc_len, self.max_query_len = max_doc_len, max_query_len
        self.instruction, self.prompt, self.bos = instruction, prompt, bos
        # pre-tokenise the fixed scaffold once
        enc = lambda x: self.tok.encode(x, add_special_tokens=False)
        if bos != BOS:          # a non-nanochat BOS must be ONE token, not text split into pieces
            assert bos == "" or len(enc(bos)) == 1, f"bos {bos!r} encodes to {enc(bos)}, not one token"
        if prompt == "chat":
            self._pre = enc(f"{bos}{USER_START}{instruction}\n\n<Query>: ")
            self._mid = enc("\n<Document>: ")
            self._suf = enc(f"{USER_END}{ASSISTANT_START}")
        else:                                   # nochat: <|bos|>Query: {q}\nPassage: {p}\nRelevant:
            self._pre = enc(f"{bos}Query: ")
            self._mid = enc("\nPassage: ")
            self._suf = enc("\nRelevant:") + list(PROMPTS[prompt].get("append_ids", []))
        # nochat_qdq repeats the query after the passage; everything else has no second query
        self._mid2 = enc("\nQuery: ") if prompt == "nochat_qdq" else []
        self.query_factor = 2 if prompt == "nochat_qdq" else 1   # GroupSampler bills the query twice
        self.overhead = len(self._pre) + len(self._mid) + len(self._mid2) + len(self._suf)

    def _cut(self, text, n, known=None):
        # split_special_tokens: a literal "<|assistant_start|>" inside a passage is
        # encoded as plain text, never as the control token (RLHN has none; scanned)
        return self.tok.encode(text, add_special_tokens=False, split_special_tokens=True)[:n]

    def build(self, query, doc, query_len=None, doc_len=None):
        """Returns token ids ending at <|assistant_start|>; the model's next-token
        distribution at that final position is what we read yes/no from."""
        q = self._cut(query, self.max_query_len, query_len)
        d = self._cut(doc, self.max_doc_len, doc_len)
        if self._mid2:
            return self._pre + q + self._mid + d + self._mid2 + q + self._suf
        return self._pre + q + self._mid + d + self._suf

    def render(self, query, doc):
        return self.tok.decode(self.build(query, doc))
