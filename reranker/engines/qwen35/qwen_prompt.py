"""Plain base-model relevance prompt with Qwen's own tokenizer (no chat markers)."""
PROMPTS = {"nochat": {"answers": (" true", " false")}}
def setup_tokenizer(tok):
    assert tok.eos_token_id is not None
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
def yes_no_ids(tok, prompt="nochat", answers=None):
    pos, neg = answers or PROMPTS[prompt]["answers"]
    y, n = (tok.encode(s, add_special_tokens=False) for s in (pos, neg))
    assert len(y) == len(n) == 1 and y != n, (y, n)
    return y[0], n[0]
class PairBuilder:
    def __init__(self, tokenizer, max_doc_len=512, max_query_len=128, prompt="nochat"):
        assert prompt == "nochat"
        self.query_factor = 1
        self.tok = setup_tokenizer(tokenizer)
        self.max_doc_len, self.max_query_len = max_doc_len, max_query_len
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self._pre = enc("Query: ")
        self._mid = enc("\nPassage: ")
        self._suf = enc("\nRelevant:")
        self.overhead = len(self._pre) + len(self._mid) + len(self._suf)
    def build(self, query, doc, query_len=None, doc_len=None):
        q = self.tok.encode(query, add_special_tokens=False, split_special_tokens=True)[:self.max_query_len]
        d = self.tok.encode(doc, add_special_tokens=False, split_special_tokens=True)[:self.max_doc_len]
        return self._pre + q + self._mid + d + self._suf
    def render(self, query, doc): return self.tok.decode(self.build(query, doc))
