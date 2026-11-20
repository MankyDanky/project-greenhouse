"""Prompt construction and truncation, one rule set for every backbone.

    [BOS] "Query: " q "\nPassage: " p "\nRelevant:"   -> score at the last token (":")

* The scaffold pieces are tokenized separately and concatenated as ids, exactly as greenhouse
  `prompt_builder_nc.PairBuilder` does for run G (so R0 is token-identical to G).
* BOS is the backbone's native convention: nanochat <|bos|>, Gemma <bos>, MiniCPM <s>, Marin <|begin_of_text|>,
  Qwen none, Gravity none
  (GLM-4.5 tokenizer: no BOS token and no post-processor prefix; [gMASK]<sop> only opens chat templates).
* Answers " true"/" false" must be single tokens in every tokenizer (checked).
* Length rules:
    caps    query[:max_query_len], passage[:max_doc_len]                       (R0: 128 / 512, as G)
    budget  query[:max_query_len], passage[:seq_budget - scaffold - len(query)] (C runs: 512 / 2,048)
  Query and passage are encoded with split_special_tokens=True, so text can never forge a control token.
"""
SCAFFOLD = ("Query: ", "\nPassage: ", "\nRelevant:")
ANSWERS = (" true", " false")
NANOCHAT_BOS, NANOCHAT_PAD = "<|bos|>", "<|assistant_end|>"


def setup_tokenizer(tok, backbone):
    if backbone == "nanochat":                 # greenhouse setup_tokenizer: pad = eos = <|assistant_end|>
        if tok.eos_token is None: tok.eos_token = NANOCHAT_PAD
        if tok.pad_token is None: tok.pad_token = NANOCHAT_PAD
    else:
        assert tok.eos_token_id is not None
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def bos_id(tok, backbone):
    """Native BOS id or None. Decided per backbone, not from add_bos flags (MiniCPM's tokenizer_config says
    add_bos False while its tokenizer.json post-processor prepends <s>)."""
    if backbone == "nanochat":
        i = tok.convert_tokens_to_ids(NANOCHAT_BOS)
    elif backbone in ("gemma", "minicpm", "marin"):
        i = tok.bos_token_id
    elif backbone in ("qwen", "gravity"):
        return None
    else:
        raise ValueError(backbone)
    assert i is not None and i != tok.unk_token_id, (backbone, i)
    return i


def answer_ids(tok):
    ids = [tok.encode(s, add_special_tokens=False) for s in ANSWERS]
    assert all(len(x) == 1 for x in ids) and ids[0] != ids[1], ids
    return ids[0][0], ids[1][0]


class PairBuilder:
    def __init__(self, tok, backbone, length_rule, max_query_len, max_doc_len=None, seq_budget=None):
        assert length_rule in ("caps", "budget"), length_rule
        self.tok, self.rule = tok, length_rule
        self.max_query_len, self.max_doc_len, self.seq_budget = max_query_len, max_doc_len, seq_budget
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        b = bos_id(tok, backbone)
        self._pre = ([b] if b is not None else []) + enc(SCAFFOLD[0])
        self._mid = enc(SCAFFOLD[1])
        self._suf = enc(SCAFFOLD[2])
        self.overhead = len(self._pre) + len(self._mid) + len(self._suf)
        if length_rule == "budget":
            assert seq_budget > self.overhead + max_query_len, (seq_budget, self.overhead, max_query_len)

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False, split_special_tokens=True)

    def build(self, query, doc, query_len=None, doc_len=None):
        q = self._enc(query)[:self.max_query_len]
        room = self.max_doc_len if self.rule == "caps" else self.seq_budget - self.overhead - len(q)
        d = self._enc(doc)[:room]
        # budget rule: the passage gets exactly what is left of seq_budget after prompt + query, so the whole
        # sequence is <= seq_budget by construction (no runtime check; tests/test_cpu.py verifies it). Since
        # max_query_len 512 + scaffold ~15 << 2,048, the passage always has >= 1,521 tokens of room (> G's 512).
        return self._pre + q + self._mid + d + self._suf

    def render(self, query, doc):
        return self.tok.decode(self.build(query, doc))
