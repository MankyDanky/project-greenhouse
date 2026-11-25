"""Configuration for the Gaggle pointwise reranker (nanochat backbones: the d34, and other shapes of the same
architecture, e.g. d32 x 1152 with value embeddings on every 4th layer).

The architecture is karpathy/nanochat @ 92d63d4, which is NOT any stock HuggingFace model: it has
relu^2 MLPs, parameter-free RMSNorm, QK-norm with a fixed 1.2 scale, per-layer sliding windows,
ResFormer value embeddings with per-head gates, embedding smearing, a mid-layer backout and
per-layer residual/x0 scalars. Hence a `trust_remote_code` config/model pair rather than a
conversion into an existing architecture.

Field names mirror nanochat's GPTConfig so that a checkpoint's meta_*.json maps across 1:1; the
HuggingFace aliases (hidden_size, ...) are provided through attribute_map.
"""

from transformers.configuration_utils import PretrainedConfig


class GaggleRerankerConfig(PretrainedConfig):
    model_type = "gaggle_reranker"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_kv_head",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "sequence_len",
    }

    def __init__(
        self,
        vocab_size: int = 32768,
        n_layer: int = 34,
        n_head: int = 17,
        n_kv_head: int = 17,
        n_embd: int = 2176,
        sequence_len: int = 2048,
        window_pattern: str = "SSSL",
        ve_every: int = 2,              # value embeddings on every ve_every-th layer, counted back from the last
        rope_theta: float = 100000.0,
        qk_scale: float = 1.2,
        mlp_ratio: int = 4,
        ve_gate_channels: int = 12,
        smear_channels: int = 24,
        # --- reranker-specific -------------------------------------------------------------
        max_query_len: int = 128,
        max_doc_len: int = 512,
        prompt: str = "nochat",
        prompt_template: str = "<|bos|>Query: {query}\nPassage: {passage}\nRelevant:",
        prompt_prefix_ids=None,      # token ids of "<|bos|>Query: "
        prompt_mid_ids=None,         # token ids of "\nPassage: "
        prompt_suffix_ids=None,      # token ids of "\nRelevant:" (+ any appended marker)
        true_token_id: int = None,   # row 0 of `score`: the " true" LM-head row
        false_token_id: int = None,  # row 1 of `score`: the " false" LM-head row
        readout_offset: int = 0,     # tokens back from the last real token at which to read
        pad_token_id: int = 32763,   # <|assistant_end|>, which nanochat has no pad token for
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.sequence_len = sequence_len
        self.window_pattern = window_pattern
        assert int(ve_every) >= 1, f"ve_every must be >= 1, got {ve_every}"
        self.ve_every = int(ve_every)
        self.rope_theta = rope_theta
        self.qk_scale = qk_scale
        self.mlp_ratio = mlp_ratio
        self.ve_gate_channels = ve_gate_channels
        self.smear_channels = smear_channels
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.prompt = prompt
        self.prompt_template = prompt_template
        self.prompt_prefix_ids = list(prompt_prefix_ids or [])
        self.prompt_mid_ids = list(prompt_mid_ids or [])
        self.prompt_suffix_ids = list(prompt_suffix_ids or [])
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.readout_offset = int(readout_offset)
        kwargs.setdefault("num_labels", 1)
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    # nanochat's own derivations, kept here so the model and the converter cannot disagree ------
    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def window_sizes(self):
        """Left attention window per layer. Tiles `window_pattern`; the last layer is always long."""
        pattern = self.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"invalid window_pattern {self.window_pattern!r}"
        long_window = self.sequence_len
        short_window = -(-long_window // 4 // 128) * 128     # ceil to the FA3 tile size (2048 -> 512)
        sizes = [long_window if pattern[i % len(pattern)] == "L" else short_window
                 for i in range(self.n_layer)]
        sizes[-1] = long_window
        return sizes

    def has_value_embedding(self, layer_idx: int) -> bool:
        """ResFormer value embeddings sit on every `ve_every`-th layer counting back from the last, which always has
        one: nanochat's has_ve(). ve_every = 2 (the default, and upstream nanochat's only option) is the alternating
        rule layer_idx % 2 == (n_layer - 1) % 2, so configs written before ve_every existed keep their layer set."""
        return (self.n_layer - 1 - layer_idx) % self.ve_every == 0
