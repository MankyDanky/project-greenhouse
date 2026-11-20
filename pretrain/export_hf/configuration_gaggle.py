"""Configuration for the Gaggle nanochat base model (karpathy/nanochat @ 92d63d4, depth 34).

The architecture is nanochat's own, not any stock `transformers` model: relu^2 MLPs, parameter-free RMSNorm,
QK-norm with a fixed 1.2 scale, per-layer sliding windows, ResFormer value embeddings with per-head gates,
embedding smearing, a mid-layer backout, per-layer residual/x0 scalars and a 15 * tanh(logits / 15) soft-cap.
Hence a `trust_remote_code` config/model pair rather than a conversion into an existing architecture.

Field names mirror nanochat's GPTConfig, so a checkpoint's meta_*.json maps across 1:1; the usual
`transformers` names (hidden_size, ...) are aliases through attribute_map.
"""

from transformers.configuration_utils import PretrainedConfig


class GaggleConfig(PretrainedConfig):
    model_type = "gaggle"
    keys_to_ignore_at_inference = ["past_key_values"]
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
        rope_theta: float = 100000.0,
        qk_scale: float = 1.2,
        mlp_ratio: int = 4,
        ve_gate_channels: int = 12,
        smear_channels: int = 24,
        logit_softcap: float = 15.0,
        bos_token_id: int = 32759,      # <|bos|>: every document starts with it, as in training
        eos_token_id: int = 32759,      # a base model ends a document by starting the next one
        pad_token_id: int = 32763,      # <|assistant_end|>; nanochat has no pad token
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.sequence_len = sequence_len
        self.window_pattern = window_pattern
        self.rope_theta = rope_theta
        self.qk_scale = qk_scale
        self.mlp_ratio = mlp_ratio
        self.ve_gate_channels = ve_gate_channels
        self.smear_channels = smear_channels
        self.logit_softcap = logit_softcap
        self.use_cache = use_cache
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)

    # nanochat's own derivations, kept here so the model, the converter and the vLLM port cannot disagree ----
    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def window_sizes(self):
        """Left attention window per layer (a query sees itself plus `window` earlier tokens).

        Tiles `window_pattern`; the last layer is always long. Same arithmetic as nanochat's
        GPT._compute_window_sizes: 2048 -> 512 for the short layers.
        """
        pattern = self.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"invalid window_pattern {self.window_pattern!r}"
        long_window = self.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        sizes = [long_window if pattern[i % len(pattern)] == "L" else short_window
                 for i in range(self.n_layer)]
        sizes[-1] = long_window
        return sizes

    def has_value_embedding(self, layer_idx: int) -> bool:
        """ResFormer value embeddings sit on alternating layers; the last layer always has one."""
        return layer_idx % 2 == (self.n_layer - 1) % 2
