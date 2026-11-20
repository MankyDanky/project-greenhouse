"""Gaggle nanochat base model, as a self-contained `transformers` causal LM.

A port of karpathy/nanochat @ 92d63d4 (`nanochat/gpt.py` + the SDPA path of `nanochat/flash_attention.py`).
It imports nothing beyond `torch` and `transformers`. Differences from the training-time code, all deliberate:

  * attention always runs through `torch.nn.functional.scaled_dot_product_attention`, written to dispatch
    exactly as nanochat's own SDPA fallback does; nanochat itself uses Flash Attention 3 when it is
    available (Hopper GPUs, bf16), which differs from SDPA only by floating-point rounding;
  * `transformers` generation: a standard `DynamicCache`, and the previous token that the embedding
    "smear" needs during decoding is passed explicitly instead of being stored in nanochat's own cache;
  * padding: right padding behaves exactly like nanochat (which assumes it); left padding, which nanochat
    does not support, masks pad keys and stops the smear from reading a pad.

The logits carry nanochat's 15 * tanh(logits / 15) soft-cap and are returned in fp32, as nanochat returns them.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:                                    # repo layout (trust_remote_code)
    from .configuration_gaggle import GaggleConfig
except ImportError:                     # flat directory
    from configuration_gaggle import GaggleConfig


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """nanochat's parameter-free RMSNorm (runs in the activation dtype, as upstream)."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """nanochat's rotation: the TRANSPOSE of the textbook/Llama convention (it rotates by -theta).

    Not interchangeable with Llama's `rotate_half`, which would flip the sign of `sin` and silently give a
    different model.
    """
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def sdpa_like_nanochat(q, k, v, window: int, mask: Optional[torch.Tensor]):
    """Attention for q (B, H, Tq, D) against k, v (B, H, Tk, D); the queries are the last Tq positions.

    Without a mask this is nanochat's `_sdpa_attention`, branch for branch, so the same SDPA kernel runs on the
    same tensors. `window` is the number of earlier tokens a query may see (nanochat's left window).
    """
    gqa = q.size(1) != k.size(1)
    if mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
    Tq, Tk = q.size(2), k.size(2)
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    if Tq == 1:
        if 0 <= window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:], v[:, :, start:]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=gqa)
    row = (Tk - Tq) + torch.arange(Tq, device=q.device).unsqueeze(1)
    col = torch.arange(Tk, device=q.device).unsqueeze(0)
    allowed = col <= row
    if 0 <= window < Tk:
        allowed = allowed & ((row - col) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=gqa)


class GaggleAttention(nn.Module):
    def __init__(self, config: GaggleConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.head_dim
        self.qk_scale = config.qk_scale
        self.ve_gate_channels = config.ve_gate_channels
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = (nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
                        if config.has_value_embedding(layer_idx) else None)

    def forward(self, x, ve, cos, sin, window, mask, past_key_values: Optional[Cache] = None):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:                                  # ResFormer value residual, gated per head
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rmsnorm(q), rmsnorm(k)                       # QK norm
        q, k = q * self.qk_scale, k * self.qk_scale         # sharper attention (upstream's 1.2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # (B, H, T, D)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        y = sdpa_like_nanochat(q, k, v, window, mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class GaggleMLP(nn.Module):
    """relu^2 feed-forward, 4x expansion, ungated (not SwiGLU)."""

    def __init__(self, config: GaggleConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class GaggleBlock(nn.Module):
    def __init__(self, config: GaggleConfig, layer_idx: int):
        super().__init__()
        self.attn = GaggleAttention(config, layer_idx)
        self.mlp = GaggleMLP(config)

    def forward(self, x, ve, cos, sin, window, mask, past_key_values=None):
        x = x + self.attn(rmsnorm(x), ve, cos, sin, window, mask, past_key_values)
        x = x + self.mlp(rmsnorm(x))
        return x


class GagglePreTrainedModel(PreTrainedModel):
    config_class = GaggleConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GaggleBlock"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # Only used by `from_config`; released checkpoints always load real weights.
        std = self.config.n_embd ** -0.5
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class GaggleModel(GagglePreTrainedModel):
    """The nanochat trunk: everything up to and including the final norm."""

    def __init__(self, config: GaggleConfig):
        super().__init__(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([GaggleBlock(config, i) for i in range(config.n_layer)]),
        })
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = nn.Linear(config.smear_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        kv_dim = config.n_kv_head * config.head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if config.has_value_embedding(i)})
        self.window_sizes = config.window_sizes
        self.gradient_checkpointing = False
        self._rope = None          # built lazily, see rotary(); NOT a buffer (meta-device loading would leave it empty)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer["wte"]

    def set_input_embeddings(self, value):
        self.transformer["wte"] = value

    def rotary(self, device, dtype):
        """Rotary tables, built the way nanochat builds them: fp32 on the model's device, 10x the context, then
        cast to the compute dtype. The model is sensitive to the last bit of these tables, so the construction
        is copied rather than re-derived."""
        key = (str(device), dtype)
        if self._rope is None or self._rope[0] != key:
            cfg = self.config
            channels = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=device)
            inv_freq = 1.0 / (cfg.rope_theta ** (channels / cfg.head_dim))
            steps = torch.arange(cfg.sequence_len * 10, dtype=torch.float32, device=device)
            freqs = torch.outer(steps, inv_freq)
            self._rope = (key, freqs.cos().to(dtype), freqs.sin().to(dtype))
        return self._rope[1], self._rope[2]

    @staticmethod
    def _is_right_padded(attention_mask: torch.Tensor) -> bool:
        """True when every row is 1s followed by 0s, which includes no padding at all."""
        m = attention_mask.to(torch.long)
        return bool((m[:, :-1] >= m[:, 1:]).all())

    def _key_masks(self, T, Tk, key_valid, device):
        """One boolean mask (B, 1, T, Tk) per distinct window, for batches with left (or interior) padding."""
        out = {}
        row = (Tk - T) + torch.arange(T, device=device).unsqueeze(1)
        col = torch.arange(Tk, device=device).unsqueeze(0)
        for window in set(self.window_sizes):
            allowed = col <= row
            if 0 <= window < Tk:
                allowed = allowed & ((row - col) <= window)
            allowed = allowed[None, None] & key_valid[:, None, None, :]
            # a query row with every key masked (a pad query) would give NaN; its output is never read, so keep
            # its own position open
            diag = torch.zeros(T, Tk, dtype=torch.bool, device=device)
            diag[torch.arange(T, device=device), row.squeeze(1)] = True
            out[window] = allowed | diag[None, None]
        return out

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        prev_input_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is None:
            raise ValueError("Gaggle needs input_ids (its value embeddings are looked up by token id); "
                             "inputs_embeds is not supported")
        B, T = input_ids.shape
        use_cache = use_cache if use_cache is not None else (getattr(self.config, "use_cache", True) and not self.training)
        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:                                # older transformers
                past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        Tk = past + T
        assert Tk <= self.config.sequence_len * 10, f"sequence length {Tk} exceeds the rotary table"

        # padding: right padding (or none) needs nothing, exactly as in nanochat; anything else masks pad keys
        key_valid = None
        if attention_mask is not None:
            am = attention_mask
            if am.size(1) < Tk:                              # a mask for the new tokens only: cache is all real
                am = torch.cat([am.new_ones(B, Tk - am.size(1)), am], dim=1)
            am = am[:, -Tk:]
            if not bool(am.all()) and not (past == 0 and self._is_right_padded(am)):
                key_valid = am.to(torch.bool)
                if position_ids is None and past == 0:       # left padding: real tokens keep their unpadded positions
                    position_ids = (am.long().cumsum(-1) - 1).clamp(min=0)

        dtype = self.transformer["wte"].weight.dtype
        cos, sin = self.rotary(input_ids.device, dtype)
        if position_ids is None:
            cos, sin = cos[None, past:Tk, None, :], sin[None, past:Tk, None, :]
        else:
            cos, sin = cos[position_ids][:, :, None, :], sin[position_ids][:, :, None, :]
        masks = (self._key_masks(T, Tk, key_valid, input_ids.device) if key_valid is not None
                 else {w: None for w in set(self.window_sizes)})

        x = rmsnorm(self.transformer["wte"](input_ids).to(dtype))
        # smear: mix the previous token's embedding into each position (none for the first token)
        c = self.config.smear_channels
        if past == 0:
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :c]))
                prev = x[:, :-1]
                if key_valid is not None:                    # never smear from a pad
                    prev = prev * key_valid[:, :T - 1, None].to(prev.dtype)
                x = torch.cat([x[:, :1], x[:, 1:] + gate * prev], dim=1)
        else:
            if prev_input_ids is None:
                prev_input_ids = getattr(past_key_values, "_gaggle_last_ids", None)
            if prev_input_ids is None:
                raise ValueError("decoding with a cache needs prev_input_ids (the token before input_ids); "
                                 "generate() passes it automatically")
            x_prev = rmsnorm(self.transformer["wte"](prev_input_ids[:, -1:]).to(dtype))
            prev = torch.cat([x_prev, x[:, :-1]], dim=1)
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., :c]))
            x = x + gate * prev
        if past_key_values is not None:
            past_key_values._gaggle_last_ids = input_ids[:, -1:]

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        all_hidden = () if output_hidden_states else None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if output_hidden_states:
                all_hidden += (x,)
            ve = (self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None)
            window = self.window_sizes[i]
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, ve, cos, sin, window, masks[window], None,
                                                      use_reentrant=False)
            else:
                x = block(x, ve, cos, sin, window, masks[window], past_key_values)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:                            # drop low-level features before the readout
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = rmsnorm(x)
        if output_hidden_states:
            all_hidden += (x,)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values if use_cache else None,
                                       hidden_states=all_hidden)


class GaggleForCausalLM(GagglePreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: GaggleConfig):
        super().__init__(config)
        self.model = GaggleModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        prev_input_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         prev_input_ids=prev_input_ids, output_hidden_states=output_hidden_states)
        h = out.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = F.linear(h[:, keep], self.lm_head.weight.to(h.dtype)).float()   # fp32, as nanochat returns
        cap = self.config.logit_softcap
        if cap:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:                               # standard shift: position t predicts labels[t + 1]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values,
                                      hidden_states=out.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      cache_position=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            cache_position=cache_position, **kwargs)
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past > 0:                                         # the smear needs the token before the new ones
            model_inputs["prev_input_ids"] = input_ids[:, past - 1:past]
        model_inputs.pop("inputs_embeds", None)
        return model_inputs


__all__ = ["GaggleConfig", "GaggleModel", "GaggleForCausalLM", "GagglePreTrainedModel"]
