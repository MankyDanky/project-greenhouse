"""Gaggle pointwise reranker, as a self-contained `transformers` model.

This is a faithful port of karpathy/nanochat @ 92d63d4 (`nanochat/gpt.py`) plus the two-row scoring
head of the reranker, with two deliberate differences from the training-time code:

  * attention runs through `torch.nn.functional.scaled_dot_product_attention` only -- no FA3, no
    `kernels` download. The training/eval harness already pinned the SDPA path, so this is the same
    computation, not an approximation;
  * padding is handled explicitly. The harness assumes right padding under causal attention (a real
    token can then never attend to a pad), whereas this port masks pad keys and makes the embedding
    "smear" skip a padded predecessor, so left- and right-padded batches give identical scores.

The score is read as

    s(q, d) = logit(" true") - logit(" false")

at the last real token, from the two pretrained LM-head rows kept in `score` (fp32). nanochat's
final `15 * tanh(logits / 15)` logit softcap is deliberately NOT applied: the reranker was trained
on the raw difference, so scores are unbounded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

try:                                    # repo layout (trust_remote_code)
    from .configuration_gaggle import GaggleRerankerConfig
except ImportError:                     # flat directory
    from configuration_gaggle import GaggleRerankerConfig


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """nanochat's parameter-free RMSNorm (runs in the activation dtype, as upstream)."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """nanochat's rotation: the TRANSPOSE of the textbook/HF convention (it rotates by -theta).

    Only the relative q/k rotation matters, so this is functionally equivalent to the usual form,
    but it is not interchangeable with it: `rotate_half` from the Llama family would flip the sign
    of `sin` and silently produce a different model.
    """
    assert x.ndim == 4                                      # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class GaggleAttention(nn.Module):
    def __init__(self, config: GaggleRerankerConfig, layer_idx: int):
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

    def forward(self, x, ve, cos, sin, attn_mask, window):
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

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # (B, H, T, D) for SDPA
        gqa = q.size(1) != k.size(1)
        if attn_mask is None and (window < 0 or window >= T):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class GaggleMLP(nn.Module):
    """relu^2 feed-forward, 4x expansion, ungated (not SwiGLU)."""

    def __init__(self, config: GaggleRerankerConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class GaggleBlock(nn.Module):
    def __init__(self, config: GaggleRerankerConfig, layer_idx: int):
        super().__init__()
        self.attn = GaggleAttention(config, layer_idx)
        self.mlp = GaggleMLP(config)

    def forward(self, x, ve, cos, sin, attn_mask, window):
        x = x + self.attn(rmsnorm(x), ve, cos, sin, attn_mask, window)
        x = x + self.mlp(rmsnorm(x))
        return x


class GagglePreTrainedModel(PreTrainedModel):
    config_class = GaggleRerankerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GaggleBlock"]
    _supports_sdpa = True
    # Note on the two scoring rows: they are deliberately NOT pinned to fp32. The training and
    # evaluation wrapper casts the whole model to the compute dtype and only then upcasts them
    # (`m.to(dtype)` followed by `m.head.data.float()`), so every number in the model card was
    # produced with rows rounded to bf16 and a fp32 matmul. Keeping them in fp32 here instead was
    # measurably *more* precise and therefore wrong: it moved scores by up to 0.24 and reordered
    # near-ties. `forward` does the matmul in fp32, which reproduces the wrapper exactly.

    def _init_weights(self, module):
        # Only used by `from_config`; released checkpoints always load real weights.
        std = self.config.n_embd ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class GaggleModel(GagglePreTrainedModel):
    """The nanochat trunk: everything up to and including the final norm."""

    def __init__(self, config: GaggleRerankerConfig):
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
        self._rope = None          # built lazily, see rotary(); NOT a buffer, see below
        self.post_init()

    def rotary(self, T, device, dtype):
        """Rotary tables for the first T positions, cached per (device, dtype).

        Deliberately not a registered buffer: `from_pretrained` builds the model on the meta device
        and materialises it from the checkpoint, and these tables are not in the checkpoint, so a
        non-persistent buffer comes back as uninitialised memory -- which is how the first version
        of this file produced NaN from the first layer on, with every weight loaded correctly.
        Computing them here costs one 2048x64 outer product per dtype.
        """
        key = (str(device), dtype)
        if self._rope is None or self._rope[0] != key:
            cfg = self.config
            # Built exactly the way upstream builds it, because the table is not bit-exact across
            # construction choices and the model is sensitive to the last bit: on CPU, in fp32, at
            # 10x the context (upstream over-computes the cache), cast to the compute dtype, and
            # only then moved to the device. Computing it directly on the GPU at 1x length changes
            # ONE sin entry, at position 365, by one bf16 ULP -- which is enough to shift the score
            # of a 547-token pair by 0.14 and nDCG@10 by ~0.001 against the published numbers.
            cpu = torch.device("cpu")
            channels = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=cpu)
            inv_freq = 1.0 / (cfg.rope_theta ** (channels / cfg.head_dim))
            steps = torch.arange(cfg.sequence_len * 10, dtype=torch.float32, device=cpu)
            freqs = torch.outer(steps, inv_freq)
            cos = freqs.cos().to(dtype)[None, :, None, :].to(device)
            sin = freqs.sin().to(dtype)[None, :, None, :].to(device)
            self._rope = (key, cos, sin)
        _, cos, sin = self._rope
        return cos[:, :T], sin[:, :T]

    def get_input_embeddings(self):
        return self.transformer["wte"]

    def set_input_embeddings(self, value):
        self.transformer["wte"] = value

    def _masks(self, T, attention_mask, device):
        """One boolean attention mask per distinct window; None means "plain causal, let SDPA do it".

        With no padding this reproduces upstream's own fallback (nanochat/flash_attention.py)
        exactly, down to the 2-D mask shape, so that SDPA dispatches to the same kernel and the
        scores are bit-identical to the harness every published number came from. Padding is what
        upstream does not handle: it assumes right padding under causal attention, where a real
        token can never attend to a pad anyway. Masking pad keys is mathematically the same thing
        for right padding, and is what makes left padding work too.
        """
        key_valid = None
        if attention_mask is not None and not self._is_right_padded(attention_mask):
            key_valid = attention_mask.to(torch.bool)
        out = {}
        for window in set(self.window_sizes):
            if key_valid is None and (window < 0 or window >= T):
                out[window] = None                      # upstream's fast path: is_causal=True
                continue
            i = torch.arange(T, device=device).unsqueeze(1)
            j = torch.arange(T, device=device).unsqueeze(0)
            allowed = j <= i
            if 0 <= window < T:
                allowed = allowed & ((i - j) <= window)  # upstream uses <=, i.e. window+1 keys
            if key_valid is not None:
                allowed = allowed[None, None] & key_valid[:, None, None, :]
                # a fully padded query row would make softmax produce NaN; its output is never read,
                # so keep the diagonal open to keep the row finite
                allowed = allowed | torch.eye(T, dtype=torch.bool, device=device)[None, None]
            out[window] = allowed                        # (T, T) unpadded, (B, 1, T, T) padded
        return out

    @staticmethod
    def _is_right_padded(attention_mask: torch.Tensor) -> bool:
        """True when every row is 1s followed by 0s (which includes no padding at all).

        Such a batch needs no key mask and no masking in the smear: under causal attention a real
        token only ever attends to positions at or before itself, and those are all real. Taking
        that route keeps this path bit-identical to the harness every published number came from.
        The pad ROWS do then attend to real tokens and carry junk, exactly as upstream, which is
        harmless because the score is read at the last real token -- but it does mean the hidden
        states at padded positions are not meaningful if you pool over them yourself.
        """
        m = attention_mask.to(torch.long)
        return bool((m[:, :-1] >= m[:, 1:]).all())

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.sequence_len, (
            f"sequence length {T} exceeds the backbone context {self.config.sequence_len}")
        dtype = self.transformer["wte"].weight.dtype
        cos, sin = self.rotary(T, input_ids.device, dtype)
        masks = self._masks(T, attention_mask, input_ids.device)

        x = rmsnorm(self.transformer["wte"](input_ids).to(dtype))
        if T > 1:                                        # smear: mix in the previous token's embedding
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :self.config.smear_channels]))
            prev = x[:, :-1]
            if attention_mask is not None and not self._is_right_padded(attention_mask):
                # skip a padded predecessor, which only arises with left padding; for right padding
                # this multiplication is a no-op at every position whose output is read, so it is
                # skipped to stay bit-identical to upstream
                prev = prev * attention_mask[:, :-1, None].to(prev.dtype)
            x = torch.cat([x[:, :1], x[:, 1:] + gate * prev], dim=1)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (self.value_embeds[str(i)](input_ids).to(x.dtype)
                  if str(i) in self.value_embeds else None)
            window = self.window_sizes[i]
            args = (x, ve, cos, sin, masks[window], window)
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=True)
            else:
                x = block(*args)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:                        # drop low-level features before the readout
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return rmsnorm(x)


@dataclass
class GaggleRerankerOutput(ModelOutput):
    logits: torch.FloatTensor = None            # (B, 1) the relevance score
    true_false_logits: torch.FloatTensor = None  # (B, 2) the two LM-head rows, fp32
    last_hidden_state: Optional[torch.FloatTensor] = None


class GaggleRerankerForSequenceClassification(GagglePreTrainedModel):
    """Pointwise (query, passage) reranker. `logits[:, 0]` is the score; higher is more relevant."""

    def __init__(self, config: GaggleRerankerConfig):
        super().__init__(config)
        self.model = GaggleModel(config)
        self.score = nn.Linear(config.n_embd, 2, bias=False)   # rows: " true", " false" (fp32)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True, **kwargs) -> GaggleRerankerOutput:
        h = self.model(input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            last = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
        else:
            # index of the LAST real token, not attention_mask.sum(1) - 1: those agree only for
            # right padding, and reading sum-1 under left padding lands on a pad, which cost ~6
            # points of score in the first version of this file
            m = attention_mask.to(torch.long)
            last = (m.size(1) - 1) - m.flip(1).argmax(1)
        last = last - self.config.readout_offset
        h_last = h[torch.arange(h.size(0), device=h.device), last]
        # fp32 matmul whatever dtype the rows are stored in, so the difference of the two logits
        # is exact -- this is the wrapper's `h_last.float() @ self.head.t()`
        yn = h_last.float() @ self.score.weight.float().t()    # (B, 2), fp32
        logits = (yn[:, 0] - yn[:, 1]).unsqueeze(-1)           # (B, 1)
        return GaggleRerankerOutput(logits=logits, true_false_logits=yn,
                                    last_hidden_state=h if output_hidden_states else None)

    # -- convenience ------------------------------------------------------------------------------
    def build_input_ids(self, tokenizer, query: str, passage: str) -> List[int]:
        """The training prompt, assembled from the token ids recorded in the config."""
        cfg = self.config
        enc = lambda text, cap: tokenizer.encode(
            text, add_special_tokens=False, split_special_tokens=True)[:cap]
        return (list(cfg.prompt_prefix_ids) + enc(query, cfg.max_query_len)
                + list(cfg.prompt_mid_ids) + enc(passage, cfg.max_doc_len)
                + list(cfg.prompt_suffix_ids))

    @torch.inference_mode()
    def compute_score(self, pairs: Sequence[Tuple[str, str]], tokenizer=None,
                      batch_size: int = 16) -> List[float]:
        """Score (query, passage) pairs. Independent per pair, so order and batching are free."""
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        pad = self.config.pad_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        device = next(self.parameters()).device
        scores: List[float] = []
        for s0 in range(0, len(pairs), batch_size):
            seqs = [self.build_input_ids(tokenizer, q, d) for q, d in pairs[s0:s0 + batch_size]]
            n = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                att[i, :len(s)] = 1
            out = self(ids.to(device), att.to(device))
            scores += out.logits[:, 0].float().tolist()
        return scores

    def rerank(self, query: str, passages: Sequence[str], tokenizer=None, batch_size: int = 16):
        """Passages sorted by score, most relevant first, as (score, passage) pairs."""
        scores = self.compute_score([(query, p) for p in passages], tokenizer, batch_size)
        return sorted(zip(scores, passages), key=lambda t: -t[0])


__all__ = ["GaggleRerankerConfig", "GaggleModel", "GaggleRerankerForSequenceClassification",
           "GaggleRerankerOutput"]
