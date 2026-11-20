"""Native NanoChat GPT with its complete LM head, adapted to HF Trainer's API.

The upstream forward (including vocabulary padding removal and logit softcap)
is unchanged. Only the causal label shift and standard CE loss live here.
"""
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


def load_upstream(nanochat_src, dtype):
    source = Path(nanochat_src or os.environ.get('NANOCHAT_SRC', '')).resolve()
    if not (source / 'nanochat/gpt.py').is_file():
        raise ValueError('Set nanochat_src or NANOCHAT_SRC to the nanochat checkout @ 92d63d4')
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Native NanoChat supports fp32 or bf16 in this recipe')
    if 'nanochat.common' not in sys.modules:
        os.environ['NANOCHAT_DTYPE'] = str(dtype).removeprefix('torch.')
    sys.path.insert(0, str(source))
    from nanochat.common import COMPUTE_DTYPE
    if COMPUTE_DTYPE != dtype:
        raise ValueError('NanoChat compute dtype is fixed on import; use a new process to change it')
    import nanochat.flash_attention as fa
    fa.USE_FA3 = False  # match LCE's SDPA path at both training and evaluation
    from nanochat.gpt import GPT, GPTConfig
    return GPT, GPTConfig


class UpstreamCEConfig(PretrainedConfig):
    model_type = 'nanochat_upstream_ce'

    def __init__(self, upstream_config=None, nanochat_src=None, **kwargs):
        kwargs.pop('tie_word_embeddings', None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        self.upstream_config = upstream_config or {}
        self.nanochat_src = nanochat_src
        self.vocab_size = self.upstream_config.get('vocab_size', 32768)
        self.max_position_embeddings = self.upstream_config.get('sequence_len', 2048)
        self.use_cache = False


class UpstreamCausalLM(nn.Module):
    """No new model parameters: the only child module is the original GPT."""
    accepts_loss_kwargs = True

    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.config = config
        GPT, GPTConfig = load_upstream(config.nanochat_src, dtype)
        names = {f.name for f in fields(GPTConfig)}
        model_cfg = GPTConfig(**{k: v for k, v in config.upstream_config.items() if k in names})
        with torch.device('meta'):
            self.backbone = GPT(model_cfg)
        # Nonpersistent rotary buffers are not part of the state dict.
        head_dim = model_cfg.n_embd // model_cfg.n_head
        self.backbone.cos, self.backbone.sin = self.backbone._precompute_rotary_embeddings(
            self.backbone.rotary_seq_len, head_dim, device=torch.device('cpu'))
        self._original_forwards = []

    def get_input_embeddings(self):
        return self.backbone.transformer.wte

    def get_output_embeddings(self):
        return self.backbone.lm_head

    @property
    def lm_head(self):
        return self.backbone.lm_head

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # PEFT's causal-LM wrapper expects this interface; ranking uses forward directly.
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    @property
    def is_gradient_checkpointing(self):
        return bool(self._original_forwards)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers=1):
        if every_n_layers != 1:
            raise ValueError('This recipe checkpoints every layer, matching LCE')
        if self._original_forwards:
            return
        for block in self.backbone.transformer.h:
            original = block.forward
            self._original_forwards.append((block, original))

            def forward(*args, _original=original, **kwargs):
                if self.training and torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)

            block.forward = forward

    def gradient_checkpointing_disable(self):
        for block, original in self._original_forwards:
            block.forward = original
        self._original_forwards = []

    def forward(self, input_ids, attention_mask=None, labels=None,
                num_items_in_batch=None, **kwargs):
        # Right padding + causal attention: real tokens cannot attend to padding.
        # The collator masks padded labels; inference gathers the last real position.
        logits = self.backbone(input_ids)
        loss = None
        if labels is not None:
            targets = labels[:, 1:].contiguous().reshape(-1)
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                   targets, ignore_index=-100, reduction='sum')
            denominator = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
            loss = loss / torch.as_tensor(denominator, device=loss.device).clamp_min(1)
        return CausalLMOutput(loss=loss, logits=logits)


def native_file(cfg, pattern):
    """Resolve one native asset locally or from the authenticated HF repository."""
    root = Path(cfg['model_name'])
    if root.is_dir():
        matches = list(root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {pattern} in {root}, found {len(matches)}')
        return matches[0]
    from fnmatch import fnmatch
    from huggingface_hub import HfApi, hf_hub_download
    revision = cfg.get('model_revision')
    names = HfApi().list_repo_files(cfg['model_name'], revision=revision)
    matches = [name for name in names if fnmatch(name, pattern)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {pattern} in {cfg["model_name"]}, found {matches}')
    return Path(hf_hub_download(cfg['model_name'], matches[0], revision=revision))


def load_native_model(cfg, dtype=torch.float32):
    meta = json.loads(native_file(cfg, 'meta_*.json').read_text())['model_config']
    config = UpstreamCEConfig(upstream_config=meta,
                             nanochat_src=os.environ.get('NANOCHAT_SRC') or cfg.get('nanochat_src'))
    config._name_or_path = cfg['model_name']
    model = UpstreamCausalLM(config, dtype=dtype)
    state = torch.load(native_file(cfg, 'model_*.pt'), map_location='cpu', weights_only=True, mmap=True)
    state = {k: v.to(dtype).contiguous() for k, v in state.items()}
    model.backbone.load_state_dict(state, strict=True, assign=True)
    return model


def load_saved_native_model(directory, nanochat_src=None, dtype=torch.float32):
    """Reload full fine-tuned weights without downloading the original base again."""
    from safetensors.torch import load_file
    directory = Path(directory)
    config = UpstreamCEConfig.from_json_file(directory / 'upstream_config.json')
    config.nanochat_src = nanochat_src or os.environ.get('NANOCHAT_SRC') or config.nanochat_src
    model = UpstreamCausalLM(config, dtype=dtype)
    state = load_file(directory / 'model.safetensors')
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
    return model
