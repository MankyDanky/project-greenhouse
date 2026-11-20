"""Uniform model/tokenizer loading for CORE and MMLU: nanochat checkpoints or HuggingFace models.
Every model gets its own tokenizer's BOS token prepended and a 2,048-token context window."""
import os, sys, torch

def native_prefix(t):
    """Token the tokenizer prepends by default (add_special_tokens=True), or None if it prepends nothing."""
    s = "The quick brown fox"
    with_sp = t(s, add_special_tokens=True)["input_ids"]; plain = t(s, add_special_tokens=False)["input_ids"]
    for k in range(len(with_sp) - len(plain) + 1):  # tolerate an appended EOS
        if with_sp[k:k + len(plain)] == plain:
            assert k <= 1, f"tokenizer prepends {k} tokens"
            return with_sp[0] if k == 1 else None
    raise ValueError("plain encoding not found inside the special-token encoding")

class HFTok:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.bos = self.t.bos_token_id if self.t.bos_token_id is not None else self.t.eos_token_id
        # EVAL_NO_PREFIX=1: prepend nothing (check for models trained with document masking and no BOS).
        if os.environ.get("EVAL_NO_PREFIX") == "1":
            self.bos = None
        # EVAL_PREFIX=native: prepend exactly what the tokenizer itself adds by default (lm-eval convention):
        # its BOS if it adds one, otherwise nothing.
        if os.environ.get("EVAL_PREFIX") == "native":
            self.bos = native_prefix(self.t)
    def get_bos_token_id(self):
        return self.bos
    def encode(self, text, prepend=None):
        ids = self.t(text, add_special_tokens=False)["input_ids"]
        return ([prepend] if prepend is not None else []) + ids
    def __call__(self, texts, prepend=None):
        if isinstance(texts, str):
            return self.encode(texts, prepend)
        enc = self.t(list(texts), add_special_tokens=False)["input_ids"]
        return [([prepend] if prepend is not None else []) + ids for ids in enc]

def _host_copy(t, device):
    """GPU->GPU copy staged through host memory (for nodes whose peer-to-peer copies are broken, e.g. watgpu108)."""
    device = torch.device(device)
    if isinstance(t, torch.Tensor) and t.is_cuda and device.type == "cuda" and t.device != device:
        return t.cpu().to(device)
    return t.to(device)

def _route_cross_gpu_copies_via_host():
    """Make accelerate's device_map hooks move activations between GPUs through the CPU instead of peer copies."""
    import accelerate.hooks as H, accelerate.utils.operations as O
    orig = O.send_to_device
    def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor):
            return _host_copy(tensor, device)
        return orig(tensor, device, non_blocking=False, skip_keys=skip_keys)
    O.send_to_device = send_to_device; H.send_to_device = send_to_device

class HFModel(torch.nn.Module):
    def __init__(self, name, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM
        # EVAL_DEVICE_MAP=auto: split a model too large for one GPU across the visible GPUs (layer placement only; the
        # arithmetic is unchanged). Unset = the single-GPU path every earlier number used.
        dm = os.environ.get("EVAL_DEVICE_MAP")
        if dm and os.environ.get("EVAL_P2P_VIA_HOST") == "1":
            _route_cross_gpu_copies_via_host()
        kw = dict(torch_dtype=dtype, trust_remote_code=True, **({"device_map": dm} if dm else {}))
        place = (lambda m: m.eval()) if dm else (lambda m: m.cuda().eval())
        # EVAL_HF_CLASS=vl: load as the image-text model even though AutoModelForCausalLM accepts the config
        # (multimodal releases store the LM under model.language_model.*, which the text-only class does not map).
        try:
            if os.environ.get("EVAL_HF_CLASS") == "vl":
                raise ValueError("forced image-text-to-text class")
            self.m = place(AutoModelForCausalLM.from_pretrained(name, **kw))
        except ValueError:
            # Multimodal checkpoints (Qwen3.5, Gemma 4) without a causal-LM mapping: load the full model, text-only input.
            from transformers import AutoModelForImageTextToText
            self.m = place(AutoModelForImageTextToText.from_pretrained(name, **kw))
        self.multi = bool(dm)
        cfg = self.m.config
        tcfg = getattr(cfg, "text_config", None) or getattr(cfg, "get_text_config", lambda: cfg)()
        self.ctx = getattr(cfg, "max_position_embeddings", None) or getattr(tcfg, "max_position_embeddings", None) or 2048
    @torch.no_grad()
    def forward(self, idx):
        if self.multi:
            dev = self.m.get_input_embeddings().weight.device
            if os.environ.get("EVAL_P2P_VIA_HOST") == "1":
                return _host_copy(self.m(input_ids=_host_copy(idx, dev), use_cache=False).logits, idx.device)
            return self.m(input_ids=idx.to(dev), use_cache=False).logits.to(idx.device)
        return self.m(input_ids=idx, use_cache=False).logits

def load(kind, src=None, ckpt=None, step=None, hf=None):
    """Returns (model, tokenizer, native_context_len, description)."""
    if kind == "hf":
        tok = HFTok(hf); model = HFModel(hf)
        return model, tok, model.ctx, hf
    os.environ["NANOCHAT_BASE_DIR"] = ckpt
    sys.path.insert(0, src)
    from nanochat.checkpoint_manager import build_model
    model, tok, meta = build_model(ckpt, step, torch.device("cuda"), phase="eval")
    if os.environ.get("EVAL_AUTOCAST_BF16") == "1":
        # Older nanochat (e.g. karpathy/nanochat-d34 @ f37d45c) keeps fp32 weights and was evaluated under bf16 autocast.
        inner = model
        class Autocast(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.inner = inner
            @torch.no_grad()
            def forward(self, idx):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.inner(idx).float()
        model = Autocast()
    return model, tok, meta["model_config"]["sequence_len"], f"nanochat:{ckpt}@{step}"
