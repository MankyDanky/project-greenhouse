"""Load the original causal LM, retaining its entire vocabulary head."""
import torch
from transformers import AutoModelForCausalLM


def load_model(cfg, dtype=torch.float32, add_lora=True):
    if cfg.get('backbone_impl', 'hf') == 'upstream':
        from modeling_nanochat_upstream_ce import load_native_model
        model = load_native_model(cfg, dtype=dtype)
        default_targets = ['c_q', 'c_k', 'c_v', 'c_proj', 'c_fc']
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_name'], dtype=dtype, revision=cfg.get('model_revision'),
            attn_implementation=cfg.get('attn_implementation', 'sdpa'),
        )
        default_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2']
    model.config.use_cache = False
    # Preserve LCE's base-model config, including the original logit softcap.
    # No replaced/sliced head, scoring layer, or newly added tokens.
    if cfg.get('lora', False) and add_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type='CAUSAL_LM', r=cfg.get('lora_r', 32),
            lora_alpha=cfg.get('lora_alpha', 64), lora_dropout=0.0,
            target_modules=cfg.get('lora_targets', default_targets),
            bias='none',
        ))
    return model


def load_tokenizer(cfg):
    """Use LCE's HF tokenizer interface, preserving the native encoding exactly."""
    from pathlib import Path
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from prompt_builder_nc import setup_tokenizer, yes_no_ids
    if cfg.get('backbone_impl', 'hf') != 'upstream' or (Path(cfg['model_name']) / 'tokenizer.json').exists():
        return setup_tokenizer(AutoTokenizer.from_pretrained(
            cfg['model_name'], revision=cfg.get('model_revision')))
    # The private native repository supplies a tiktoken encoding, not tokenizer.json.
    # This is tokenizer conversion only; GPT weights/architecture are never converted.
    import pickle
    import tempfile
    from transformers.integrations.tiktoken import convert_tiktoken_to_fast
    from modeling_nanochat_upstream_ce import native_file
    with native_file(cfg, 'tokenizer/tokenizer.pkl').open('rb') as source:
        encoding = pickle.load(source)
    with tempfile.TemporaryDirectory(prefix='nanochat-ce-tokenizer-') as tmp:
        convert_tiktoken_to_fast(encoding, tmp)
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(tmp) / 'tokenizer.json'))
    # Match LCE's conversion: special markers in the scaffold, ordinary encoding in text.
    for text in ['<|bos|>Query: ', '\nPassage: ', '\nRelevant:', ' true', ' false',
                 'café 中文 😀\t12345', '<|assistant_start|>literal marker']:
        assert tok.encode(text, add_special_tokens=False) == encoding.encode(text, allowed_special='all')
        assert tok.encode(text, add_special_tokens=False, split_special_tokens=True) == encoding.encode_ordinary(text)
    tok = setup_tokenizer(tok)
    yes_no_ids(tok, 'nochat')
    return tok
