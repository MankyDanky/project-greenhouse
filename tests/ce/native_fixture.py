"""Shared tiny native NanoChat fixture for grouped CE and model parity tests."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reranker/engines/ce'))
from modeling_nanochat_upstream_ce import load_upstream


class NativeFixture:
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        vocab = {s:i for i,s in enumerate(['[PAD]', '[EOS]', '[UNK]', 'true', 'false',
            'q1', 'q2', 'p1', 'p2', 'p3', 'n1', 'n2', 'n3', 'word'])}
        backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        self.tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                           eos_token='[EOS]', unk_token='[UNK]')
        GPT, GPTConfig = load_upstream(os.environ['NANOCHAT_SRC'], torch.float32)
        config = GPTConfig(sequence_len=256, vocab_size=len(self.tok), n_layer=4,
                           n_head=2, n_kv_head=1, n_embd=32, window_pattern='SSSL')
        self.reference = GPT(config)
        self.reference.init_weights()
        # Nontrivial values exercise every part of the native forward.
        with torch.no_grad():
            self.reference.smear_lambda.fill_(0.3)
            for block in self.reference.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
        base = self.root / 'native'
        base.mkdir()
        self.tok.save_pretrained(base)
        (base / 'meta_1.json').write_text(json.dumps({'model_config': asdict(config)}))
        torch.save(self.reference.state_dict(), base / 'model_1.pt')
        self.cfg = dict(model_name=str(base), train_parquet=str(self.root/'train.parquet'),
            output_dir=str(self.root/'out'), prompt='nochat', max_doc_len=8,
            max_query_len=8, lora=False, attn_implementation='sdpa', precision='fp32',
            grad_checkpointing=True, per_device_batch_size=2,
            seed=17, epochs=1, lr=0.01, weight_decay=0.01, warmup_steps=1, clip=12.,
            num_workers=0, resume='auto', save_every=1, keep_ckpts=1, log_every=1,
            wandb_project=None)

        self.cfg.update(backbone_impl='upstream', nanochat_src=os.environ['NANOCHAT_SRC'])

    def tearDown(self):
        self.tmp.cleanup()
