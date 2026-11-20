"""Score LCE-format cached BM25 runs using the original causal LM head."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reranker/engines/ce'))
from prompt_builder_nc import PairBuilder, yes_no_ids, setup_tokenizer
from modeling_nanochat_ce import load_model

QRELS = {
    **{f'dl{y}':f'qrels.dl{y}-passage.txt' for y in range(19,24)},
    'covid':'qrels.beir-v1.0.0-trec-covid.test.txt',
    'news':'qrels.beir-v1.0.0-trec-news.test.txt',
    'robust04':'qrels.beir-v1.0.0-robust04.test.txt',
    'nfc':'qrels.beir-v1.0.0-nfcorpus.test.txt',
    'scifact':'qrels.beir-v1.0.0-scifact.test.txt',
    'scidocs':'qrels.beir-v1.0.0-scidocs-test.txt',
    'fiqa':'qrels.beir-v1.0.0-fiqa-test.txt',
}


def doc_text(candidate):
    doc = candidate.get('doc') or {}
    text = doc.get('contents') or doc.get('text') or ''
    title = doc.get('title') or ''
    if title and not text.startswith(title):
        text = f'{title}\n{text}'
    return ' '.join(text.split())


class CEScorer:
    def __init__(self, checkpoint, device='cuda', dtype=torch.bfloat16):
        checkpoint = Path(checkpoint)
        cfg = json.loads((checkpoint / 'recipe_config.json').read_text())
        tok = setup_tokenizer(AutoTokenizer.from_pretrained(checkpoint))
        self.builder = PairBuilder(tok, max_doc_len=cfg['max_doc_len'],
                                   max_query_len=cfg['max_query_len'], prompt='nochat')
        self.true_id, self.false_id = yes_no_ids(tok, 'nochat')
        if (checkpoint / 'adapter_config.json').exists():
            from peft import PeftModel
            base = load_model(cfg, dtype=dtype, add_lora=False)
            self.model = PeftModel.from_pretrained(base, checkpoint)
        elif cfg.get('backbone_impl', 'hf') == 'upstream':
            from modeling_nanochat_upstream_ce import load_saved_native_model
            self.model = load_saved_native_model(checkpoint, dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def score(self, query, docs, batch_size=16):
        scores=[]
        for start in range(0,len(docs),batch_size):
            seqs=[self.builder.build(query,doc) for doc in docs[start:start+batch_size]]
            n=max(map(len,seqs))
            ids=torch.full((len(seqs),n),self.builder.tok.pad_token_id,dtype=torch.long,device=self.device)
            mask=torch.zeros_like(ids)
            for i,seq in enumerate(seqs):
                ids[i,:len(seq)]=torch.tensor(seq,device=self.device)
                mask[i,:len(seq)]=1
            # Preserve the model's native full vocabulary head and logit softcap.
            logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            final=logits[torch.arange(len(seqs),device=self.device),mask.sum(1)-1].float()
            scores.extend((final[:,self.true_id]-final[:,self.false_id]).cpu().tolist())
        return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,help='HF final/ or checkpoint-N directory')
    p.add_argument('--dataset',required=True,choices=list(QRELS))
    p.add_argument('--tag',default='nanochat-ce')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--dtype',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--out',default='eval/runs')
    args=p.parse_args()
    if args.dtype=='fp32':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    runs=Path(os.environ.get('EVAL_RUNS','data/bm25_top100'))
    qrels=Path(os.environ.get('EVAL_QRELS_DIR','eval/qrels'))/QRELS[args.dataset]
    scorer=CEScorer(args.model,dtype=torch.float32 if args.dtype=='fp32' else torch.bfloat16)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    run=out/f'{args.tag}_{args.dataset}.trec'
    with (runs/f'retrieve_results_{args.dataset}_top100.jsonl').open() as source,run.open('w') as dest:
        for line in source:
            row=json.loads(line);candidates=row['candidates']
            scores=scorer.score(row['query']['text'],[doc_text(c) for c in candidates],args.batch_size)
            for rank,i in enumerate(sorted(range(len(scores)),key=lambda i:-scores[i]),1):
                dest.write(f"{row['query']['qid']} Q0 {candidates[i]['docid']} {rank} {scores[i]:.10f} {args.tag}\n")
    result=subprocess.check_output([os.environ.get('EVAL_TREC_EVAL','trec_eval'),'-c','-m','ndcg_cut.10',
        '-m','map','-m','recip_rank',str(qrels),str(run)],text=True)
    run.with_suffix('.metrics.txt').write_text(result)
    print(result)


if __name__=='__main__':
    main()
