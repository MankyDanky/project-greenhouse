"""Real native-CE gradient, LCE sampling, and distributed-resume checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys

import torch
from datasets import Dataset
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reranker/engines/ce'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'eval/backends'))
from dataset_nc import PairGroups, GroupSampler, CESteps, make_plan, microbatches
from train_nc import run
from modeling_nanochat_ce import load_model
from prompt_builder_nc import PairBuilder


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--worker'); args=parser.parse_args()
    if args.worker:
        run(json.loads(Path(args.worker).read_text()), use_cpu=True)
        torch.distributed.destroy_process_group()
        return
    from native_fixture import NativeFixture
    fixture=NativeFixture(); fixture.setUp()
    try:
        rows=21
        Dataset.from_dict({'query':['q1']*rows, 'pos_texts':[['p1','p2']]*rows,
            'neg_texts':[['n1','n2'] if i%2 else ['n3 word word'] for i in range(rows)],
            'pos_lens':[[1,1]]*rows, 'neg_lens':[[1,1] if i%2 else [3] for i in range(rows)],
            'query_len':[1]*rows,'subset':['test']*rows}).to_parquet(fixture.cfg['train_parquet'])
        cfg={**fixture.cfg,'k_choices':[1,2],'global_batch_groups':4,'reference_world_size':2,
             'reference_tokens_per_batch':128,'max_groups':1,'per_device_batch_size':2,'save_first_step':True}
        builder=PairBuilder(fixture.tok,prompt='nochat',max_doc_len=8,max_query_len=8)
        groups=PairGroups(cfg['train_parquet'],builder,seed=17)
        plan=make_plan(groups,builder,cfg)
        assert len(plan)>=3
        # Compare each CE step with the actual local batches yielded on both LCE ranks.
        references=[]
        for rank in range(2):
            sampler=GroupSampler(groups.avail,groups.neg_lens,groups.query_lens,max_tokens=128,
                max_groups=1,max_doc_len=8,max_query_len=8,k_choices=[1,2],seed=17,
                world_size=2,rank=rank,global_groups=4,scaffold=builder.overhead)
            packed=[]; cur=[]
            for mb,(_,last,_) in zip(sampler._batches,sampler.meta):
                cur.extend(mb)
                if last:packed.append(cur);cur=[]
            references.append(packed)
        assert len(plan)==len(references[0])==len(references[1])
        for i,step in enumerate(plan):
            assert sorted(step)==sorted(references[0][i]+references[1][i])
        data=CESteps(groups,plan,builder)
        for i in range(len(data)):
            expected=[]
            for item in plan[i]:
                g=groups[item]
                for slot,prompt in enumerate(g['seqs']):
                    expected.append((prompt,fixture.tok.convert_tokens_to_ids('true' if slot==g['pos_idx'] else 'false')))
            assert [(r['input_ids'][:-1],r['labels'][-1]) for r in data[i]['pairs']]==expected
        batch=data[0]
        model=load_model(cfg); reference=copy.deepcopy(model)
        inputs,_,_=next(microbatches(batch,10000,1,fixture.tok.pad_token_id,'cpu'))
        out=reference(**inputs)
        direct=torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,len(fixture.tok)),
            inputs['labels'][:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(out.loss,direct)
        direct.backward()
        for inputs,n,last in microbatches(batch,2,1,fixture.tok.pad_token_id,'cpu'):
            (model(**inputs).loss*n/batch['global_pairs']).backward()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-4)
        assert model.lm_head.weight.grad[5].abs().sum()>0
        print('SAMPLING_AND_FULL_VOCABULARY_GRADIENT_PASSED',flush=True)
        ref_cfg={**cfg,'output_dir':str(fixture.root/'reference')}
        run(ref_cfg,use_cpu=True)
        # Recover a crash between the last checkpoint save and the final export.
        shutil.rmtree(Path(ref_cfg['output_dir'])/'final')
        run(ref_cfg,use_cpu=True)
        assert (Path(ref_cfg['output_dir'])/'final/COMPLETE').exists()
        run(cfg,use_cpu=True,stop_after=2)
        # Resume with both a different microbatch and a different world size.
        resumed={**cfg,'per_device_batch_size':3}
        path=fixture.root/'resume.json';path.write_text(json.dumps(resumed))
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        subprocess.run([sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',
            '--master-addr=127.0.0.1',f'--master-port={port}',__file__,'--worker',str(path)],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},check=True,timeout=180)
        a=load_file(Path(ref_cfg['output_dir'])/'final/model.safetensors')
        b=load_file(Path(cfg['output_dir'])/'final/model.safetensors')
        err=max((a[k]-b[k]).abs().max().item() for k in a)
        for k in a:torch.testing.assert_close(a[k],b[k],atol=2e-5,rtol=2e-4)
        oa=torch.load(Path(ref_cfg['output_dir'])/f'checkpoint-{len(plan)}/optimizer.pt',weights_only=True)
        ob=torch.load(Path(cfg['output_dir'])/f'checkpoint-{len(plan)}/optimizer.pt',weights_only=True)
        for k in oa['state']:
            for field in ['exp_avg','exp_avg_sq','step']:
                torch.testing.assert_close(oa['state'][k][field],ob['state'][k][field],atol=2e-5,rtol=2e-4)
        la=[json.loads(x) for x in (Path(ref_cfg['output_dir'])/'train.jsonl').read_text().splitlines()]
        lb=[json.loads(x) for x in (Path(cfg['output_dir'])/'train.jsonl').read_text().splitlines()]
        assert [(x['step'],x['lr'],x['pairs']) for x in la]==[(x['step'],x['lr'],x['pairs']) for x in lb]
        print('DISTRIBUTED_RESUME_PASSED max_weight_error=',err,flush=True)
        # Optional adapters still leave the original full LM head frozen and reloadable.
        lora_cfg={**cfg,'output_dir':str(fixture.root/'lora'),'lora':True,'lora_r':2,'lora_alpha':4}
        run(lora_cfg,use_cpu=True,stop_after=1)
        from ce import CEScorer
        scorer=CEScorer(Path(lora_cfg['output_dir'])/'checkpoint-1',device='cpu',dtype=torch.float32)
        assert all(torch.isfinite(torch.tensor(scorer.score('q1',['p1','n1']))))
        original=load_model(cfg).lm_head.weight
        torch.testing.assert_close(scorer.model.get_base_model().lm_head.weight,original,atol=0,rtol=0)
        print('GROUPED_LORA_RELOAD_PASSED',flush=True)
    finally:fixture.tearDown()


if __name__=='__main__':main()
