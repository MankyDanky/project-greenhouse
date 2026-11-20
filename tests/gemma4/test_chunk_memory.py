"""Throwaway full-size maximum-length execution/checkpoint probe; never trains the real output."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reranker/engines/gemma4"))
import argparse, json, os, gc, shutil
from pathlib import Path
import torch
import torch.distributed as dist
import yaml
from shared import reference, setup_precision, total_loss, FP32Guard, assert_fp32_params
from modeling_gemma4_lce import Gemma4Reranker
from optimizer_sharding import build_optimizer, prepare_checkpoint, finish_checkpoint, ownership_report


def main():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dev=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=dev)
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='diagnostics')
    args=parser.parse_args()
    run=Path(args.out); run.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path(args.config).read_text())
    out=Path(cfg['output_dir']); cp=reference.find_latest_ckpt(out)
    if cp is None and cfg.get("require_resume_min_step", 0) > 0:
        raise RuntimeError("Required resume checkpoint is missing")
    dtype=setup_precision(cfg,print)
    torch.manual_seed(cfg['seed']+rank)
    model=Gemma4Reranker(cfg['model_name'],yes_id=1847,no_id=2416,dtype=dtype,
                         attn_implementation=cfg['attn_implementation'], model_revision=cfg.get('model_revision')).to(dev).train()
    ck=None;source_step=0
    if cp is not None:
        ck=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
        assert ck['step']>=cfg.get('require_resume_min_step',0)
        assert ck['packing']['plan_hash']==cfg['expected_plan_hash']
        source_step=ck['step'];model.load_state_dict(ck['model'],strict=True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert_fp32_params(model)
    ddp=reference.make_ddp(model);opt=build_optimizer(model,cfg)
    if ck is not None:opt.load_state_dict(ck['optimizer'])
    del ck;gc.collect()
    length=cfg['max_query_len']+cfg['max_doc_len']+12
    budget=cfg['execution_tokens_per_batch'];assert budget>=length and length==652
    guard=FP32Guard(model)
    for n in [24,budget//length]:
        ids=torch.randint(10,1000,(n,length),device=dev);ids[:,0]=2
        am=torch.ones_like(ids);pos=torch.zeros(1,dtype=torch.long,device=dev)
        fn=lambda scores:total_loss(scores,pos,tau=cfg['tau'])
        if n*length>budget:
            reference.gradcache_backward(model,ddp,ids,am,1,n,budget,fn,1.,True,True)
        else:
            loss,_=fn(ddp(ids,am).scores.view(1,n));loss.backward()
        if guard is not None:
            assert guard.remove()>0;guard=None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
        assert torch.isfinite(norm)
        opt.step();opt.zero_grad(set_to_none=False)
        del ids,am,pos
    scratch=run/'chunk_memory_probe';scratch.mkdir(exist_ok=True)
    path=scratch/'throwaway_checkpoint.pt'
    prepare_checkpoint(opt,path)
    if rank==0:
        reference.save_ckpt(path,model,opt,source_step+2,0,0,
                            {**cfg,'THROWAWAY_MEMORY_PROBE':True},None,{'probe':True})
    finish_checkpoint(opt,path)
    if rank==0:
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert len(saved['optimizer']['state'])==505
        del saved;gc.collect();path.unlink()
    dist.barrier()
    peak=torch.cuda.max_memory_allocated()/2**30
    record={'passed':True,'rank':rank,'source_checkpoint':str(cp) if cp is not None else None,'source_step':source_step,
            'max_sequence_length':length,'execution_tokens':budget,'gradcache_and_direct_backward':True,
            'full_checkpoint_save':True,'throwaway_updates_discarded':True,
            'peak_allocated_GiB':peak,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
            'device_total_GiB':torch.cuda.get_device_properties(dev).total_memory/2**30,
            **ownership_report(opt)}
    assert record['device_total_GiB']-record['peak_allocated_GiB']>=3,record
    (run/f'chunk_memory_validation_rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'GEMMA_CHUNK_MEMORY_PASS rank={rank} '+json.dumps(record),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
