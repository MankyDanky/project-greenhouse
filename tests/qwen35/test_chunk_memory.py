from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reranker/engines/qwen35"))
"""Isolated full-size Qwen chunk/optimizer/checkpoint memory test; no real training updates."""
import datetime, gc, json, os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from shared import reference, setup_precision, total_loss, FP32Guard, assert_fp32_params
from modeling_qwen35_lce import Qwen35Reranker
from qwen_prompt import PairBuilder, setup_tokenizer

def main():
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local);dev=torch.device("cuda",local)
    dist.init_process_group("nccl",device_id=dev,timeout=datetime.timedelta(minutes=15))
    import argparse, yaml
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="diagnostics")
    args=ap.parse_args()
    run=Path(args.out);run.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path(args.config).read_text())
    from shared import validate_native_config
    validate_native_config(cfg)
    dtype=setup_precision(cfg,print);torch.manual_seed(cfg["seed"]+rank)
    model=Qwen35Reranker(cfg["model_name"],dtype=dtype,attn_implementation=cfg["attn_implementation"],model_revision=cfg.get("model_revision")).to(dev).train()
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    assert_fp32_params(model)
    ddp=reference.make_ddp(model)
    opt=torch.optim.AdamW(list(model.parameters()),lr=cfg["lr"],weight_decay=cfg["weight_decay"],betas=(0.9,0.95),fused=True)
    source_step=0;cp=None
    if cfg.get("require_resume_min_step"):
        cp=reference.find_latest_ckpt(Path(cfg["output_dir"]))
        assert cp is not None
        ck=torch.load(cp,map_location="cpu",mmap=True,weights_only=False)
        assert ck["step"]>=cfg["require_resume_min_step"] and ck["packing"]["plan_hash"]==cfg["expected_plan_hash"]
        assert ck["packing"]["objective"]=="lce_qwen35_tied_gap"
        source_step=ck["step"];model.load_state_dict(ck["model"],strict=True);opt.load_state_dict(ck["optimizer"])
        del ck;gc.collect()
    tok=setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model_name"],revision=cfg.get("model_revision")))
    builder=PairBuilder(tok,max_query_len=cfg["max_query_len"],max_doc_len=cfg["max_doc_len"],prompt=cfg["prompt"])
    length=cfg["max_query_len"]+cfg["max_doc_len"]+builder.overhead
    budget=cfg["execution_tokens_per_batch"];assert budget >= length
    guard=FP32Guard(model)
    for n in [max(24,2*(budget//length)),budget//length]:
        ids=torch.randint(10,1000,(n,length),device=dev);am=torch.ones_like(ids)
        pos=torch.zeros(1,dtype=torch.long,device=dev)
        fn=lambda scores:total_loss(scores,pos,tau=cfg["tau"])
        if ids.numel()>budget:
            reference.gradcache_backward(model,ddp,ids,am,1,n,budget,fn,1.,True,True)
        else:
            loss,_=fn(ddp(ids,am).scores.view(1,n));loss.backward()
        if guard is not None:
            assert guard.remove()>0;guard=None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["clip"])
        assert torch.isfinite(norm)
        opt.step();opt.zero_grad(set_to_none=False)
        del ids,am,pos
    scratch=run/"chunk_memory_probe";scratch.mkdir(exist_ok=True)
    path=scratch/"throwaway_checkpoint.pt"
    if rank==0:
        reference.save_ckpt(path,model,opt,source_step+2,0,0,{**cfg,"THROWAWAY_MEMORY_PROBE":True},None,{"probe":True})
        saved=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
        states=saved["optimizer"]["state"]
        assert len(states)==len(list(model.parameters()))
        assert sum(x["exp_avg"].numel() for x in states.values())==sum(p.numel() for p in model.parameters())
        assert all(x["exp_avg"].dtype==torch.float32 and x["exp_avg_sq"].dtype==torch.float32 and x["step"].item()==source_step+2 for x in states.values())
        del saved,states;gc.collect();path.unlink()
    dist.barrier()
    peak=torch.cuda.max_memory_allocated()/2**30
    record=dict(passed=True,rank=rank,source_checkpoint=str(cp) if cp else None,source_step=source_step,max_sequence_length=length,execution_tokens=budget,
        gradcache_and_direct_backward=True,full_checkpoint_save=True,
        throwaway_updates_discarded=True,peak_allocated_GiB=peak,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        device_total_GiB=torch.cuda.get_device_properties(dev).total_memory/2**30,
        trainable_parameters=sum(p.numel() for p in model.parameters()),optimizer="torch.optim.AdamW",
        fused=True,optimizer_state_sharding=False)
    assert record["device_total_GiB"]-peak>=3,record
    (run/f"chunk_memory_validation_rank{rank}.json").write_text(json.dumps(record,indent=2)+"\n")
    print(f"QWEN_CHUNK{budget}_MEMORY_PASS rank={rank} "+json.dumps(record),flush=True)
    dist.destroy_process_group()
if __name__=="__main__":main()
