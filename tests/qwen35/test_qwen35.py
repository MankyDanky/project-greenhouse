from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reranker/engines/qwen35"))
"""Validate native tied-head LCE gradients, padding, GradCache, and checkpoint reload."""
import copy,json,sys,tempfile,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
from modeling_qwen35_lce import Qwen35Reranker
from train import gradcache_backward
from shared import total_loss

def run(device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    c=Qwen3_5TextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,
        linear_value_head_dim=16,linear_conv_kernel_dim=4,
        layer_types=["linear_attention"]*3+["full_attention"],
        tie_word_embeddings=True,attention_dropout=0.,
        rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.,
                         "mrope_section":[2,3,3],"mrope_interleaved":True})
    c._attn_implementation="sdpa"
    native=Qwen3_5ForCausalLM(c).float().to(device).train()
    model=Qwen35Reranker(full=copy.deepcopy(native),yes_id=7,no_id=9).to(device).train()
    ids=torch.randint(10,128,(8,80),device=device)
    ids[:,0]=7
    am=torch.ones_like(ids);am[1,67:]=0;ids[1,67:]=0
    pos=torch.tensor([1,3],device=device)
    logits=native(input_ids=ids,attention_mask=am,use_cache=False).logits
    idx=am.sum(1)-1
    yn=logits[torch.arange(8,device=device),idx][:,[7,9]]
    score=yn[:,0]-yn[:,1]
    out=model(ids,am)
    torch.testing.assert_close(out.yes_no,yn,atol=2e-5,rtol=2e-5)
    loss_fn=lambda s:total_loss(s,pos,tau=1.)
    loss=F.cross_entropy(score.view(2,4),pos)
    loss.backward()
    mloss,_=loss_fn(out.scores.view(2,4));mloss.backward()
    for name,p in model.backbone.named_parameters():
        p0=dict(native.model.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,p0.grad,atol=3e-5,rtol=2e-4,msg=name)
    with torch.no_grad():
        alone=model(ids[1:2,:67],am[1:2,:67]).scores
        torch.testing.assert_close(alone,out.scores[1:2],atol=2e-5,rtol=2e-5)
    cached=copy.deepcopy(model);cached.zero_grad(set_to_none=True)
    cached.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    gradcache_backward(cached,cached,ids,am,2,4,160,loss_fn,1.,True,False)
    for (name,p),(_,q) in zip(model.named_parameters(),cached.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4,msg=name)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
    opt.step()
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"state.pt"
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict()},path)
        loaded=torch.load(path,weights_only=True,map_location=device)
        restored=copy.deepcopy(model);restored.load_state_dict(loaded["model"],strict=True)
        opt2=torch.optim.AdamW(restored.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
        opt2.load_state_dict(loaded["optimizer"])
        for obj,op in [(model,opt),(restored,opt2)]:
            op.zero_grad()
            l,_=loss_fn(obj(ids,am).scores.view(2,4));l.backward();op.step()
        for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    return {"native_tied_head_logits_and_gradients":True,"right_padding":True,
            "gradcache_with_gradient_checkpointing":True,"adamw_checkpoint_roundtrip":True,
            "all_text_parameters_trainable":model.trainable_report(),"device":device,
            "initial_lce_loss":float(loss.detach())}

if __name__=="__main__":
    device="cuda" if "--cuda" in sys.argv else "cpu"
    result=run(device)
    out=Path("diagnostics")/("validation_gpu.json" if device=="cuda" else "validation_cpu.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print("QWEN35_LCE_VALIDATION_PASSED",json.dumps(result),flush=True)
