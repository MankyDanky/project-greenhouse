"""Native full-vocabulary parity, PLE/shared-KV gradients, GradCache, AdamW reload."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reranker/engines/gemma4"))
import copy,json,sys,tempfile,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Gemma4ForCausalLM,Gemma4TextConfig
from modeling_gemma4_lce import Gemma4Reranker
from shared import total_loss,reference

def run(device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    c=Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,
        hidden_size=64,intermediate_size=128,num_hidden_layers=10,
        num_attention_heads=4,num_key_value_heads=1,head_dim=16,global_head_dim=16,
        hidden_size_per_layer_input=8,num_kv_shared_layers=5,
        layer_types=["sliding_attention"]*4+["full_attention"]+["sliding_attention"]*4+["full_attention"],
        sliding_window=32,use_double_wide_mlp=True,final_logit_softcapping=30.,
        tie_word_embeddings=True,attention_dropout=0.)
    c._attn_implementation="sdpa"
    native=Gemma4ForCausalLM(c).float().to(device).train()
    model=Gemma4Reranker(full=copy.deepcopy(native),yes_id=7,no_id=9).to(device).train()
    ids=torch.randint(10,128,(8,80),device=device);ids[:,0]=2
    am=torch.ones_like(ids);am[1,67:]=0;ids[1,67:]=0
    pos=torch.tensor([1,3],device=device)
    logits=native(input_ids=ids,attention_mask=am,use_cache=False).logits
    yn=logits[torch.arange(8,device=device),am.sum(1)-1][:,[7,9]]
    scores=yn[:,0]-yn[:,1];out=model(ids,am)
    torch.testing.assert_close(out.yes_no,yn,atol=2e-5,rtol=2e-5)
    loss_fn=lambda s:total_loss(s,pos,tau=1.)
    loss=F.cross_entropy(scores.view(2,4),pos);loss.backward()
    mloss,_=loss_fn(out.scores.view(2,4));mloss.backward()
    for name,p in model.backbone.named_parameters():
        orig=dict(native.model.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,orig.grad,atol=3e-5,rtol=2e-4,msg=lambda detail: name + ": " + detail)
    with torch.no_grad():
        torch.testing.assert_close(model(ids[1:2,:67],am[1:2,:67]).scores,
                                   out.scores[1:2],atol=2e-5,rtol=2e-5)
    cached=copy.deepcopy(model);cached.zero_grad(set_to_none=True)
    cached.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    reference.gradcache_backward(cached,cached,ids,am,2,4,160,loss_fn,1.,True,False)
    # Retain every chunk graph in an independent ordinary-autograd oracle.
    # This isolates GradCache from GPU GEMM batch-shape rounding differences.
    oracle=copy.deepcopy(model);oracle.zero_grad(set_to_none=True)
    oracle_scores=torch.cat([oracle(ids[a:a+2],am[a:a+2]).scores for a in range(0,len(ids),2)])
    oracle_loss,_=loss_fn(oracle_scores.view(2,4));oracle_loss.backward()
    batch_shape_diagnostics=[]
    for (name,p),(_,q),(_,o) in zip(model.named_parameters(),cached.named_parameters(),oracle.named_parameters()):
        assert q.grad is not None and o.grad is not None,name
        torch.testing.assert_close(o.grad,q.grad,atol=3e-5,rtol=3e-4,msg=lambda detail: name + ": " + detail)
        batch_shape_diagnostics.append({"name":name,
            "full_vs_oracle_max_abs":float((p.grad-o.grad).abs().max()),
            "oracle_vs_cache_max_abs":float((o.grad-q.grad).abs().max())})
        if device=="cpu":
            torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4,msg=lambda detail: name + ": " + detail)
    print("BATCH_SHAPE_DIAGNOSTICS",json.dumps(sorted(batch_shape_diagnostics,
          key=lambda x:x["full_vs_oracle_max_abs"],reverse=True)[:5]),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=0.00005,betas=(.9,.95),weight_decay=.01)
    opt.step()
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"state.pt"
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict()},path)
        state=torch.load(path,weights_only=True,map_location=device)
        restored=copy.deepcopy(model);restored.load_state_dict(state["model"],strict=True)
        opt2=torch.optim.AdamW(restored.parameters(),lr=0.00005,betas=(.9,.95),weight_decay=.01)
        opt2.load_state_dict(state["optimizer"])
        for obj,op in [(model,opt),(restored,opt2)]:
            op.zero_grad();l,_=loss_fn(obj(ids,am).scores.view(2,4));l.backward();op.step()
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    return {"native_full_vocab_softcapped_logits_and_gradients":True,"right_padding":True,
            "PLE_and_shared_KV_gradients":True,"gradcache_with_gradient_checkpointing":True,"ordinary_autograd_chunk_oracle":True,
            "adamw_checkpoint_roundtrip":True,"all_text_parameters_trainable":model.trainable_report(),
            "device":device,"initial_lce_loss":float(loss.detach())}
if __name__=="__main__":
    device="cuda" if "--cuda" in sys.argv else "cpu"
    result=run(device)
    d=Path(os.environ.get("GEMMA_RUN_DIR",Path.cwd()))
    (d/("validation_gpu.json" if device=="cuda" else "validation_cpu.json")).write_text(json.dumps(result,indent=2)+"\n")
    print("GEMMA4_LCE_VALIDATION_PASSED",json.dumps(result),flush=True)
