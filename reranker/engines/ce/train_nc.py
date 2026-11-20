"""Standard full-vocabulary CE with LCE-matched samples and optimizer steps."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
from safetensors.torch import load_file, save_file

from dataset_nc import PairGroups, CESteps, make_plan, plan_digest, microbatches, identity
from modeling_nanochat_ce import load_model, load_tokenizer
from prompt_builder_nc import PairBuilder
from stepping import setup_precision, assert_fp32_params, FP32Guard, lr_at


def checkpoint_dirs(out):
    return sorted((p for p in out.glob('checkpoint-[0-9]*') if (p/'COMPLETE').exists()),
                  key=lambda p: int(p.name.split('-')[-1]))


def save_checkpoint(out, model, optimizer, tok, cfg, signature, step, total, rank, world, epoch, names):
    dest = out/f'checkpoint-{step}'
    temp = out/f'.checkpoint-{step}.tmp'
    if rank == 0:
        if temp.exists(): shutil.rmtree(temp)
        temp.mkdir()
    if world > 1: dist.barrier()
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
    if torch.cuda.is_available(): rng['cuda'] = torch.cuda.get_rng_state()
    torch.save(rng, temp/('rng_state.pth' if world == 1 else f'rng_state_{rank}.pth'))
    if cfg.get('fsdp', False):
        from sharding import full_checkpoint_states
        weights, optim_state = full_checkpoint_states(model, optimizer, names)
    elif rank == 0:
        weights, optim_state = model.state_dict(), optimizer.state_dict()
    if rank == 0:
        save_file({k:v.detach().cpu().contiguous() for k,v in weights.items()}, temp/'model.safetensors')
        torch.save(optim_state, temp/'optimizer.pt')
        del weights, optim_state
        torch.save({'step':step, 'total':total, 'peak':cfg['lr'], 'warmup':cfg['warmup_steps']}, temp/'scheduler.pt')
        tok.save_pretrained(temp)
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        if base.config.model_type == 'nanochat_upstream_ce':
            base.config.to_json_file(temp/'upstream_config.json')
        elif not cfg.get('lora'):
            base.config.save_pretrained(temp)
        if cfg.get('lora'): model.save_pretrained(temp)
        (temp/'recipe_config.json').write_text(json.dumps(cfg, indent=2))
        (temp/'training_signature.json').write_text(json.dumps(dict(signature, world_size=world), indent=2))
        (temp/'trainer_state.json').write_text(json.dumps({'global_step':step, 'max_steps':total, 'epoch':epoch}, indent=2))
    if world > 1: dist.barrier()
    if rank == 0:
        (temp/'COMPLETE').write_text(f'step={step}\n')
        temp.rename(dest)
        for old in checkpoint_dirs(out)[:-cfg.get('keep_ckpts', 1)]:
            shutil.rmtree(old)
        print(f'SAVED {dest}', flush=True)
    if world > 1: dist.barrier()
    return dest


def run(cfg, use_cpu=False, stop_after=None):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if not use_cpu:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cpu' if use_cpu else f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group('gloo' if use_cpu else 'nccl', timeout=datetime.timedelta(minutes=30))
    log = print if rank == 0 else lambda *a, **k: None
    setup_precision(cfg, log)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if cfg.get('prompt', 'nochat') != 'nochat': raise ValueError('Expected the plain true/false prompt')
    out = Path(cfg['output_dir'])
    if (out/'final/COMPLETE').exists() and cfg.get('resume', 'auto') == 'auto': return
    tok = load_tokenizer(cfg)
    builder = PairBuilder(tok, prompt='nochat', max_doc_len=cfg['max_doc_len'], max_query_len=cfg['max_query_len'])
    groups = PairGroups(cfg['train_parquet'], builder, seed=cfg['seed'])
    plans = [make_plan(groups, builder, cfg, e) for e in range(cfg['epochs'])]
    total = sum(map(len, plans))
    if cfg.get('expected_steps_per_epoch') and any(len(p) != cfg['expected_steps_per_epoch'] for p in plans):
        raise ValueError('LCE reference step count differs; refusing to silently repack')
    keys = ['model_name', 'model_revision', 'backbone_impl', 'nanochat_src', 'prompt', 'max_doc_len',
            'max_query_len', 'lora', 'seed', 'epochs', 'lr', 'weight_decay', 'warmup_steps', 'clip',
            'precision', 'k_choices', 'global_batch_groups', 'reference_world_size',
            'reference_tokens_per_batch', 'max_groups']
    if cfg.get('lora'): keys += ['lora_r', 'lora_alpha', 'lora_targets']
    signature = {k:cfg.get(k) for k in keys}
    import inspect
    signature['sampling_code_sha256'] = hashlib.sha256(Path(inspect.getfile(PairGroups)).read_bytes()).hexdigest()
    with open(cfg['train_parquet'], 'rb') as f:
        signature.update(data_sha256=hashlib.file_digest(f, 'sha256').hexdigest(),
                         objective='full_vocabulary_ce_lce_groups', plans=[plan_digest(p) for p in plans],
                         total_steps=total)
    candidates = checkpoint_dirs(out)
    resume = cfg.get('resume', 'auto')
    resume = candidates[-1] if resume == 'auto' and candidates else (None if resume in ('auto', 'off', None, False) else Path(resume))
    if not resume and out.exists() and any(out.iterdir()): raise ValueError('Fresh run needs an empty output directory')
    start = 0
    if resume:
        old = json.loads((resume/'training_signature.json').read_text()); old.pop('world_size', None)
        if old != signature: raise ValueError('Checkpoint is not the identical grouped-CE experiment')
        start = json.loads((resume/'trainer_state.json').read_text())['global_step']
    # All ranks must finish the empty-directory check before any rank creates output.
    if world > 1: dist.barrier()
    fsdp = cfg.get('fsdp', False)
    if fsdp and (use_cpu or world < 2 or cfg.get('lora') or cfg.get('backbone_impl') != 'upstream'):
        raise ValueError('FSDP requires multiple CUDA GPUs and the full native model')
    model = load_model(cfg)
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if builder.overhead + cfg['max_query_len'] + cfg['max_doc_len'] + 1 > model.config.max_position_embeddings:
        raise ValueError('Prompt plus answer exceeds the model context')
    if resume: model.load_state_dict(load_file(resume/'model.safetensors'), strict=True)
    model.to(device).train()
    if cfg.get('grad_checkpointing', True) and not fsdp: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert_fp32_params(model)
    # FSDP reduces every microbatch to keep accumulated gradients sharded.
    if fsdp:
        from sharding import shard_model
        wrapped = shard_model(model, cfg.get('grad_checkpointing', True))
    # Reuse LCE's DDP bucket-view memory behavior.
    elif world > 1:
        for p in model.parameters():
            if p.requires_grad: p.grad = torch.zeros_like(p)
        wrapped = DDP(model, device_ids=None if use_cpu else [device.index],
                      broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    else: wrapped = model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'], betas=(0.9,0.95), fused=not use_cpu)
    if resume:
        if fsdp:
            from sharding import restore_optimizer
            restore_optimizer(model, optimizer, resume/'optimizer.pt', names)
        else:
            state = torch.load(resume/'optimizer.pt', map_location='cpu', weights_only=True, mmap=True)
            optimizer.load_state_dict(state); del state
        rngfile = resume/('rng_state.pth' if world == 1 else f'rng_state_{rank}.pth')
        if not rngfile.exists(): rngfile = next(resume.glob('rng_state*.pth'))
        rng = torch.load(rngfile, map_location='cpu', weights_only=False)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['cpu'])
        if not use_cpu and 'cuda' in rng: torch.cuda.set_rng_state(rng['cuda'])
    out.mkdir(parents=True, exist_ok=True)
    wb = None
    if rank == 0:
        (out/'training_signature.json').write_text(json.dumps(dict(signature, world_size=world), indent=2))
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
        (out/'plan_summary.json').write_text(json.dumps({'steps':total, 'plan_hashes':signature['plans'],
             'groups':sum(len(s) for p in plans for s in p),
             'pairs':sum(k+1 for p in plans for s in p for _,k,_ in s)}, indent=2))
        if cfg.get('wandb_project'):
            import wandb
            wid = out/'wandb_id.txt'
            wb = wandb.init(project=cfg['wandb_project'], entity=cfg.get('wandb_entity'),
                            name=cfg.get('wandb_run_name'), config=cfg,
                            id=wid.read_text().strip() if resume and wid.exists() else None,
                            resume='allow' if resume else None)
            wid.write_text(wb.id)
    log(f'GROUPED_CE total_steps={total} start={start} world={world} fsdp={fsdp} micro={cfg["per_device_batch_size"]}', flush=True)
    guard = FP32Guard(model)
    dest = resume  # also allows final export after a crash immediately after the last save
    step, epoch_start, began = start, 0, time.monotonic()
    for e, plan in enumerate(plans):
        skip = max(0, min(len(plan), start - epoch_start))
        data = CESteps(groups, plan, builder, rank, world, skip)
        loader = DataLoader(data, batch_size=None, collate_fn=identity, num_workers=cfg.get('num_workers', 6))
        for batch in loader:
            t0 = time.monotonic()
            summed = torch.zeros((), device=device)
            for inputs, n, last in microbatches(batch, cfg['per_device_batch_size'], world, tok.pad_token_id, device):
                sync = wrapped.no_sync() if world > 1 and not fsdp and not last else contextlib.nullcontext()
                with sync:
                    loss = wrapped(**inputs).loss
                    # DDP/FSDP average ranks; this gives the global mean over ALL pairs.
                    (loss * (n * world / batch['global_pairs'])).backward()
                summed += loss.detach() * n
                if guard is not None: guard.remove(); guard = None
                del loss, inputs
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            lr = lr_at(step, total, cfg['lr'], cfg['warmup_steps'])
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.step(); optimizer.zero_grad(set_to_none=fsdp); step += 1
            if world > 1: dist.all_reduce(summed)
            record = dict(step=step, total_steps=total, loss=float(summed)/batch['global_pairs'],
                          lr=lr, grad_norm=float(norm), groups=batch['groups'], pairs=batch['global_pairs'],
                          step_seconds=time.monotonic()-t0, elapsed_seconds=time.monotonic()-began)
            if not use_cpu:
                record['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
            if rank == 0:
                with (out/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                if step == start+1 or step % cfg.get('log_every',20) == 0: print(json.dumps(record), flush=True)
                if wb: wb.log(record, step=step)
            save = step == total or step % cfg['save_every'] == 0 or (step == 1 and cfg.get('save_first_step', True)) or step == stop_after
            if save:
                dest = save_checkpoint(out, model, optimizer, tok, cfg, signature, step, total, rank, world,
                                       e+(batch['index']+1)/len(plan), names)
            if stop_after is not None and step >= stop_after:
                if wb: wb.finish()
                return
        epoch_start += len(plan)
    if rank == 0:
        final = out/'final'; final.mkdir()
        for p in dest.iterdir():
            if p.suffix in ('.json','.safetensors') or p.name == 'COMPLETE': os.link(p, final/p.name)
        log(f'COMPLETE {final}', flush=True)
        if wb: wb.finish()
    if world > 1: dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/G_climbmix0924_rlhn250k.yaml')
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.override:
        key,value = item.split('=',1); cfg[key] = yaml.safe_load(value)
    run(cfg)
    if dist.is_initialized(): dist.destroy_process_group()
