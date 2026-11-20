"""Two-rank AdamW equivalence and ordinary <-> ZeRO-1 checkpoint round-trips."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reranker/engines/gemma4"))
import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from optimizer_sharding import build_optimizer, prepare_checkpoint, finish_checkpoint, ownership_report
from shared import total_loss, lr_at, reference


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cpu', action='store_true'); args = ap.parse_args()
    rank = int(os.environ['RANK'])
    device = 'cpu' if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}"
    if not args.cpu:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
    dist.init_process_group('gloo' if args.cpu else 'nccl')
    torch.manual_seed(17)
    baseline = torch.nn.Sequential(torch.nn.Linear(31, 113), torch.nn.Tanh(),
                                   torch.nn.Linear(113, 1)).float().to(device)
    sharded = copy.deepcopy(baseline)
    ddp = DDP(baseline, device_ids=None if args.cpu else [int(os.environ['LOCAL_RANK'])],
              gradient_as_bucket_view=True)
    cfg = dict(lr=0.00005, weight_decay=.01)
    ordinary = build_optimizer(baseline, cfg)
    zero = build_optimizer(sharded, {**cfg, 'optimizer_state_sharding': True})
    work = Path(os.environ.get('GEMMA_RUN_DIR', tempfile.gettempdir()))
    path = work / 'optimizer_validation_checkpoint.pt'
    max_parameter_diff = 0.0
    # One ordinary update represents the existing checkpoint. Then resume sharded.
    for step in range(7):
        torch.manual_seed(700 + 13 * step + rank)
        x = torch.randn(16, 31, device=device)
        labels = torch.tensor([0, 2], device=device)
        loss, _ = total_loss(ddp(x).view(2, 8), labels, tau=1.)
        (loss * 1000).backward()  # deliberately exercises active clipping
        gn = torch.nn.utils.clip_grad_norm_(baseline.parameters(), 12.)
        assert float(gn) > 12
        for p, q in zip(baseline.parameters(), sharded.parameters()):
            q.grad = p.grad.detach().clone()
        lr = lr_at([0, 1, 149, 150, 700, 1700, 1930][step], total=1931, peak=.00005, warmup=150)
        for opt in (ordinary, zero):
            for group in opt.param_groups: group['lr'] = lr
        ordinary.step()
        if step == 0:
            sharded.load_state_dict(baseline.state_dict())
            zero.load_state_dict(copy.deepcopy(ordinary.state_dict()))
        else:
            zero.step()
        for p, q in zip(baseline.parameters(), sharded.parameters()):
            max_parameter_diff = max(max_parameter_diff, float((p-q).abs().max()))
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        prepare_checkpoint(zero, path)
        if rank == 0:
            saved = zero.state_dict(); expected = ordinary.state_dict()
            assert saved['param_groups'] == expected['param_groups']
            for i, state in expected['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(saved['state'][i][key].cpu(), value.cpu(), atol=0, rtol=0)
            # Exercise the same NanoChat serializer used for real checkpoints.
            reference.save_ckpt(path, sharded, zero, step+1, 0, step+1, cfg, None, {'test': True})
        finish_checkpoint(zero, path)
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert checkpoint['step'] == step+1
        # Both loaders must accept the global state and continue identically.
        ordinary.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        zero.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        del checkpoint
        ordinary.zero_grad(set_to_none=False); zero.zero_grad(set_to_none=False)
        dist.barrier()
    report = {'passed': True, 'device': device, 'steps': 7,
              'max_parameter_abs_difference': max_parameter_diff,
              'moments_and_step_counters_bitwise_equal': True,
              'ordinary_to_sharded_to_ordinary_reload': True,
              'native_checkpoint_writer': True, **ownership_report(zero)}
    (work / f'optimizer_validation_{"cpu" if args.cpu else "cuda"}_rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    dist.barrier()
    if rank == 0: path.unlink()
    print(f'OPTIMIZER_SHARDING_VALIDATION_PASS rank={rank} '+json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
