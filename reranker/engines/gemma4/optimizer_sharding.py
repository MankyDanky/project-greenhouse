"""ZeRO-1 ownership only; native fused AdamW and ordinary NanoChat checkpoints."""
import copy
import gc
import os
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer


class ShardedAdamW(ZeroRedundancyOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, optimizer_class=torch.optim.AdamW,
                         overlap_with_ddp=False, parameters_as_bucket_view=False, **kwargs)

    def load_state_dict(self, state):
        # Let native AdamW place its own moments/step counters correctly for fused=True.
        # Loading only the owned subset avoids a second full GPU optimizer copy.
        full_params = [p for g in self.param_groups for p in g['params']]
        saved_ids = [i for g in state['param_groups'] for i in g['params']]
        if len(full_params) != len(saved_ids):
            raise ValueError('Checkpoint optimizer parameter count differs')
        global_id = {id(p): i for p, i in zip(full_params, saved_ids)}
        local = self.optim.state_dict()
        if len(local['param_groups']) != len(state['param_groups']):
            raise ValueError('Checkpoint optimizer group count differs')
        restored = {'state': {}, 'param_groups': []}
        for exposed, actual, local_group, saved in zip(
                self.param_groups, self.optim.param_groups, local['param_groups'], state['param_groups']):
            settings = {k: copy.deepcopy(v) for k, v in saved.items() if k != 'params'}
            exposed.update(settings)
            restored['param_groups'].append({**settings, 'params': local_group['params']})
            for p, lid in zip(actual['params'], local_group['params']):
                gid = global_id[id(p)]
                if gid in state['state']:
                    restored['state'][lid] = state['state'][gid]
        self.optim.load_state_dict(restored)


def build_optimizer(model, cfg):
    cls = ShardedAdamW if cfg.get('optimizer_state_sharding', False) else torch.optim.AdamW
    return cls(list(model.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
               betas=(0.9, 0.95), fused=True)


def prepare_checkpoint(opt, path):
    """All ranks stage states; rank 0 maps them on CPU for the unchanged saver.

    PyTorch's default NCCL object consolidation serializes an entire shard into
    GPU buffers. File-backed consolidation avoids those large transient copies.
    The resulting optimizer.state_dict() is PyTorch's ordinary global format.
    """
    if not isinstance(opt, ShardedAdamW):
        return
    n = sum(p.numel() for g in opt.param_groups for p in g['params'])
    # Global moments staged on disk plus the new full checkpoint coexist briefly.
    if shutil.disk_usage(Path(path).parent).free < 20 * n + 5 * 2**30:
        raise RuntimeError('Insufficient disk for staged AdamW state and atomic full checkpoint')
    opt._sync_param_groups(opt.param_groups, opt.optim.param_groups)
    base = str(path) + '.optimizer-stage'
    rank = dist.get_rank()
    shard = Path(base + f'.rank{rank}.state')
    torch.save(opt.optim.state_dict(), str(shard) + '.tmp')
    os.replace(str(shard) + '.tmp', shard)
    dist.barrier()
    if rank == 0:
        opt._all_state_dicts = [torch.load(base + f'.rank{r}.state', map_location='cpu',
            weights_only=False, mmap=True) for r in range(dist.get_world_size())]


def finish_checkpoint(opt, path):
    if not isinstance(opt, ShardedAdamW):
        return
    opt._all_state_dicts = []
    gc.collect()
    dist.barrier()
    Path(str(path) + f'.optimizer-stage.rank{dist.get_rank()}.state').unlink()
    dist.barrier()


def ownership_report(opt):
    if not isinstance(opt, ShardedAdamW):
        return {'sharded': False}
    local = [p for g in opt.optim.param_groups for p in g['params']]
    return {'sharded': True, 'rank': dist.get_rank(), 'world': dist.get_world_size(),
            'owned_parameters': sum(p.numel() for p in local),
            'owned_parameter_tensors': len(local), 'optimizer': 'torch.optim.AdamW',
            'fused': True, 'overlap_with_ddp': False}
