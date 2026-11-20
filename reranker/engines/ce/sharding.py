"""Keep ordinary full checkpoints interoperable with FSDP2 training."""
from pathlib import Path
import torch


def named_optimizer_state(state, names):
    ids = [i for group in state['param_groups'] for i in group['params']]
    if len(ids) != len(names):
        raise ValueError('Optimizer parameter count differs from the checkpoint')
    mapping = dict(zip(ids, names))
    return {'state': {mapping[k]: v for k, v in state['state'].items()},
            'param_groups': [{**g, 'params': [mapping[k] for k in g['params']]}
                             for g in state['param_groups']]}


def numbered_optimizer_state(state, names):
    mapping = {name: i for i, name in enumerate(names)}
    return {'state': {mapping[k]: v for k, v in state['state'].items()},
            'param_groups': [{**g, 'params': [mapping[k] for k in g['params']]}
                             for g in state['param_groups']]}



def shard_model(model, activation_checkpointing):
    """FSDP2 changes parameter storage, not the native forward or CE objective."""
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    for i, block in enumerate(model.backbone.transformer.h):
        if activation_checkpointing:
            block = checkpoint_wrapper(block)
            model.backbone.transformer.h[i] = block
        fully_shard(block, reshard_after_forward=True)
    fully_shard(model, reshard_after_forward=True)
    return model


def full_checkpoint_states(model, optimizer, names):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    weights = get_model_state_dict(model, options=options)
    state = get_optimizer_state_dict(model, optimizer, options=options)
    if torch.distributed.get_rank() == 0:
        state = numbered_optimizer_state(state, names)
    return weights, state


def restore_optimizer(model, optimizer, path, names):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict
    state = {}
    if torch.distributed.get_rank() == 0:
        state = named_optimizer_state(
            torch.load(path, map_location='cpu', weights_only=True, mmap=True), names)
    set_optimizer_state_dict(model, optimizer, state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
