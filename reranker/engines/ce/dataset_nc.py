"""LCE's exact sampling/step plan, with independent causal-CE labels."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stepping import pack_steps
from reranker.src.dataset_nc import PairGroups, GroupSampler


def make_plan(groups, builder, cfg, epoch=0):
    sampler = GroupSampler(
        groups.avail, groups.neg_lens, groups.query_lens,
        max_tokens=cfg['reference_tokens_per_batch'], max_groups=cfg['max_groups'],
        max_doc_len=cfg['max_doc_len'], max_query_len=cfg['max_query_len'],
        k_choices=cfg['k_choices'], seed=cfg['seed'],
        world_size=cfg['reference_world_size'], global_groups=cfg['global_batch_groups'],
        scaffold=builder.overhead)
    steps = pack_steps(sampler._build(epoch), cfg['global_batch_groups'], cfg['reference_world_size'])
    return [[item for mb in step for item in mb] for step in steps]


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()


class CESteps(Dataset):
    """One item is one optimizer step. GPU microbatches never repack it."""
    def __init__(self, groups, plan, builder, rank=0, world=1, start=0):
        self.groups, self.plan = groups, plan
        self.rank, self.world, self.start = rank, world, start
        self.answers = [builder.tok.encode(s, add_special_tokens=False) for s in (' true', ' false')]
        if any(len(x) != 1 for x in self.answers):
            raise ValueError('Expected one-token true/false targets')

    def __len__(self):
        return len(self.plan) - self.start

    def __getitem__(self, i):
        index = self.start + i
        pairs, count = [], 0
        for item in self.plan[index]:
            group = self.groups[item]  # unchanged LCE positive/negative selection and slot shuffle
            for slot, prompt in enumerate(group['seqs']):
                if count % self.world == self.rank:
                    answer = self.answers[0 if slot == group['pos_idx'] else 1]
                    pairs.append({'input_ids': prompt + answer, 'labels': [-100] * len(prompt) + answer})
                count += 1
        return {'pairs': pairs, 'global_pairs': count, 'groups': len(self.plan[index]), 'index': index}


def identity(item):
    return item


def microbatches(step, micro, world, pad, device):
    pairs = step['pairs']
    # Equal forward/backward counts on every DDP rank. A possible empty last
    # chunk contributes zero loss, never an extra training example.
    chunks = math.ceil(math.ceil(step['global_pairs'] / world) / micro)
    for j in range(chunks):
        rows = pairs[j * micro:(j + 1) * micro]
        valid = len(rows)
        if not rows:
            rows = pairs[:1]
        if not rows:
            raise ValueError('More ranks than pairs in an optimizer step')
        width = max(len(row['input_ids']) for row in rows)
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for k, row in enumerate(rows):
            n = len(row['input_ids'])
            ids[k, :n] = torch.tensor(row['input_ids'], device=device)
            labels[k, :n] = torch.tensor(row['labels'], device=device)
            mask[k, :n] = 1
        yield {'input_ids': ids, 'labels': labels, 'attention_mask': mask}, valid, j == chunks - 1
