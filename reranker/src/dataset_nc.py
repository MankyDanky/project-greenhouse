"""
Groups for pointwise contrastive training.

A GROUP = 1 query + 1 positive + K negatives, rendered as K+1 INDEPENDENT
(query, doc) sequences. The collator flattens B groups into B*(K+1) rows; the
trainer reshapes the resulting scores back to [B, K+1] for InfoNCE.

Shared with the jina pipeline: the same data/rlhn_prepared/train.parquet, the
same one-positive-per-epoch resampling, the same positive-slot shuffling (a
fixed slot would be learnable), and the same teacher-logit permutation.
"""
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset, Sampler
from stepping import pack_steps, shard_steps

SCAFFOLD = 43          # chat template overhead; GroupSampler takes the builder's actual value


class PairGroups(Dataset):
    def __init__(self, parquet_path: str, builder, seed: int = 0,
                 subsets: Optional[List[str]] = None):
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=parquet_path, split="train")
        if subsets:
            ds = ds.filter(lambda r: r["subset"] in subsets, num_proc=4)
        self.ds, self.builder = ds, builder
        self.seed, self.epoch = seed, 0
        self.has_teacher = "neg_teacher" in ds.column_names
        self.has_trank = "neg_trank" in ds.column_names          # listwise teacher ranks (merge_teacher_ranks.py)
        self.avail = [len(r) for r in ds["neg_lens"]]
        self.neg_lens = ds["neg_lens"]
        self.query_lens = ds["query_len"]

    def set_epoch(self, e): self.epoch = e
    def __len__(self): return len(self.ds)

    def __getitem__(self, item):
        # the epoch comes WITH the item: persistent DataLoader workers never see a
        # set_epoch() made in the main process
        i, K, epoch = item
        r = self.ds[i]
        rng = random.Random((self.seed * 1_000_003 + epoch) * 1_000_003 + i)
        pi = rng.randrange(len(r["pos_texts"]))
        K = min(K, len(r["neg_texts"]))
        ni = rng.sample(range(len(r["neg_texts"])), K)
        docs = [r["neg_texts"][j] for j in ni]
        lens = [r["neg_lens"][j] for j in ni]
        tsc = None
        if self.has_teacher and r.get("neg_teacher") and r.get("pos_teacher"):
            tsc = [r["neg_teacher"][j] for j in ni]
        slot = rng.randrange(K + 1)
        trk = None
        if self.has_trank:                                  # same rng draws above: order is unchanged
            trk = [r["neg_trank"][j] for j in ni]; trk.insert(slot, r["pos_trank"][pi])
        docs.insert(slot, r["pos_texts"][pi]); lens.insert(slot, r["pos_lens"][pi])
        if tsc is not None:
            tsc.insert(slot, r["pos_teacher"][pi])
        seqs = [self.builder.build(r["query"], d, query_len=r["query_len"], doc_len=L)
                for d, L in zip(docs, lens)]
        return {"seqs": seqs, "pos_idx": slot, "K": K,
                "subset": r["subset"], "teacher": tsc, "trank": trk}


class GroupCollator:
    """Flattens B groups of K+1 sequences into B*(K+1) padded rows."""
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, feats):
        ks = {f["K"] for f in feats}
        assert len(ks) == 1, f"ragged K within a batch: {ks}"
        K = ks.pop(); G = K + 1; B = len(feats)
        flat = [s for f in feats for s in f["seqs"]]
        n = max(len(s) for s in flat)
        ids = torch.full((len(flat), n), self.pad, dtype=torch.long)
        att = torch.zeros((len(flat), n), dtype=torch.long)
        for i, s in enumerate(flat):                       # RIGHT padding
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long); att[i, :len(s)] = 1
        out = {"input_ids": ids, "attention_mask": att, "B": B, "G": G, "K": K,
               "pos_idx": torch.tensor([f["pos_idx"] for f in feats], dtype=torch.long),
               "subset": [f["subset"] for f in feats]}
        if any(f.get("teacher") is not None for f in feats):
            t = torch.zeros(B, G, dtype=torch.float); m = torch.zeros(B, dtype=torch.bool)
            for i, f in enumerate(feats):
                if f.get("teacher") is not None:
                    t[i, :len(f["teacher"])] = torch.tensor(f["teacher"], dtype=torch.float)
                    m[i] = True
            out["teacher"] = t; out["teacher_mask"] = m
        if feats[0].get("trank") is not None:
            out["trank"] = torch.tensor([f["trank"] for f in feats], dtype=torch.long)
        return out


class GroupSampler(Sampler):
    """Token-budget batching over GROUPS, with a deterministic K per row.

    InfoNCE needs a rectangular [B, K+1], so every group in a batch must have the
    same candidate count -- but rows carry anywhere from 7 to 100 negatives.

    Rule: each row gets the LARGEST k in k_choices it can supply.
        k_choices = [7, 15, 23]
        50 negatives -> K=23      20 -> K=15      9 -> K=7
    Rows with fewer than min(k_choices) negatives are dropped.

    A "bucket" is just the set of rows sharing the same K, so a batch never mixes
    rows that disagree. Nothing is random and no row's negatives are wasted
    beyond its bucket ceiling.
    """
    def __init__(self, avail, neg_lens, query_lens, max_tokens, max_groups=16,
                 max_doc_len=512, max_query_len=128, k_choices=(7, 15, 23),
                 seed=0, mega=50, rank=0, world_size=1, safety=1.30, global_groups=128,
                 scaffold=SCAFFOLD, pack_world=None, query_factor=1):
        self.avail, self.max_tokens, self.max_groups = avail, max_tokens, max_groups
        self.scaffold = scaffold           # prompt overhead in tokens (PairBuilder.overhead)
        self.global_groups = global_groups
        # MEAN length is the wrong estimator: a group is padded to its LONGEST
        # document, so a row averaging 150 tokens with one 512-token doc costs
        # 2x what the mean predicts. Measured: 34.5% of K=23 rows exceeded a
        # mean-based estimate, up to 2.07x -- which OOMed jobs 1534529 and
        # 1534729. Use max, so est_tokens is an upper bound.
        self.proxy = [max((min(x, max_doc_len) for x in r), default=0)
                      for r in neg_lens]
        self.proxy_mean = [sum(min(x, max_doc_len) for x in r) / max(1, len(r))
                           for r in neg_lens]      # kept for length-sorted batching
        self.query_factor = query_factor          # prompts that repeat the query (nochat_qdq) bill it twice
        self.qlen = [min(q, max_query_len) * query_factor for q in query_lens]
        self.k_choices = sorted(k_choices)
        self.seed, self.mega, self.safety = seed, mega, safety
        self.rank, self.world_size, self.epoch, self._skip = rank, world_size, 0, 0
        # pack steps as if there were pack_world GPUs (a multiple of the real world size), so a run
        # on fewer GPUs sees exactly the same groups in every optimizer step as the original run
        self.pack_world = pack_world or world_size
        assert self.pack_world % world_size == 0, (pack_world, world_size)
        floor = self.k_choices[0]
        self.buckets, self.dropped = {}, 0
        for i, a in enumerate(avail):
            el = [k for k in self.k_choices if k <= a]
            if not el:                      # fewer negatives than the floor
                self.dropped += 1
                continue
            self.buckets.setdefault(el[-1], []).append(i)   # largest K it supports
        self.set_epoch(0)

    def set_epoch(self, e, skip: int = 0):
        # local microbatches (memory-sized) -> global steps of ~global_groups groups
        self.epoch = e
        steps = pack_steps(self._build(e), self.global_groups, self.pack_world)
        self.n_steps = len(steps)
        self._batches, self.meta = shard_steps(steps, self.rank, self.world_size)
        self._skip = max(0, min(skip, len(self._batches)))

    def est_tokens(self, i, k):
        return (k + 1) * (self.scaffold + self.qlen[i] + self.proxy[i]) * self.safety

    def _build(self, epoch):
        g = random.Random(self.seed + epoch); batches = []
        for cap, idxs in self.buckets.items():
            idxs = list(idxs); g.shuffle(idxs)
            chunk = self.max_groups * self.mega
            for s0 in range(0, len(idxs), chunk):
                block = sorted(idxs[s0:s0 + chunk], key=lambda i: self.proxy_mean[i])
                cur, cur_max, pos = [], 0.0, 0
                k = cap                      # the bucket IS the K
                while pos < len(block):
                    i = block[pos]
                    e = self.est_tokens(i, min(k, self.avail[i]))
                    nm = max(cur_max, e)
                    # cap groups by K: a K=23 group is 24 sequences, so allowing
                    # the same group count as K=15 quadruples the batch and OOMs
                    cap_groups = max(1, self.max_groups * 16 // (k + 1))
                    if cur and ((len(cur) + 1) * nm > self.max_tokens
                                or len(cur) >= cap_groups):
                        batches.append([(j, k, epoch) for j in cur]); cur, cur_max = [], 0.0
                        continue
                    cur.append(i); cur_max = nm; pos += 1
                if cur: batches.append([(j, k, epoch) for j in cur])
        g.shuffle(batches)
        return batches                     # ALL microbatches; pack_steps + shard_steps split them

    def __iter__(self): return iter(self._batches[self._skip:])
    def __len__(self): return len(self._batches) - self._skip
