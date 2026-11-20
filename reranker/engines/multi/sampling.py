"""Frozen logical plans on top of the greenhouse GroupSampler (imported unmodified).

Two length rules:
  caps    -- G's rule: query <= max_query_len, passage <= max_doc_len, independently (R0: 128 / 512).
  budget  -- the comparison rule: query <= max_query_len (512), passage gets whatever is left of a
             seq_budget-token sequence (2,048, nanochat's pretraining context) after the scaffold and query.
The plan (which groups share a microbatch and a step) is computed once from the nanochat-ClimbMix length
columns and is identical for every backbone; each backbone then tokenizes the text natively.
"""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # reranker/
LCE_SRC = ROOT / "src"
if str(LCE_SRC) not in sys.path:
    sys.path.insert(0, str(LCE_SRC))
from dataset_nc import GroupSampler          # noqa: E402  (greenhouse, unmodified)
from stepping import pack_steps, shard_steps  # noqa: E402

PLAN_KEYS = ("max_tokens_per_batch", "pack_world", "packing_scaffold", "max_groups", "global_batch_groups",
             "k_choices", "seed", "max_query_len", "max_doc_len", "length_rule", "seq_budget")


class BudgetGroupSampler(GroupSampler):
    """GroupSampler whose per-row length estimate follows the plan's length rule, sharded for any world size.

    With length_rule == "caps" the estimate is exactly GroupSampler's. With "budget", a passage's estimate is clamped
    to seq_budget - scaffold - query, mirroring how PairBuilder truncates it, then the epoch is re-packed.

    World size: the plan's steps are always packed for pack_world (G's plan: 2), so every world sees the same groups in
    every optimizer step. Greenhouse also requires pack_world % world_size == 0, because DDP needs every rank to run the
    same number of backward passes. This trainer has no per-microbatch collective (it all-reduces gradients once per
    step), so any world size works as long as every step has at least world_size microbatches: shard_steps deals a
    step's microbatches round-robin (counts may differ by one between ranks) with loss weights world*len/G, and the
    all-reduce mean is still the exact mean over the step's groups. A step with fewer microbatches than ranks would
    leave a rank without the step's last microbatch and deadlock, so it is refused up front, identically on every rank.
    For world sizes dividing pack_world this is greenhouse's sampler unchanged (tests/test_world.py checks both)."""

    def __init__(self, *args, length_rule="caps", seq_budget=None, rank=0, world_size=1, pack_world=None, **kw):
        assert 0 <= rank < world_size, (rank, world_size)
        # build as a single rank (greenhouse's divisibility assert), then shard for the real rank and world
        super().__init__(*args, rank=0, world_size=1, pack_world=pack_world or world_size, **kw)
        self.rank, self.world_size = rank, world_size
        self.length_rule, self.seq_budget = length_rule, seq_budget
        if length_rule == "budget":
            neg_lens = args[1] if len(args) > 1 else kw["neg_lens"]
            room = [max(1, seq_budget - self.scaffold - q) for q in self.qlen]
            self.proxy = [max((min(x, r) for x in negs), default=0) for negs, r in zip(neg_lens, room)]
            self.proxy_mean = [sum(min(x, r) for x in negs) / max(1, len(negs)) for negs, r in zip(neg_lens, room)]
        else:
            assert length_rule == "caps", length_rule
        self.set_epoch(0)

    def set_epoch(self, e, skip: int = 0):
        # greenhouse GroupSampler.set_epoch, plus the one-microbatch-per-rank-per-step check
        self.epoch = e
        steps = pack_steps(self._build(e), self.global_groups, self.pack_world)
        fewest = min(len(st) for st in steps)
        if fewest < self.world_size:
            raise ValueError(f"world size {self.world_size}: a step of the plan has only {fewest} microbatches, "
                             f"so some rank would get none and the step's all-reduce would deadlock")
        self.n_steps = len(steps)
        self._batches, self.meta = shard_steps(steps, self.rank, self.world_size)
        self._skip = max(0, min(skip, len(self._batches)))


def step_cursor(sampler, step):
    """Index into this rank's microbatches where optimizer step `step` + 1 begins (0 for step 0). Ranks run different
    numbers of microbatches when the world size does not divide pack_world, so a resume needs each rank's own cursor."""
    if step == 0:
        return 0
    ends = [i for i, m in enumerate(sampler.meta) if m[1]]
    assert len(ends) == sampler.n_steps and step <= len(ends), (len(ends), sampler.n_steps, step)
    return ends[step - 1] + 1


def make_sampler(cfg, avail, neg_lens, query_lens, rank=0, world=1):
    return BudgetGroupSampler(
        avail, neg_lens, query_lens, max_tokens=int(cfg["max_tokens_per_batch"]),
        max_groups=int(cfg["max_groups"]), max_doc_len=int(cfg["max_doc_len"]),
        max_query_len=int(cfg["max_query_len"]), k_choices=tuple(cfg["k_choices"]), seed=int(cfg["seed"]),
        scaffold=int(cfg["packing_scaffold"]), global_groups=int(cfg["global_batch_groups"]),
        rank=rank, world_size=world, pack_world=int(cfg["pack_world"]), query_factor=1,
        length_rule=cfg.get("length_rule", "caps"), seq_budget=cfg.get("seq_budget"))


def plan_of(sampler, cfg, epoch=0):
    steps = pack_steps(sampler._build(epoch), int(cfg["global_batch_groups"]), int(cfg["pack_world"]))
    return [[item for mb in st for item in mb] for st in steps]


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()


def milestones(plan, every):
    """1-based optimizer steps at which cumulative groups first reach every, 2*every, ...; plus the last step."""
    out, cum, nxt = [], 0, every
    for s, st in enumerate(plan, 1):
        cum += len(st)
        while cum >= nxt:
            out.append({"step": s, "groups": cum, "target": nxt}); nxt += every
    if not out or out[-1]["step"] != len(plan):
        out.append({"step": len(plan), "groups": cum, "target": "final"})
    return out


def summarize(plan, every=None):
    s = {"steps": len(plan), "groups": sum(map(len, plan)),
         "pairs": sum(k + 1 for st in plan for _, k, _ in st), "sha256": plan_hash(plan)}
    if every:
        s["milestones"] = milestones(plan, every)
    return s


def load_lengths(parquet):
    import pyarrow.parquet as pq
    t = pq.ParquetFile(parquet).read(columns=["query_len", "neg_lens"])
    q, n = t.column("query_len").to_pylist(), t.column("neg_lens").to_pylist()
    return [len(r) for r in n], n, q
