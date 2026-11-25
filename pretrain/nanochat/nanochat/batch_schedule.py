"""
Phased total batch size for pretraining, e.g. 262k -> 524k -> 1M -> 2M tokens over a fixed token horizon, as in
the d34 ClimbMix production run.

A schedule string lists phases as BATCH:END, comma-separated, where END is the cumulative number of training tokens
at which the phase ends; the last phase has no END and runs to the horizon:

    "262144:10e9,524288:30e9,1048576:92.5e9,2097152"

The LR / momentum / weight-decay schedules advance in *canonical steps*: a step of batch B counts
max(1, B / canonical_batch). With canonical_batch = 2^20 this is the d34 run's own accounting (every ramp step
counts 1, every 2M step counts 2: its metadata has canonical step 318,000 at optimizer step 227,158 with the 2M
switch at 136,316). For a constant batch <= canonical_batch a canonical step is just a step, so the schedules
reduce exactly to nanochat's step-based ones.
"""
from dataclasses import dataclass


@dataclass
class BatchSchedule:
    phases: list            # [(batch_tokens, end_tokens or None)]
    target_tokens: int
    canonical_batch: int
    batch: list             # batch[i] = total batch (tokens) of optimizer step i, i in [0, num_steps)
    tokens_before: list     # tokens trained before step i, i in [0, num_steps]
    canonical_before: list  # canonical steps before step i, i in [0, num_steps]
    overshoot: int = 0      # tokens trained past target_tokens (0 when the last step lands exactly)

    @property
    def num_steps(self):
        return len(self.batch)

    @property
    def canonical_horizon(self):
        return self.canonical_before[-1]

    def phase_ranges(self):
        """[(batch, first_step, last_step)] for the phases that actually occur."""
        out = []
        for i, b in enumerate(self.batch):
            if out and out[-1][0] == b:
                out[-1][2] = i
            else:
                out.append([b, i, i])
        return [tuple(r) for r in out]


def parse_batch_schedule(spec):
    phases = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    assert parts, "empty --batch-schedule"
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if ":" in part:
            assert not last, f"the last phase runs to the horizon and takes no END: {part!r}"
            b, end = part.split(":")
            phases.append((int(b), int(float(end))))
        else:
            assert last, f"every phase but the last needs BATCH:END: {part!r}"
            phases.append((int(part), None))
    ends = [e for _, e in phases if e is not None]
    assert all(b > 0 for b, _ in phases), f"batch sizes must be positive: {spec!r}"
    assert ends == sorted(ends) and len(set(ends)) == len(ends) and all(e > 0 for e in ends), \
        f"phase ends must be positive and increasing: {spec!r}"
    return phases


def build_batch_schedule(spec, target_tokens, canonical_batch=2**20, granularity=None):
    """Simulate the run: one entry per optimizer step until target_tokens is reached.

    The last step trains only the tokens that are left when they are a whole number of micro-batches (`granularity`:
    tokens per forward/backward over all ranks), so the run lands exactly on the horizon, as the d34 run did with its
    half-batch last update. Otherwise (or without `granularity`) the last step is a full batch and overshoots by less
    than one batch; `overshoot` records by how much."""
    phases = parse_batch_schedule(spec)
    assert target_tokens > 0, "the batch schedule needs a token horizon (--target-tokens)"
    batch, tokens_before, canonical_before = [], [0], [0.0]
    tokens, canonical, k = 0, 0.0, 0
    while tokens < target_tokens:
        while phases[k][1] is not None and tokens >= phases[k][1]:
            k += 1
        b = phases[k][0]
        left = target_tokens - tokens
        if left < b and granularity and left % granularity == 0:
            b = left                       # exact horizon: a short last step
        batch.append(b)
        tokens += b
        canonical += max(1.0, b / canonical_batch)
        tokens_before.append(tokens)
        canonical_before.append(canonical)
    return BatchSchedule(phases, target_tokens, canonical_batch, batch, tokens_before, canonical_before,
                         overshoot=tokens - target_tokens)
