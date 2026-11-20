"""Pure token-clock rules for production; no Torch or cluster dependencies."""
REFERENCE_BATCH = 1048576
RAMP_SOURCE_STEP = 136316
RAMP_SOURCE_TOKENS = 92537618432
ABLATION_TOKENS = 2147483648
HORIZON = 321817
FINAL_TOKENS = RAMP_SOURCE_TOKENS + (HORIZON - RAMP_SOURCE_STEP) * REFERENCE_BATCH


def source_clock(meta):
    if "production" in meta:
        state = meta["production"]
        assert state["version"] == 1
        canonical = state["canonical_step"]
        tokens = state["total_training_tokens"]
    else:
        state = meta["ablation"]
        assert state["source_step"] == RAMP_SOURCE_STEP
        assert state["added_tokens"] == ABLATION_TOKENS
        assert state["token_budget"] == ABLATION_TOKENS
        canonical = state["canonical_step"]
        tokens = RAMP_SOURCE_TOKENS + state["added_tokens"]
        assert canonical == 138364
    assert canonical == int(canonical) and RAMP_SOURCE_STEP <= canonical <= HORIZON
    assert tokens == RAMP_SOURCE_TOKENS + (canonical - RAMP_SOURCE_STEP) * REFERENCE_BATCH
    return int(canonical), int(tokens)


class TokenClock:
    def __init__(self, source_actual_step, source_canonical_step, source_tokens, batch):
        assert batch in (REFERENCE_BATCH, 2 * REFERENCE_BATCH)
        self.start_actual = source_actual_step
        self.start_canonical = source_canonical_step
        self.start_tokens = source_tokens
        self.batch = batch
        assert source_tokens == RAMP_SOURCE_TOKENS + (source_canonical_step - RAMP_SOURCE_STEP) * REFERENCE_BATCH
        self.remaining = FINAL_TOKENS - source_tokens
        assert self.remaining >= 0
        self.updates_to_finish = (self.remaining + batch - 1) // batch
        self.final_actual = source_actual_step + self.updates_to_finish

    def added(self, actual_step):
        assert self.start_actual <= actual_step <= self.final_actual
        return min((actual_step - self.start_actual) * self.batch, self.remaining)

    def canonical(self, actual_step):
        return self.start_canonical + self.added(actual_step) // REFERENCE_BATCH

    def total(self, actual_step):
        return self.start_tokens + self.added(actual_step)

    def update_tokens(self, actual_step):
        return min(self.batch, self.remaining - self.added(actual_step))

    def finished(self, actual_step):
        return self.added(actual_step) == self.remaining


def lr_multiplier(canonical_step, warmup=40, warmdown_ratio=0.5528, final_lr_frac=0.05):
    warmdown = round(warmdown_ratio * HORIZON)
    if canonical_step < warmup:
        return (canonical_step + 1) / warmup
    if canonical_step <= HORIZON - warmdown:
        return 1.0
    progress = (HORIZON - canonical_step) / warmdown
    return progress + (1 - progress) * final_lr_frac
