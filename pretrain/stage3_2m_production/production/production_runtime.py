"""Resumable production instrumentation; import only inside an allocation."""
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist

from production_clock import REFERENCE_BATCH, HORIZON, FINAL_TOKENS, TokenClock, source_clock


def add_arguments(parser):
    parser.add_argument("--production-checkpoint-source-dir", required=True)
    parser.add_argument("--production-metrics-path", required=True)
    parser.add_argument("--production-matrix-lr", type=float, default=None,
                        help="Optional assertion of the winner's saved Muon initial LR; never overrides it")
    parser.add_argument("--production-keep-latest", type=int, default=3)
    parser.add_argument("--production-release", default=None)
    parser.add_argument("--production-winner-checkpoint-dir", default=None)
    parser.add_argument("--production-winner-completion", default=None)
    parser.add_argument("--production-winner-completion-sha256", default=None)


def validate_arguments(args, world_size):
    assert args.resume_from_step >= 0
    assert args.num_iterations == HORIZON
    assert args.total_batch_size in (REFERENCE_BATCH, 2 * REFERENCE_BATCH)
    assert (world_size, args.device_batch_size, args.max_seq_len, args.depth) == (8, 8, 2048, 34)
    assert args.fp8 and args.fp8_recipe == "tensorwise"
    assert args.warmdown_ratio == 0.5528 and args.warmup_steps == 40 and args.final_lr_frac == 0.05
    assert args.eval_tokens == 41943040 and args.eval_every > 0 and args.run == "dummy"
    assert args.production_keep_latest >= 2
    assert args.model_tag and args.model_tag.startswith("d34prod"), "Use a separate d34prod output tag"
    assert args.stop_at_step == -1, "Production stopping is owned by token horizon and deadline"


def group_settings(optimizer):
    return [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups]


def restore_optimizer_for_production(optimizer, optimizer_data, requested_lr):
    optimizer.load_state_dict(optimizer_data)
    settings = group_settings(optimizer)
    rates = {g["initial_lr"] for g in settings if g["kind"] == "muon"}
    assert len(rates) == 1
    actual_lr = rates.pop()
    assert math.isfinite(actual_lr) and actual_lr > 0
    if requested_lr is not None:
        assert math.isclose(actual_lr, requested_lr, rel_tol=1e-13, abs_tol=0), "Winner LR mismatch"
    return settings


def resume_loader_state(meta, rank, initial_evidence=None):
    if "production" in meta:
        states = meta["production"]["loader_states"]
    else:
        assert initial_evidence is not None, "First resume requires validated per-rank COMPLETE evidence"
        states = initial_evidence["loader_states"]
    assert len(states) == 8 and sorted(row["rank"] for row in states) == list(range(8))
    return next(row["state"] for row in states if row["rank"] == rank)


@contextmanager
def preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


class ProductionRuntime:
    def __init__(self, args, rank, world_size, checkpoint_dir, meta_data,
                 optimizer, loaded_settings, first_prefetched_state, initial_evidence=None):
        self.args, self.rank, self.world_size = args, rank, world_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_path = Path(args.production_metrics_path)
        self.loaded_settings = loaded_settings
        canonical, tokens = source_clock(meta_data)
        assert meta_data["total_batch_size"] == args.total_batch_size, "Preserve winning batch size"
        if "production" in meta_data:
            assert meta_data["production"]["horizon"] == HORIZON
            assert meta_data["production"]["world_size"] == world_size
        else:
            assert self.checkpoint_dir.resolve() != Path(args.production_checkpoint_source_dir).resolve()
        self.clock = TokenClock(meta_data["step"], canonical, tokens, args.total_batch_size)
        self.matrix_lr = next(g["initial_lr"] for g in loaded_settings if g["kind"] == "muon")
        expected_lr = meta_data.get("production", meta_data.get("ablation"))["actual_matrix_lr"]
        assert math.isclose(self.matrix_lr, expected_lr, rel_tol=1e-13, abs_tol=0)
        self.started_at = time.time()
        self.rank_tokens = self.updates = 0
        self.interval_loss_sum = self.interval_tokens = 0
        self.last_val = None
        self.complete_written = False
        self.last_saved_step = None
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")

        def prepare():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            assert not (self.metrics_path.parent / "BASE_DONE").exists()
            for meta_path in self.checkpoint_dir.glob("meta_*.json"):
                assert "production" in json.loads(meta_path.read_text()), "Unrelated checkpoint in production directory"
        self.rank_zero_checked(prepare)
        states = self.gather({"rank": rank, "loaded_state": resume_loader_state(meta_data, rank, initial_evidence),
                              "first_prefetched_state": first_prefetched_state})
        self.emit("start", source_dir=str(Path(args.production_checkpoint_source_dir).resolve()),
                  source_actual_step=meta_data["step"], source_canonical_step=canonical,
                  source_total_training_tokens=tokens, horizon=HORIZON, final_tokens=FINAL_TOKENS,
                  actual_matrix_lr=self.matrix_lr, total_batch_size=args.total_batch_size,
                  final_actual_step=self.clock.final_actual, loader_states=states,
                  optimizer=loaded_settings, config=vars(args).copy(),
                  initial_completion_evidence=initial_evidence,
                  resume_note="Per-rank row-group cursors; packing buffers and prefetched batch are not serialized")

    def gather(self, value):
        if not dist.is_initialized():
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def rank_zero_checked(self, action):
        error = [None]
        if self.rank == 0:
            try:
                action()
            except Exception as exception:
                error[0] = f"{type(exception).__name__}: {exception}"
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])

    def emit(self, event, **fields):
        if self.rank == 0:
            with self.metrics_path.open("a") as stream:
                stream.write(json.dumps({"event": event, "time_unix": time.time(),
                                        "job_id": self.job_id, **fields}, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()

    def added_tokens(self, step):
        return self.clock.added(step)

    def canonical_step(self, step):
        return self.clock.canonical(step)

    def consumed_batch(self, x, y):
        self.rank_tokens += x.numel()

    def assert_static_optimizer_settings(self, optimizer):
        for old, current in zip(self.loaded_settings, group_settings(optimizer)):
            allowed = {"lr", "momentum", "weight_decay"} if old["kind"] == "muon" else {"lr"}
            assert {k: v for k, v in old.items() if k not in allowed} == {
                k: v for k, v in current.items() if k not in allowed}

    def record_step(self, step, mean_loss, dt, optimizer, lrm):
        assert math.isfinite(mean_loss)
        self.assert_static_optimizer_settings(optimizer)
        self.updates += 1
        batch = self.clock.update_tokens(step)
        self.interval_loss_sum += mean_loss * batch
        self.interval_tokens += batch
        if self.updates == 1 or self.canonical_step(step + 1) % 100 == 0 or self.clock.finished(step + 1):
            self.emit("train", step_before_update=step, canonical_step_before_update=self.canonical_step(step),
                      total_training_tokens=self.clock.total(step + 1), update_tokens=batch,
                      mean_train_loss=mean_loss, dt_seconds=dt, lr_multiplier=lrm,
                      optimizer=group_settings(optimizer))

    def record_eval(self, step, val_bpb, optimizer, loader_state):
        assert math.isfinite(val_bpb)
        assert self.rank_tokens * self.world_size == self.added_tokens(step)
        self.assert_static_optimizer_settings(optimizer)
        self.emit("eval", step=step, canonical_step=self.canonical_step(step),
                  total_training_tokens=self.clock.total(step), val_bpb=float(val_bpb),
                  interval_train_mean_loss=(self.interval_loss_sum / self.interval_tokens if self.interval_tokens else None),
                  interval_train_tokens=self.interval_tokens, optimizer=group_settings(optimizer))
        self.last_val = float(val_bpb)
        self.interval_loss_sum = self.interval_tokens = 0

    def checkpoint_metadata(self, step, loader_state):
        return {"version": 1, "actual_step": step, "canonical_step": self.canonical_step(step),
                "total_training_tokens": self.clock.total(step), "horizon": HORIZON,
                "final_training_tokens": FINAL_TOKENS, "actual_matrix_lr": self.matrix_lr,
                "total_batch_size": self.args.total_batch_size, "world_size": self.world_size,
                "loader_states": self.gather({"rank": self.rank, "state": loader_state}),
                "resume_semantics": "approximate_per_rank_row_group_cursor"}

    def save_checkpoint(self, save_function, step, model_state, optimizer_state, metadata):
        metadata = dict(metadata)
        metadata["production"] = self.checkpoint_metadata(step, metadata["dataloader_state_dict"])
        if step == self.clock.start_actual and "production" in json.loads(
                (Path(self.args.production_checkpoint_source_dir) / f"meta_{step:06d}.json").read_text()):
            # A no-progress deadline or recovery after final save needs no rewrite.
            if self.checkpoint_dir.resolve() == Path(self.args.production_checkpoint_source_dir).resolve():
                self.last_saved_step = step
                self.emit("checkpoint_reused", step=step, canonical_step=self.canonical_step(step))
                return
        pending = self.checkpoint_dir / f".pending_{step:06d}_{self.job_id}"
        self.rank_zero_checked(lambda: pending.mkdir(parents=True, exist_ok=True))
        save_function(str(pending), step, model_state, optimizer_state, metadata, rank=self.rank)
        if dist.is_initialized():
            dist.barrier()

        def publish():
            names = [f"model_{step:06d}.pt"] + [f"optim_{step:06d}_rank{r}.pt" for r in range(self.world_size)]
            names.append(f"meta_{step:06d}.json")  # Publish metadata last as the commit record.
            for name in names:
                assert (pending / name).is_file() and (pending / name).stat().st_size > 0
            assert not (self.checkpoint_dir / names[-1]).exists(), "Committed checkpoint already exists"
            for name in names:
                os.replace(pending / name, self.checkpoint_dir / name)
            pending.rmdir()
            self.prune_checkpoints()
        self.rank_zero_checked(publish)
        self.last_saved_step = step
        self.emit("checkpoint", step=step, **metadata["production"])

    def prune_checkpoints(self):
        # Only complete, explicitly marked production checkpoints in this output directory.
        from checkpoint_state import complete_checkpoints
        checkpoints = complete_checkpoints(self.checkpoint_dir)
        for entry in checkpoints[:-self.args.production_keep_latest]:
            assert entry["metadata"].get("production", {}).get("version") == 1
            for name in entry["filenames"]:
                (self.checkpoint_dir / name).unlink()

    def finish(self, step, optimizer):
        assert self.clock.finished(step) and self.last_saved_step == step and self.last_val is not None
        summary = {"status": "complete", "actual_step": step, "canonical_step": self.canonical_step(step),
                   "total_training_tokens": self.clock.total(step), "terminal_val_bpb": self.last_val,
                   "actual_matrix_lr": self.matrix_lr, "total_batch_size": self.args.total_batch_size,
                   "checkpoint_dir": str(self.checkpoint_dir), "optimizer": group_settings(optimizer)}
        def commit():
            for name in ("summary.json", "BASE_DONE"):
                target = self.metrics_path.parent / name
                temporary = target.with_name(target.name + ".tmp")
                temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, target)
        self.rank_zero_checked(commit)
        self.emit("complete", **summary)
        self.complete_written = True
