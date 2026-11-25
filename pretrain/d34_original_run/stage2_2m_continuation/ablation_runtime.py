"""Instrumentation for the isolated, equal-token checkpoint-136316 screen."""

import copy
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REFERENCE_BATCH = 1048576
START_STEP = 136316
HORIZON = 321817
DEFAULT_TOKEN_BUDGET = 2147483648
EVAL_GRID = (0, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048)


def add_arguments(parser):
    parser.add_argument("--ablation-matrix-lr", type=float, required=True,
                        help="Actual Muon matrix learning rate; no batch multiplier is applied")
    parser.add_argument("--ablation-token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--ablation-checkpoint-source-dir", type=str, required=True)
    parser.add_argument("--ablation-metrics-path", type=str, required=True)


def validate_arguments(args, world_size):
    assert args.resume_from_step == START_STEP
    assert args.num_iterations == HORIZON, "Keep the original schedule horizon"
    assert args.total_batch_size in (REFERENCE_BATCH, 2 * REFERENCE_BATCH)
    assert (world_size, args.device_batch_size, args.max_seq_len, args.depth) == (8, 8, 2048, 34)
    assert args.fp8 and args.fp8_recipe == "tensorwise", "Preserve the source run's precision"
    assert args.warmdown_ratio == 0.5528
    assert args.final_lr_frac == 0.05 and args.warmup_steps == 40
    assert args.eval_tokens == 41943040
    assert args.ablation_matrix_lr > 0 and math.isfinite(args.ablation_matrix_lr)
    assert args.ablation_token_budget > 0
    assert args.ablation_token_budget % args.total_batch_size == 0
    assert args.ablation_token_budget <= DEFAULT_TOKEN_BUDGET, "This screen is capped at 2 Gi tokens"
    assert args.model_tag and args.model_tag not in {"d34veramp", "d34ve1m", "d34ve"}
    assert args.run == "dummy", "Disable external logging for this isolated screen"
    # This script owns the matched-token eval grid and terminal checkpoint.
    args.core_metric_every = -1
    args.sample_every = -1
    args.eval_every = -1
    args.save_every = -1
    args.stop_at_step = -1


def group_settings(optimizer):
    return [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups]


def restore_optimizer_for_ablation(optimizer, optimizer_data, actual_matrix_lr):
    """Load every saved setting/buffer, then change only Muon lr/initial_lr."""
    optimizer.load_state_dict(optimizer_data)
    before = group_settings(optimizer)
    muon_count = 0
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            group["lr"] = actual_matrix_lr
            group["initial_lr"] = actual_matrix_lr
            muon_count += 1
    assert muon_count > 0
    after = group_settings(optimizer)
    for old, new in zip(before, after):
        allowed = {"lr", "initial_lr"} if old["kind"] == "muon" else set()
        assert {k: v for k, v in old.items() if k not in allowed} == {
            k: v for k, v in new.items() if k not in allowed
        }, "Unexpected optimizer setting changed while restoring the checkpoint"
    return before


@contextmanager
def preserve_rng_state():
    """Validation (including loader construction) must not change training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
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


class AblationRuntime:
    def __init__(self, args, rank, world_size, checkpoint_dir, meta_data,
                 optimizer, loaded_settings, first_prefetched_state):
        self.args, self.rank, self.world_size = args, rank, world_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_path = Path(args.ablation_metrics_path)
        self.loaded_settings = loaded_settings
        self.hasher = hashlib.sha256()
        self.microbatches = self.rank_tokens = self.updates = 0
        self.interval_loss_sum = self.interval_tokens = 0
        self.total_loss_sum = self.total_loss_tokens = 0
        self.last_val = None
        self.last_eval_streams = None
        self.started_at = time.time()
        self.complete_written = False
        self.eval_tokens = {n * REFERENCE_BATCH for n in EVAL_GRID
                            if n * REFERENCE_BATCH <= args.ablation_token_budget}
        self.eval_tokens.add(args.ablation_token_budget)
        assert self.checkpoint_dir.resolve() != Path(args.ablation_checkpoint_source_dir).resolve()
        assert meta_data["step"] == START_STEP and meta_data["total_batch_size"] == REFERENCE_BATCH

        def prepare_output():
            assert not self.metrics_path.exists(), "Refusing to overwrite an earlier metrics stream"
            assert not self.checkpoint_dir.exists() or not any(self.checkpoint_dir.iterdir()), "Output checkpoint directory must be empty"
            assert not (self.metrics_path.parent / "COMPLETE.json").exists()
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.open("x").close()

        self.rank_zero_checked(prepare_output)
        start_states = self.gather({
            "rank": rank,
            "checkpoint_loader_state": meta_data["dataloader_state_dict"],
            "first_prefetched_loader_state": first_prefetched_state,
        })
        self.emit("start", source_dir=str(Path(args.ablation_checkpoint_source_dir).resolve()),
                  source_step=START_STEP, source_model_config=meta_data["model_config"],
                  total_batch_size=args.total_batch_size, token_budget=args.ablation_token_budget,
                  eval_token_grid=sorted(self.eval_tokens), eval_tokens=args.eval_tokens,
                  actual_matrix_lr=args.ablation_matrix_lr, world_size=world_size,
                  original_horizon=HORIZON, schedule_reference_batch=REFERENCE_BATCH,
                  loader_states=start_states, loaded_optimizer=loaded_settings,
                  applied_optimizer=group_settings(optimizer), config=vars(args).copy(),
                  resume_note="Original row-group resume is approximate; all branches share that restart")

    def gather(self, data):
        if not dist.is_initialized():
            return [data]
        result = [None] * self.world_size
        dist.all_gather_object(result, data)
        return result

    def rank_zero_checked(self, action):
        """Broadcast rank-zero filesystem failures instead of stranding peers."""
        status = [None]
        if self.rank == 0:
            try:
                action()
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        if dist.is_initialized():
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])

    def emit(self, event, **fields):
        if self.rank != 0:
            return
        row = {"event": event, "time_unix": time.time(), **fields}
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

    def consumed_batch(self, x, y):
        # x/y alias the loader's persistent buffer: hash BEFORE next(train_loader).
        self.hasher.update(x.detach().cpu().contiguous().numpy().tobytes(order="C"))
        self.hasher.update(y.detach().cpu().contiguous().numpy().tobytes(order="C"))
        self.microbatches += 1
        self.rank_tokens += x.numel()

    def added_tokens(self, step):
        return (step - START_STEP) * self.args.total_batch_size

    def canonical_step(self, step):
        return START_STEP + self.added_tokens(step) / REFERENCE_BATCH

    def assert_non_muon_unchanged(self, optimizer):
        for old, current in zip(self.loaded_settings, group_settings(optimizer)):
            if old["kind"] != "muon":
                assert current == old, "A non-Muon optimizer group changed"

    def record_step(self, step, mean_loss, dt, optimizer, lrm):
        if not math.isfinite(mean_loss):
            self.emit("failed", reason="nonfinite_train_loss", step=step)
            raise FloatingPointError("Nonfinite mean training loss; no optimizer update was permitted")
        self.assert_non_muon_unchanged(optimizer)
        self.updates += 1
        batch = self.args.total_batch_size
        self.interval_loss_sum += mean_loss * batch
        self.interval_tokens += batch
        self.total_loss_sum += mean_loss * batch
        self.total_loss_tokens += batch
        self.emit("train", step_before_update=step, completed_updates=self.updates,
                  added_tokens=self.added_tokens(step + 1),
                  canonical_step_before_update=self.canonical_step(step),
                  mean_train_loss=mean_loss, loss_reduction="all microbatches and all ranks, token weighted",
                  dt_seconds=dt, lr_multiplier=lrm, optimizer=group_settings(optimizer))

    def record_eval(self, step, val_bpb, optimizer, loader_state):
        assert math.isfinite(val_bpb), "Nonfinite validation loss"
        self.assert_non_muon_unchanged(optimizer)
        tokens = self.added_tokens(step)
        assert self.rank_tokens * self.world_size == tokens
        streams = self.gather({"rank": self.rank, "sha256": self.hasher.hexdigest(),
                               "consumed_microbatches": self.microbatches,
                               "consumed_x_tokens": self.rank_tokens,
                               "prefetched_loader_state": loader_state})
        self.emit("eval", step=step, added_tokens=tokens, canonical_step=self.canonical_step(step),
                  val_bpb=float(val_bpb), consumed_streams=streams,
                  interval_train_mean_loss=(self.interval_loss_sum / self.interval_tokens
                                            if self.interval_tokens else None),
                  interval_train_tokens=self.interval_tokens, optimizer=group_settings(optimizer))
        self.last_val, self.last_eval_streams = float(val_bpb), streams
        self.interval_loss_sum = self.interval_tokens = 0

    def checkpoint_metadata(self, step):
        return {"source_step": START_STEP, "added_tokens": self.added_tokens(step),
                "canonical_step": self.canonical_step(step),
                "token_budget": self.args.ablation_token_budget,
                "actual_matrix_lr": self.args.ablation_matrix_lr}

    def finish(self, step, optimizer):
        assert self.added_tokens(step) == self.args.ablation_token_budget
        assert self.last_val is not None and self.total_loss_tokens == self.args.ablation_token_budget
        if dist.is_initialized():
            dist.barrier()  # Every rank has returned from save_checkpoint.
        def finalize_output():
            names = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"] + [
                f"optim_{step:06d}_rank{r}.pt" for r in range(self.world_size)]
            for name in names:
                path = self.checkpoint_dir / name
                assert path.is_file() and path.stat().st_size > 0, f"Incomplete terminal checkpoint: {path}"
            assert json.loads((self.checkpoint_dir / f"meta_{step:06d}.json").read_text())["step"] == step
            summary = {"status": "complete", "completed_updates": self.updates,
                       "terminal_step": step, **self.checkpoint_metadata(step),
                       "terminal_val_bpb": self.last_val,
                       "mean_train_loss": self.total_loss_sum / self.total_loss_tokens,
                       "consumed_streams": self.last_eval_streams,
                       "checkpoint_dir": str(self.checkpoint_dir), "checkpoint_files": names,
                       "wall_seconds": time.time() - self.started_at,
                       "optimizer": group_settings(optimizer)}
            self.emit("complete", **summary)
            for name in ("summary.json", "COMPLETE.json"):
                temporary = self.metrics_path.parent / (name + ".tmp")
                temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
                os.replace(temporary, self.metrics_path.parent / name)
        self.rank_zero_checked(finalize_output)
        self.complete_written = True
