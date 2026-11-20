"""State for an explicitly released production chain. No submission side effects."""
import argparse
import json
from pathlib import Path
import time

from checkpoint_state import complete_checkpoints
from completion_evidence import released_initial_evidence
from production_clock import ABLATION_TOKENS, RAMP_SOURCE_TOKENS, source_clock


def atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def start_leg(directory, job_id, before_tokens):
    directory = Path(directory)
    path = directory / "chain_state.json"
    state = json.loads(path.read_text()) if path.exists() else {"consecutive_failures": 0}
    if state.get("status") == "running" and state.get("job_id") != job_id:
        state["consecutive_failures"] += 1  # Previous job died before its EXIT trap.
    if state["consecutive_failures"] >= 3:
        atomic(directory / "CHAIN_STOP", {"reason": "three_consecutive_failures", **state})
        return False
    state.update(status="running", job_id=job_id, before_tokens=before_tokens, started_at=time.time())
    atomic(path, state)
    return True


def finish_leg(directory, job_id, exit_code, after_tokens, done=False):
    directory = Path(directory)
    path = directory / "chain_state.json"
    state = json.loads(path.read_text())
    assert state["job_id"] == job_id, "Refusing to update a different leg's state"
    progressed = after_tokens > state["before_tokens"]
    success = exit_code == 0 and (progressed or done)
    state["consecutive_failures"] = 0 if success else state["consecutive_failures"] + 1
    state.update(status="complete" if done else ("checkpointed" if success else "failed"),
                 exit_code=exit_code, after_tokens=after_tokens, finished_at=time.time())
    atomic(path, state)
    if state["consecutive_failures"] >= 3:
        atomic(directory / "CHAIN_STOP", {"reason": "three_consecutive_failures", **state})
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("check-release")
    release.add_argument("release")
    release.add_argument("source_dir")
    release.add_argument("source_step", type=int)
    release.add_argument("batch", type=int)
    release.add_argument("matrix_lr", type=float)
    start = commands.add_parser("start")
    start.add_argument("directory")
    start.add_argument("job_id")
    start.add_argument("source_dir")
    start.add_argument("source_step", type=int)
    finish = commands.add_parser("finish")
    finish.add_argument("directory")
    finish.add_argument("job_id")
    finish.add_argument("exit_code", type=int)
    finish.add_argument("checkpoint_dir")
    args = parser.parse_args()
    if args.command == "check-release":
        value = json.loads(Path(args.release).read_text())
        assert value["status"] == "approved_for_production"
        assert value["validation_status"] == "all_four_arms_complete_and_comparable"
        assert value["actual_checkpoint_step"] == args.source_step
        assert value["total_batch_size"] == args.batch and value["actual_matrix_lr"] == args.matrix_lr
        meta = json.loads((Path(args.source_dir) / f"meta_{args.source_step:06d}.json").read_text())
        canonical, tokens = source_clock(meta)
        assert canonical == 138364 and tokens == RAMP_SOURCE_TOKENS + ABLATION_TOKENS
        assert meta["total_batch_size"] == args.batch
        assert meta["ablation"]["actual_matrix_lr"] == args.matrix_lr
        released_initial_evidence(meta, args.source_dir, args.source_dir, args.release,
                                  value["winner_completion"], value["winner_completion_sha256"])
        print("Production release and initial winning checkpoint agree")
    elif args.command == "start":
        meta = json.loads((Path(args.source_dir) / f"meta_{args.source_step:06d}.json").read_text())
        _, tokens = source_clock(meta)
        raise SystemExit(0 if start_leg(args.directory, args.job_id, tokens) else 3)
    else:
        checkpoints = complete_checkpoints(args.checkpoint_dir)
        state_path = Path(args.directory) / "chain_state.json"
        previous = json.loads(state_path.read_text())
        after = checkpoints[-1]["total_training_tokens"] if checkpoints else previous["before_tokens"]
        print(json.dumps(finish_leg(args.directory, args.job_id, args.exit_code, after,
                                   done=(Path(args.directory) / "BASE_DONE").is_file()), sort_keys=True))
