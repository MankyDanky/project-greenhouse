"""Lightweight checkpoint discovery; never loads model/optimizer tensors."""
import argparse
import json
from pathlib import Path


def complete_checkpoints(directory):
    directory = Path(directory)
    result = []
    for meta_path in directory.glob("meta_*.json"):
        try:
            metadata = json.loads(meta_path.read_text())
            state = metadata["production"]
            step = metadata["step"]
            assert state["version"] == 1 and state["actual_step"] == step and state["world_size"] == 8
            assert meta_path.name == f"meta_{step:06d}.json"
            filenames = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"] + [f"optim_{step:06d}_rank{rank}.pt" for rank in range(8)]
            if all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in filenames):
                result.append({"step": step, "canonical_step": state["canonical_step"],
                               "total_training_tokens": state["total_training_tokens"],
                               "filenames": filenames, "metadata": metadata})
        except (OSError, ValueError, KeyError, AssertionError, TypeError):
            continue
    return sorted(result, key=lambda entry: entry["total_training_tokens"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--initial-dir", required=True)
    parser.add_argument("--initial-step", type=int, required=True)
    args = parser.parse_args()
    checkpoints = complete_checkpoints(args.directory)
    if checkpoints:
        print(str(Path(args.directory).resolve()))
        print(checkpoints[-1]["step"])
    else:
        print(str(Path(args.initial_dir).resolve()))
        print(args.initial_step)
