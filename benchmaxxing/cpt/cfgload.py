"""YAML configs with single inheritance: `extends: other.yaml` (path relative to the file) is loaded first and this
file's keys are merged over it (dicts recursively, everything else replaced)."""
from pathlib import Path

import yaml


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    return _merge(load_config(path.parent / parent), cfg) if parent else cfg
