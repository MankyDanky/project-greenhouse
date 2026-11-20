"""YAML configs with single inheritance and command-line overrides.

`extends: other.yaml` (a path relative to the file) is loaded first and this file's keys are merged over it (dicts
recursively, everything else replaced). Overrides are KEY=VALUE strings: KEY is dotted (replay.fraction,
groups.msmarco-v1.ratio), VALUE is parsed as YAML (0.05, true, [scifact], [])."""
from pathlib import Path

import yaml


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg, items):
    for item in items or ():
        key, eq, val = item.partition("=")
        assert key and eq, f"--set expects KEY=VALUE, got {item!r}"
        node, parts = cfg, key.split(".")
        for k in parts[:-1]:
            if node.get(k) is None:
                node[k] = {}
            node = node[k]
            assert isinstance(node, dict), f"--set {key}: {k} is not a mapping"
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    cfg = _merge(load_config(path.parent / parent), cfg) if parent else cfg
    return apply_overrides(cfg, overrides)
