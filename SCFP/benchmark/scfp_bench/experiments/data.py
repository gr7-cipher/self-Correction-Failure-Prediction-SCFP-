"""Load benchmark splits and write prediction files."""

import json
import os
from typing import Dict, List

from ..stages import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test")


def load_splits(run_dir: str) -> Dict[str, List[dict]]:
    """Prefer final/ (has failure_type after annotation); fall back to splits/."""
    base = "final" if os.path.exists(os.path.join(run_dir, "final", "train.jsonl")) else "splits"
    return {s: read_jsonl(os.path.join(run_dir, base, f"{s}.jsonl")) for s in SPLITS}


def partial_text(r: dict, sep: str = "\n\n[SEP]\n\n") -> str:
    """The predictor's input: prompt, initial response and critique. Never Rf."""
    return sep.join([r["prompt"], r["initial_response"], r["critique"]])


def write_predictions(run_dir: str, model: str, task: str, rows: List[dict]):
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    p = os.path.join(run_dir, "predictions", f"{model}__{task}.jsonl")
    write_jsonl(p, rows)
    return p


def load_features(run_dir: str, name: str) -> Dict[str, dict]:
    p = os.path.join(run_dir, "features", f"{name}.jsonl")
    return {r["trace_id"]: r for r in read_jsonl(p)} if os.path.exists(p) else {}


def dump_json(run_dir: str, name: str, obj):
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    with open(os.path.join(run_dir, "results", name), "w") as f:
        json.dump(obj, f, indent=2)
