"""Grouped k-fold cross-validation over all labeled traces.

The single test split has 108 traces, so its 95% interval on accuracy is about
+/-9 points. Cross-validation predicts every one of the 716 traces exactly once
(out-of-fold), which narrows the interval to about +/-3.5 points and allows
paired tests between models on the same 716 traces.

Folds are grouped by problem (all traces of a problem are in the same fold) and
stratified by source. Within each fold, 15% of the training problems are held
out for early stopping. Each fold is run with the already tested scripts
(train_encoder, baselines) on a fold-specific run directory.

  python -m scfp_bench.experiments.cv run --config CFG --name deberta-large-scfp \\
      -- --arch scfp --model microsoft/deberta-v3-large
  python -m scfp_bench.experiments.cv tfidf --config CFG
  python -m scfp_bench.experiments.cv summary --config CFG
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import yaml

from ..stages import read_jsonl, write_jsonl
from .data import load_splits
from .metrics import bootstrap, mcnemar, summarize

K = 5


def folds(rows, k=K, seed=42):
    """problem_id -> fold, stratified by source."""
    rng = random.Random(seed)
    out = {}
    for src in sorted({r["source"] for r in rows}):
        pids = sorted({r["problem_id"] for r in rows if r["source"] == src})
        rng.shuffle(pids)
        for i, p in enumerate(pids):
            out[p] = i % k
    return out


def fold_dir(cfg, k, seed=42):
    """Create (once) the run directory of fold k: train/val from the other folds, test = fold k."""
    base = cfg["run_dir"]
    d = os.path.join(base, "cv", f"fold{k}")
    if os.path.exists(os.path.join(d, "splits", "test.jsonl")):
        return d
    S = load_splits(base)
    rows = [r for v in S.values() for r in v]
    f = folds(rows)
    rest = sorted({r["problem_id"] for r in rows if f[r["problem_id"]] != k})
    rng = random.Random(seed + k)
    rng.shuffle(rest)
    val = set(rest[: round(0.15 * len(rest))])
    parts = {"test": [r for r in rows if f[r["problem_id"]] == k],
             "val": [r for r in rows if r["problem_id"] in val],
             "train": [r for r in rows if f[r["problem_id"]] != k and r["problem_id"] not in val]}
    os.makedirs(os.path.join(d, "splits"), exist_ok=True)
    for name, part in parts.items():
        write_jsonl(os.path.join(d, "splits", f"{name}.jsonl"), part)
    feat = os.path.join(base, "features")
    if os.path.exists(feat) and not os.path.exists(os.path.join(d, "features")):
        shutil.copytree(feat, os.path.join(d, "features"))
    return d


def fold_cfg(cfg, d):
    c = dict(cfg, run_dir=d)
    p = os.path.join(d, "config.yaml")
    yaml.safe_dump(c, open(p, "w"))
    return p


def cmd_run(cfg, name, extra):
    for k in range(K):
        d = fold_dir(cfg, k)
        if os.path.exists(os.path.join(d, "predictions", f"{name}__binary.jsonl")):
            print(f"=== {name} fold {k}: done, skipping", flush=True)
            continue
        print(f"=== {name} fold {k}", flush=True)
        cmd = [sys.executable, "-m", "scfp_bench.experiments.train_encoder",
               "--config", fold_cfg(cfg, d), "--name", name] + extra
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {name} fold {k} failed (exit {r.returncode})", flush=True)


def cmd_tfidf(cfg):
    from .baselines import run as run_baselines
    for k in range(K):
        d = fold_dir(cfg, k)
        print(f"=== baselines fold {k}", flush=True)
        run_baselines(dict(cfg, run_dir=d))


def oof(cfg, name):
    """Out-of-fold predictions of one model, or None if any fold is missing."""
    rows = []
    for k in range(K):
        p = os.path.join(cfg["run_dir"], "cv", f"fold{k}", "predictions", f"{name}__binary.jsonl")
        if not os.path.exists(p):
            return None
        rows += read_jsonl(p)
    return {r["trace_id"]: r for r in rows}


def cmd_summary(cfg, out=None):
    base = cfg["run_dir"]
    names = sorted({f[: -len("__binary.jsonl")]
                    for k in range(K)
                    for f in os.listdir(os.path.join(base, "cv", f"fold{k}", "predictions"))
                    if f.endswith("__binary.jsonl")} if os.path.exists(os.path.join(base, "cv", "fold0", "predictions")) else [])
    ref = oof(cfg, "tfidf_lr")
    maj = oof(cfg, "majority")
    res = {}
    for n in names:
        P = oof(cfg, n)
        if P is None:
            continue
        ids = sorted(P)
        y = [P[i]["y_true"] for i in ids]; yp = [P[i]["y_pred"] for i in ids]; pf = [P[i]["p_fail"] for i in ids]
        r = {**summarize(y, yp, pf), "ci95": bootstrap(y, yp, pf)}
        if ref and n != "tfidf_lr":
            r["mcnemar_vs_tfidf_lr"] = mcnemar(y, [ref[i]["y_pred"] for i in ids], yp)
        if maj and n != "majority":
            r["mcnemar_vs_majority"] = mcnemar(y, [maj[i]["y_pred"] for i in ids], yp)
        res[n] = r
    os.makedirs(os.path.join(base, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(base, "results", "cv_summary.json"), "w"), indent=2)
    lines = ["| Model | n | Accuracy [95% CI] | Macro F1 | AUC | p vs TF-IDF | p vs majority |",
             "|---|---|---|---|---|---|---|"]
    for n, r in sorted(res.items(), key=lambda kv: -kv[1]["accuracy"]):
        ci = r["ci95"]["accuracy"]
        f = lambda key: (f"{r[key]['p_value']:.3f}" if key in r else "--")
        lines.append(f"| {n} | {r['n']} | {100 * r['accuracy']:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
                     f"{r['macro_f1']:.3f} | {r.get('auc', float('nan')):.3f} | {f('mcnemar_vs_tfidf_lr')} | {f('mcnemar_vs_majority')} |")
    table = "\n".join(lines)
    open(os.path.join(base, "results", "cv_summary.md"), "w").write(
        "# 5-fold grouped cross-validation (out-of-fold predictions on all labeled traces)\n\n" + table + "\n")
    print(table)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tfidf", "summary"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None)
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []   # arguments for train_encoder
    a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    cfg = yaml.safe_load(open(a.config))
    if a.cmd == "run":
        if not a.name:
            ap.error("run needs --name")
        cmd_run(cfg, a.name, extra)
    elif a.cmd == "tfidf":
        cmd_tfidf(cfg)
    else:
        cmd_summary(cfg)


if __name__ == "__main__":
    main()
