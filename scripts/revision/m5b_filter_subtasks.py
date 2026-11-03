"""M5 addendum: within-source metrics for the filter effect (SCFP v1.0 vs natural) and for the sub-tasks.

(a) uses the saved out-of-fold predictions (set == "v1" and "natural").
(b) recomputes the sub-task out-of-fold predictions with the paper_v7 code (GBM on signals + verifier, same
    folds), checks them against results.json, and saves them per trace. See REVISION_PLAN.md (M5 addendum).

    .venv/bin/python scripts/revision/m5b_filter_subtasks.py --out results/revision/m5
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
from m5_deconfound import B, SEED, auc, within  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows, oof  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.signals import grouped_folds  # noqa: E402

CONFIGS = [os.path.join(BENCH, "configs", c) for c in ("signals_mac_big.yaml", "signals_kaggle.yaml")]
PRED = os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")
RESULTS = os.path.join(BENCH, "scfp_runs", "paper_v7", "results.json")
TASKS = {"r0_wrong": (lambda r: True, lambda r: int(not r["r0_correct"])),
         "changed": (lambda r: True, lambda r: r["changed"]),
         "repair": (lambda r: not r["r0_correct"], lambda r: int(r["rf_correct"])),
         "harm": (lambda r: r["r0_correct"], lambda r: int(not r["rf_correct"]))}
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_within(y, p, strata, clusters, rng, n=B):
    cl = list(clusters.values())
    out = []
    for _ in range(n):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        out.append(within(y[idx], p[idx], strata[idx])[0])
    return np.array(out)


def clusters_of(rows, key="problem_id"):
    c = defaultdict(list)
    for i, r in enumerate(rows):
        c[r[key]].append(i)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m5")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cwd = os.getcwd()
    os.chdir(BENCH)  # configs use paths relative to the benchmark directory
    cfgs = [yaml.safe_load(open(c)) for c in CONFIGS]

    # problem ids for every trace of both sets
    pid = {}
    for cfg in cfgs:
        for f in ("natural/traces.jsonl",):
            for l in open(os.path.join(cfg["run_dir"], f)):
                t = json.loads(l)
                pid[t["trace_id"]] = t["problem_id"]
        for l in open(os.path.join(cfg["signals"]["v1_run_dir"], "labeled.jsonl")):
            t = json.loads(l)
            pid[t["trace_id"]] = t["problem_id"]

    # (a) filter effect
    preds = defaultdict(list)
    for l in open(PRED):
        r = json.loads(l)
        if r["target"] == "rf_wrong" and r["predictor"] in ("text_tfidf", "signals"):
            r["problem_id"] = pid[r["trace_id"]]
            preds[(r["set"], r["generator"], r["predictor"])].append(r)
    filt = {}
    for g in ("qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b"):
        filt[g] = {}
        for k in ("text_tfidf", "signals"):
            pts, draws = {}, {}
            for s in ("v1", "natural"):
                rows = sorted(preds[(s, g, k)], key=lambda r: r["trace_id"])
                y = np.array([r["y"] for r in rows])
                p = np.array([r["p"] for r in rows])
                st = np.array([r["source"] for r in rows])
                pts[s] = {"within_auc": within(y, p, st)[0], "pooled_auc": float(auc(y, p)), "n": len(rows),
                          "n_problems": len({r["problem_id"] for r in rows})}
                draws[s] = boot_within(y, p, st, clusters_of(rows), rng)
                pts[s]["within_auc_ci95"] = q(draws[s])
            d = draws["natural"] - draws["v1"]
            filt[g][k] = {**pts, "natural_minus_v1": pts["natural"]["within_auc"] - pts["v1"]["within_auc"],
                          "natural_minus_v1_ci95": q(d)}
            print(f"filter {g} {k}: v1 within {pts['v1']['within_auc']:.3f} {[round(x, 3) for x in pts['v1']['within_auc_ci95']]}"
                  f" (pooled {pts['v1']['pooled_auc']:.3f}, n={pts['v1']['n']}) | natural within "
                  f"{pts['natural']['within_auc']:.3f} | diff {filt[g][k]['natural_minus_v1']:+.3f} "
                  f"{[round(x, 3) for x in q(d)]}")

    # (b) sub-tasks
    R = json.load(open(RESULTS))
    sub, sub_rows = {}, []
    for cfg in cfgs:
        for model in cfg["signals"]["models"]:
            rows = generator_rows(cfg, "natural", model)
            for r in rows:
                r["problem_id"] = pid[r["trace_id"]]
            sub[model] = {}
            for t, (keep, lab) in TASKS.items():
                s_ = [r for r in rows if keep(r)]
                yt = np.array([lab(r) for r in s_])
                p = oof("gbm", {"text": None, "hidden": None, "scalar": scalar_matrix(s_)}, yt, grouped_folds(s_))
                pooled = float(auc(yt, p))
                ref = R["subtasks"][f"natural|{model}|{t}"]["auc"]
                assert abs(pooled - ref) < 1e-9, f"sub-task {model} {t}: {pooled} vs results.json {ref}"
                st = np.array([r["source"] for r in s_])
                w = within(yt, p, st)
                dr = boot_within(yt, p, st, clusters_of(s_), rng)
                sub[model][t] = {"pooled_auc": pooled, "within_auc": w[0], "stratified_auc": w[1],
                                 "within_auc_ci95": q(dr), "n": len(s_)}
                for r, v, yy in zip(s_, p, yt):
                    sub_rows.append({"generator": model, "trace_id": r["trace_id"], "problem_id": r["problem_id"],
                                     "source": r["source"], "target": t, "y": int(yy), "p": float(v)})
            print(f"subtasks {model}: " + ", ".join(f"{t} within {v['within_auc']:.3f} (pooled {v['pooled_auc']:.3f})"
                                                   for t, v in sub[model].items()))
    rule = {g: all(sub[g][t]["within_auc"] > sub[g]["r0_wrong"]["within_auc"] for t in ("changed", "repair", "harm"))
            for g in sub}
    print("sub-task decision rule (behaviour > r0_wrong within source):", rule)
    os.chdir(cwd)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"filter_effect": filt, "subtasks": sub, "subtask_rule_holds": rule},
              open(os.path.join(a.out, "filter_subtasks.json"), "w"), indent=1)
    with open(os.path.join(a.out, "subtask_predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in sub_rows)


if __name__ == "__main__":
    main()
