"""T7 (round 2): the appendix accuracy table of the natural set with problem-level statistics. AUC over all traces
with a problem-level cluster bootstrap (10,000 resamples), accuracy and macro-F1 at 0.5, and the problem-level paired
permutation test of accuracy against text taken from M7 (results/revision/m7/stats_m7.json), so that the same p-value
appears in every table. Replaces the exact McNemar test, which treats the traces of a problem as independent and is
invalid for the pooled row.

    .venv/bin/python scripts/revision/t7_nat_main.py --out results/revision/t7
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
from m5_deconfound import auc  # noqa: E402
from m7_stats import cluster_ci, cluster_index, macro_f1  # noqa: E402

B, SEED = 10000, 0
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GENS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b", "qwen2.5-3b", "pooled"]
PRED = ["text_tfidf", "signals_noverifier", "signals", "signals_gbm", "concat_text_signals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/t7")
    a = ap.parse_args()
    pid = {}
    for l in open(os.path.join(ROOT, "results", "revision", "m5", "identity_predictions.jsonl")):
        r = json.loads(l)
        pid[(r["generator"], r["trace_id"])] = r["problem_id"]
    base = defaultdict(dict)
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong" and r["predictor"] in PRED:
            k = (r["generator"], r["trace_id"])
            for row in (r["generator"], "pooled"):
                base[row].setdefault(k, {"y": r["y"], "problem_id": pid[k]})[r["predictor"]] = r["p"]
    stats = json.load(open(os.path.join(ROOT, "results", "revision", "m7", "stats_m7.json")))["natural"]
    res = {}
    for g in GENS:
        rows = sorted(base[g].values(), key=lambda r: r["problem_id"])
        y = np.array([r["y"] for r in rows])
        cl = cluster_index([r["problem_id"] for r in rows])
        res[g] = {"n": len(rows), "majority": float(max(y.mean(), 1 - y.mean())), "failure_rate": float(y.mean())}
        for k in PRED:
            p = np.array([r[k] for r in rows])
            ci, _ = cluster_ci(lambda i: auc(y[i], p[i]), cl, np.random.default_rng(SEED), n=B)
            yp = (p >= .5).astype(int)
            res[g][k] = {"auc": float(auc(y, p)), "auc_ci95": ci, "accuracy": float(np.mean(yp == y)),
                         "macro_f1": macro_f1(y, yp)}
            if k != "text_tfidf":
                res[g][k]["perm_p_vs_text"] = stats[g][k]["accuracy_perm_p"]
                assert abs(stats[g][k]["accuracy"] - res[g][k]["accuracy"]) < 1e-12, (g, k)
        print(g, {k: round(res[g][k]["auc"], 3) for k in PRED})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "nat_main_perm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
