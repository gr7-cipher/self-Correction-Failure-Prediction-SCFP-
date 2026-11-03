"""M5 post-hoc sensitivity: AUC within source x fold (x generator for pooled rows). See REVISION_PLAN.md.

    .venv/bin/python scripts/revision/m5c_fold_strata.py --out results/revision/m5
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
from m5_deconfound import GENS, PREDICTORS, within  # noqa: E402
from scfp_bench.experiments.signals import grouped_folds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m5")
    a = ap.parse_args()
    base = defaultdict(dict)
    pid = {}
    for l in open(os.path.join(a.out, "identity_predictions.jsonl")):
        r = json.loads(l)
        pid[r["trace_id"]] = r["problem_id"]
        base[r["row"]][(r["generator"], r["trace_id"])] = {**r, "identity": r["p"]}
    BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong":
            for row in (r["generator"], "pooled"):
                base[row][(r["generator"], r["trace_id"])][r["predictor"]] = r["p"]
    out = {}
    for row in GENS + ["pooled"]:
        rows = sorted(base[row].values(), key=lambda r: (r["generator"], r["trace_id"]))
        fold = {}
        for g in GENS:  # the folds each prediction was made with: per generator, grouped by problem
            gr = [r for r in rows if r["generator"] == g]
            if gr:
                for r, f in zip(gr, grouped_folds(gr)):
                    fold[(g, r["trace_id"])] = f
        y = np.array([r["y"] for r in rows])
        st = np.array([f"{r['source']}|{r['generator']}|{fold[(r['generator'], r['trace_id'])]}" for r in rows])
        out[row] = {k: within(y, np.array([r[k] for r in rows]), st)[0] for k in PREDICTORS}
        print(row, {k: round(v, 3) for k, v in out[row].items()})
    json.dump(out, open(os.path.join(a.out, "within_source_fold.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
