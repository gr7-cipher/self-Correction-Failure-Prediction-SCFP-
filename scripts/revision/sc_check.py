"""Post-hoc check: the Qwen2.5-0.5B natural-set predictors with and without the undisclosed sc_* features.
Compares the saved predictions of the submitted analysis (results/revision/sc_check/predictions_with_sc.jsonl)
with the rerun of paper_v7 after the fix. See REVISION_PLAN.md (post-hoc finding).

    .venv/bin/python scripts/revision/sc_check.py --out results/revision/sc_check
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
from m5_deconfound import within  # noqa: E402

PRED = ["signals_noverifier", "signals", "signals_gbm", "concat_text_signals", "text_tfidf"]
B, SEED = 1000, 0


def load(path):
    d = defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r["target"] == "rf_wrong":
            d[(r["set"], r["generator"], r["predictor"])][r["trace_id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/sc_check")
    a = ap.parse_args()
    old = load(os.path.join(a.out, "predictions_with_sc.jsonl"))
    new = load(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl"))
    res = {"unchanged_other_generators": {}, "qwen2.5-0.5b": {}}
    for k in old:
        if k[1] != "qwen2.5-0.5b":
            same = all(abs(old[k][t]["p"] - new[k][t]["p"]) < 1e-12 for t in old[k])
            res["unchanged_other_generators"]["|".join(k)] = same
    rng = np.random.default_rng(SEED)
    for st in ("natural", "v1"):
        ids = sorted(old[(st, "qwen2.5-0.5b", "signals")])
        rows = old[(st, "qwen2.5-0.5b", "signals")]
        y = np.array([rows[t]["y"] for t in ids])
        s = np.array([rows[t]["source"] for t in ids])
        grp = np.array([rows[t].get("problem_id", t) for t in ids])
        cl = defaultdict(list)
        for i, g in enumerate(grp):
            cl[g].append(i)
        cl = [np.array(v) for v in cl.values()]
        D = [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(B)]
        for p in PRED:
            po = np.array([old[(st, "qwen2.5-0.5b", p)][t]["p"] for t in ids])
            pn = np.array([new[(st, "qwen2.5-0.5b", p)][t]["p"] for t in ids])
            w = lambda q, i: within(y[i], q[i], s[i])[0]
            d = [w(po, i) - w(pn, i) for i in D]
            res["qwen2.5-0.5b"][f"{st}|{p}"] = {"with_sc": w(po, np.arange(len(y))), "without_sc": w(pn, np.arange(len(y))),
                                                 "diff": w(po, np.arange(len(y))) - w(pn, np.arange(len(y))),
                                                 "diff_ci95": list(np.percentile(d, [2.5, 97.5])),
                                                 "max_abs_prob_change": float(np.max(np.abs(po - pn)))}
    for k, v in res["qwen2.5-0.5b"].items():
        print(f"{k:35s} with {v['with_sc']:.3f} without {v['without_sc']:.3f} diff {v['diff']:+.3f} "
              f"[{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]")
    print("other generators unchanged:", all(res["unchanged_other_generators"].values()))
    json.dump(res, open(os.path.join(a.out, "sc_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
