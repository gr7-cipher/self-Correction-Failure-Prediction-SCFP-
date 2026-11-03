"""M5: remove source and generator confounding from the natural-set failure-prediction metrics.

Reads the saved out-of-fold predictions (scfp_runs/paper_v7/predictions.jsonl; target rf_wrong) and adds one
new fit, the identity-only baseline (one-hot source; pooled: source and generator) with the same grouped
5-fold CV. Reports within-source AUC (headline), stratified AUC, pooled AUC, AUPRC and Brier score with a
problem-level cluster bootstrap. See REVISION_PLAN.md (M5).

    .venv/bin/python scripts/revision/m5_deconfound.py --out results/revision/m5
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
from scfp_bench.experiments.signals import grouped_folds  # noqa: E402

RUNS = [os.path.join(BENCH, "scfp_runs", d) for d in ("signals_mac_q05_big", "signals_kaggle")]
PRED = os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")
GENS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b", "qwen2.5-3b"]
SOURCES = ["gsm8k", "strategyqa", "logiqa"]
PREDICTORS = ["identity", "text_tfidf", "signals_noverifier", "signals", "signals_gbm", "concat_text_signals"]
B, SEED = 1000, 0


def auc(y, p):
    """Mann-Whitney AUC (ties count 1/2); nan if one class is missing."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within(y, p, strata):
    """Support-weighted mean of per-stratum AUC, and pair-weighted (stratified) AUC."""
    a, w, num, den = [], [], 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        v = auc(y[m], p[m])
        if np.isnan(v):
            continue
        a.append(v)
        w.append(m.sum())
        pairs = y[m].sum() * (m.sum() - y[m].sum())
        num += v * pairs
        den += pairs
    return float(np.average(a, weights=w)), float(num / den)


def ap(y, p):
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def metrics(y, p, strata):
    w, st = within(y, p, strata)
    return {"within_auc": w, "stratified_auc": st, "pooled_auc": float(auc(y, p)), "auprc": ap(y, p),
            "brier": float(np.mean((p - y) ** 2)), "failure_rate": float(y.mean()), "n": int(len(y))}


def identity_oof(rows, cols):
    from sklearn.linear_model import LogisticRegressionCV
    X = np.array([[r[c] == v for c, vals in cols for v in vals] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    folds = grouped_folds(rows)
    p = np.zeros(len(rows))
    for k in range(5):
        tr, te = folds != k, folds == k
        m = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=3, max_iter=5000, class_weight="balanced")
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="results/revision/m5")
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pid = {}
    for rd in RUNS:
        for l in open(os.path.join(rd, "natural", "traces.jsonl")):
            t = json.loads(l)
            pid[t["trace_id"]] = t["problem_id"]
    by = defaultdict(dict)
    for l in open(PRED):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong":
            by[r["generator"]].setdefault(r["trace_id"], {"trace_id": r["trace_id"], "generator": r["generator"],
                                                          "source": r["source"], "y": r["y"],
                                                          "problem_id": pid[r["trace_id"]]})[r["predictor"]] = r["p"]
    data = {}
    for g in GENS:
        rows = sorted(by[g].values(), key=lambda r: r["trace_id"])
        p = identity_oof(rows, [("source", SOURCES)])
        for r, v in zip(rows, p):
            r["identity"] = float(v)
        data[g] = rows
    pooled = [r for g in GENS for r in data[g]]
    p = identity_oof(pooled, [("source", SOURCES), ("generator", GENS)])
    pooled_identity = dict(zip([(r["generator"], r["trace_id"]) for r in pooled], p))
    data["pooled"] = [{**r, "identity": float(pooled_identity[(r["generator"], r["trace_id"])])} for r in pooled]

    res, rng = {}, np.random.default_rng(SEED)
    for g, rows in data.items():
        y = np.array([r["y"] for r in rows])
        strata = np.array([r["source"] + ("|" + r["generator"] if g == "pooled" else "") for r in rows])
        P = {k: np.array([r[k] for r in rows]) for k in PREDICTORS}
        clusters = defaultdict(list)
        for i, r in enumerate(rows):
            clusters[r["problem_id"]].append(i)
        cl = list(clusters.values())
        point = {k: metrics(y, P[k], strata) for k in PREDICTORS}
        draws = {k: defaultdict(list) for k in PREDICTORS}
        diff = {k: [] for k in PREDICTORS}
        for _ in range(B):
            idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
            yb, sb = y[idx], strata[idx]
            wt = within(yb, P["text_tfidf"][idx], sb)[0]
            for k in PREDICTORS:
                m = metrics(yb, P[k][idx], sb)
                for mk in ("within_auc", "stratified_auc", "pooled_auc", "auprc", "brier"):
                    draws[k][mk].append(m[mk])
                diff[k].append(m["within_auc"] - wt)
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[g] = {k: {**point[k], "ci95": {mk: q(v) for mk, v in draws[k].items()},
                      "within_auc_diff_vs_text": point[k]["within_auc"] - point["text_tfidf"]["within_auc"],
                      "within_auc_diff_vs_text_ci95": q(diff[k])} for k in PREDICTORS}
        res[g]["_n_clusters"] = len(cl)
        print(g, f"n={len(rows)} clusters={len(cl)}")
        for k in PREDICTORS:
            v = res[g][k]
            print(f"  {k:22s} within={v['within_auc']:.3f} {[round(x, 3) for x in v['ci95']['within_auc']]} "
                  f"strat={v['stratified_auc']:.3f} pooled={v['pooled_auc']:.3f} auprc={v['auprc']:.3f} "
                  f"brier={v['brier']:.3f} d-vs-text={v['within_auc_diff_vs_text']:+.3f} "
                  f"{[round(x, 3) for x in v['within_auc_diff_vs_text_ci95']]}")
    json.dump(res, open(os.path.join(a.out, "deconfounded.json"), "w"), indent=1)
    with open(os.path.join(a.out, "identity_predictions.jsonl"), "w") as f:
        for g, rows in data.items():
            for r in rows:
                f.write(json.dumps({"row": g, "generator": r["generator"], "trace_id": r["trace_id"],
                                    "problem_id": r["problem_id"], "source": r["source"], "y": r["y"],
                                    "p": r["identity"]}) + "\n")


if __name__ == "__main__":
    main()
