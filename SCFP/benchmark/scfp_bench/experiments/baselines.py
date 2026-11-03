"""CPU baselines for binary failure prediction (no GPU, no annotation needed).

  majority    always predicts the training-set majority class
  random      predicts failure with the training-set failure rate
  length      critique length; direction and threshold fit on train
  confidence  negative mean token log-prob of the critique (needs logprobs.py)
  tfidf_lr    TF-IDF word/bigram features of (P, R0, F) + logistic regression,
              C chosen on validation macro-F1

Thresholds and hyperparameters are fit on train/val only; test is touched once.

  python -m scfp_bench.experiments.baselines --config configs/laptop_local.yaml
"""

import argparse

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import dump_json, load_features, load_splits, partial_text, write_predictions
from .metrics import bootstrap, mcnemar, summarize


def fit_threshold(score, y):
    """Choose the sign and cut that maximize train accuracy for a 1-D score."""
    best = (-1, 1, 0.0)
    for sign in (1, -1):
        s = sign * np.asarray(score)
        for cut in np.unique(s):
            acc = np.mean((s >= cut).astype(int) == y)
            if acc > best[0]:
                best = (acc, sign, cut)
    return best[1], best[2]


def rank01(x):
    """Map scores to [0, 1] by rank so AUC is defined for any scale."""
    x = np.asarray(x, dtype=float)
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def run(cfg, seed=42):
    run_dir = cfg["run_dir"]
    S = load_splits(run_dir)
    y = {k: np.array([r["label"] for r in v]) for k, v in S.items()}
    lp = load_features(run_dir, "critique_logprob")
    preds = {}

    prior = y["train"].mean()
    preds["majority"] = (np.full(len(y["test"]), int(prior >= 0.5)), np.full(len(y["test"]), prior))
    rng = np.random.default_rng(seed)
    preds["random"] = ((rng.random(len(y["test"])) < prior).astype(int), rng.random(len(y["test"])))

    def heuristic(feature):
        tr = np.array([feature(r) for r in S["train"]])
        te = np.array([feature(r) for r in S["test"]])
        sign, cut = fit_threshold(tr, y["train"])
        return (sign * te >= cut).astype(int), rank01(sign * te)

    preds["length"] = heuristic(lambda r: len(r["critique"].split()))
    if lp and all(r["trace_id"] in lp for v in S.values() for r in v):
        preds["confidence"] = heuristic(lambda r: -lp[r["trace_id"]]["mean_logprob"])
    else:
        print("  confidence: skipped (run scfp_bench.experiments.logprobs first)")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([partial_text(r) for r in S["train"]])
    Xva = vec.transform([partial_text(r) for r in S["val"]])
    Xte = vec.transform([partial_text(r) for r in S["test"]])
    best_c, best_f1 = None, -1
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(Xtr, y["train"])
        f1 = f1_score(y["val"], m.predict(Xva), average="macro")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    m = LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced").fit(Xtr, y["train"])
    preds["tfidf_lr"] = (m.predict(Xte), m.predict_proba(Xte)[:, 1])

    results = {"test_n": int(len(y["test"])), "test_failure_rate": float(y["test"].mean()),
               "tfidf_lr_C": best_c, "models": {}}
    for name, (yp, p) in preds.items():
        rows = [{"trace_id": r["trace_id"], "model": name, "task": "binary", "split": "test",
                 "source_model": r["source_model"], "source": r["source"],
                 "y_true": int(t), "y_pred": int(a), "p_fail": float(b)}
                for r, t, a, b in zip(S["test"], y["test"], yp, p)]
        write_predictions(run_dir, name, "binary", rows)
        results["models"][name] = {**summarize(y["test"], yp, p), "ci95": bootstrap(y["test"], yp, p)}
        if name != "tfidf_lr":
            results["models"][name]["mcnemar_vs_tfidf_lr"] = mcnemar(y["test"], preds["tfidf_lr"][0], yp)
    dump_json(run_dir, "baselines_binary.json", results)
    for name, r in results["models"].items():
        print(f"  {name:11s} acc={r['accuracy']:.3f} {r['ci95']['accuracy']} "
              f"macroF1={r['macro_f1']:.3f} auc={r.get('auc', float('nan')):.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(yaml.safe_load(open(ap.parse_args().config)))


if __name__ == "__main__":
    main()
