"""Metrics, bootstrap intervals and paired tests on prediction files.

verification/verify_metrics.py recomputes the same quantities with its own
code, so the paper's tables are checked by two independent implementations.
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def summarize(y: Sequence[int], yp: Sequence[int], p: Sequence[float] = None) -> Dict[str, float]:
    y, yp = np.asarray(y), np.asarray(yp)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, yp)),
           "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
           "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0))}
    if p is not None and len(set(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def bootstrap(y, yp, p=None, n=1000, seed=0) -> Dict[str, List[float]]:
    """95% percentile intervals over test examples."""
    y, yp = np.asarray(y), np.asarray(yp)
    p = None if p is None else np.asarray(p)
    rng = np.random.default_rng(seed)
    draws = {"accuracy": [], "macro_f1": [], "auc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        draws["accuracy"].append(accuracy_score(y[i], yp[i]))
        draws["macro_f1"].append(f1_score(y[i], yp[i], average="macro", zero_division=0))
        if p is not None and len(set(y[i])) > 1:
            draws["auc"].append(roc_auc_score(y[i], p[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items() if v}


def mcnemar(y, pred_a, pred_b) -> Dict[str, float]:
    """Exact McNemar test on paired correctness of two models on the same examples."""
    y = np.asarray(y)
    a, b = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}
