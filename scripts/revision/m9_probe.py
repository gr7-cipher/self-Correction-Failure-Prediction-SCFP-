"""M9: hidden-state probes at R0's answer tokens, at every layer, and a stacked model with the layer chosen
inside each outer training fold. Reads the job-A arrays (results/revision/gpuA/hidden_<generator>.npz).
See REVISION_PLAN.md (M9).

    .venv/bin/python scripts/revision/m9_probe.py --hidden results/revision/gpuA --out results/revision/m9
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import yaml
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from m5_deconfound import within  # noqa: E402
from scfp_bench.experiments.improve import Base, scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows, oof  # noqa: E402
from scfp_bench.experiments.signals import grouped_folds  # noqa: E402

CONFIGS = {"qwen2.5-0.5b": "signals_mac_big.yaml", "qwen2.5-1.5b": "signals_kaggle.yaml",
           "smollm2-1.7b": "signals_kaggle.yaml", "qwen2.5-3b": "signals_kaggle.yaml"}
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1]
B, SEED, JOBS = 1000, 0, 6


def probe(X, y, groups):
    """Standardize, then class-balanced L2 logistic regression with C chosen by 3-fold CV grouped by problem."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = list(GroupKFold(3).split(X, y, groups))
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=cv, scoring="roc_auc", class_weight="balanced",
                                                             max_iter=3000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(X, y)


def layer_oof(H, l, y, folds, groups, ok):
    """Out-of-fold probe probabilities at layer l (rows without an answer state stay nan)."""
    p = np.full(len(y), np.nan)
    X = H[:, l].astype(np.float32)
    for k in range(5):
        tr, te = np.where((folds != k) & ok)[0], np.where((folds == k) & ok)[0]
        p[te] = probe(X[tr], y[tr], groups[tr]).predict_proba(X[te])[:, 1]
    return p


def inner_layer_scores(H, l, y, groups, strata, tr):
    """Inner 3-fold (grouped) out-of-fold probe probabilities at layer l on the outer training rows."""
    from sklearn.model_selection import GroupKFold
    X = H[tr, l].astype(np.float32)
    q = np.zeros(len(tr))
    for a, b in GroupKFold(3).split(X, y[tr], groups[tr]):
        q[b] = probe(X[a], y[tr][a], groups[tr][a]).predict_proba(X[b])[:, 1]
    return within(y[tr], q, strata[tr])[0], q


def stacked(H, rows, y, folds, groups, strata, ok, Xsig):
    """Signals + verifier LR with the probe probability added; the layer is chosen inside each outer training fold."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scfp_bench.experiments.improve import _lr
    p = np.zeros(len(y))
    chosen = []
    L = H.shape[1]
    for k in range(5):
        tr = np.where((folds != k) & ok)[0]
        te = np.where(folds == k)[0]
        res = Parallel(n_jobs=JOBS)(delayed(inner_layer_scores)(H, l, y, groups, strata, tr) for l in range(L))
        best = int(np.argmax([r[0] for r in res]))
        chosen.append(best)
        f = np.full(len(y), np.nan)
        f[tr] = res[best][1]
        te_ok = te[ok[te]]
        f[te_ok] = probe(H[tr, best].astype(np.float32), y[tr], groups[tr]).predict_proba(H[te_ok, best].astype(np.float32))[:, 1]
        X = np.hstack([Xsig, f[:, None]])
        tr_all = np.where(folds != k)[0]  # rows without an answer state keep a missing (imputed) probe feature
        f_tr_missing = np.setdiff1d(tr_all, tr)
        X[f_tr_missing, -1] = np.nan
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _lr([0.001, 0.01, 0.1, 1]))
        m.fit(X[tr_all], y[tr_all])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p, chosen


def boot_within(y, ps, strata, rng):
    """Bootstrap over traces (= problems: one trace per problem per generator) of within-source AUC and differences."""
    out = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append([within(y[i], p[i], strata[i])[0] for p in ps])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="results/revision/gpuA")
    ap.add_argument("--out", default="results/revision/m9")
    ap.add_argument("--models", nargs="+", default=list(CONFIGS), help="rerun these generators, keep the others")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from m9.json")
    a = ap.parse_args()
    if a.plot_only:
        plot(json.load(open(os.path.join(a.out, "m9.json"))), os.path.join(a.out, "fig_layers.pdf"))
        return
    hid_dir, out = os.path.abspath(a.hidden), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(BENCH)
    old_p, old_pr = os.path.join(out, "m9.json"), os.path.join(out, "m9_predictions.jsonl")
    keep = set(CONFIGS) - set(a.models)
    res = {g: v for g, v in json.load(open(old_p)).items() if g in keep} if keep else {}
    pred_rows = [r for r in map(json.loads, open(old_pr)) if r["generator"] in keep] if keep else []
    for g, cfg_name in CONFIGS.items():
        if g not in a.models:
            continue
        rng = np.random.default_rng(SEED)
        cfg = yaml.safe_load(open(os.path.join("configs", cfg_name)))
        rows = generator_rows(cfg, "natural", g)
        tr_ = {t["trace_id"]: t for t in (json.loads(l) for l in open(os.path.join(cfg["run_dir"], "natural", "traces.jsonl")))}
        z = np.load(os.path.join(hid_dir, f"hidden_{g}.npz"))
        idx = {t: i for i, t in enumerate(z["trace_id"])}
        assert set(idx) == {r["trace_id"] for r in rows}, g
        o = np.array([idx[r["trace_id"]] for r in rows])
        A, Lt, ok = z["ans"][o], z["last"][o], z["has_ans"][o].astype(bool)
        y = np.array([r["label"] for r in rows])
        y0 = np.array([int(not tr_[r["trace_id"]]["r0_correct"]) for r in rows])
        folds = grouped_folds(rows)
        groups = np.array([r["problem_id"] for r in rows])
        strata = np.array([r["source"] for r in rows])
        L = A.shape[1]
        print(f"{g}: {len(rows)} traces, {L} layers, {int((~ok).sum())} without an answer-span state", flush=True)
        per = {}
        for tgt, yy in (("rf_wrong", y), ("r0_wrong", y0)):
            P = Parallel(n_jobs=JOBS)(delayed(layer_oof)(A, l, yy, folds, groups, ok) for l in range(L))
            per[tgt] = [within(yy[ok], p[ok], strata[ok])[0] for p in P]
            if tgt == "rf_wrong":
                P_rf = P
            print(f"  {tgt} within-source AUC by layer:", " ".join(f"{v:.3f}" for v in per[tgt]), flush=True)
        best = int(np.argmax(per["rf_wrong"]))
        # the paper's probe: last R0 token, last layer, PCA-64 (recomputed from the new extraction)
        p_last = oof("hidden", {"text": None, "scalar": None, "hidden": Lt[:, -1].astype(np.float32)}, y, folds)
        # signals + verifier LR (the paper's "signals" predictor) and the stacked model
        Xsig = scalar_matrix(rows)
        p_sig = oof("scalar", {"text": None, "hidden": None, "scalar": Xsig}, y, folds)
        p_stk, chosen = stacked(A, rows, y, folds, groups, strata, ok, Xsig)
        w = lambda p: within(y, p, strata)[0]
        bt = boot_within(y[ok], [P_rf[best][ok]], strata[ok], rng)
        bs = boot_within(y, [p_stk, p_sig], strata, rng)
        d = bs[:, 0] - bs[:, 1]
        res[g] = {"n": len(rows), "layers": L, "n_without_answer_state": int((~ok).sum()),
                  "per_layer_within_auc": per, "best_layer_post_hoc": best,
                  "best_layer_auc": per["rf_wrong"][best], "best_layer_ci95": [float(np.percentile(bt[:, 0], 2.5)), float(np.percentile(bt[:, 0], 97.5))],
                  "last_layer_answer_auc": per["rf_wrong"][-1],
                  "last_token_pca64_within_auc": w(p_last), "signals_within_auc": w(p_sig), "stacked_within_auc": w(p_stk),
                  "stacked_minus_signals": float(w(p_stk) - w(p_sig)),
                  "stacked_minus_signals_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "layers_chosen_per_fold": chosen}
        for r, a1, a2, a3 in zip(rows, P_rf[best], p_stk, p_sig):
            pred_rows.append({"generator": g, "trace_id": r["trace_id"], "source": r["source"], "y": int(r["label"]),
                              "probe_best_layer": None if np.isnan(a1) else float(a1), "stacked": float(a2), "signals": float(a3)})
        v = res[g]
        print(f"  best layer {best} (post hoc): {v['best_layer_auc']:.3f} {[round(x, 3) for x in v['best_layer_ci95']]}; "
              f"last-token PCA-64 {v['last_token_pca64_within_auc']:.3f}; signals {v['signals_within_auc']:.3f}; "
              f"stacked {v['stacked_within_auc']:.3f}; diff {v['stacked_minus_signals']:+.3f} "
              f"{[round(x, 3) for x in v['stacked_minus_signals_ci95']]}; layers chosen {chosen}", flush=True)
    res = {g: res[g] for g in CONFIGS}
    res["_rule_count"] = sum(res[g]["stacked_minus_signals_ci95"][0] > 0 for g in CONFIGS)
    json.dump(res, open(os.path.join(out, "m9.json"), "w"), indent=1)
    with open(os.path.join(out, "m9_predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in pred_rows)
    plot(res, os.path.join(out, "fig_layers.pdf"))
    print("generators where stacking adds (CI > 0):", res["_rule_count"])


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    names = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "smollm2-1.7b": "SmolLM2-1.7B", "qwen2.5-3b": "Qwen2.5-3B"}
    for j, (tgt, title) in enumerate((("rf_wrong", "Final answer wrong"), ("r0_wrong", "Initial answer wrong"))):
        for g, n in names.items():
            v = res[g]["per_layer_within_auc"][tgt]
            ax[j].plot(np.arange(1, len(v) + 1) / len(v), v, label=n, lw=1.5)
        ax[j].axhline(0.5, color="0.6", lw=0.8, ls=":")
        ax[j].set_title(title, fontsize=10)
        ax[j].set_xlabel("Relative layer depth")
    ax[0].set_ylabel("Within-source AUC")
    h, lab = ax[0].get_legend_handles_labels()  # legend below the panels, off the curves
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)


if __name__ == "__main__":
    main()
