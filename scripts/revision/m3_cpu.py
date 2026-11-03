"""M3 (CPU part): Delta-label, deployable two-stage gate, and the answer-span log-prob alone, on the natural set.
Same folds and features as the paper (signals + verifier); within-source metrics and cluster bootstrap as in M5.
See REVISION_PLAN.md (M3).

    .venv/bin/python scripts/revision/m3_cpu.py --out results/revision/m3
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from m5_deconfound import B, SEED, auc, within  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows, oof  # noqa: E402
from scfp_bench.experiments.signals import grouped_folds  # noqa: E402

CONFIGS = {"qwen2.5-0.5b": "signals_mac_big.yaml", "qwen2.5-1.5b": "signals_kaggle.yaml",
           "smollm2-1.7b": "signals_kaggle.yaml", "qwen2.5-3b": "signals_kaggle.yaml"}
CLASSES = ["helped", "neutral", "harmed"]


def lr_pipe(multi=False):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=3, max_iter=5000, class_weight="balanced"))


def delta_label(r):
    if not r["r0_correct"] and r["rf_correct"]:
        return 0
    if r["r0_correct"] and not r["rf_correct"]:
        return 2
    return 1


def macro_f1(y, yp, k):
    f = []
    for c in range(k):
        tp = np.sum((yp == c) & (y == c)); fp = np.sum((yp == c) & (y != c)); fn = np.sum((yp != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot(fn, n_items, rng, groups=None):
    idx_all = np.arange(n_items)
    vals = []
    if groups is None:
        for _ in range(B):
            v = fn(rng.integers(0, n_items, n_items))
            if not np.isnan(v):
                vals.append(v)
    else:
        cl = {}
        for i, g in enumerate(groups):
            cl.setdefault(g, []).append(i)
        cl = [np.array(v) for v in cl.values()]
        for _ in range(B):
            v = fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]))
            if not np.isnan(v):
                vals.append(v)
    del idx_all
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m3")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.chdir(BENCH)
    pid = {}
    for run in ("signals_mac_q05_big", "signals_kaggle"):
        for l in open(os.path.join("scfp_runs", run, "natural", "traces.jsonl")):
            t = json.loads(l)
            pid[t["trace_id"]] = t["problem_id"]
    res, preds, rng = {}, [], np.random.default_rng(SEED)
    pooled = {k: [] for k in ("y_rf", "y_r0", "y_d", "src", "gen", "pid", "p_direct", "p_gate", "p_lp", "P_delta")}
    for g, cname in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join("configs", cname)))
        rows = generator_rows(cfg, "natural", g)
        X = scalar_matrix(rows)
        folds = grouped_folds(rows)
        y_rf = np.array([r["label"] for r in rows])
        y_r0 = np.array([int(not r["r0_correct"]) for r in rows])
        y_d = np.array([delta_label(r) for r in rows])
        src = np.array([r["source"] for r in rows])
        n = len(rows)
        # (a) Delta-label: multinomial, class-balanced
        Pd = np.zeros((n, 3))
        for k in range(5):
            tr, te = folds != k, folds == k
            m = lr_pipe().fit(X[tr], y_d[tr])
            Pd[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
        # (b) two-stage gate
        p0 = 1 - oof("scalar", {"text": None, "hidden": None, "scalar": X}, y_r0, folds)   # P(R0 correct)
        Xg = np.hstack([X, p0[:, None]])
        p_rep, p_harm = np.zeros(n), np.zeros(n)
        r0c = 1 - y_r0
        rf_c = 1 - y_rf
        for k in range(5):
            tr, te = folds != k, folds == k
            w = tr & (r0c == 0)   # R0 wrong in training: P(repair | R0 wrong, signals, p0_hat)
            p_rep[te] = lr_pipe().fit(Xg[w], rf_c[w]).predict_proba(Xg[te])[:, 1]
            c = tr & (r0c == 1)   # R0 right in training: P(harm | R0 right, signals, p0_hat)
            p_harm[te] = lr_pipe().fit(Xg[c], y_rf[c]).predict_proba(Xg[te])[:, 1]
        p_gate = (1 - p0) * (1 - p_rep) + p0 * p_harm
        p_direct = oof("scalar", {"text": None, "hidden": None, "scalar": X}, y_rf, folds)
        # (c) answer-span log-prob alone (failure score = minus log-prob; missing -> median)
        lp = np.array([r.get("lp_answer_mean", np.nan) for r in rows], float)
        n_lp_missing = int(np.isnan(lp).sum())
        p_lp = -np.where(np.isnan(lp), np.nanmedian(lp), lp)

        def wa(y, p, i=None):
            i = np.arange(n) if i is None else i
            return within(y[i], p[i], src[i])[0]

        def delta_auc(i=None):
            i = np.arange(n) if i is None else i
            return float(np.mean([within((y_d[i] == c).astype(int), Pd[i, c], src[i])[0] for c in range(3)]))
        r = {"n": n, "class_rates": {c: float(np.mean(y_d == j)) for j, c in enumerate(CLASSES)},
             "delta": {"macro_within_auc": delta_auc(), "macro_within_auc_ci95": boot(delta_auc, n, rng),
                       "per_class_within_auc": {c: within((y_d == j).astype(int), Pd[:, j], src)[0] for j, c in enumerate(CLASSES)},
                       "macro_f1": macro_f1(y_d, Pd.argmax(1), 3)},
             "gate": {"within_auc": wa(y_rf, p_gate), "within_auc_ci95": boot(lambda i: wa(y_rf, p_gate, i), n, rng),
                      "direct_within_auc": wa(y_rf, p_direct),
                      "diff": wa(y_rf, p_gate) - wa(y_rf, p_direct),
                      "diff_ci95": boot(lambda i: wa(y_rf, p_gate, i) - wa(y_rf, p_direct, i), n, rng),
                      "stage1_r0_wrong_within_auc": wa(y_r0, 1 - p0)},
             "answer_logprob": {"rf_wrong_within_auc": wa(y_rf, p_lp), "rf_wrong_ci95": boot(lambda i: wa(y_rf, p_lp, i), n, rng),
                                "r0_wrong_within_auc": wa(y_r0, p_lp), "r0_wrong_ci95": boot(lambda i: wa(y_r0, p_lp, i), n, rng),
                                "missing_imputed": n_lp_missing}}
        res[g] = r
        print(g, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in
                             {"delta_macro_auc": r["delta"]["macro_within_auc"], "delta_f1": r["delta"]["macro_f1"],
                              "gate": r["gate"]["within_auc"], "direct": r["gate"]["direct_within_auc"],
                              "gate_diff": r["gate"]["diff"], "lp_rf": r["answer_logprob"]["rf_wrong_within_auc"],
                              "lp_r0": r["answer_logprob"]["r0_wrong_within_auc"]}.items()}),
              {c: round(v, 3) for c, v in r["delta"]["per_class_within_auc"].items()}, r["gate"]["diff_ci95"])
        for i, row in enumerate(rows):
            preds.append({"generator": g, "trace_id": row["trace_id"], "problem_id": pid[row["trace_id"]], "source": src[i],
                          "y_rf": int(y_rf[i]), "y_r0": int(y_r0[i]), "y_delta": CLASSES[y_d[i]],
                          "p_direct": float(p_direct[i]), "p_gate": float(p_gate[i]), "p_r0_correct": float(p0[i]),
                          "p_delta": Pd[i].tolist(), "score_answer_logprob": float(p_lp[i])})
        for k_, v in (("y_rf", y_rf), ("y_r0", y_r0), ("y_d", y_d), ("p_direct", p_direct), ("p_gate", p_gate), ("p_lp", p_lp)):
            pooled[k_].append(v)
        pooled["P_delta"].append(Pd)
        pooled["src"].append(np.array([f"{s}|{g}" for s in src]))
        pooled["pid"].append(np.array([pid[x["trace_id"]] for x in rows]))
    Y = {k: np.concatenate(v) for k, v in pooled.items() if v}
    st, gr = Y["src"], Y["pid"]
    wa = lambda y, p, i: within(y[i], p[i], st[i])[0]
    N = len(st)
    allix = np.arange(N)
    res["pooled"] = {
        "n": int(N),
        "delta": {"macro_within_auc": float(np.mean([within((Y["y_d"] == c).astype(int), Y["P_delta"][:, c], st)[0] for c in range(3)]))},
        "gate": {"within_auc": wa(Y["y_rf"], Y["p_gate"], allix), "direct_within_auc": wa(Y["y_rf"], Y["p_direct"], allix),
                 "diff": wa(Y["y_rf"], Y["p_gate"], allix) - wa(Y["y_rf"], Y["p_direct"], allix),
                 "diff_ci95": boot(lambda i: wa(Y["y_rf"], Y["p_gate"], i) - wa(Y["y_rf"], Y["p_direct"], i), N, rng, gr)},
        "answer_logprob": {"rf_wrong_within_auc": wa(Y["y_rf"], Y["p_lp"], allix), "r0_wrong_within_auc": wa(Y["y_r0"], Y["p_lp"], allix)}}
    print("pooled", res["pooled"])
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "m3_cpu.json"), "w"), indent=1)
    with open(os.path.join(out, "m3_cpu_predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)


if __name__ == "__main__":
    main()
