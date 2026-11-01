"""M8 (3): exclusion sensitivity. Traces without an explicit final answer are counted as failures instead of being
excluded. SCFP v1.0: TF-IDF baseline refit with the same code path. Natural set: text, signals without the verifier
and signals + verifier (LR), within-source AUC, with the excluded traces' teacher-forced features from job A.
See REVISION_PLAN.md (M8, and the implementation details for (3)).

    .venv/bin/python scripts/revision/m8_exclusion.py --gpua results/revision/gpuA --out results/revision/m8
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
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from m5_deconfound import within  # noqa: E402
from m7_stats import cluster_index, macro_f1, perm_test  # noqa: E402
from scfp_bench.experiments.data import partial_text  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import SC_FEATURES, generator_rows, oof  # noqa: E402
from scfp_bench.experiments.signals import VERIFIER, grouped_folds  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

CONFIGS = {"qwen2.5-0.5b": "signals_mac_big.yaml", "qwen2.5-1.5b": "signals_kaggle.yaml",
           "smollm2-1.7b": "signals_kaggle.yaml", "qwen2.5-3b": "signals_kaggle.yaml"}
B, SEED = 1000, 0


def cluster_draws(groups, rng, n=B):
    cl = defaultdict(list)
    for i, g in enumerate(groups):
        cl[g].append(i)
    cl = [np.array(v) for v in cl.values()]
    return [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(n)]


def tfidf(train, val, test):
    """The baselines.py TF-IDF procedure: C chosen on validation macro-F1, then refit on train."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    y = {k: np.array([r["label"] for r in v]) for k, v in (("train", train), ("val", val), ("test", test))}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([partial_text(r) for r in train])
    Xva, Xte = vec.transform([partial_text(r) for r in val]), vec.transform([partial_text(r) for r in test])
    best_c, best_f1 = None, -1
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(Xtr, y["train"])
        f1 = f1_score(y["val"], m.predict(Xva), average="macro")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    m = LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced").fit(Xtr, y["train"])
    return m.predict(Xte), m.predict_proba(Xte)[:, 1], best_c


def scfp(rng):
    from m5_deconfound import auc
    run = os.path.join(BENCH, "scfp_runs", "laptop_v1")
    S = {s: read_jsonl(os.path.join(run, "splits", f"{s}.jsonl")) for s in ("train", "val", "test")}
    split_of = {r["problem_id"]: s for s, v in S.items() for r in v}
    ex = read_jsonl(os.path.join(run, "excluded.jsonl"))
    E = defaultdict(list)
    for r in ex:
        E[split_of[r["problem_id"]]].append({**r, "label": 1})
    # the original model must be reproduced exactly from the same code path
    saved = {json.loads(l)["trace_id"]: json.loads(l) for l in open(os.path.join(run, "predictions", "tfidf_lr__binary.jsonl"))}
    yp0, p0, _ = tfidf(S["train"], S["val"], S["test"])
    assert all(abs(saved[r["trace_id"]]["p_fail"] - p) < 1e-9 for r, p in zip(S["test"], p0))
    A = {s: S[s] + E[s] for s in S}
    yp1, p1, c1 = tfidf(A["train"], A["val"], A["test"])
    out = {"n_excluded_by_split": {s: len(E[s]) for s in S}, "C_with_exclusions": c1}
    for name, rows, yp, p in (("original", S["test"], yp0, p0), ("with_exclusions", A["test"], yp1, p1),
                              ("with_exclusions_on_original_test", S["test"], yp1[:len(S["test"])], p1[:len(S["test"])])):
        y = np.array([r["label"] for r in rows])
        D = cluster_draws([r["problem_id"] for r in rows], rng)
        out[name] = {"n": len(rows), "failure_rate": float(y.mean()), "accuracy": float(np.mean(yp == y)),
                     "majority_accuracy": float(max(y.mean(), 1 - y.mean())), "macro_f1": macro_f1(y, yp),
                     "auc": float(auc(y, p)),
                     "accuracy_ci95": list(np.percentile([np.mean(yp[i] == y[i]) for i in D], [2.5, 97.5])),
                     "auc_ci95": list(np.percentile([auc(y[i], p[i]) for i in D], [2.5, 97.5]))}
        # problem-level paired permutation test against the training-majority prediction (as in M7)
        tr = S["train"] if name == "original" else A["train"]
        maj = int(np.mean([r["label"] for r in tr]) >= 0.5)
        cl = cluster_index([r["problem_id"] for r in rows])
        out[name]["train_majority_class"] = maj
        out[name]["p_vs_majority"] = perm_test((yp == y).astype(float), (np.full(len(y), maj) == y).astype(float), cl, rng)
    return out


def natural(gpua, rng):
    res = {}
    preds = defaultdict(dict)
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong":
            preds[(r["generator"], r["predictor"])][r["trace_id"]] = r
    os.chdir(BENCH)
    for g, cfg_name in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join("configs", cfg_name)))
        kept = generator_rows(cfg, "natural", g)
        feats = {json.loads(l)["trace_id"]: json.loads(l) for l in open(os.path.join(gpua, f"excluded_features_{g}.jsonl"))}
        exc = []
        for t in read_jsonl(os.path.join(cfg["run_dir"], "natural", "excluded.jsonl")):
            if t["source_model"] != g:
                continue
            f = {k: (np.nan if v is None else v) for k, v in feats[t["trace_id"]].items()}
            ver = {k: np.nan for k in VERIFIER}
            if t["source"] == "gsm8k" and not t["r0_parsed_from_final_line"]:
                ver["pot_agree_r0"] = 0.0
            exc.append({**f, **ver, **{k: np.nan for k in SC_FEATURES}, "trace_id": t["trace_id"], "problem_id": t["problem_id"],
                        "source": t["source"], "source_model": g, "label": 1,
                        "text": " [SEP] ".join([t["prompt"], t["initial_response"], t["critique"]])})
        assert len(exc) == len(feats), g
        allr = kept + exc
        y = np.array([r["label"] for r in allr])
        strata = np.array([r["source"] for r in allr])
        folds = grouped_folds(allr)
        Xv = scalar_matrix(allr)
        Xn = scalar_matrix([{k: v for k, v in r.items() if k not in VERIFIER} for r in allr])
        P = {"text_tfidf": oof("text", {"text": [r["text"] for r in allr], "scalar": None, "hidden": None}, y, folds),
             "signals_noverifier": oof("scalar", {"text": None, "hidden": None, "scalar": Xn}, y, folds),
             "signals": oof("scalar", {"text": None, "hidden": None, "scalar": Xv}, y, folds)}
        D = cluster_draws([r["problem_id"] for r in allr], rng)
        yk = np.array([r["label"] for r in kept])
        sk = np.array([r["source"] for r in kept])
        out = {"kept": len(kept), "excluded": len(exc), "failure_rate_excluding": float(yk.mean()), "failure_rate_with": float(y.mean())}
        for k, p in P.items():
            orig = np.array([preds[(g, k)][r["trace_id"]]["p"] for r in kept])
            out[k] = {"within_auc_with_exclusions": within(y, p, strata)[0],
                      "within_auc_with_exclusions_ci95": list(np.percentile([within(y[i], p[i], strata[i])[0] for i in D], [2.5, 97.5])),
                      "within_auc_original": within(yk, orig, sk)[0],
                      "within_auc_with_model_on_kept": within(yk, p[:len(kept)], sk)[0]}
        out["signals_minus_text_with_exclusions"] = out["signals"]["within_auc_with_exclusions"] - out["text_tfidf"]["within_auc_with_exclusions"]
        d = [within(y[i], P["signals"][i], strata[i])[0] - within(y[i], P["text_tfidf"][i], strata[i])[0] for i in D]
        out["signals_minus_text_with_exclusions_ci95"] = list(np.percentile(d, [2.5, 97.5]))
        res[g] = out
        print(g, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if not isinstance(v, dict)}),
              {k: (round(out[k]["within_auc_original"], 3), round(out[k]["within_auc_with_exclusions"], 3)) for k in P}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpua", default="results/revision/gpuA")
    ap.add_argument("--out", default="results/revision/m8")
    a = ap.parse_args()
    gpua, out = os.path.abspath(a.gpua), os.path.abspath(a.out)
    rng = np.random.default_rng(SEED)
    res = {"scfp": scfp(rng)}
    print("SCFP:", json.dumps({k: v for k, v in res["scfp"].items()}, default=str)[:900], flush=True)
    res["natural"] = natural(gpua, rng)
    json.dump(res, open(os.path.join(out, "exclusion_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
