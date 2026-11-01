"""Results for the natural-distribution section of the paper (v7), from released features.

For every generator (Qwen2.5-0.5B on the Mac run, Qwen2.5-1.5B / SmolLM2-1.7B / Qwen2.5-3B on the
Kaggle run) this recomputes, with 5-fold cross-validation grouped by problem:

  * failure prediction (final answer wrong) with text, generator signals with and without the
    program-of-thought verifier, and gradient boosting; pooled = the per-generator out-of-fold
    predictions concatenated;
  * the same per source, to show where the verifier helps;
  * sub-tasks: initial answer wrong, answer changed by the revision, repair (initial wrong ->
    final right) and harm (initial right -> final wrong);
  * outcome statistics of the natural set and of SCFP v1.

Hidden-state probes are not recomputed here (their arrays are not kept with the results); the
paper takes them from each run's signals_summary.json.

    PYTHONPATH=. python -m scfp_bench.experiments.paper_v7 \
        --configs configs/signals_mac_big.yaml configs/signals_kaggle.yaml --out scfp_runs/paper_v7

Writes results.json and predictions.jsonl (one row per trace, predictor and target).
"""

import argparse
import json
import os

import numpy as np
import yaml

from ..stages import read_jsonl
from .improve import Base, scalar_matrix
from .metrics import bootstrap, mcnemar, summarize
from .signals import D, VERIFIER, grouped_folds, load_set

PREDICTORS = {  # name -> (Base kind, use verifier features)
    "text_tfidf": ("text", False),
    "signals_noverifier": ("scalar", False),
    "signals": ("scalar", True),
    "signals_gbm": ("gbm", True),
    "concat_text_signals": ("concat", True),
}
SOURCES = ["gsm8k", "strategyqa", "logiqa"]
SC_FEATURES = ["sc_agree", "sc_distinct", "sc_entropy", "sc_majority_is_r0"]


def oof(kind, data, y, folds):
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        p[te] = Base(kind, data).fit(tr, y).proba(te)
    return p


def score(y, p, base=None):
    yp = (p >= .5).astype(int)
    s, ci = summarize(y, yp, p), bootstrap(y, yp, p)
    maj = np.full(len(y), int(y.mean() >= .5))
    out = {**s, "ci": ci, "majority": float(max(y.mean(), 1 - y.mean())), "failure_rate": float(y.mean()),
           "p_vs_majority": mcnemar(y, yp, maj)["p_value"]}
    if base is not None:
        out["p_vs_tfidf"] = mcnemar(y, yp, (base >= .5).astype(int))["p_value"]
    return out


def generator_rows(cfg, set_name, model):
    """Feature rows of one generator, with verifier features and trace fields merged in."""
    feats = [r for r in read_jsonl(D(cfg, "features", f"{set_name}.jsonl")) if r["source_model"] == model]
    vp = D(cfg, "features", f"{set_name}_verifier.jsonl")
    ver = {v["trace_id"]: v for v in read_jsonl(vp)} if os.path.exists(vp) else {}
    tr = {r["trace_id"]: r for r in load_set(cfg, set_name)}
    if not cfg["signals"].get("samples"):
        # revision: a run configured without self-consistency must not use the sc_* features that reused calls
        # from an earlier run left on part of its traces (Qwen2.5-0.5B; REVISION_PLAN.md, post-hoc finding)
        feats = [{**r, **{k: np.nan for k in SC_FEATURES}} for r in feats]
    rows = []
    for r in feats:
        t = tr[r["trace_id"]]
        rows.append({**r, **{k: ver.get(r["trace_id"], {}).get(k, np.nan) for k in VERIFIER},
                     "text": " [SEP] ".join([t["prompt"], t["initial_response"], t["critique"]]),
                     "changed": int(str(t["r0_answer"]) != str(t["rf_answer"])), "rf_correct": t["rf_correct"]})
    return rows


def predict_all(rows, y, folds):
    data_v = {"text": [r["text"] for r in rows], "scalar": scalar_matrix(rows), "hidden": None}
    data_n = {**data_v, "scalar": scalar_matrix([{k: v for k, v in r.items() if k not in VERIFIER} for r in rows])}
    return {name: oof(kind, data_v if ver else data_n, y, folds) for name, (kind, ver) in PREDICTORS.items()}


def outcome_stats(rows):
    r0 = np.array([r["r0_correct"] for r in rows], bool)
    rf = np.array([r["rf_correct"] for r in rows], bool)
    return {"n": len(rows), "r0_acc": float(r0.mean()), "rf_acc": float(rf.mean()),
            "repair": float(rf[~r0].mean()) if (~r0).any() else None,
            "harm": float((~rf[r0]).mean()) if r0.any() else None,
            "changed": float(np.mean([r["changed"] for r in rows])),
            "by_source": {s: int(sum(r["source"] == s for r in rows)) for s in SOURCES}}


def run(cfg, set_name, model, res, pred_out):
    rows = generator_rows(cfg, set_name, model)
    if len(rows) < 30:
        return
    folds = grouped_folds(rows)
    key = f"{set_name}|{model}"
    res["outcomes"][key] = outcome_stats(rows)
    if set_name == "natural":
        ex = [r for r in read_jsonl(D(cfg, "natural", "excluded.jsonl")) if r["source_model"] == model]
        res["outcomes"][key]["excluded"] = len(ex)
        res["outcomes"][key]["excluded_by_source"] = {s: sum(r["source"] == s for r in ex) for s in SOURCES}
    y = np.array([r["label"] for r in rows])
    preds = predict_all(rows, y, folds)
    for name, p in preds.items():
        res["main"][f"{key}|{name}"] = score(y, p, None if name == "text_tfidf" else preds["text_tfidf"])
        for r, v in zip(rows, p):
            pred_out.append({"set": set_name, "generator": model, "trace_id": r["trace_id"], "source": r["source"],
                             "target": "rf_wrong", "predictor": name, "y": int(r["label"]), "p": float(v)})
    for s in SOURCES:
        m = np.array([r["source"] == s for r in rows])
        if m.sum() >= 30 and 0 < y[m].mean() < 1:
            for name in ("text_tfidf", "signals_noverifier", "signals"):
                res["by_source"][f"{key}|{s}|{name}"] = score(y[m], preds[name][m])
    if set_name != "natural":
        return
    gbm = {"text": None, "hidden": None}
    tasks = {"r0_wrong": (lambda r: True, lambda r: int(not r["r0_correct"])),
             "changed": (lambda r: True, lambda r: r["changed"]),
             "repair": (lambda r: not r["r0_correct"], lambda r: int(r["rf_correct"])),
             "harm": (lambda r: r["r0_correct"], lambda r: int(not r["rf_correct"]))}
    for t, (keep, lab) in tasks.items():
        sub = [r for r in rows if keep(r)]
        yt = np.array([lab(r) for r in sub])
        if len(sub) < 30 or not 0 < yt.mean() < 1:
            continue
        p = oof("gbm", {**gbm, "scalar": scalar_matrix(sub)}, yt, grouped_folds(sub))
        res["subtasks"][f"{key}|{t}"] = score(yt, p)


def pooled(pred_out, res):
    """Concatenated per-generator out-of-fold predictions on the natural set."""
    for name in PREDICTORS:
        r = [x for x in pred_out if x["set"] == "natural" and x["predictor"] == name]
        b = [x for x in pred_out if x["set"] == "natural" and x["predictor"] == "text_tfidf"]
        y, p = np.array([x["y"] for x in r]), np.array([x["p"] for x in r])
        res["main"][f"natural|pooled|{name}"] = score(y, p, None if name == "text_tfidf" else np.array([x["p"] for x in b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default="scfp_runs/paper_v7")
    a = ap.parse_args()
    res, pred_out = {"main": {}, "by_source": {}, "subtasks": {}, "outcomes": {}, "configs": a.configs}, []
    for c in a.configs:
        cfg = yaml.safe_load(open(c))
        for model in cfg["signals"]["models"]:
            for set_name in ("natural", "v1"):
                print(f"  {c}: {set_name} {model}", flush=True)
                run(cfg, set_name, model, res, pred_out)
    pooled(pred_out, res)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in pred_out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
