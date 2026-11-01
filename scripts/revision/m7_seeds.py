"""M7 seeds: test accuracy, macro-F1 and AUC of every Table 8 encoder and Table 9 ablation over 5 seeds
(seed 42 = the original M1 run; seeds 1-4 retrained on a Kaggle T4 with m7_train_seeds.sh). See REVISION_PLAN.md (M7).

    .venv/bin/python scripts/revision/m7_seeds.py --seeds results/revision/m7/seeds --out results/revision/m7
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m5_deconfound import auc  # noqa: E402
from m7_stats import RUN, T8, T9, macro_f1  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]


def metrics(path, order):
    rows = {r["trace_id"]: r for r in (json.loads(l) for l in open(path)) if r["split"] == "test"}
    assert set(rows) == set(order), path
    y = np.array([rows[t]["y_true"] for t in order])
    yp = np.array([rows[t]["y_pred"] for t in order])
    p = np.array([rows[t]["p_fail"] for t in order], float)
    return {"accuracy": float(np.mean(yp == y)), "macro_f1": macro_f1(y, yp), "auc": float(auc(y, p)),
            "positive_rate": float(np.mean(yp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="results/revision/m7/seeds")
    ap.add_argument("--out", default="results/revision/m7")
    a = ap.parse_args()
    order = sorted(json.loads(l)["trace_id"] for l in open(os.path.join(RUN, "splits", "test.jsonl")))
    ref = {n: metrics(os.path.join(RUN, "predictions", f"{n}__binary.jsonl"), order) for n in ("tfidf_lr", "majority")}
    res = {"reference": ref, "models": {}}
    for name in T8 + T9:
        per = {}
        for s in SEEDS:
            if s == 42:
                p = os.path.join(RUN, "predictions", f"{name}__binary.jsonl")
                meta = json.load(open(os.path.join(RUN, "results", f"{name}.json")))
            else:
                p = os.path.join(a.seeds, "predictions", f"{name}__seed{s}__binary.jsonl")
                meta = json.load(open(os.path.join(a.seeds, "results", f"{name}__seed{s}.json")))
                assert meta["args"]["seed"] == s and meta["args"]["no_amp"], (name, s)
            per[s] = {**metrics(p, order), "device": meta.get("gpu") or meta.get("device"),
                      "best_val_macro_f1": meta.get("best_val_macro_f1")}
        agg = {}
        for k in ("accuracy", "macro_f1", "auc"):
            v = np.array([per[s][k] for s in SEEDS])
            agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                      "seed42": float(per[42][k]), "seeds_above_tfidf": int(np.sum(v > ref["tfidf_lr"][k]))}
        res["models"][name] = {"per_seed": {str(s): per[s] for s in SEEDS}, "summary": agg}
        print(f"{name:20s} acc {agg['accuracy']['mean']:.3f}±{agg['accuracy']['sd']:.3f} (42: {agg['accuracy']['seed42']:.3f}) "
              f"F1 {agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['sd']:.3f} AUC {agg['auc']['mean']:.3f}±{agg['auc']['sd']:.3f} "
              f"| >TF-IDF acc {agg['accuracy']['seeds_above_tfidf']}/5 auc {agg['auc']['seeds_above_tfidf']}/5")
    full = res["models"]["deberta-scfp"]["per_seed"]
    res["ablation_minus_full"] = {}
    for name in T9:
        d = {k: [res["models"][name]["per_seed"][str(s)][k] - full[str(s)][k] for s in SEEDS] for k in ("accuracy", "auc")}
        res["ablation_minus_full"][name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                                "seeds_above_full": int(np.sum(np.array(v) > 0))} for k, v in d.items()}
    print("reference:", {n: {k: round(v, 3) for k, v in m.items()} for n, m in ref.items()})
    print("ablation - full (AUC mean, seeds above):",
          {n: (round(v["auc"]["mean"], 3), v["auc"]["seeds_above_full"]) for n, v in res["ablation_minus_full"].items()})
    json.dump(res, open(os.path.join(a.out, "seeds_m7.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
