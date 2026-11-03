# By-source AUC/accuracy of out-of-fold scalar predictions, with and without the verifier.
# Usage: PYTHONPATH=. python -m scfp_bench.experiments.by_source configs/<run>.yaml
import sys, os, yaml, numpy as np
from sklearn.metrics import roc_auc_score
from scfp_bench.stages import read_jsonl
from scfp_bench.experiments.improve import Base, scalar_matrix
from scfp_bench.experiments.signals import D, VERIFIER, grouped_folds
cfg = yaml.safe_load(open(sys.argv[1]))
feats = read_jsonl(D(cfg, "features", "natural.jsonl"))
ver = {v["trace_id"]: v for v in read_jsonl(D(cfg, "features", "natural_verifier.jsonl"))}
y = np.array([r["label"] for r in feats]); folds = grouped_folds(feats)
src = np.array([r["source"] for r in feats])
for tag, rows in [("no verifier", feats),
                  ("verifier", [{**r, **{k: ver.get(r["trace_id"], {}).get(k, np.nan) for k in VERIFIER}} for r in feats])]:
    data = {"scalar": scalar_matrix(rows)}; p = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        p[te] = Base("scalar", data).fit(tr, y).proba(te)
    for s in ["gsm8k", "strategyqa", "logiqa", "all"]:
        m = np.ones(len(y), bool) if s == "all" else src == s
        print(f"{tag:12s} {s:10s} n={m.sum():4d} AUC {roc_auc_score(y[m], p[m]):.3f} acc {((p[m]>.5)==y[m]).mean():.1%} maj {max(y[m].mean(),1-y[m].mean()):.1%}")
g = [(ver[r["trace_id"]], r) for r in feats if r["trace_id"] in ver and ver[r["trace_id"]].get("pot_ran")]
for a in (1, 0):
    sub = [r for v, r in g if v["pot_agree_r0"] == a]
    print(f"program {'agrees' if a else 'disagrees'} with R0: n={len(sub)} R0 correct {np.mean([r['r0_correct'] for r in sub]):.1%} final wrong {np.mean([r['label'] for r in sub]):.1%}")
