"""A1 (round 2): Delta-label leakage check. Most traces are 'neutral', which nearly means 'the answer did not change',
and one critique cue (f_proposes_other_answer) says whether F states an answer different from R0. Same rows, folds
and LR model as M3 (m3_cpu.py). See REVISION_PLAN.md (Round 2, A1).
  (i)  Delta-label macro within-source AUC with that cue removed from the features
  (ii) helped vs harmed only, on the traces whose final answer differs from R0's (within-source AUC of P(harmed))

    .venv/bin/python scripts/revision/a1_delta.py --out results/revision/a1
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
from m3_cpu import CONFIGS, delta_label, lr_pipe  # noqa: E402
from m5_deconfound import within  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows  # noqa: E402
from scfp_bench.experiments.signals import SCALARS, grouped_folds  # noqa: E402

B, SEED = 10000, 0
CUE = "f_proposes_other_answer"


def boot(fn, n, rng):
    """Bootstrap over traces; within one generator each problem has one trace, so this is the problem-level bootstrap."""
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    v = [x for x in v if not np.isnan(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def multinomial_oof(X, y, folds):
    P = np.zeros((len(y), 3))
    for k in range(5):
        tr, te = folds != k, folds == k
        m = lr_pipe().fit(X[tr], y[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/a1")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.chdir(BENCH)
    rng = np.random.default_rng(SEED)
    res = {}
    cue_col = SCALARS.index(CUE)
    for g, cname in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join("configs", cname)))
        rows = generator_rows(cfg, "natural", g)
        X = scalar_matrix(rows)
        Xnc = np.delete(X, cue_col, axis=1)
        folds = grouped_folds(rows)
        y_d = np.array([delta_label(r) for r in rows])
        src = np.array([r["source"] for r in rows])
        changed = np.array([r["changed"] for r in rows], bool)
        n = len(rows)
        macro = lambda P, i: float(np.mean([within((y_d[i] == c).astype(int), P[i, c], src[i])[0] for c in range(3)]))
        P_full, P_nc = multinomial_oof(X, y_d, folds), multinomial_oof(Xnc, y_d, folds)
        allix = np.arange(n)
        # (ii) helped (0) vs harmed (2) among changed answers; target = harmed
        sel = np.where(changed & (y_d != 1))[0]
        yh = (y_d[sel] == 2).astype(int)
        ph = np.zeros(len(sel))
        fs = folds[sel]
        for k in range(5):
            tr, te = fs != k, fs == k
            ph[te] = lr_pipe().fit(X[sel][tr], yh[tr]).predict_proba(X[sel][te])[:, 1]
        ph_nc = np.zeros(len(sel))
        for k in range(5):
            tr, te = fs != k, fs == k
            ph_nc[te] = lr_pipe().fit(Xnc[sel][tr], yh[tr]).predict_proba(Xnc[sel][te])[:, 1]
        ssel = src[sel]
        hh = lambda p, i: within(yh[i], p[i], ssel[i])[0]
        by_src = {}
        for s in np.unique(ssel):
            m = ssel == s
            by_src[s] = {"n": int(m.sum()), "harmed": int(yh[m].sum()), "helped": int((1 - yh[m]).sum())}
        r = {"n": n, "share_neutral": float(np.mean(y_d == 1)),
             "share_neutral_unchanged": float(np.mean((y_d == 1) & ~changed)),
             "share_unchanged_among_neutral": float(np.mean(~changed[y_d == 1])),
             "delta_macro": {"with_cue": macro(P_full, allix), "with_cue_ci95": boot(lambda i: macro(P_full, i), n, rng),
                             "without_cue": macro(P_nc, allix), "without_cue_ci95": boot(lambda i: macro(P_nc, i), n, rng),
                             "diff_ci95": boot(lambda i: macro(P_full, i) - macro(P_nc, i), n, rng)},
             "helped_vs_harmed": {"n": int(len(sel)), "n_harmed": int(yh.sum()), "n_helped": int(len(yh) - yh.sum()),
                                  "by_source": by_src,
                                  "within_auc": hh(ph, np.arange(len(sel))),
                                  "within_auc_ci95": boot(lambda i: hh(ph, i), len(sel), rng),
                                  "within_auc_without_cue": hh(ph_nc, np.arange(len(sel))),
                                  "within_auc_without_cue_ci95": boot(lambda i: hh(ph_nc, i), len(sel), rng)}}
        res[g] = r
        print(g, f"neutral {r['share_neutral']:.3f} (unchanged among neutral {r['share_unchanged_among_neutral']:.3f})",
              f"macro {r['delta_macro']['with_cue']:.3f} -> no cue {r['delta_macro']['without_cue']:.3f}",
              f"| helped/harmed n={len(sel)} ({int(yh.sum())} harmed) AUC {r['helped_vs_harmed']['within_auc']:.3f}",
              [round(x, 3) for x in r['helped_vs_harmed']['within_auc_ci95']], flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "a1_delta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
