"""M4: routing baselines without self-correction, Pareto sweep, risk-coverage, Overall-Score sensitivity.

Everything is computed from the existing routing log (laptop_v1/routing/traces.jsonl): 50 tuning
problems (file order) and 150 evaluation problems, unchanged. Nothing is generated. See REVISION_PLAN.md (M4).

    .venv/bin/python scripts/revision/m4_routing.py --out results/revision/m4
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
from scfp_bench.answers import extract, is_correct  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

RUN = os.path.join(BENCH, "scfp_runs", "laptop_v1")
GRID = np.round(np.arange(0.10, 0.9001, 0.05), 2)
PAIRS = [(lo, hi) for lo in GRID for hi in GRID if lo < hi]
B, SEED = 1000, 0
FIXED_NEW = ["No correction (R0 only)", "Agreement tool", "Agreement tool + human"]


def agreement(r):
    """(answer source, program answered, agrees) for the agreement policies."""
    r0 = extract(r["R0"], "number").value
    ext = r["ext_answer"]
    if ext is None:
        return False, False
    return True, r0 is not None and is_correct(r0, ext, "number")


def paths(rows, hc, hl):
    """Per-problem cost (relative to mean intrinsic tokens of these rows), latency, correctness, per path."""
    t = lambda r, *k: sum(r["tok"][x] for x in k)
    l = lambda r, *k: sum(r["lat"][x] for x in k)
    unit = np.mean([t(r, "R0", "F", "Rf") for r in rows])
    ag = [agreement(r) for r in rows]
    esc = np.array([not (ran and agr) for ran, agr in ag])           # disagreement or no program answer
    C = {"int": np.array([t(r, "R0", "F", "Rf") for r in rows]) / unit,
         "ext": np.array([t(r, "R0", "F", "program") for r in rows]) / unit,
         "hum": np.array([t(r, "R0", "F") for r in rows]) / unit + hc,
         "r0": np.array([t(r, "R0") for r in rows]) / unit,
         "agr": np.array([t(r, "R0", "program") for r in rows]) / unit}
    C["agrh"] = C["agr"] + hc * esc
    L = {"int": np.array([l(r, "R0", "F", "Rf") for r in rows]),
         "ext": np.array([l(r, "R0", "F", "program", "exec") for r in rows]),
         "hum": np.array([l(r, "R0", "F") for r in rows]) + hl,
         "r0": np.array([l(r, "R0") for r in rows]),
         "agr": np.array([l(r, "R0", "program", "exec") for r in rows])}
    L["agrh"] = L["agr"] + hl * esc
    r0c = np.array([r["r0_correct"] for r in rows], float)
    extc = np.array([r["ext_correct"] for r in rows], float)
    ran = np.array([a[0] for a in ag])
    A = {"int": np.array([r["int_correct"] for r in rows], float), "ext": extc, "hum": np.ones(len(rows)),
         "r0": r0c, "agr": np.where(ran, extc, r0c), "agrh": np.where(esc, 1.0, r0c)}
    return C, L, A, unit, esc


def route(s, lo, hi):
    return np.array(["int" if v < lo else "ext" if v < hi else "hum" for v in s])


def pick(choice, D):
    return np.array([D[k][i] for i, k in enumerate(choice)])


def score(c, l, a, Cn, Ln, w):
    return 1 - (w[0] * c.mean() / Cn + w[1] * l.mean() / Ln + w[2] * (1 - a.mean()) ** 2)


def fit(rows, key, lat, hc, hl, w):
    C, L, A, _, _ = paths(rows, hc, hl)
    Cn, Ln = C["hum"].mean(), L["hum"].mean()
    s = np.array([r[key] for r in rows])
    best = None
    for lo, hi in PAIRS:
        ch = route(s, lo, hi)
        v = score(pick(ch, C), pick(ch, L) + lat, pick(ch, A), Cn, Ln, w)
        if best is None or v > best[0]:
            best = (v, lo, hi)
    return best[1], best[2]


def conf_scores(rows, tune):
    tn = np.sort([-r["critique_logprob"] for r in tune])
    return np.array([np.searchsorted(tn, -r["critique_logprob"], side="right") / len(tn) for r in rows])


def policies(tune, test, meta, hc, hl, w):
    """Every policy on the eval problems: name -> (cost, latency, correct) arrays, plus router thresholds."""
    for rows in (tune, test):
        cs = conf_scores(rows, tune)
        for r, v in zip(rows, cs):
            r["conf_score"] = float(v)
    th_dyn = fit(tune, "p_fail", meta["predictor_latency_s"], hc, hl, w)
    th_conf = fit(tune, "conf_score", meta["scorer_latency_s"], hc, hl, w)
    C, L, A, unit, esc = paths(test, hc, hl)
    out, share = {}, {}
    for name, k in (("Always Intrinsic", "int"), ("Always Extrinsic", "ext"), ("Always Human (oracle)", "hum"),
                    ("No correction (R0 only)", "r0"), ("Agreement tool", "agr"), ("Agreement tool + human", "agrh")):
        out[name] = (C[k], L[k], A[k])
        share[name] = {"hum": 1.0 if k == "hum" else float(esc.mean()) if k == "agrh" else 0.0,
                       "tool": float(k in ("ext", "agr", "agrh"))}
    share["Random (uniform, expected)"] = {"hum": 1 / 3, "tool": 1 / 3}
    out["Random (uniform, expected)"] = tuple((D["int"] + D["ext"] + D["hum"]) / 3 for D in (C, L, A))
    for name, key, lat, th in (("Confidence-based", "conf_score", meta["scorer_latency_s"], th_conf),
                               ("Dynamic (our predictor)", "p_fail", meta["predictor_latency_s"], th_dyn)):
        ch = route([r[key] for r in test], *th)
        out[name] = (pick(ch, C), pick(ch, L) + lat, pick(ch, A))
        share[name] = {"hum": float(np.mean(ch == "hum")), "tool": float(np.mean(ch == "ext"))}
    out["_share"] = share
    return out, {"Confidence-based": th_conf, "Dynamic (our predictor)": th_dyn}, C, L, A, unit, esc


def boot_ci(a, n=B, seed=SEED):
    rng = np.random.default_rng(seed)
    d = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def paired(a, b, n=B, seed=SEED):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        d.append(a[i].mean() - b[i].mean())
    n01, n10 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "a_only": n01, "b_only": n10, "mcnemar_p": float(p)}


def risk_coverage(score_, wrong):
    order = np.argsort(score_, kind="stable")
    w = wrong[order]
    k = np.arange(1, len(w) + 1)
    risk = np.cumsum(w) / k
    return k / len(w), risk, float(risk.mean())


def aurc_ci(score_, wrong, n=B, seed=SEED):
    rng = np.random.default_rng(seed)
    d, dr = [], []
    for _ in range(n):
        i = np.sort(rng.integers(0, len(wrong), len(wrong)))  # keep problem order for tie-breaking
        a = risk_coverage(score_[i], wrong[i])[2]
        d.append(a)
        dr.append(a - wrong[i].mean())
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(d), q(dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = read_jsonl(os.path.join(RUN, "routing", "traces.jsonl"))
    meta = json.load(open(os.path.join(RUN, "routing", "score_meta.json")))
    old = json.load(open(os.path.join(RUN, "results", "routing.json")))
    tune, test = rows[:old["n_tune"]], rows[old["n_tune"]:]
    HC, HL, W = 15.0, 45.0, (0.4, 0.2, 0.4)

    pol, th, C, L, A, unit, esc = policies(tune, test, meta, HC, HL, W)
    shares = pol.pop("_share")
    Cn, Ln = C["hum"].mean(), L["hum"].mean()

    # 1) reproduce the published strategies exactly before adding anything
    repro = {}
    for name, v in old["strategies"].items():
        c, l, acc = pol[name]
        mine = {"accuracy": acc.mean(), "relative_cost": c.mean(), "latency_s": l.mean(),
                "overall_score": score(c, l, acc, Cn, Ln, W)}
        repro[name] = all(abs(mine[k] - v[k]) < 1e-9 for k in mine)
    assert all(repro.values()), f"cannot reproduce the published routing table: {repro}"

    # 2) main table: every policy, with CIs
    main_rows = {}
    for name, (c, l, acc) in pol.items():
        rand = name.startswith("Random")
        main_rows[name] = {"accuracy": float(acc.mean()), "accuracy_ci95": None if rand else boot_ci(acc),
                           "relative_cost": float(c.mean()), "latency_s": float(l.mean()),
                           "overall_score": float(score(c, l, acc, Cn, Ln, W)),
                           "thresholds": [float(x) for x in th[name]] if name in th else None,
                           "share_human": shares[name]["hum"],
                           "paired_vs_r0_only": None if rand or name.startswith("No correction")
                           else paired(acc, pol["No correction (R0 only)"][2])}
    comps = {}
    for new in FIXED_NEW:
        for ref in ("Always Intrinsic", "Always Extrinsic", "Dynamic (our predictor)", "Confidence-based",
                    "No correction (R0 only)"):
            if new != ref:
                comps[f"{new} vs {ref}"] = paired(pol[new][2], pol[ref][2])
    agr = {"program_answered": int(sum(agreement(r)[0] for r in test)),
           "program_agrees_with_r0": int(sum(agreement(r)[1] for r in test)),
           "escalated_to_human": int(esc.sum()),
           "agreement_tool_answers_differ_from_extrinsic": int(np.sum(pol["Agreement tool"][2] != pol["Always Extrinsic"][2]))}

    # 3) Pareto sweep (descriptive, eval set)
    sweep = {}
    for name, key, lat in (("Dynamic (our predictor)", "p_fail", meta["predictor_latency_s"]),
                           ("Confidence-based", "conf_score", meta["scorer_latency_s"])):
        s = np.array([r[key] for r in test])
        pts = []
        for lo, hi in PAIRS:
            ch = route(s, lo, hi)
            pts.append({"tau_low": float(lo), "tau_high": float(hi), "relative_cost": float(pick(ch, C).mean()),
                        "latency_s": float((pick(ch, L) + lat).mean()), "accuracy": float(pick(ch, A).mean()),
                        "share_hum": float(np.mean(ch == "hum"))})
        sweep[name] = pts

    # 4) risk-coverage of the intrinsic answer
    wrong = 1 - A["int"]
    rc = {}
    for name, key in (("DeBERTa p_fail", "p_fail"), ("Critique confidence", "conf_score")):
        s = np.array([r[key] for r in test])
        cov, risk, aurc = risk_coverage(s, wrong)
        ci, dci = aurc_ci(s, wrong)
        rc[name] = {"aurc": aurc, "aurc_ci95": ci, "diff_vs_random": aurc - float(wrong.mean()),
                    "diff_vs_random_ci95": dci, "coverage": cov.tolist(), "risk": risk.tolist()}
    k = np.arange(1, len(wrong) + 1)
    n_wrong = int(wrong.sum())
    orisk = np.maximum(0, k - (len(wrong) - n_wrong)) / k
    rc["Random (expected)"] = {"aurc": float(wrong.mean()), "coverage": (k / len(k)).tolist(),
                               "risk": [float(wrong.mean())] * len(k)}
    rc["Oracle"] = {"aurc": float(orisk.mean()), "coverage": (k / len(k)).tolist(), "risk": orisk.tolist()}

    # 5) sensitivity of the Overall Score
    simplex = [(i / 10, j / 10, round(1 - i / 10 - j / 10, 1)) for i in range(11) for j in range(11 - i)]
    sens, wins = [], {n: 0 for n in pol}
    dyn_beats_fixed = 0
    for hc, hl, w in itertools.product((5.0, 15.0, 50.0), (15.0, 45.0, 120.0), simplex):
        P, T, C2, L2, _, _, _ = policies(tune, test, meta, hc, hl, w)
        P.pop("_share")
        cn, ln = C2["hum"].mean(), L2["hum"].mean()
        sc = {n: float(score(*v, cn, ln, w)) for n, v in P.items()}
        best = max(sc, key=sc.get)
        wins[best] += 1
        fixed_best = max(v for n, v in sc.items() if n not in ("Dynamic (our predictor)", "Confidence-based"))
        dyn_beats_fixed += sc["Dynamic (our predictor)"] > fixed_best
        sens.append({"human_cost_x": hc, "human_latency_s": hl, "w": w, "best": best, "scores": sc})
    n_set = len(sens)
    by_assump = {}
    for hc, hl in itertools.product((5.0, 15.0, 50.0), (15.0, 45.0, 120.0)):
        sub = [s for s in sens if s["human_cost_x"] == hc and s["human_latency_s"] == hl]
        by_assump[f"{hc:g}x/{hl:g}s"] = {n: sum(s["best"] == n for s in sub) for n in pol}

    # non-dominated policies on (cost, accuracy) at the tuned operating points; and which sweep points
    # (descriptive, eval set) are not dominated by any fixed or tuned policy
    named = {n: (v["relative_cost"], v["accuracy"]) for n, v in main_rows.items() if not n.startswith("Random")}
    dom = lambda c, a_, others: any(oc <= c and oa >= a_ and (oc < c or oa > a_) for oc, oa in others)
    frontier = sorted([n for n, (c, a_) in named.items() if not dom(c, a_, [v for m, v in named.items() if m != n])],
                      key=lambda n: named[n][0])
    sweep_nd = {n: sum(not dom(p["relative_cost"], p["accuracy"], named.values()) for p in pts)
                for n, pts in sweep.items()}
    nd_share = min(p["share_hum"] for pts in sweep.values() for p in pts
                   if not dom(p["relative_cost"], p["accuracy"], named.values()))

    res = {"n_tune": len(tune), "pareto_frontier_policies": frontier, "sweep_points_not_dominated": sweep_nd,
           "sweep_points_total": len(PAIRS), "sweep_not_dominated_min_share_human": float(nd_share), "n_eval": len(test), "cost_unit_tokens": float(unit),
           "assumptions": {"human_cost_x": HC, "human_latency_s": HL, "human_accuracy": 1.0, "weights": W,
                           "note": "human cost, latency and accuracy are assumptions, not measurements"},
           "reproduces_published_table": repro, "policies": main_rows, "paired": comps, "agreement": agr,
           "r0_accuracy_eval": float(A["r0"].mean()),
           "risk_coverage": rc,
           "sensitivity": {"n_settings": n_set, "wins": wins, "dynamic_beats_best_fixed": int(dyn_beats_fixed),
                           "wins_by_assumption": by_assump}}
    json.dump(res, open(os.path.join(a.out, "routing_m4.json"), "w"), indent=1)
    json.dump(sweep, open(os.path.join(a.out, "pareto_sweep.json"), "w"), indent=1)
    json.dump(sens, open(os.path.join(a.out, "sensitivity_settings.json"), "w"))
    plot(res, sweep, a.out)
    for n, v in main_rows.items():
        ci = v["accuracy_ci95"]
        print(f"{n:28s} acc={v['accuracy']:.3f} {ci and [round(x, 3) for x in ci]} cost={v['relative_cost']:.2f}x "
              f"lat={v['latency_s']:.1f}s score={v['overall_score']:.3f}")
    print("agreement:", agr)
    print("frontier:", frontier, "sweep points not dominated:", sweep_nd, "of", len(PAIRS))
    for k_, v in comps.items():
        print(f"  {k_}: diff {v['diff']:+.3f} CI {[round(x, 3) for x in v['ci95']]} McNemar p={v['mcnemar_p']:.3f}")
    for n, v in rc.items():
        print(f"  AURC {n}: {v['aurc']:.3f} {v.get('aurc_ci95')} diff-vs-random {v.get('diff_vs_random_ci95')}")
    print("sensitivity wins:", {k_: v for k_, v in wins.items() if v}, f"dynamic beats best fixed in {dyn_beats_fixed}/{n_set}")


def plot(res, sweep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, color in (("Dynamic (our predictor)", "#4C72B0"), ("Confidence-based", "#DD8452")):
        pts = sweep[name]
        x, y = np.array([p["relative_cost"] for p in pts]), np.array([p["accuracy"] for p in pts])
        ax.scatter(x, y, s=8, alpha=0.35, color=color, label=f"{name.split(' (')[0]} router, all $\\tau$ pairs")
        o = np.argsort(x)
        front, best = [], -1
        for i in o:
            if y[i] > best:
                front.append(i)
                best = y[i]
        ax.plot(x[front], y[front], color=color, lw=1)
    marks = {"Always Intrinsic": "s", "Always Extrinsic": "^", "No correction (R0 only)": "*",
             "Agreement tool": "D", "Agreement tool + human": "P", "Random (uniform, expected)": "x",
             "Confidence-based": "o", "Dynamic (our predictor)": "o", "Always Human (oracle)": "v"}
    for name, m in marks.items():
        v = res["policies"][name]
        face = "none" if name in ("Confidence-based", "Dynamic (our predictor)") else "k"
        ax.scatter(v["relative_cost"], v["accuracy"], marker=m, s=60 if m == "*" else 36, facecolors=face,
                   edgecolors="k", zorder=5, label=name + (" (tuned)" if face == "none" else ""))
    ax.set_xscale("log")
    ax.set_xlabel("Relative cost (tokens; human path assumed 15$\\times$), log scale")
    ax.set_ylabel("Accuracy on 150 evaluation problems")
    ax.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_pareto.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, st in (("DeBERTa p_fail", "-"), ("Critique confidence", "--"), ("Random (expected)", ":"),
                     ("Oracle", "-.")):
        v = res["risk_coverage"][name]
        ax.plot(v["coverage"], v["risk"], st, label=f"{name} (AURC {v['aurc']:.3f})")
    ax.set_xlabel("Coverage (share of problems kept on intrinsic correction)")
    ax.set_ylabel("Risk (intrinsic answer wrong)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_risk_coverage.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
