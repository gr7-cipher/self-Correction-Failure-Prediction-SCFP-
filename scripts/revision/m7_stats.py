"""M7 (CPU part): problem-level cluster bootstrap intervals, problem-level paired permutation tests,
Holm-Bonferroni per table family, power analysis for the SCFP test split, and calibration.
See REVISION_PLAN.md (M7). Seeds are handled by m7_seeds.py.

    .venv/bin/python scripts/revision/m7_stats.py --out results/revision/m7
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
from m5_deconfound import auc, within  # noqa: E402

RUN = os.path.join(BENCH, "scfp_runs", "laptop_v1")
B, NPERM, SEED = 1000, 10000, 0
T7 = ["majority", "random", "length", "confidence", "tfidf_lr"]
T8 = ["deberta-scfp", "deberta-plain", "roberta-base", "bert-base"]
T9 = ["abl-no-segments", "abl-no-attention", "abl-inputs-P-R0", "abl-inputs-P-F", "abl-inputs-R0-F", "abl-deberta-xsmall"]
NAT_GENS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b", "qwen2.5-3b", "pooled"]
NAT_PRED = ["identity", "signals_noverifier", "signals", "signals_gbm", "concat_text_signals"]


def macro_f1(y, yp):
    f = []
    for c in (0, 1):
        tp = np.sum((yp == c) & (y == c))
        fp, fn = np.sum((yp == c) & (y != c)), np.sum((yp != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def cluster_index(groups):
    c = defaultdict(list)
    for i, g in enumerate(groups):
        c[g].append(i)
    return [np.array(v) for v in c.values()]


def cluster_ci(fn, cl, rng, n=B):
    vals = []
    for _ in range(n):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], np.array(vals)


def perm_test(ca, cb, cl, rng, n=NPERM):
    """Problem-level paired permutation test of the accuracy difference (sign flips of per-problem sums)."""
    d = np.array([np.sum(ca[i].astype(float) - cb[i]) for i in cl])
    obs = abs(d.sum())
    if obs == 0:
        return 1.0
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.sum(np.abs(flips @ d) >= obs - 1e-12) + 1) / (n + 1))


def holm(p):
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def ece(y, p, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    rel = []
    for k in range(bins):
        m = (p >= edges[k]) & ((p < edges[k + 1]) if k < bins - 1 else (p <= edges[k + 1]))
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
            rel.append({"bin": k, "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return float(e), rel


# ------------------------------------------------------------------ SCFP test split (Tables 7-10)
def scfp(rng):
    test = [json.loads(l) for l in open(os.path.join(RUN, "splits", "test.jsonl"))]
    pid = {r["trace_id"]: r["problem_id"] for r in test}
    order = sorted(pid)

    def load(name):
        rows = {r["trace_id"]: r for r in (json.loads(l) for l in open(os.path.join(RUN, "predictions", f"{name}__binary.jsonl")))}
        return (np.array([rows[t]["y_true"] for t in order]), np.array([rows[t]["y_pred"] for t in order]),
                np.array([rows[t].get("p_fail", np.nan) for t in order], float))
    cl = cluster_index([pid[t] for t in order])
    out, corr = {}, {}
    for name in T7 + T8 + T9:
        y, yp, p = load(name)
        corr[name] = (yp == y)
        has_p = not np.all(np.isnan(p)) and len(set(np.round(p, 12))) > 1
        acc_ci, _ = cluster_ci(lambda i: np.mean(yp[i] == y[i]), cl, rng)
        f1_ci, _ = cluster_ci(lambda i: macro_f1(y[i], yp[i]), cl, rng)
        auc_ci = cluster_ci(lambda i: auc(y[i], p[i]), cl, rng)[0] if has_p else None
        out[name] = {"accuracy": float(np.mean(yp == y)), "accuracy_ci95": acc_ci, "macro_f1": macro_f1(y, yp),
                     "macro_f1_ci95": f1_ci, "auc": float(auc(y, p)) if has_p else None, "auc_ci95": auc_ci}
    fams = {"t7_vs_tfidf": [(n, "tfidf_lr") for n in T7 if n != "tfidf_lr"],
            "t7_vs_majority": [(n, "majority") for n in T7 if n != "majority"],
            "t8_vs_tfidf": [(n, "tfidf_lr") for n in T8], "t8_vs_majority": [(n, "majority") for n in T8],
            "t9_vs_full": [(n, "deberta-scfp") for n in T9]}
    tests = {}
    for fam, pairs in fams.items():
        ps = [perm_test(corr[a].astype(float), corr[b].astype(float), cl, rng) for a, b in pairs]
        tests[fam] = [{"a": a, "b": b, "p": p, "p_holm": h} for (a, b), p, h in zip(pairs, ps, holm(ps))]
    # cross-model (Table 10): Qwen2.5 and SmolLM2 test subsets
    rows = {r["trace_id"]: r for r in (json.loads(l) for l in open(os.path.join(RUN, "predictions", "xmodel-qwen-to-all__binary.jsonl")))}
    xm = {}
    for sub, keep in (("qwen", lambda r: r["source_model"] != "smollm2-1.7b"), ("smollm2", lambda r: r["source_model"] == "smollm2-1.7b")):
        ts = [t for t in order if t in rows and keep(rows[t])]
        y = np.array([rows[t]["y_true"] for t in ts]); yp = np.array([rows[t]["y_pred"] for t in ts])
        c2 = cluster_index([pid[t] for t in ts])
        xm[sub] = {"n": len(ts), "n_problems": len(c2), "accuracy": float(np.mean(y == yp)),
                   "accuracy_ci95": cluster_ci(lambda i: np.mean(yp[i] == y[i]), c2, rng)[0]}
    return {"n": len(order), "n_problems": len(cl), "predictors": out, "tests": tests, "cross_model": xm}, corr, cl, load


# ------------------------------------------------------------------ power (SCFP test split)
def icc_oneway(x, cl):
    """ANOVA estimator of the intraclass correlation of x within clusters (unequal sizes)."""
    k = len(cl); n = sum(len(c) for c in cl); grand = np.mean(x)
    msb = sum(len(c) * (x[c].mean() - grand) ** 2 for c in cl) / (k - 1)
    msw = sum(((x[c] - x[c].mean()) ** 2).sum() for c in cl) / (n - k)
    n0 = (n - sum(len(c) ** 2 for c in cl) / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)) if (msb + (n0 - 1) * msw) > 0 else 0.0


def hanley_se(A, n1, n0):
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0))


def power(load, cl, rng):
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    y, yd, pd = load("deberta-scfp")
    _, yt, pt = load("tfidf_lr")
    n = len(y); mbar = n / len(cl)
    d = (yd == y).astype(float) - (yt == y).astype(float)
    icc_d = max(0.0, icc_oneway(d, cl))
    icc_c = max(0.0, icc_oneway((yd == y).astype(float), cl))
    deff_d, deff_c = 1 + (mbar - 1) * icc_d, 1 + (mbar - 1) * icc_c
    psi = float(np.mean(d != 0))
    mde_acc = float((za + zb) * np.sqrt(psi * deff_d / n))
    # AUC vs 0.5 (one model) and paired AUC difference, Hanley-McNeil with the design effect
    n1, n0 = int(y.sum()), int(n - y.sum())
    A = 0.5
    for _ in range(100):
        A = 0.5 + (za + zb) * hanley_se(A, n1, n0) * np.sqrt(deff_c)
    pos, neg = y == 1, y == 0
    r = float(np.mean([stats.pearsonr(pd[pos], pt[pos])[0], stats.spearmanr(pd[neg], pt[neg])[0]]))
    se0 = hanley_se(0.6, n1, n0)
    mde_auc_diff = float((za + zb) * np.sqrt(2 * se0 ** 2 * (1 - r) * deff_c))
    # empirical check: cluster-bootstrap SE of the observed differences
    _, accd = cluster_ci(lambda i: np.mean(d[i]), cl, rng)
    _, aucd = cluster_ci(lambda i: auc(y[i], pd[i]) - auc(y[i], pt[i]), cl, rng)
    # simulation: set the true paired accuracy difference (model A minus TF-IDF) to delta by flipping model A's
    # answers (wrong->right to raise it, right->wrong to lower it), resample problems, and test with a
    # cluster-robust z on per-problem summed differences
    sim = {}
    ct = (yt == y).astype(float)
    for delta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        hits = 0
        for _ in range(500):
            ca = (yd == y).astype(float)
            need = int(round((delta - (ca.mean() - ct.mean())) * n))
            pool = np.where(ca == (0.0 if need > 0 else 1.0))[0]
            ca[rng.choice(pool, min(len(pool), abs(need)), replace=False)] = 1.0 if need > 0 else 0.0
            idx_cl = [cl[j] for j in rng.integers(0, len(cl), len(cl))]
            s_ = np.array([np.sum(ca[c] - ct[c]) for c in idx_cl])
            z = s_.sum() / np.sqrt(np.sum((s_ - s_.mean()) ** 2) * len(s_) / (len(s_) - 1)) if np.sum(s_ ** 2) > 0 else 0
            hits += abs(z) > za
        sim[f"{delta:.2f}"] = hits / 500
    return {"n": n, "n_problems": len(cl), "mean_cluster_size": mbar, "icc_correctness": icc_c,
            "icc_paired_difference": icc_d, "deff_correctness": deff_c, "deff_difference": deff_d,
            "discordance": psi, "mde_accuracy_paired": mde_acc, "mde_auc_vs_half": float(A - 0.5),
            "auc_score_correlation": r, "mde_auc_paired": mde_auc_diff,
            "bootstrap_se_accuracy_diff": float(accd.std()), "bootstrap_mde_accuracy": float((za + zb) * accd.std()),
            "bootstrap_se_auc_diff": float(aucd.std()), "bootstrap_mde_auc": float((za + zb) * aucd.std()),
            "simulated_power_accuracy": sim}


# ------------------------------------------------------------------ natural set (Table 14 family)
def natural(rng, res_dir):
    base = defaultdict(dict)
    for l in open(os.path.join(res_dir, "m5", "identity_predictions.jsonl")):
        r = json.loads(l)
        base[r["row"]][(r["generator"], r["trace_id"])] = {**r, "identity": r["p"]}
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong":
            for row in (r["generator"], "pooled"):
                base[row][(r["generator"], r["trace_id"])][r["predictor"]] = r["p"]
    out, auc_p, acc_p, keys = {}, [], [], []
    for g in NAT_GENS:
        rows = sorted(base[g].values(), key=lambda r: (r["generator"], r["trace_id"]))
        y = np.array([r["y"] for r in rows])
        st = np.array([r["source"] + ("|" + r["generator"] if g == "pooled" else "") for r in rows])
        cl = cluster_index([r["problem_id"] for r in rows])
        P = {k: np.array([r[k] for r in rows]) for k in NAT_PRED + ["text_tfidf"]}
        out[g] = {}
        for k in NAT_PRED:
            # 10,000 resamples: with 1,000 the smallest attainable p (2/1001) times 25 comparisons is 0.05,
            # so Holm could never reject (resolution, not a result)
            _, dd = cluster_ci(lambda i: within(y[i], P[k][i], st[i])[0] - within(y[i], P["text_tfidf"][i], st[i])[0], cl,
                               np.random.default_rng(SEED), n=NPERM)
            pa = float(min(1.0, 2 * (min(np.sum(dd <= 0), np.sum(dd >= 0)) + 1) / (len(dd) + 1)))
            pacc = perm_test(((P[k] >= .5) == y).astype(float), ((P["text_tfidf"] >= .5) == y).astype(float), cl, rng)
            out[g][k] = {"within_auc_diff_p": pa, "accuracy_perm_p": pacc,
                         "accuracy": float(np.mean((P[k] >= .5) == y)),
                         "accuracy_text": float(np.mean((P["text_tfidf"] >= .5) == y))}
            auc_p.append(pa); acc_p.append(pacc); keys.append((g, k))
    for (g, k), h1, h2 in zip(keys, holm(auc_p), holm(acc_p)):
        out[g][k]["within_auc_diff_p_holm"], out[g][k]["accuracy_perm_p_holm"] = h1, h2
    return out


# ------------------------------------------------------------------ routing (Table 11 family)
def routing(rng, res_dir):
    sys.path.insert(0, BENCH)
    import m4_routing as m4
    from scfp_bench.stages import read_jsonl
    rows = read_jsonl(os.path.join(RUN, "routing", "traces.jsonl"))
    meta = json.load(open(os.path.join(RUN, "routing", "score_meta.json")))
    pol, *_ = m4.policies(rows[:50], rows[50:], meta, 15.0, 45.0, (0.4, 0.2, 0.4))
    pol.pop("_share")
    cl = [np.array([i]) for i in range(150)]
    names = [n for n in pol if not n.startswith("Random") and n != "No correction (R0 only)"]
    ps = [perm_test(pol[n][2], pol["No correction (R0 only)"][2], cl, rng) for n in names]
    return [{"policy": n, "p": p, "p_holm": h} for n, p, h in zip(names, ps, holm(ps))]


# ------------------------------------------------------------------ calibration
def calibration(res_dir, load):
    out, curves = {}, {}
    # natural set: the best within-source predictor per generator (from M5) and pooled
    D = json.load(open(os.path.join(res_dir, "m5", "deconfounded.json")))
    preds = defaultdict(list)
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong":
            preds[(r["generator"], r["predictor"])].append(r)
    for g in NAT_GENS:
        best = max((k for k in D[g] if not k.startswith("_") and k != "identity"), key=lambda k: D[g][k]["within_auc"])
        rr = [r for (gg, k), v in preds.items() if k == best and (g == "pooled" or gg == g) for r in v]
        y, p = np.array([r["y"] for r in rr]), np.array([r["p"] for r in rr])
        e, rel = ece(y, p)
        out[f"natural|{g}|{best}"] = {"ece": e, "brier": float(np.mean((p - y) ** 2)), "mean_p": float(p.mean()),
                                      "failure_rate": float(y.mean()), "n": len(y)}
        curves[f"natural|{g}|{best}"] = rel
    y, _, p = load("deberta-scfp")
    e, rel = ece(y, p)
    out["scfp_test|deberta-scfp"] = {"ece": e, "brier": float(np.mean((p - y) ** 2)), "mean_p": float(p.mean()),
                                     "failure_rate": float(y.mean()), "n": len(y)}
    curves["scfp_test|deberta-scfp"] = rel
    rows = [json.loads(l) for l in open(os.path.join(RUN, "routing", "traces.jsonl"))][50:]
    y = np.array([1 - r["int_correct"] for r in rows], float); p = np.array([r["p_fail"] for r in rows])
    e, rel = ece(y, p)
    out["routing_eval|deberta-scfp"] = {"ece": e, "brier": float(np.mean((p - y) ** 2)), "mean_p": float(p.mean()),
                                        "failure_rate": float(y.mean()), "n": len(y)}
    curves["routing_eval|deberta-scfp"] = rel
    out["routing_eval|confidence"] = {"note": "rank score, not a probability: no ECE"}
    return out, curves


def plot_reliability(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(curves)
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.6), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, keys):
        rel = curves[k]
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        ax.scatter([r["mean_p"] for r in rel], [r["frac_pos"] for r in rel], s=[8 + r["n"] / 15 for r in rel])
        ax.plot([r["mean_p"] for r in rel], [r["frac_pos"] for r in rel], lw=0.8)
        t = k.split("|")
        ax.set_title((t[1] if t[0] == "natural" else t[0].replace("_", " ")) + "\n" + t[-1], fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.supxlabel("Mean predicted failure probability (15 bins)", fontsize=8)
    fig.supylabel("Observed failure rate", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--out", default="results/revision/m7")
    a = a_.parse_args()
    res_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    S, corr, cl, load = scfp(rng)
    P = power(load, cl, np.random.default_rng(SEED))
    N = natural(np.random.default_rng(SEED), res_dir)
    R = routing(np.random.default_rng(SEED), res_dir)
    C, curves = calibration(res_dir, load)
    json.dump({"scfp": S, "power": P, "natural": N, "routing": R, "calibration": C},
              open(os.path.join(a.out, "stats_m7.json"), "w"), indent=1)
    json.dump(curves, open(os.path.join(a.out, "reliability_bins.json"), "w"), indent=1)
    plot_reliability(curves, os.path.join(a.out, "fig_reliability.pdf"))
    for n, v in S["predictors"].items():
        print(f"{n:20s} acc {v['accuracy']:.3f} {[round(x, 3) for x in v['accuracy_ci95']]} auc {v['auc'] and round(v['auc'], 3)} {v['auc_ci95'] and [round(x, 3) for x in v['auc_ci95']]}")
    for fam, t in S["tests"].items():
        print(fam, [(x["a"], round(x["p"], 3), round(x["p_holm"], 3)) for x in t])
    print("power:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items()})
    for g, v in N.items():
        print(g, {k: (round(x["within_auc_diff_p_holm"], 4), round(x["accuracy_perm_p_holm"], 4)) for k, x in v.items()})
    print("routing:", [(x["policy"], round(x["p"], 4), round(x["p_holm"], 4)) for x in R])
    print("calibration:", {k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in C.items()})


if __name__ == "__main__":
    main()
