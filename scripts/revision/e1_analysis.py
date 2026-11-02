"""E1 analysis (round 2): routing on the 1,000-problem hold-out with the natural-set predictors as routers.

Reads the GPU job's traces (results/revision/e1/gpu/<generator>/), fits the paper's LR "scalar" models on the
natural-set GSM8K traces of the same generator (signals + PoT verifier; signals + SC3 from the M6 samples), recalibrates
them with isotonic regression on the tuning split, chooses thresholds on the tuning split, and evaluates every policy on
the evaluation split. See REVISION_PLAN.md (Round 2, E1, and the E1 addition).

    .venv/bin/python scripts/revision/e1_analysis.py --gpu results/revision/e1/gpu --out results/revision/e1
"""

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
import m4_routing as m4  # noqa: E402
from m6_analysis import CONFIGS, samples_by_prompt  # noqa: E402
from m7_stats import holm, perm_test  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.answers import extract, is_correct, normalize_number  # noqa: E402
from scfp_bench.experiments.improve import Base, scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows  # noqa: E402
from scfp_bench.experiments.signals import VERIFIER, verdict_features  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

GENS = ["qwen2.5-0.5b", "smollm2-1.7b", "qwen2.5-1.5b"]
B, SEED = 10000, 0
HC, HL, W = 15.0, 45.0, (0.4, 0.2, 0.4)
SC3 = ["sc3_agree_frac", "sc3_majority_is_r0"]
ROUTERS = {"Router: signals + PoT": "p_pot", "Router: signals + SC3": "p_sc3",
           "Router: SCFP v1.0 DeBERTa": "p_deb", "Confidence-based": "conf_score"}
FIXED = {"Always Intrinsic": "int", "Always Extrinsic": "ext", "Always Human (oracle)": "hum",
         "No correction (R0 only)": "r0", "Agreement tool": "agr", "Agreement tool + human": "agrh"}
NATURAL_ROUTERS = ("Router: signals + PoT", "Router: signals + SC3")


# ------------------------------------------------------------------ splits
def splits(probs):
    orig = read_jsonl(os.path.join(m4.RUN, "routing", "traces.jsonl"))
    tune0, eval0 = [r["problem_id"] for r in orig[:50]], [r["problem_id"] for r in orig[50:]]
    rest = sorted(set(probs) - set(tune0) - set(eval0))
    extra = sorted(random.Random(SEED).sample(rest, 250))
    tune = set(tune0) | set(extra)
    return tune, set(eval0)


# ------------------------------------------------------------------ features
def sc_feats(r0_answer, vals):
    agree = [v is not None and v == r0_answer for v in vals]     # unparsable sample counts as disagreeing
    parsed = [v for v in vals if v is not None]
    return {"sc3_agree_frac": float(np.mean(agree)),
            "sc3_majority_is_r0": float(bool(parsed) and max(set(parsed), key=parsed.count) == r0_answer)}


def holdout_rows(gpu, g):
    rows = read_jsonl(os.path.join(gpu, g, "routing_traces.jsonl"))
    feats = {f["problem_id"]: f for f in read_jsonl(os.path.join(gpu, g, "features.jsonl"))}
    out = []
    for r in rows:
        f = feats[r["problem_id"]]
        t = {"initial_response": r["R0"], "critique": r["F"], "source": "gsm8k", "r0_answer": r["r0_answer"]}
        ex = r["exec"]
        pot = normalize_number(ex["stdout"]) if ex["stdout"].strip() else None
        r0n, fa = extract(r["R0"], "number").value, extract(r["F"], "number").value
        v = {"pot_ran": int(pot is not None),
             "pot_agree_r0": float(is_correct(r0n, pot, "number")) if pot is not None else np.nan,
             "pot_agree_f": float(is_correct(fa, pot, "number")) if pot is not None else np.nan}
        vals = [extract(x, "number", accept_boxed=True).value for x in r["samples"]]
        sig = {k: (np.nan if f.get(k) is None else f[k]) for k in
               ("lp_r0_mean", "lp_r0_min", "lp_answer_mean", "lp_f_mean", "lp_f_min", "n_tok_r0", "n_tok_f")}
        assert all(f[k] == verdict_features(t)[k] for k in verdict_features(t))
        out.append({**r, **sig, **verdict_features(t), **v, **sc_feats(r["r0_answer"], vals), "source": "gsm8k",
                    "tf_latency_s": f["tf_latency_s"], "fail": int(not r["int_correct"])})
    return out


def natural_rows(g, m6_gen):
    os.chdir(BENCH)
    cfg = yaml.safe_load(open(os.path.join("configs", CONFIGS[g])))
    rows = [r for r in generator_rows(cfg, "natural", g) if r["source"] == "gsm8k"]
    trace = {t["trace_id"]: t for t in read_jsonl(os.path.join(cfg["run_dir"], "natural", "traces.jsonl"))}
    S = samples_by_prompt(os.path.join(m6_gen, f"calls_{g}.jsonl"))
    for r in rows:
        t = trace[r["trace_id"]]
        msg = prompts.initial_messages(t["prompt"], "number")[0]["content"]
        vals = [extract(S[(msg, k)], "number", accept_boxed=True).value for k in (1, 2, 3)]
        r.update(sc_feats(t["r0_answer"], vals))
    return rows


def fit_predict(train, test, with_verifier):
    """The paper's LR 'scalar' model, fitted on all natural-set GSM8K traces, applied to the hold-out rows."""
    strip = (lambda r: r) if with_verifier else (lambda r: {k: v for k, v in r.items() if k not in VERIFIER})
    rows = [strip(r) for r in train] + [strip(r) for r in test]
    X = scalar_matrix(rows)
    if not with_verifier:
        X = np.hstack([X, np.array([[r[c] for c in SC3] for r in rows], float)])
    y = np.array([r["label"] for r in train] + [0] * len(test))
    n = len(train)
    m = Base("scalar", {"scalar": X, "text": None, "hidden": None}).fit(np.arange(n), y)
    return m.proba(np.arange(n, len(rows)))


def deberta_scores(rows, g, cache):
    """SCFP v1.0 predictor's failure probability for each hold-out trace (Apple M1), cached on disk."""
    if os.path.exists(cache):
        c = json.load(open(cache))
        return np.array([c["p"][r["problem_id"]] for r in rows]), c["latency_s"]
    import torch
    from transformers import AutoTokenizer
    from scfp_bench.experiments.train_encoder import build, encode
    run = m4.RUN
    res = json.load(open(os.path.join(run, "results", "deberta-scfp.json")))
    a = res["args"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a["model"])
    model = build(a["arch"], a["model"], special_attention=not a.get("no_special_attention", False))
    model.load_state_dict(torch.load(os.path.join(run, "results", "deberta-scfp.pt"), map_location="cpu"))
    model.to(device).eval()
    p, lat = {}, []
    for r in rows:
        x = {"trace_id": r["problem_id"], "source_model": g, "source": "gsm8k", "prompt": r["question"],
             "initial_response": r["R0"], "critique": r["F"]}
        ids, seg = encode(tok, x, a["max_len"])
        t0 = time.time()
        with torch.no_grad():
            xi = torch.tensor([ids], device=device)
            out = model(input_ids=xi, attention_mask=torch.ones_like(xi), token_type_ids=torch.tensor([seg], device=device))["binary_logits"]
            p[r["problem_id"]] = float(torch.softmax(out.float(), -1)[0, 1])
        lat.append(time.time() - t0)
    json.dump({"p": p, "latency_s": float(np.mean(lat)), "device": device}, open(cache, "w"))
    return np.array([p[r["problem_id"]] for r in rows]), float(np.mean(lat))


# ------------------------------------------------------------------ routing
def route(s, lo, hi):
    return np.where(s < lo, 0, np.where(s < hi, 1, 2))   # 0 int, 1 ext, 2 hum


def router_paths(C, L, A, extra_c, extra_l, pot_paid):
    """Per-path arrays (int, ext, hum) for a router that pays extra_c / extra_l on every problem."""
    Cr = np.stack([C["int"] + extra_c, C["ext"] + (0 if pot_paid else extra_c), C["hum"] + extra_c])
    Lr = np.stack([L["int"] + extra_l, L["ext"] + (0 if pot_paid else extra_l), L["hum"] + extra_l])
    Ar = np.stack([A["int"], A["ext"], A["hum"]])
    return Cr, Lr, Ar


def take(M, ch):
    return M[ch, np.arange(M.shape[1])]


def extras(rows, unit, name, lat):
    tok = lambda k: np.array([r["tok"][k] for r in rows], float) / unit
    lt = lambda k: np.array([r["lat"][k] for r in rows], float)
    tf = np.array([r["tf_latency_s"] for r in rows])
    if name == "Router: signals + PoT":
        return tok("program"), lt("program") + lt("exec") + tf, True
    if name == "Router: signals + SC3":
        return tok("samples"), lt("samples") + tf, False
    if name == "Router: SCFP v1.0 DeBERTa":
        return np.zeros(len(rows)), np.full(len(rows), lat), False
    return np.zeros(len(rows)), tf, False                    # confidence: teacher-forced pass for lp_f_mean


def tune_thresholds(rows, key, name, lat, hc, hl, w):
    C, L, A, unit, _ = m4.paths(rows, hc, hl)
    Cn, Ln = C["hum"].mean(), L["hum"].mean()
    ec, el, pot = extras(rows, unit, name, lat)
    Cr, Lr, Ar = router_paths(C, L, A, ec, el, pot)
    s = np.array([r[key] for r in rows])
    best = None
    for lo, hi in m4.PAIRS:
        ch = route(s, lo, hi)
        v = m4.score(take(Cr, ch), take(Lr, ch), take(Ar, ch), Cn, Ln, w)
        if best is None or v > best[0]:
            best = (v, lo, hi)
    return best[1], best[2]


def evaluate(tune, test, lat, hc, hl, w, keep_choice=False):
    C, L, A, unit, esc = m4.paths(test, hc, hl)
    Cn, Ln = C["hum"].mean(), L["hum"].mean()
    pol, th, choice = {}, {}, {}
    for name, k in FIXED.items():
        pol[name] = (C[k], L[k], A[k])
    pol["Random (uniform, expected)"] = tuple((D["int"] + D["ext"] + D["hum"]) / 3 for D in (C, L, A))
    for name, key in ROUTERS.items():
        th[name] = tune_thresholds(tune, key, name, lat, hc, hl, w)
        ec, el, pot = extras(test, unit, name, lat)
        Cr, Lr, Ar = router_paths(C, L, A, ec, el, pot)
        ch = route(np.array([r[key] for r in test]), *th[name])
        pol[name] = (take(Cr, ch), take(Lr, ch), take(Ar, ch))
        choice[name] = ch
    return pol, th, choice, (C, L, A, unit, esc, Cn, Ln)


def boot_ci(a, rng):
    d = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def aurc_ci(score_, wrong, rng):
    d, dr = [], []
    for _ in range(B):
        i = np.sort(rng.integers(0, len(wrong), len(wrong)))
        a = m4.risk_coverage(score_[i], wrong[i])[2]
        d.append(a)
        dr.append(a - wrong[i].mean())
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(d), q(dr)


def ece(y, p, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    from sklearn.isotonic import IsotonicRegression
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="results/revision/e1/gpu")
    ap.add_argument("--m6-gen", default="results/revision/m6/gen")
    ap.add_argument("--out", default="results/revision/e1")
    ap.add_argument("--gens", nargs="+", default=GENS)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    gpu, out, m6_gen = (os.path.abspath(x) for x in (a.gpu, a.out, a.m6_gen))
    os.makedirs(out, exist_ok=True)
    res, all_p, keys, sweeps = {}, [], [], {}
    for g in a.gens:
        rows = holdout_rows(gpu, g)
        tune_ids, eval0 = splits({r["problem_id"] for r in rows})
        nat = natural_rows(g, m6_gen)
        p_pot = fit_predict(nat, rows, True)
        p_sc3 = fit_predict(nat, rows, False)
        p_deb, deb_lat = deberta_scores(rows, g, os.path.join(out, f"deberta_scores_{g}.json"))
        tune = [r for r in rows if r["problem_id"] in tune_ids]
        test = [r for r in rows if r["problem_id"] not in tune_ids]
        it = np.array([r["problem_id"] in tune_ids for r in rows])
        y = np.array([r["fail"] for r in rows])
        cal = {}
        for name, p in (("p_pot", p_pot), ("p_sc3", p_sc3), ("p_deb", p_deb)):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[it], y[it])
            pc = iso.predict(p)
            for r, raw, c in zip(rows, p, pc):
                r[name], r[name + "_raw"] = float(c), float(raw)
            cal[name] = {"ece_raw_eval": ece(y[~it], p[~it]), "ece_cal_eval": ece(y[~it], pc[~it]),
                         "auc_raw_eval": float(m4_auc(y[~it], p[~it])), "auc_cal_eval": float(m4_auc(y[~it], pc[~it]))}
        tn = np.sort([-r["lp_f_mean"] for r in tune])
        for r in rows:
            r["conf_score"] = float(np.searchsorted(tn, -r["lp_f_mean"], side="right") / len(tn))
        pol, th, choice, (C, L, A, unit, esc, Cn, Ln) = evaluate(tune, test, deb_lat, HC, HL, W)
        rng = np.random.default_rng(SEED)
        r0c = pol["No correction (R0 only)"][2]
        main_rows = {}
        for name, (c, l, acc) in pol.items():
            rand = name.startswith("Random")
            main_rows[name] = {"accuracy": float(acc.mean()), "accuracy_ci95": None if rand else boot_ci(acc, rng),
                               "relative_cost": float(c.mean()), "latency_s": float(l.mean()),
                               "overall_score": float(m4.score(c, l, acc, Cn, Ln, W)),
                               "thresholds": [float(x) for x in th[name]] if name in th else None,
                               "share": ({k: float(np.mean(choice[name] == i)) for i, k in enumerate(("int", "ext", "hum"))}
                                         if name in choice else None),
                               "diff_vs_r0": float(acc.mean() - r0c.mean())}
            if not rand and name != "No correction (R0 only)":
                p_ = perm_test(acc, r0c, [np.array([i]) for i in range(len(acc))], rng)
                main_rows[name]["perm_p_vs_r0"] = p_
                all_p.append(p_)
                keys.append((g, name))
        # risk-coverage of the intrinsic answer (eval split), calibrated scores
        wrong = 1 - A["int"]
        rc = {}
        for name, key in (("Signals + PoT", "p_pot"), ("Signals + SC3", "p_sc3"), ("SCFP v1.0 DeBERTa", "p_deb"),
                          ("Critique confidence", "conf_score")):
            s_ = np.array([r[key] for r in test])
            cov, risk, aurc = m4.risk_coverage(s_, wrong)
            ci, dci = aurc_ci(s_, wrong, np.random.default_rng(SEED))
            rc[name] = {"aurc": aurc, "aurc_ci95": ci, "diff_vs_random_ci95": dci, "coverage": cov.tolist(), "risk": risk.tolist()}
        k_ = np.arange(1, len(wrong) + 1)
        nw = int(wrong.sum())
        orisk = np.maximum(0, k_ - (len(wrong) - nw)) / k_
        rc["Random (expected)"] = {"aurc": float(wrong.mean()), "coverage": (k_ / len(k_)).tolist(), "risk": [float(wrong.mean())] * len(k_)}
        rc["Oracle"] = {"aurc": float(orisk.mean()), "coverage": (k_ / len(k_)).tolist(), "risk": orisk.tolist()}
        # Pareto sweep (descriptive, eval split)
        sweep = {}
        for name, key in ROUTERS.items():
            ec, el, pot = extras(test, unit, name, deb_lat)
            Cr, Lr, Ar = router_paths(C, L, A, ec, el, pot)
            s_ = np.array([r[key] for r in test])
            sweep[name] = [{"tau_low": float(lo), "tau_high": float(hi), "relative_cost": float(take(Cr, route(s_, lo, hi)).mean()),
                            "accuracy": float(take(Ar, route(s_, lo, hi)).mean()),
                            "share_hum": float(np.mean(route(s_, lo, hi) == 2))} for lo, hi in m4.PAIRS]
        sweeps[g] = sweep
        # Overall-Score sensitivity (594 settings), thresholds re-tuned in each
        sens = None
        if not a.no_sensitivity:
            simplex = [(i / 10, j / 10, round(1 - i / 10 - j / 10, 1)) for i in range(11) for j in range(11 - i)]
            wins = {n: 0 for n in pol}
            beats = {n: 0 for n in ROUTERS}
            nat_wins = 0
            for hc, hl, w in itertools.product((5.0, 15.0, 50.0), (15.0, 45.0, 120.0), simplex):
                P, _, _, (C2, L2, _, _, _, cn, ln) = evaluate(tune, test, deb_lat, hc, hl, w)
                sc = {n: float(m4.score(*v, cn, ln, w)) for n, v in P.items()}
                best = max(sc, key=sc.get)
                wins[best] += 1
                fixed_best = max(v for n, v in sc.items() if n not in ROUTERS)
                for n in ROUTERS:
                    beats[n] += sc[n] > fixed_best
                nat_wins += best in NATURAL_ROUTERS
            sens = {"n_settings": len(simplex) * 9, "wins": wins, "router_beats_best_fixed": beats,
                    "natural_router_best": nat_wins}
        # reproduction check (registered): Qwen2.5-1.5B R0 accuracy on the original 150 evaluation problems
        repro = None
        if g == "qwen2.5-1.5b":
            r0_150 = np.array([r["r0_correct"] for r in rows if r["problem_id"] in eval0], float)
            pub = json.load(open(os.path.join(HERE, "..", "..", "results", "revision", "m4", "routing_m4.json")))
            pci = pub["policies"]["No correction (R0 only)"]["accuracy_ci95"]
            repro = {"n": int(len(r0_150)), "r0_accuracy_new": float(r0_150.mean()),
                     "published": pub["policies"]["No correction (R0 only)"]["accuracy"], "published_ci95": pci,
                     "inside_ci": bool(pci[0] <= r0_150.mean() <= pci[1])}
        res[g] = {"n_tune": len(tune), "n_eval": len(test), "n_natural_train": len(nat), "cost_unit_tokens": float(unit),
                  "deberta_latency_s": deb_lat, "calibration": cal,
                  "accuracy_eval": {"r0": float(A["r0"].mean()), "int": float(A["int"].mean()), "ext": float(A["ext"].mean())},
                  "agreement": {"program_answered": int(sum(m4.agreement(r)[0] for r in test)),
                                "escalated_to_human": int(esc.sum())},
                  "policies": main_rows, "risk_coverage": rc, "sensitivity": sens, "reproduction": repro}
        print(g, f"tune {len(tune)} eval {len(test)} | R0 {A['r0'].mean():.3f} int {A['int'].mean():.3f} ext {A['ext'].mean():.3f}", flush=True)
        for n, v in main_rows.items():
            print(f"  {n:30s} acc {v['accuracy']:.3f} cost {v['relative_cost']:.2f} score {v['overall_score']:.3f} "
                  f"share {v['share']} p {v.get('perm_p_vs_r0')}", flush=True)
        print("  AURC:", {n: round(v["aurc"], 3) for n, v in rc.items()}, "cal:", cal, flush=True)
        if sens:
            print("  sensitivity wins:", {k: v for k, v in sens["wins"].items() if v}, "beats:", sens["router_beats_best_fixed"], flush=True)
        if repro:
            print("  reproduction:", repro, flush=True)
    for (g, n), h in zip(keys, holm(all_p)):
        res[g]["policies"][n]["perm_p_vs_r0_holm"] = h
    # registered H3 decision per generator
    for g in a.gens:
        s_ = res[g]["sensitivity"]
        pol = res[g]["policies"]
        best_nat = max(s_["wins"][n] for n in NATURAL_ROUTERS) if s_ else 0
        sig = any(pol[n]["perm_p_vs_r0_holm"] < 0.05 and pol[n]["diff_vs_r0"] > 0
                  and pol[n]["relative_cost"] < pol["Always Intrinsic"]["relative_cost"] for n in NATURAL_ROUTERS)
        res[g]["h3"] = ("supported" if s_ and best_nat > s_["n_settings"] / 2 else
                        "partly supported" if best_nat >= 1 or sig else "not supported")
        print(g, "H3:", res[g]["h3"], flush=True)
    json.dump(res, open(os.path.join(out, "e1_routing.json"), "w"), indent=1)
    json.dump(sweeps, open(os.path.join(out, "e1_sweeps.json"), "w"))
    plot(res, sweeps, out, a.gens)


def m4_auc(y, p):
    from m5_deconfound import auc
    return auc(y, p)


NAMES = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "smollm2-1.7b": "SmolLM2-1.7B", "qwen2.5-1.5b": "Qwen2.5-1.5B"}


def plot(res, sweeps, out, gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, len(gens), figsize=(3.3 * len(gens), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    colors = {"Router: signals + PoT": "#4C72B0", "Router: signals + SC3": "#55A868",
              "Router: SCFP v1.0 DeBERTa": "#C44E52", "Confidence-based": "#DD8452"}
    marks = {"Always Intrinsic": "s", "Always Extrinsic": "^", "No correction (R0 only)": "*", "Agreement tool": "D",
             "Agreement tool + human": "P", "Always Human (oracle)": "v", "Random (uniform, expected)": "x"}
    for ax, g in zip(axes, gens):
        for name, col in colors.items():
            pts = sweeps[g][name]
            x, y = np.array([p["relative_cost"] for p in pts]), np.array([p["accuracy"] for p in pts])
            ax.scatter(x, y, s=4, alpha=0.25, color=col)
            v = res[g]["policies"][name]
            ax.scatter(v["relative_cost"], v["accuracy"], marker="o", s=34, facecolors="none", edgecolors=col, lw=1.4,
                       zorder=5, label=name.replace("Router: ", "") + " (tuned)")
        for name, m in marks.items():
            v = res[g]["policies"][name]
            ax.scatter(v["relative_cost"], v["accuracy"], marker=m, s=50 if m == "*" else 26, color="k", zorder=6, label=name)
        ax.set_xscale("log")
        ax.set_title(NAMES[g])
        ax.set_xlabel("Relative cost (log scale)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Accuracy ({res[gens[0]]['n_eval']} evaluation problems)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, "fig_e1_pareto.pdf"))
    plt.close(fig)
    fig, axes = plt.subplots(1, len(gens), figsize=(3.3 * len(gens), 3.0))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, gens):
        for name, st in (("Signals + PoT", "-"), ("Signals + SC3", "-"), ("SCFP v1.0 DeBERTa", "--"),
                         ("Critique confidence", "--"), ("Random (expected)", ":"), ("Oracle", "-.")):
            v = res[g]["risk_coverage"][name]
            ax.plot(v["coverage"], v["risk"], st, lw=1, label=f"{name} ({v['aurc']:.3f})")
        ax.set_title(NAMES[g])
        ax.set_xlabel("Coverage (kept on intrinsic correction)")
        ax.legend(fontsize=5.5, title="AURC", title_fontsize=6)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Risk (revised answer wrong)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e1_risk_coverage.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
