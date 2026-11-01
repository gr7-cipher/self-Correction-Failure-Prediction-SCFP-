#!/usr/bin/env python3
"""
verify_metrics.py -- consistency audit for the SCFP manuscript.

Two kinds of checks:

  Part A  Internal-consistency checks. These use only numbers printed in the
          manuscript (or transcribed from its figures) and test whether they
          are arithmetically possible. No experimental artifacts are needed.

  Part B  Artifact-based verification. Given per-example prediction files
          (--predictions), recompute confusion matrices, accuracy, macro-F1,
          bootstrap CIs and paired significance tests. If no prediction
          files are supplied, Part B reports every empirical claim as
          NOT VERIFIABLE instead of estimating it.

  Part C  Dataset checks on the released SCFP data (--repo).

  Part D  v5/v6 tables recomputed from the run directory (--run-dir, --tables).

  Part E  v7 natural-set results recomputed from per-trace predictions (--v7-dir,
          --v7-runs, --manuscript).

Usage:
  python verify_metrics.py --repo ../repo/SCFP
  python verify_metrics.py --repo ../repo/SCFP --predictions preds/*.jsonl

Prediction file format (one JSON object per line, one file per model):
  {"trace_id": "...", "model": "deberta", "task": "binary"|"multiclass",
   "y_true": int, "y_pred": int, "p_fail": float (binary only, optional)}
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

RESULTS = []  # (section, check, verdict, detail)


def record(section, check, ok, detail):
    verdict = {True: "PASS", False: "FAIL", None: "NOT VERIFIABLE"}[ok]
    RESULTS.append((section, check, verdict, detail))


def h(probs):
    p = np.asarray([x for x in probs if x > 0], dtype=float)
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------
# Numbers as printed in enhanced_manuscript_v2.tex
# --------------------------------------------------------------------------
PAPER = {
    "binary": {  # Table tab:main_results, accuracy (mean, 95% CI half-width)
        "ours": (0.824, 0.021), "llama3_sft": (0.809, 0.022),
        "qwen_sft": (0.805, 0.022), "roberta": (0.798, 0.023),
        "bert": (0.776, 0.024), "gpt4o_cot": (0.765, 0.024),
        "llama70b_cot": (0.748, 0.025), "gpt4o": (0.742, 0.025),
        "confidence": (0.623, 0.028), "length": (0.587, 0.029),
        "random": (0.551, 0.029),
    },
    "multiclass_acc_ours": 0.756,
    "macro_f1_binary_ours": 0.814,
    "recall_JH": 0.81, "recall_RM": 0.81, "BA_to_OC": 0.17,
    "test_failure_dist": {"JH": .21, "CM": .18, "BA": .16, "OC": .19, "RM": .26},
    "test_traces": 1800, "test_failure_rate": 0.54,
    "failure_dist_total": {"JH": .22, "CM": .18, "BA": .15, "OC": .20, "RM": .25},
    "info": {  # Table tab:info_theory
        "binary": {"H": 0.996, "H_cond": 0.267, "I": 0.729, "bound": 0.866, "acc": 0.824},
        "multi": {"H": 2.234, "H_cond": 1.512, "I": 0.722, "bound": 0.678, "acc": 0.756},
    },
    "routing": {  # Table tab:gsm8k_results: accuracy, latency s, cost x, human share
        "Always Intrinsic": (0.682, 3.5, 1.0, 0.00),
        "Always Extrinsic": (0.891, 8.2, 2.5, 0.00),
        "Always Human": (0.968, 45.0, 15.0, 1.00),
        "Random Routing": (0.785, 12.3, 4.2, 0.33),
        "Confidence-Based": (0.823, 6.8, 1.8, 0.15),
        "Dynamic Routing": (0.874, 4.8, 1.4, 0.08),
    },
    "readme_binary_acc": {  # README.md results table of the released repo
        "roberta": 0.789, "bert": 0.752, "gpt4o": 0.736,
        "confidence": 0.643, "length": 0.587, "random": 0.501,
    },
}

# Raw counts transcribed from enhanced_confusion_matrix.png (left panel).
# Rows = true class, columns = predicted class, order JH, CM, BA, OC, RM.
FIGURE_CM = np.array([
    [145, 12, 8, 10, 5],
    [15, 125, 20, 15, 5],
    [10, 25, 110, 30, 5],
    [8, 18, 35, 115, 4],
    [5, 8, 12, 10, 145],
])
CLASSES = ["JH", "CM", "BA", "OC", "RM"]


# --------------------------------------------------------------------------
# Part A -- internal consistency
# --------------------------------------------------------------------------
def check_confusion_matrix():
    s = "A1 confusion matrix (Fig. confusion_matrix)"
    cm = FIGURE_CM
    n = cm.sum()
    acc = np.trace(cm) / n
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    f1 = 2 * precision * recall / (precision + recall)
    rows = cm.sum(1)
    expected_n = round(PAPER["test_traces"] * PAPER["test_failure_rate"])
    expected_rows = {c: round(expected_n * p) for c, p in PAPER["test_failure_dist"].items()}

    record(s, "row sums follow test-set class distribution",
           len(set(rows)) > 1,
           f"row sums = {rows.tolist()} (N={n}); Table 2 test split implies "
           f"N~{expected_n} with rows ~{list(expected_rows.values())}")
    record(s, "Accuracy = trace(CM)/sum(CM) matches Table 3 multi-class accuracy",
           abs(acc - PAPER["multiclass_acc_ours"]) < 0.005,
           f"trace/sum = {np.trace(cm)}/{n} = {acc:.3f}; paper reports {PAPER['multiclass_acc_ours']}")
    record(s, "Macro-F1 from CM matches Table 3 multi-class macro-F1 (0.682)",
           abs(f1.mean() - 0.682) < .005,
           f"figure implies macro-F1 = {f1.mean():.3f} (paper Table 3 multi-class: 0.682); "
           f"per-class recall {dict(zip(CLASSES, recall.round(3)))}")
    record(s, "JH / RM recall = 0.81 as stated in text",
           abs(recall[0] - .81) < .005 and abs(recall[4] - .81) < .005,
           f"JH={recall[0]:.3f}, RM={recall[4]:.3f}")
    record(s, "BA->OC confusion = 17% as stated in text",
           abs(cm[2, 3] / rows[2] - .17) < .005, f"{cm[2, 3]}/{rows[2]} = {cm[2, 3] / rows[2]:.3f}")


def check_routing():
    s = "A2 routing cost/latency (Table gsm8k_results)"
    r = PAPER["routing"]
    acc_i, lat_i, c_i, _ = r["Always Intrinsic"]
    acc_e, lat_e, c_e, _ = r["Always Extrinsic"]
    acc_h, lat_h, c_h, _ = r["Always Human"]
    for name in ["Dynamic Routing", "Confidence-Based", "Random Routing"]:
        acc, lat, cost, p_h = r[name]
        # Expected value over paths: C = sum_k p_k C_k.  With p_h fixed and the
        # rest split between intrinsic and extrinsic in any proportion:
        c_lo, c_hi = p_h * c_h + (1 - p_h) * c_i, p_h * c_h + (1 - p_h) * c_e
        l_lo, l_hi = p_h * lat_h + (1 - p_h) * lat_i, p_h * lat_h + (1 - p_h) * lat_e
        record(s, f"{name}: cost {cost}x within feasible [{c_lo:.2f}, {c_hi:.2f}]x",
               c_lo - 1e-9 <= cost <= c_hi + 1e-9,
               f"p_h={p_h}: C >= {p_h}*{c_h} + {1 - p_h:.2f}*{c_i} = {c_lo:.2f}x. "
               "Lower bound ignores the predictor and the R0/F generation every query pays.")
        record(s, f"{name}: latency {lat}s within feasible [{l_lo:.2f}, {l_hi:.2f}]s",
               l_lo - 1e-9 <= lat <= l_hi + 1e-9,
               f"L >= {p_h}*{lat_h} + {1 - p_h:.2f}*{lat_i} = {l_lo:.2f}s")
    # Uniform random routing over three paths has a closed-form expectation.
    acc_r, lat_r, c_r, ph_r = r["Random Routing"]
    exp = [np.mean(v) for v in ([acc_i, acc_e, acc_h], [lat_i, lat_e, lat_h], [c_i, c_e, c_h])]
    record(s, "Random Routing equals the uniform mixture of the three paths",
           abs(acc_r - exp[0]) < .01 and abs(c_r - exp[2]) < .1,
           f"uniform mixture: acc={exp[0]:.3f}, latency={exp[1]:.1f}s, cost={exp[2]:.2f}x; "
           f"paper: acc={acc_r}, latency={lat_r}s, cost={c_r}x")
    record(s, "True Dynamic Routing cost/latency", None,
           "Needs the per-query routing log (path taken, cost, latency, correctness). "
           "router.py:151 samples p_fail from np.random.beta(2,3); no such log exists.")


def check_information_theory():
    s = "A3 information-theoretic bound (Eq. accuracy_bound, Table info_theory)"
    for task, k in [("binary", 2), ("multi", 5)]:
        d = PAPER["info"][task]
        eq_bound = 1 - d["H_cond"] / math.log2(k)
        record(s, f"{task}: Eq. bound with Table entropies equals reported bound {d['bound']}",
               abs(eq_bound - d["bound"]) < .005, f"1 - {d['H_cond']}/log2({k}) = {eq_bound:.3f}")
        record(s, f"{task}: model accuracy {d['acc']} <= bound from Eq.",
               d["acc"] <= eq_bound, f"{d['acc']} vs {eq_bound:.3f}")
        record(s, f"{task}: I = H(Y) - H(Y|T)", abs(d["H"] - d["H_cond"] - d["I"]) < .002,
               f"{d['H']} - {d['H_cond']} = {d['H'] - d['H_cond']:.3f}; reported {d['I']}")

        # Correct Fano bound: H(Y|T) <= H_b(Pe) + Pe*log2(k-1)  =>  Pe >= Pe_min.
        def fano(pe):
            hb = 0.0 if pe in (0, 1) else -(pe * math.log2(pe) + (1 - pe) * math.log2(1 - pe))
            return hb + pe * math.log2(k - 1) - d["H_cond"]
        pe_min = optimize.brentq(fano, 1e-12, (k - 1) / k)
        record(s, f"{task}: accuracy within the correct Fano bound", d["acc"] <= 1 - pe_min,
               f"Fano: Acc <= {1 - pe_min:.3f}. Entropy estimates themselves have no artifact "
               "in the repo, so this bound is not empirically grounded either.")
    stated = PAPER["failure_dist_total"]
    record(s, "multi-class H(Y) matches stated class distribution",
           abs(h(stated.values()) - PAPER["info"]["multi"]["H"]) < .01,
           f"H(22/18/15/20/25%) = {h(stated.values()):.3f} bits; reported 2.234")
    record(s, "binary H(Y) matches 45/55 split",
           abs(h([.45, .55]) - PAPER["info"]["binary"]["H"]) < .005,
           f"H(45/55) = {h([.45, .55]):.3f} bits; reported 0.996")


def check_intervals():
    s = "A4 confidence intervals and claimed improvements"
    b = PAPER["binary"]
    m1, w1 = b["ours"]
    for k in ["llama3_sft", "qwen_sft", "roberta"]:
        m2, w2 = b[k]
        overlap = (m1 - w1) <= (m2 + w2)
        record(s, f"95% CI of ours does not overlap {k}", not overlap,
               f"[{m1 - w1:.3f}, {m1 + w1:.3f}] vs [{m2 - w2:.3f}, {m2 + w2:.3f}]")
    # Approximate unpaired z-test from summary statistics (SE = halfwidth/1.96).
    m2, w2 = b["llama3_sft"]
    z = (m1 - m2) / math.hypot(w1 / 1.96, w2 / 1.96)
    p = 2 * stats.norm.sf(abs(z))
    record(s, "ours vs Llama-3 SFT significant at 0.05 (approximate)", p < .05,
           f"z={z:.2f}, p={p:.3f} from summary statistics only. Both models use the same test "
           "set, so the correct test is McNemar on per-example predictions (Part B).")
    rel = {k: (m1 / v[0] - 1) * 100 for k, v in b.items() if k != "ours"}
    in_range = [k for k, v in rel.items() if 7.7 - .05 <= v <= 11.1 + .05]
    record(s, "abstract: 7.7-11.1% relative gain covers SFT Llama-3/Qwen",
           "llama3_sft" in in_range and "qwen_sft" in in_range,
           "relative accuracy gains: " + ", ".join(f"{k}={v:.1f}%" for k, v in rel.items()))
    for k, v in PAPER["readme_binary_acc"].items():
        record(s, f"README accuracy for {k} matches paper", abs(v - b[k][0]) < 5e-4,
               f"README {v:.3f} vs paper {b[k][0]:.3f}")


# --------------------------------------------------------------------------
# Part B -- artifact-based verification
# --------------------------------------------------------------------------
def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def bootstrap_ci(y_true, y_pred, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = [fn(y_true[s], y_pred[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def verify_predictions(paths):
    s = "B  artifact-based metrics"
    if not paths:
        for claim in ["Table main_results (all rows)", "Fig. roc_curves", "Fig. confusion_matrix",
                      "Table ablation_studies", "Table cross_model_generalization",
                      "attention analysis t(1798)=14.2", "Table info_theory entropies",
                      "ECE (listed as a metric, never reported)"]:
            record(s, claim, None, "no per-example prediction artifacts supplied or found in repo")
        return
    by_model = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            by_model[(r["model"], r["task"])][r["trace_id"]] = r
    for (model, task), rows in sorted(by_model.items()):
        y = np.array([r["y_true"] for r in rows.values()])
        yp = np.array([r["y_pred"] for r in rows.values()])
        cm = confusion_matrix(y, yp)
        acc = np.trace(cm) / cm.sum()
        f1 = f1_score(y, yp, average="macro")
        lo, hi = bootstrap_ci(y, yp, lambda a, b: np.mean(a == b))
        detail = (f"N={len(y)}, acc=trace/sum={acc:.4f} [95% CI {lo:.4f}, {hi:.4f}], "
                  f"macro-F1={f1:.4f}, row sums={cm.sum(1).tolist()}")
        if task == "binary" and all("p_fail" in r for r in rows.values()):
            p = np.array([r["p_fail"] for r in rows.values()])
            detail += f", AUC={roc_auc_score(y, p):.4f} (continuous scores)"
        record(s, f"{model} / {task}", True, detail)
    ref = ("deberta", "binary")
    if ref in by_model:
        for (model, task), rows in by_model.items():
            if task != "binary" or model == "deberta":
                continue
            common = sorted(set(rows) & set(by_model[ref]))
            a = np.array([by_model[ref][t]["y_pred"] == by_model[ref][t]["y_true"] for t in common])
            b_ = np.array([rows[t]["y_pred"] == rows[t]["y_true"] for t in common])
            n01, n10, p = mcnemar_exact(a, b_)
            record(s, f"McNemar deberta vs {model}", p < .05,
                   f"N={len(common)}, discordant {n01}/{n10}, exact p={p:.4g}")


# --------------------------------------------------------------------------
# Part C -- released dataset
# --------------------------------------------------------------------------
def check_dataset(repo):
    s = "C  released dataset"
    path = os.path.join(repo, "data", "scfp_v1.jsonl")
    if not os.path.exists(path):
        record(s, "dataset present", None, f"{path} not found")
        return
    R = [json.loads(l) for l in open(path)]
    mask = lambda t: re.sub(r"\d+", "#", t)
    by_tmpl = defaultdict(Counter)
    for r in R:
        by_tmpl[mask(r["critique"])][r["failure_type"]] += 1
    leak = sum(max(c.values()) for c in by_tmpl.values()) / len(R)
    record(s, "failure label not determined by critique template", leak < .9,
           f"{len(by_tmpl)} critique templates; majority-label-per-template accuracy = {leak:.3f}")
    record(s, "unique source problems = 3,000 (paper)", len({r["prompt"] for r in R}) == 3000,
           f"{len({r['prompt'] for r in R})} unique prompts")
    nonmath = [r for r in R if "GSM8K" not in r["domain"]]
    arith = sum(bool(re.search(r"\d+ [×x*+\-] \d+ =", r["initial_response"])) for r in nonmath)
    record(s, "non-math answers are not arithmetic templates", arith == 0,
           f"{arith}/{len(nonmath)} StrategyQA/LogicQA traces answer with an arithmetic result")
    llama = sum(any(x in r["source_model"].lower() for x in ("llama", "qwen", "mistral")) for r in R)
    record(s, "open-weight traces exist for Table cross_model_generalization", llama > 0,
           f"{llama} traces from Llama/Qwen/Mistral; models = {sorted({r['source_model'] for r in R})}")
    splits = {}
    for k in ("train", "val", "test"):
        p = os.path.join(repo, "data", "processed", f"{k}.jsonl")
        if os.path.exists(p):
            splits[k] = [json.loads(l) for l in open(p)]
    if splits:
        record(s, "split sizes = 8,400 / 1,800 / 1,800 (Table 2)",
               [len(splits[k]) for k in ("train", "val", "test")] == [8400, 1800, 1800],
               f"{[len(v) for v in splits.values()]}")
        train_p = {r["prompt"] for r in splits["train"]}
        shared = sum(r["prompt"] in train_p for r in splits["test"])
        record(s, "no test trace shares a source problem with train", shared == 0,
               f"{shared}/{len(splits['test'])} test traces share a prompt with train")
        fail = Counter(r["failure_type"] for r in splits["test"] if r["label"] == 1)
        record(s, "test failure distribution (for Fig. confusion_matrix rows)", None,
               f"N={sum(fail.values())}: {dict(fail)}")
    meta = os.path.join(repo, "data", "dataset_metadata.json")
    if os.path.exists(meta):
        k = json.load(open(meta)).get("pilot_kappa")
        record(s, "pilot kappa = 0.847 (paper)", k is not None and abs(k - .847) < 5e-4,
               f"metadata pilot_kappa = {k}")
    weights = glob.glob(os.path.join(repo, "models", "**", "*.bin"), recursive=True) + \
        glob.glob(os.path.join(repo, "models", "**", "*.safetensors"), recursive=True)
    record(s, "trained model weights released", bool(weights), f"{len(weights)} weight files found")


# --------------------------------------------------------------------------
# Part D -- v5 manuscript: recompute every generated number from the run dir
# --------------------------------------------------------------------------
def check_v5(run_dir, tables_dir):
    s = "D  v5 tables vs. run directory"
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    base = "final" if os.path.exists(os.path.join(run_dir, "final", "train.jsonl")) else "splits"
    S = {k: load(os.path.join(run_dir, base, f"{k}.jsonl")) for k in ("train", "val", "test")}
    nums = json.load(open(os.path.join(tables_dir, "numbers.json")))

    owner = {}
    leak = 0
    for k, rows in S.items():
        for r in rows:
            leak += owner.setdefault(r["problem_id"], k) != k
    record(s, "no source problem appears in two splits", leak == 0, f"{leak} conflicts")
    held = {r["problem_id"] for r in load(os.path.join(run_dir, "holdout.jsonl"))}
    used = sum(r["problem_id"] in held for v in S.values() for r in v)
    record(s, "no routing hold-out problem in the benchmark", used == 0, f"{used} traces")
    for k, rows in S.items():
        n, p = len(rows), len({r["problem_id"] for r in rows})
        t = nums["splits"][k]
        record(s, f"{k}: traces/problems match tab_composition",
               (n, p) == (t["traces"], t["problems"]), f"data {n}/{p}, table {t['traces']}/{t['problems']}")
    allr = [r for v in S.values() for r in v]
    cells = Counter(r["outcome_cell"] for r in allr)
    record(s, "outcome cells match tab_composition", dict(cells) == nums["outcomes"], f"{dict(cells)}")
    excl = load(os.path.join(run_dir, "excluded.jsonl"))
    tr = nums["traces"]
    record(s, "generated = kept + excluded", tr["generated"] == len(allr) + len(excl) and tr["kept"] == len(allr),
           f"kept {len(allr)}, excluded {len(excl)}, table {tr}")
    bad_excl = sum(r.get("r0_parsed_from_final_line", True) and r.get("rf_parsed_from_final_line", True) for r in excl)
    record(s, "every excluded trace really lacks a final-answer line", bad_excl == 0, f"{bad_excl} wrongly excluded")

    # independent recomputation of every binary result row from its prediction file
    test_ids = sorted(r["trace_id"] for r in S["test"])
    label = {r["trace_id"]: r["label"] for r in S["test"]}
    pred_dir = os.path.join(run_dir, "predictions")
    table_rows = nums["binary_results"]
    names = {"majority": "Majority class", "random": "Random (train prior)", "length": "Length heuristic",
             "confidence": "Confidence heuristic (critique log-prob)", "tfidf_lr": "TF-IDF + logistic regression"}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*__binary.jsonl"))):
        rows = load(f)
        m = rows[0]["model"]
        ids = sorted(r["trace_id"] for r in rows)
        record(s, f"{m}: predictions cover exactly the test split", ids == test_ids, f"{len(ids)} rows")
        y = np.array([label[r["trace_id"]] for r in rows])
        record(s, f"{m}: y_true equals the split label", all(r["y_true"] == label[r["trace_id"]] for r in rows), "")
        yp = np.array([r["y_pred"] for r in rows])
        acc = float(np.mean(y == yp))
        # macro-F1 by hand, independent of sklearn
        f1s = []
        for c in (0, 1):
            tp = np.sum((yp == c) & (y == c)); fp = np.sum((yp == c) & (y != c)); fn = np.sum((yp != c) & (y == c))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        mf1 = float(np.mean(f1s))
        p = np.array([r["p_fail"] for r in rows])
        pos, neg = p[y == 1], p[y == 0]
        auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))  # Mann-Whitney
        # every run's accuracy as quoted in the prose (macros.tex), recomputed here
        mac = nums.get("macros", {})
        key = "res" + "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", m) if w) + "Acc"
        if key in mac:
            record(s, f"{m}: prose macro \\{key} matches recomputed accuracy",
                   abs(float(mac[key]) - 100 * acc) < 0.051, f"macro {mac[key]}, recomputed {100 * acc:.1f}")
        t = table_rows.get(names.get(m, m))
        if t is None:
            continue  # encoder/ablation runs are checked through the macros above
        ok = abs(acc - t["accuracy"]) < 5e-4 and abs(mf1 - t["macro_f1"]) < 5e-4 and abs(auc - t.get("auc", auc)) < 5e-4
        record(s, f"{m}: accuracy / macro-F1 / AUC match the table", ok,
               f"recomputed {acc:.4f} / {mf1:.4f} / {auc:.4f}; table {t['accuracy']:.4f} / {t['macro_f1']:.4f} / {t.get('auc', float('nan')):.4f}")

    # routing: recompute every strategy from the per-problem routing log
    rlog = os.path.join(run_dir, "routing", "traces.jsonl")
    rres = os.path.join(run_dir, "results", "routing.json")
    if os.path.exists(rlog) and os.path.exists(rres):
        rows_ = load(rlog)
        R = json.load(open(rres))
        test_ = rows_[R["n_tune"]:]
        tok_int = np.array([r["tok"]["R0"] + r["tok"]["F"] + r["tok"]["Rf"] for r in test_], float)
        unit = tok_int.mean()
        acc_int = np.mean([r["int_correct"] for r in test_]); acc_ext = np.mean([r["ext_correct"] for r in test_])
        S_ = R["strategies"]
        record(s, "routing: Always Intrinsic accuracy recomputed", abs(acc_int - S_["Always Intrinsic"]["accuracy"]) < 1e-9, f"{acc_int:.4f}")
        record(s, "routing: Always Extrinsic accuracy recomputed", abs(acc_ext - S_["Always Extrinsic"]["accuracy"]) < 1e-9, f"{acc_ext:.4f}")
        record(s, "routing: Always Intrinsic cost is 1.0x by definition", abs(S_["Always Intrinsic"]["relative_cost"] - 1.0) < 1e-9, "")
        c_ext = np.mean([(r["tok"]["R0"] + r["tok"]["F"] + r["tok"]["program"]) / unit for r in test_])
        record(s, "routing: Always Extrinsic cost recomputed from token counts", abs(c_ext - S_["Always Extrinsic"]["relative_cost"]) < 1e-9, f"{c_ext:.4f}")
        hum = S_["Always Human (oracle)"]
        for name, st in S_.items():
            ph = st["share"]["hum"]
            lo_c = ph * hum["relative_cost"] + (1 - ph) * min(1.0, S_["Always Extrinsic"]["relative_cost"])
            lo_l = ph * hum["latency_s"] + (1 - ph) * min(S_["Always Intrinsic"]["latency_s"], S_["Always Extrinsic"]["latency_s"])
            record(s, f"routing: {name} satisfies the cost and latency lower bounds",
                   st["relative_cost"] >= lo_c - 0.02 and st["latency_s"] >= lo_l - 2.0,
                   f"cost {st['relative_cost']:.2f} >= {lo_c:.2f}; latency {st['latency_s']:.1f} >= {lo_l:.1f}")
        mac = nums.get("macros", {})
        for key, name in (("routeDynAcc", "Dynamic (our predictor)"), ("routeIntAcc", "Always Intrinsic"), ("routeExtAcc", "Always Extrinsic")):
            if key in mac:
                record(s, f"routing: prose macro \\{key} matches routing.json",
                       abs(float(mac[key]) - 100 * S_[name]["accuracy"]) < 0.051, f"{mac[key]} vs {100 * S_[name]['accuracy']:.1f}")

    # no hand-typed numbers left in generated tables
    for f in sorted(glob.glob(os.path.join(tables_dir, "*.tex"))):
        first = open(f).readline()
        record(s, f"{os.path.basename(f)} is generated", "generated by scfp_bench" in first, first.strip()[:60])



# --------------------------------------------------------------------------
# Part E -- v7 manuscript: natural-set results, recomputed from per-trace predictions
# --------------------------------------------------------------------------
V7_GEN = {"qwen2.5-0.5b": "QA", "qwen2.5-1.5b": "QB", "smollm2-1.7b": "SM", "qwen2.5-3b": "QC", "pooled": "Pool"}
V7_PRED = {"text_tfidf": "Tfidf", "signals_noverifier": "SigNoVer", "signals": "Sig", "signals_gbm": "Gbm",
           "concat_text_signals": "Concat"}
V7_SRC = {"gsm8k": "Gsm", "strategyqa": "Sqa", "logiqa": "Logiqa"}


def _v7_metrics(y, p):
    yp = (p >= .5).astype(int)
    return {"n": len(y), "auc": roc_auc_score(y, p), "acc": float(np.mean(yp == y)),
            "f1": f1_score(y, yp, average="macro"), "maj": max(y.mean(), 1 - y.mean())}


def check_v7(v7_dir, tables_dir, run_dirs, manuscript):
    s = "E  v7 natural-set results (recomputed from predictions.jsonl)"
    N = json.load(open(os.path.join(tables_dir, "numbers_v7.json")))
    R = json.load(open(os.path.join(v7_dir, "results.json")))
    rows = [json.loads(l) for l in open(os.path.join(v7_dir, "predictions.jsonl"))]
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["generator"], r["predictor"])].append(r)

    def arr(key):
        rr = sorted(by[key], key=lambda r: r["trace_id"])
        return np.array([r["y"] for r in rr]), np.array([r["p"] for r in rr]), rr

    def close(macro, value, nd=3, scale=1):
        if macro not in N:
            return False, f"{macro} missing"
        got = float(N[macro].replace(",", "")) if not N[macro][0] in "<=" else None
        want = round(value * scale, nd)
        return got is not None and abs(got - want) <= 10 ** -nd * 0.51, f"macro {N[macro]}, recomputed {want}"

    gens = [g for g in V7_GEN if g != "pooled"]
    # main table: every generator x predictor, plus pooled
    for g in gens + ["pooled"]:
        m = V7_GEN[g]
        if g == "pooled":
            base = {k: (np.concatenate([arr(("natural", x, k))[0] for x in gens]),
                        np.concatenate([arr(("natural", x, k))[1] for x in gens])) for k in V7_PRED}
        else:
            base = {k: arr(("natural", g, k))[:2] for k in V7_PRED}
        ty, tp = base["text_tfidf"]
        for k, pm in V7_PRED.items():
            y, p = base[k]
            mt = _v7_metrics(y, p)
            ok1, d1 = close(f"v{m}{pm}AUC", mt["auc"])
            ok2, d2 = close(f"v{m}{pm}Acc", mt["acc"], 1, 100)
            ok3, d3 = close(f"v{m}{pm}Fone", mt["f1"])
            record(s, f"{g} {k}: AUC, accuracy, macro-F1", ok1 and ok2 and ok3, f"{d1}; {d2}; {d3}")
            lo, hi = float(N[f"v{m}{pm}AUCLo"]), float(N[f"v{m}{pm}AUCHi"])
            record(s, f"{g} {k}: AUC interval contains the estimate", lo <= mt["auc"] <= hi, f"[{lo}, {hi}] vs {mt['auc']:.3f}")
            if k != "text_tfidf":
                _, _, pv = mcnemar_exact((p >= .5) == y, (tp >= .5) == ty)
                got = N[f"v{m}{pm}PTfidf"]
                want = "< 0.001" if pv < 0.001 else f"= {pv:.3f}"
                record(s, f"{g} {k}: McNemar p vs TF-IDF", got == want, f"macro {got}, recomputed {want}")
        ok, d = close(f"v{m}Maj", _v7_metrics(ty, tp)["maj"], 1, 100)
        record(s, f"{g}: majority-class rate", ok, d)

    # prose claims about the main table
    best = {g: max(V7_PRED, key=lambda k: _v7_metrics(*arr(("natural", g, k))[:2])["auc"]) for g in gens}
    named = {"qwen2.5-0.5b": "concat_text_signals", "qwen2.5-1.5b": "signals_gbm",
             "smollm2-1.7b": "concat_text_signals", "qwen2.5-3b": "signals_gbm"}
    record(s, "prose: the predictor named as best per generator has the highest AUC", best == named, str(best))
    bests = {g: float(N[f"v{V7_GEN[g]}{V7_PRED[named[g]]}AUC"]) for g in gens}
    record(s, "prose: best-AUC range runs from SmolLM2 (min) to Qwen2.5-3B (max)",
           min(bests, key=bests.get) == "smollm2-1.7b" and max(bests, key=bests.get) == "qwen2.5-3b", str(bests))
    texts = {g: float(N[f"v{V7_GEN[g]}TfidfAUC"]) for g in gens}
    record(s, "prose: text-only AUC range runs from SmolLM2 (min) to Qwen2.5-3B (max)",
           min(texts, key=texts.get) == "smollm2-1.7b" and max(texts, key=texts.get) == "qwen2.5-3b", str(texts))
    sig = {g: N[f"v{V7_GEN[g]}{V7_PRED[named[g]]}PTfidf"] for g in gens}
    sig_concat = {g: N[f"v{V7_GEN[g]}ConcatPTfidf"] for g in gens}
    is_sig = lambda v: v.startswith("<") or float(v.lstrip("= ")) < 0.05
    record(s, "prose: significantly above text for three of four generators (not Qwen2.5-1.5B)",
           all(is_sig(sig_concat[g]) for g in ("qwen2.5-0.5b", "smollm2-1.7b", "qwen2.5-3b"))
           and not any(is_sig(N[f"vQB{pm}PTfidf"]) for k, pm in V7_PRED.items() if k != "text_tfidf"),
           f"concat p: {sig_concat}; Qwen2.5-1.5B: {[N[f'vQB{pm}PTfidf'] for k, pm in V7_PRED.items() if k != 'text_tfidf']}")
    for g in ("qwen2.5-1.5b", "qwen2.5-3b"):
        above = []
        for k in V7_PRED:
            y, p, _ = arr(("natural", g, k))
            mt = _v7_metrics(y, p)
            maj_pred = np.full(len(y), int(y.mean() >= .5))
            _, _, pv = mcnemar_exact((p >= .5) == y, maj_pred == y)
            if mt["acc"] > mt["maj"] and pv < 0.05:
                above.append(k)
        record(s, f"prose: no predictor significantly above the majority class ({g})", not above, f"significant: {above}")

    # by source
    other = []
    for g in gens:
        for src, sm in V7_SRC.items():
            for k, name in (("text_tfidf", "Tfidf"), ("signals_noverifier", "NoVer"), ("signals", "Ver")):
                y, p, rr = arr(("natural", g, k))
                msk = np.array([r["source"] == src for r in rr])
                a = roc_auc_score(y[msk], p[msk])
                ok, d = close(f"src{V7_GEN[g]}{sm}{name}", a)
                record(s, f"{g} {src} {k}: AUC by source", ok, d)
                if src != "gsm8k":
                    other.append(round(a, 2))
    record(s, "prose: StrategyQA and LogiQA AUC between 0.47 and 0.60 for every feature set",
           min(other) >= 0.47 and max(other) <= 0.60, f"min {min(other)}, max {max(other)}")

    # SCFP v1 vs natural (filter effect)
    for g in ("qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b"):
        y, p, _ = arr(("v1", g, "text_tfidf"))
        ok, d = close(f"filt{V7_GEN[g]}VoneTfidf", roc_auc_score(y, p))
        record(s, f"{g}: SCFP v1 text AUC", ok and int(N[f"filt{V7_GEN[g]}VoneN"]) == len(y), f"{d}; n {len(y)}")

    # outcome statistics, from the trace files themselves
    traces = {}
    for rd in run_dirs:
        for f, kept in (("traces.jsonl", True), ("excluded.jsonl", False)):
            for l in open(os.path.join(rd, "natural", f)):
                t = json.loads(l)
                traces.setdefault(t["source_model"], []).append((kept, t))
    total = 0
    for g in gens:
        m = V7_GEN[g]
        kept = [t for k, t in traces[g] if k]
        total += len(kept)
        r0 = np.array([t["r0_correct"] for t in kept], bool)
        rf = np.array([t["rf_correct"] for t in kept], bool)
        checks = [close(f"nat{m}Gen", len(traces[g]), 0), close(f"nat{m}Kept", len(kept), 0),
                  close(f"nat{m}RzeroAcc", r0.mean(), 1, 100), close(f"nat{m}RfAcc", rf.mean(), 1, 100),
                  close(f"nat{m}Repair", rf[~r0].mean(), 1, 100), close(f"nat{m}Harm", (~rf[r0]).mean(), 1, 100),
                  close(f"nat{m}Fail", 1 - rf.mean(), 1, 100)]
        record(s, f"{g}: natural-set outcome statistics", all(c[0] for c in checks), "; ".join(c[1] for c in checks))
        if g in ("qwen2.5-0.5b", "smollm2-1.7b"):
            record(s, f"prose: self-correction raised final accuracy ({g})", rf.mean() > r0.mean(), f"{r0.mean():.3f} -> {rf.mean():.3f}")
        else:
            record(s, f"prose: self-correction lowered final accuracy ({g})", rf.mean() < r0.mean(), f"{r0.mean():.3f} -> {rf.mean():.3f}")
    ok, d = close("natTotalKept", total, 0)
    record(s, "total natural-set traces", ok, d)
    pooled_n = int(N["vPoolN"].replace(",", ""))
    record(s, "pooled n equals the sum of generators", pooled_n == total, f"{pooled_n} vs {total}")

    # sub-tasks: macros equal results.json, and prose ranges
    for t, mt in (("r0_wrong", "Rzero"), ("changed", "Changed"), ("repair", "Repair"), ("harm", "Harm")):
        vals = {g: R["subtasks"][f"natural|{g}|{t}"]["auc"] for g in gens}
        ok = all(close(f"sub{V7_GEN[g]}{mt}", vals[g])[0] for g in gens)
        record(s, f"sub-task {t}: macros match results.json", ok, str({g: round(v, 3) for g, v in vals.items()}))
    rng = {"Changed": ("QB", "QC"), "Repair": ("QB", "QC"), "Harm": ("QB", "QC"), "Rzero": ("QC", "QB")}
    for mt, (lo, hi) in rng.items():
        v = {m: float(N[f"sub{m}{mt}"]) for m in ("QA", "QB", "SM", "QC")}
        record(s, f"prose: sub-task {mt} range runs from {lo} (min) to {hi} (max)",
               min(v, key=v.get) == lo and max(v, key=v.get) == hi, str(v))

    # every v7 macro used in the manuscript is defined, and every table it inputs exists
    tex = open(manuscript).read()
    used = set(re.findall(r"\\((?:v|nat|src|sub|filt|hid)(?:QA|QB|SM|QC|Pool|Total)[A-Za-z]*)", tex))
    missing = sorted(u for u in used if u not in N)
    record(s, "every v7 macro in the manuscript is generated", not missing, f"{len(used)} used, missing {missing}")
    inputs = re.findall(r"\\input\{(paper_tables/[^}]+)\}", tex)
    absent = [i for i in inputs if not os.path.exists(os.path.join(os.path.dirname(manuscript), i if i.endswith(".tex") else i + ".tex"))]
    record(s, "every \\input table exists", not absent, f"{len(inputs)} inputs, absent {absent}")

    # v8 (stand-alone natural-set paper): extra prose claims, and no trace of SCFP v1
    f = lambda k: float(N[k])
    gain = {m: round(f(f"v{m}SigNoVerAUC") - f(f"v{m}TfidfAUC"), 3) for m in ("QA", "QB", "SM", "QC")}
    record(s, "prose: signals without the verifier add at most 0.04 AUC to text", max(gain.values()) <= 0.04, str(gain))
    qb_ver = max(f("vQBSigAUC"), f("vQBGbmAUC"), f("vQBConcatAUC")) - f("vQBSigNoVerAUC")
    record(s, "prose: for Qwen2.5-1.5B, signals without the verifier give most of the gain over text",
           gain["QB"] > qb_ver, f"no-verifier gain {gain['QB']:.3f}, verifier adds {qb_ver:.3f}")
    pool_nv, pool_v = f("vPoolSigNoVerAUC") - f("vPoolTfidfAUC"), f("vPoolConcatAUC") - f("vPoolSigNoVerAUC")
    record(s, "prose: pooled, most of the gain over text comes with the verifier", pool_v > pool_nv,
           f"no-verifier {pool_nv:.3f}, verifier {pool_v:.3f}")
    hid = {m: (f(f"hid{m}AUC"), f(f"hid{m}MidAUC"), f(f"v{m}SigNoVerAUC")) for m in ("QA", "QB", "SM", "QC")}
    record(s, "prose: no hidden-state probe exceeds the signals without the verifier",
           all(max(a, b) < c for a, b, c in hid.values()), str(hid))
    if "macros_natural" in tex:
        body = tex.split("\\begin{document}", 1)[1]
        hits = re.findall(r"SCFP v1|v1\.0|\\filt[A-Z]|tab_filter|macros_v7|macros\.tex", body)
        record(s, "stand-alone manuscript: no reference to SCFP v1 or its numbers", not hits, f"hits {hits}")


# --------------------------------------------------------------------------
# Part F -- major revision: numbers recomputed here from raw logs, independently of scripts/revision/
# --------------------------------------------------------------------------
def check_revision(rev_tables, routing_dir, manuscript):
    s = "F  revision (recomputed from raw logs)"
    N = json.load(open(os.path.join(rev_tables, "numbers_rev.json")))
    rows = [json.loads(l) for l in open(os.path.join(routing_dir, "traces.jsonl"))]
    ev = rows[50:]
    r0 = np.array([r["r0_correct"] for r in ev], float)
    intr = np.array([r["int_correct"] for r in ev], float)
    ext = np.array([r["ext_correct"] for r in ev], float)
    unit = np.mean([r["tok"]["R0"] + r["tok"]["F"] + r["tok"]["Rf"] for r in ev])
    record(s, "M4: R0-only accuracy", f"{100 * r0.mean():.1f}" == N["rtRzeroAcc"], f"{100 * r0.mean():.1f} vs {N['rtRzeroAcc']}")
    c = np.mean([r["tok"]["R0"] for r in ev]) / unit
    record(s, "M4: R0-only relative cost", f"{c:.2f}" == N["rtRzeroCost"], f"{c:.2f} vs {N['rtRzeroCost']}")
    lat = np.mean([r["lat"]["R0"] for r in ev])
    record(s, "M4: R0-only latency", f"{lat:.1f}" == N["rtRzeroLat"], f"{lat:.1f} vs {N['rtRzeroLat']}")
    for name, other, key in (("intrinsic", intr, "rtRzeroVsIntP"), ("extrinsic", ext, "rtRzeroVsExtP")):
        _, _, p = mcnemar_exact(r0 == 1, other == 1)
        want = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        record(s, f"M4: McNemar R0-only vs {name}", want == N[key], f"{want} vs {N[key]}")
    # AURC of the DeBERTa predictor for keeping the intrinsic answer (ties broken by file order)
    pf = np.array([r["p_fail"] for r in ev])
    wrong = 1 - intr
    o = np.argsort(pf, kind="stable")
    aurc = np.mean(np.cumsum(wrong[o]) / np.arange(1, len(o) + 1))
    record(s, "M4: DeBERTa AURC", f"{aurc:.3f}" == N["rtAurcDeberta"], f"{aurc:.3f} vs {N['rtAurcDeberta']}")
    record(s, "M4: random AURC equals the intrinsic error rate", f"{wrong.mean():.3f}" == N["rtAurcRandom"], "")
    # M5: within-source AUC recomputed with sklearn from the saved out-of-fold predictions
    bench = os.path.dirname(os.path.dirname(routing_dir.rstrip("/")))
    preds = defaultdict(list)
    for l in open(os.path.join(bench, "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["target"] == "rf_wrong":
            preds[(r["set"], r["generator"], r["predictor"])].append(r)

    def wauc(rows, key=lambda r: r["source"]):
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        a_ = [(roc_auc_score([x["y"] for x in v], [x["p"] for x in v]), len(v)) for v in g.values()
              if 0 < sum(x["y"] for x in v) < len(v)]
        return sum(x * n for x, n in a_) / sum(n for _, n in a_)
    codes = {"qwen2.5-0.5b": "QA", "qwen2.5-1.5b": "QB", "smollm2-1.7b": "SM", "qwen2.5-3b": "QC"}
    pc = {"text_tfidf": "Tfidf", "signals_noverifier": "SigNoVer", "signals": "Sig", "signals_gbm": "Gbm",
          "concat_text_signals": "Concat"}
    for g, m in codes.items():
        for k, mk in pc.items():
            v = wauc(preds[("natural", g, k)])
            record(s, f"M5: within-source AUC {g} {k}", f"{v:.3f}" == N[f"w{m}{mk}AUC"], f"{v:.3f} vs {N[f'w{m}{mk}AUC']}")
    for k, mk in pc.items():
        rows = [r for g in codes for r in preds[("natural", g, k)]]
        v = wauc(rows, key=lambda r: (r["source"], r["generator"]))
        record(s, f"M5: pooled within-source AUC {k}", f"{v:.3f}" == N[f"wPool{mk}AUC"], f"{v:.3f} vs {N[f'wPool{mk}AUC']}")
    v = wauc(preds[("v1", "qwen2.5-0.5b", "text_tfidf")])
    record(s, "M5: SCFP v1.0 within-source text AUC (filter model)", f"{v:.3f}" == N["fwQATfidfVone"], f"{v:.3f}")
    for g, m in codes.items():
        best = max(pc, key=lambda k: wauc(preds[("natural", g, k)]))
        lo = float(N[f"w{m}{pc[best]}DiffLo"])
        record(s, f"M5: best within-source predictor beats text, CI excludes 0 ({g})", lo > 0, f"{best}, CI low {lo}")
    sub = defaultdict(list)
    for l in open(os.path.join(os.path.dirname(rev_tables), "..", "results", "revision", "m5", "subtask_predictions.jsonl")):
        r = json.loads(l)
        sub[(r["generator"], r["target"])].append(r)
    for g, m in codes.items():
        v = wauc(sub[(g, "r0_wrong")])
        record(s, f"M5: sub-task r0_wrong within-source AUC ({g})", f"{v:.3f}" == N[f"sw{m}Rzero"], f"{v:.3f}")
    rule = sum(all(wauc(sub[(g, t)]) > wauc(sub[(g, "r0_wrong")]) for t in ("changed", "repair", "harm")) for g in codes)
    record(s, "M5: behaviour > R0 correctness holds for the stated number of generators", str(rule) == N["swRuleHolds"], str(rule))
    # M7: claims in the power paragraph and calibration sentence
    st = json.load(open(os.path.join(os.path.dirname(rev_tables), "..", "results", "revision", "m7", "stats_m7.json")))
    P_ = st["scfp"]["predictors"]
    enc = ["deberta-scfp", "deberta-plain", "roberta-base", "bert-base", "abl-no-segments", "abl-no-attention",
           "abl-inputs-P-R0", "abl-inputs-P-F", "abl-inputs-R0-F", "abl-deberta-xsmall"]
    accs = [P_[k]["accuracy"] for k in enc + ["tfidf_lr"]]
    aucs = [P_[k]["auc"] for k in enc + ["tfidf_lr"]]
    record(s, "M7: every encoder/ablation/TF-IDF accuracy difference is below the accuracy MDE",
           max(accs) - min(accs) < st["power"]["mde_accuracy_paired"], f"max diff {max(accs) - min(accs):.3f}")
    record(s, "M7: every encoder/ablation/TF-IDF AUC difference is below the paired-AUC MDE",
           max(aucs) - min(aucs) < st["power"]["mde_auc_paired"], f"max diff {max(aucs) - min(aucs):.3f}")
    cal = st["calibration"]
    record(s, "M7: mean p exceeds the failure rate for the Qwen2.5-1.5B and Qwen2.5-3B best predictors",
           all(v["mean_p"] > v["failure_rate"] for k, v in cal.items()
               if k.startswith("natural|qwen2.5-1.5b") or k.startswith("natural|qwen2.5-3b")), "")
    record(s, "M7: R0-only vs extrinsic not significant after Holm",
           [x for x in st["routing"] if x["policy"] == "Always Extrinsic"][0]["p_holm"] >= 0.05, "")
    # M6: the self-consistency samples (raw call logs) and the claims made from pot_vs_sc.json
    rev = os.path.join(os.path.dirname(rev_tables), "..", "results", "revision")
    m6 = json.load(open(os.path.join(rev, "m6", "pot_vs_sc.json")))
    pinned = {"qwen2.5-0.5b": "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2.5-1.5b": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
              "smollm2-1.7b": "31b70e2e869a7173562077fd711b654946d38674", "qwen2.5-3b": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}
    for g, m in codes.items():
        calls = [json.loads(l) for l in open(os.path.join(rev, "m6", "gen", f"calls_{g}.jsonl"))]
        ok = (len(calls) == 3 * m6[g]["n"] and all(c["ok"] and c["temperature"] == 0.7 and c["max_tokens"] == 512
                                                    and c["model_version"].endswith("@" + pinned[g]) for c in calls)
              and len({(c["messages"][0]["content"], c["sample"]) for c in calls}) == len(calls))
        record(s, f"M6: {g} has 3 logged samples per GSM8K trace at T=0.7 with the pinned commit", ok,
               f"{len(calls)} calls, n={m6[g]['n']}")
        for k, mk in (("pot - sc1", "PotSc"), ("sc1_pot - sc1", "ScPotSc"), ("sc3_pot - sc3", "ScThPotScTh")):
            c = m6[g]["compare"][k]
            record(s, f"M6: {g} {k} macro", N[f"sc{m}{mk}"] == f"{c['diff']:+.3f}" and N[f"sc{m}{mk}Lo"] == f"{c['ci95'][0]:.3f}",
                   f"{N[f'sc{m}{mk}']} vs {c['diff']:+.3f}")
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
    for k, mk in (("sc1_pot - sc1", "scRuleScPot"), ("pot - sc1", "scRulePotSc"), ("sc3_pot - sc3", "scRuleScThPot")):
        n = sum(m6[g]["compare"][k]["ci95"][0] > 0 for g in codes)
        record(s, f"M6: decision-rule count for {k}", N[mk] == words[n], f"{N[mk]} vs {n}")
    record(s, "M6: PoT is significantly worse than SC1 for no generator",
           all(m6[g]["compare"]["pot - sc1"]["ci95"][1] > 0 for g in codes), "")
    q = m6["qwen2.5-1.5b"]["compare"]["sc1_pot - sc1"]["ci95"]
    record(s, "M6: PoT adds nothing measurable beyond SC1 for Qwen2.5-1.5B (CI includes 0)", q[0] <= 0 <= q[1], "")
    # M7 seeds: recompute every seed's test accuracy and AUC from the prediction files (sklearn), then the macros
    run_v1 = os.path.join(bench, "laptop_v1")
    sd_dir = os.path.join(rev, "m7", "seeds")
    order = sorted(json.loads(l)["trace_id"] for l in open(os.path.join(run_v1, "splits", "test.jsonl")))

    def seed_metrics(path):
        rr = {r["trace_id"]: r for r in (json.loads(l) for l in open(path))}
        y = [rr[t]["y_true"] for t in order]
        return float(np.mean([rr[t]["y_pred"] == rr[t]["y_true"] for t in order])), roc_auc_score(y, [rr[t]["p_fail"] for t in order])
    tf_acc, tf_auc = seed_metrics(os.path.join(run_v1, "predictions", "tfidf_lr__binary.jsonl"))
    sdcode = {"deberta-scfp": "DebScfp", "deberta-plain": "DebPlain", "roberta-base": "Roberta", "bert-base": "Bert",
              "abl-no-segments": "AblNoSeg", "abl-no-attention": "AblNoAtt", "abl-inputs-P-R0": "AblPR", "abl-inputs-P-F": "AblPF",
              "abl-inputs-R0-F": "AblRF", "abl-deberta-xsmall": "AblXs"}
    per = {}
    for k, mk in sdcode.items():
        ms = [seed_metrics(os.path.join(run_v1, "predictions", f"{k}__binary.jsonl"))]
        ms += [seed_metrics(os.path.join(sd_dir, "predictions", f"{k}__seed{x}__binary.jsonl")) for x in (1, 2, 3, 4)]
        per[k] = np.array(ms)
        acc, au = per[k][:, 0], per[k][:, 1]
        ok = (N[f"sd{mk}AccMean"] == f"{acc.mean():.3f}" and N[f"sd{mk}AccSd"] == f"{acc.std(ddof=1):.3f}"
              and N[f"sd{mk}AUCMean"] == f"{au.mean():.3f}" and N[f"sd{mk}AUCSd"] == f"{au.std(ddof=1):.3f}")
        record(s, f"M7 seeds: {k} 5-seed mean/sd of accuracy and AUC", ok,
               f"acc {acc.mean():.3f}±{acc.std(ddof=1):.3f} auc {au.mean():.3f}±{au.std(ddof=1):.3f}")
    record(s, "M7 seeds: our model's accuracy is below TF-IDF for all five seeds", bool(np.all(per["deberta-scfp"][:, 0] < tf_acc)), "")
    record(s, "M7 seeds: our model's AUC above TF-IDF for the stated number of seeds",
           str(int(np.sum(per["deberta-scfp"][:, 1] > tf_auc))) == N["sdDebScfpAucAboveTfidf"], "")
    above = {k: int(np.sum(per[k][:, 1] > per["deberta-scfp"][:, 1])) for k in sdcode if k.startswith("abl-")}
    record(s, "M7 seeds: no-segments AUC above full for 5/5; other ablations for 2 or 3 of 5",
           above["abl-no-segments"] == 5 and all(v in (2, 3) for k, v in above.items() if k != "abl-no-segments"), str(above))
    record(s, "M7 seeds: every ablation's Holm-adjusted p vs full is 1.000",
           all(N[f"sd{sdcode[k]}PFullHolm"] == "1.000" for k in above), "")
    sds = [per[k][:, 0].std(ddof=1) for k in sdcode]
    record(s, "M7 seeds: accuracy sd range", f"{min(sds):.3f}" == N["sdAccSdMin"] and f"{max(sds):.3f}" == N["sdAccSdMax"], "")
    # M9: stacked vs signals within-source AUC recomputed from the saved predictions; signals match the paper's
    m9 = json.load(open(os.path.join(rev, "m9", "m9.json")))
    m9p = defaultdict(list)
    for l in open(os.path.join(rev, "m9", "m9_predictions.jsonl")):
        r = json.loads(l)
        m9p[r["generator"]].append(r)
    for g, m in codes.items():
        rr = m9p[g]
        a_s = wauc([{"source": r["source"], "y": r["y"], "p": r["signals"]} for r in rr])
        a_k = wauc([{"source": r["source"], "y": r["y"], "p": r["stacked"]} for r in rr])
        saved = {r["trace_id"]: r["p"] for r in preds[("natural", g, "signals")]}
        same = max(abs(saved[r["trace_id"]] - r["signals"]) for r in rr) < 1e-9
        record(s, f"M9: {g} signals predictions equal the paper's saved ones", same, "")
        record(s, f"M9: {g} stacked and signals within-source AUC", f"{a_k:.3f}" == N[f"hp{m}Stk"] and f"{a_s:.3f}" == N[f"hp{m}Sig"],
               f"{a_k:.3f}/{a_s:.3f} vs {N[f'hp{m}Stk']}/{N[f'hp{m}Sig']}")
        record(s, f"M9: {g} layers chosen per fold are all valid", all(0 <= x < m9[g]["layers"] for x in m9[g]["layers_chosen_per_fold"]), "")
    n9 = sum(m9[g]["stacked_minus_signals_ci95"][0] > 0 for g in codes)
    record(s, "M9: decision-rule count", words[n9] == N["hpRule"], f"{n9}")
    record(s, "M9: probe alone below the signals for every generator",
           all(m9[g]["best_layer_auc"] < m9[g]["signals_within_auc"] for g in codes), "")
    record(s, "M9: best initial-wrong layer AUC above best final-wrong layer AUC for every generator",
           all(max(m9[g]["per_layer_within_auc"]["r0_wrong"]) > m9[g]["best_layer_auc"] for g in codes), "")
    # M3: Delta-label, gate and baselines recomputed with sklearn from the saved predictions and raw GPU outputs
    m3p = defaultdict(list)
    for l in open(os.path.join(rev, "m3", "m3_cpu_predictions.jsonl")):
        r = json.loads(l)
        m3p[r["generator"]].append(r)
    cls = ["helped", "neutral", "harmed"]
    for g, m in codes.items():
        rr = m3p[g]
        mac = np.mean([wauc([{"source": r["source"], "y": int(r["y_delta"] == c), "p": r["p_delta"][j]} for r in rr])
                       for j, c in enumerate(cls)])
        record(s, f"M3: {g} Delta-label macro within-source AUC", f"{mac:.3f}" == N[f"dl{m}Macro"], f"{mac:.3f} vs {N[f'dl{m}Macro']}")
        dg = wauc([{"source": r["source"], "y": r["y_rf"], "p": r["p_gate"]} for r in rr]) - \
            wauc([{"source": r["source"], "y": r["y_rf"], "p": r["p_direct"]} for r in rr])
        record(s, f"M3: {g} gate minus direct", f"{dg:+.3f}" == N[f"dl{m}GateDiff"] or (abs(dg) < 5e-4 and N[f"dl{m}GateDiff"] in ("-0.000", "+0.000")),
               f"{dg:+.3f} vs {N[f'dl{m}GateDiff']}")
        direct = wauc([{"source": r["source"], "y": r["y_rf"], "p": r["p_direct"]} for r in rr])
        record(s, f"M3: {g} Delta-label macro AUC above the final-outcome AUC", mac > direct, "")
        pt = {json.loads(l)["trace_id"]: json.loads(l)["p_true"] for l in open(os.path.join(rev, "gpuA", f"ptrue_{g}.jsonl"))}
        a_pt = wauc([{"source": r["source"], "y": r["y_rf"], "p": 1 - pt[r["trace_id"]]} for r in rr])
        record(s, f"M3: {g} P(True) within-source AUC from the raw scores", f"{a_pt:.3f}" == N[f"cb{m}PTrueRf"], f"{a_pt:.3f} vs {N[f'cb{m}PTrueRf']}")
    calls = [json.loads(l) for l in open(os.path.join(rev, "gpuA", "calls_verbal_smollm2-1.7b.jsonl"))]
    bad = sum(1 for c in calls if not re.search(r"\d+", c["text"]) or int(re.search(r"\d+", c["text"]).group()) > 100)
    record(s, "M3: SmolLM2 unparsable verbalized confidences", f"{bad:,}" == N["cbVerbalUnparsedSM"], f"{bad}")
    # M8 exclusion sensitivity: counts recomputed from the raw excluded files; claims checked against the JSON
    ex8 = json.load(open(os.path.join(rev, "m8", "exclusion_sensitivity.json")))
    v1ex = [json.loads(l) for l in open(os.path.join(bench, "laptop_v1", "excluded.jsonl"))]
    test_p = {json.loads(l)["problem_id"] for l in open(os.path.join(bench, "laptop_v1", "splits", "test.jsonl"))}
    n_te = sum(r["problem_id"] in test_p for r in v1ex)
    record(s, "M8: SCFP test split with exclusions has 108 + excluded test traces", ex8["scfp"]["with_exclusions"]["n"] == 108 + n_te
           and N["exScfpWithN"] == str(108 + n_te), f"{108 + n_te}")
    record(s, "M8: TF-IDF not significantly above majority under either exclusion rule",
           ex8["scfp"]["original"]["p_vs_majority"] >= 0.05 and ex8["scfp"]["with_exclusions"]["p_vs_majority"] >= 0.05, "")
    record(s, "M8: signals beat text within source for every generator with exclusions counted as failures",
           all(ex8["natural"][g]["signals_minus_text_with_exclusions_ci95"][0] > 0 for g in codes) and N["exSigTextAll"] == "all four", "")
    for g, m in codes.items():
        run = "signals_mac_q05_big" if g == "qwen2.5-0.5b" else "signals_kaggle"
        ne = sum(1 for l in open(os.path.join(bench, run, "natural", "excluded.jsonl")) if json.loads(l)["source_model"] == g)
        record(s, f"M8: {g} excluded natural traces", ex8["natural"][g]["excluded"] == ne, f"{ne}")
    # M8 job C: outcomes recomputed from the raw labelled traces and call logs
    g8 = os.path.join(rev, "m8", "gpu")
    q7 = [json.loads(l) for l in open(os.path.join(g8, "q7b", "traces_all.jsonl"))]
    k7 = [r for r in q7 if r["kept"]]
    record(s, "M8: Qwen2.5-7B kept traces and failure rate", N["qsKept"] == str(len(k7)) and
           N["qsFail"] == f"{100 * np.mean([not r['rf_correct'] for r in k7]):.1f}", f"{len(k7)}")
    c7 = [json.loads(l) for l in open(os.path.join(g8, "q7b", "calls.jsonl"))]
    record(s, "M8: every Qwen2.5-7B call is ok and from one pinned commit", all(c["ok"] for c in c7) and
           len({c["model_version"] for c in c7}) == 1, str({c["model_version"] for c in c7}))
    sub8 = json.load(open(os.path.join(rev, "m8", "subset.json")))
    record(s, "M8: Qwen2.5-7B ran on exactly the 600 subset problems",
           {r["problem_id"] for r in q7} == {p for v in sub8["problems"].values() for p in v}, "")
    for gg, mm in (("qwen2.5-3b", "QC"), ("smollm2-1.7b", "SM")):
        alt = [json.loads(l) for l in open(os.path.join(g8, f"alt_{gg}", "traces_all.jsonl"))]
        record(s, f"M8: {gg} alternative-template failure rate", len(alt) == 600 and
               N[f"pa{mm}FailA"] == f"{100 * np.mean([not r['rf_correct'] for r in alt]):.1f}", "")
    pc = {dt: [json.loads(l) for l in open(os.path.join(g8, "precision", f"calls_{dt}.jsonl"))] for dt in ("float16", "bfloat16")}
    tb = {c["messages"][0]["content"]: c["text"] for c in pc["bfloat16"]}
    ntd = sum(c["text"] != tb[c["messages"][0]["content"]] for c in pc["float16"])
    record(s, "M8: float16 vs bfloat16 texts that differ", N["prTextDiff"] == str(ntd) and N["prN"] == str(len(pc["float16"])), f"{ntd}")
    # M3 job B: raw scores recomputed; the claims about the reward model and the judge
    m3g = json.load(open(os.path.join(rev, "m3", "m3_gpu.json")))
    rmv = [json.loads(l) for l in open(os.path.join(rev, "gpuB", "reward_model.jsonl"))]
    jdv = [json.loads(l) for l in open(os.path.join(rev, "gpuB", "judge.jsonl"))]
    record(s, "M3: reward model and judge scored all 6,824 traces, one commit each",
           len(rmv) == len(jdv) == 6824 and len({r["model_version"] for r in rmv}) == 1 and len({r["model_version"] for r in jdv}) == 1, "")
    rm_ = {(r["generator"], r["trace_id"]): -r["reward"] for r in rmv}
    a_rm = wauc([{"source": r["source"], "y": r["y_rf"], "p": rm_[("smollm2-1.7b", r["trace_id"])]} for r in m3p["smollm2-1.7b"]])
    record(s, "M3: SmolLM2 reward-model within-source AUC from the raw rewards", f"{a_rm:.3f}" == N["cbSMRewardRf"], f"{a_rm:.3f}")
    B_ = m3g["by"]
    record(s, "M3: reward model above the signals for SmolLM2 only",
           [g for g in codes if B_[g]["rf_wrong"]["reward"]["diff_vs_signals_ci95"][0] > 0] == ["smollm2-1.7b"], "")
    record(s, "M3: judge n.s. vs signals for three generators and pooled, below for Qwen2.5-3B only",
           [g for g in list(codes) + ["pooled"] if B_[g]["rf_wrong"]["judge"]["diff_vs_signals_ci95"][1] < 0] == ["qwen2.5-3b"]
           and not any(B_[g]["rf_wrong"]["judge"]["diff_vs_signals_ci95"][0] > 0 for g in list(codes) + ["pooled"]), "")
    record(s, "M3: judge beats text (Holm) for all but Qwen2.5-3B; reward model only for SmolLM2 and pooled",
           [g for g in list(codes) + ["pooled"] if B_[g]["rf_wrong"]["judge"]["p_vs_text_holm"] >= 0.05] == ["qwen2.5-3b"]
           and [g for g in list(codes) + ["pooled"] if B_[g]["rf_wrong"]["reward"]["p_vs_text_holm"] < 0.05
                and B_[g]["rf_wrong"]["reward"]["diff_vs_text"] > 0] == ["smollm2-1.7b", "pooled"], "")
    record(s, "M3: pooled reward model above the signals for a wrong initial answer",
           B_["pooled"]["r0_wrong"]["reward"]["diff_vs_signals_ci95"][0] > 0, "")
    # M1: prose numbers about SCFP v1.0, recomputed from the released labelled traces
    lab1 = [json.loads(l) for l in open(os.path.join(bench, "laptop_v1", "labeled.jsonl"))]
    r0w = [r for r in lab1 if not r["r0_correct"]]
    r0r = [r for r in lab1 if r["r0_correct"]]
    ok = (N["psRzeroWrong"] == str(len(r0w)) and N["psRepaired"] == str(sum(r["rf_correct"] for r in r0w))
          and N["psRzeroRight"] == str(len(r0r)) and N["psBroken"] == str(sum(not r["rf_correct"] for r in r0r))
          and N["psFailPct"] == f"{100 * np.mean([not r['rf_correct'] for r in lab1]):.1f}"
          and N["psExcl"] == str(len(v1ex)) and N["psGen"] == str(len(v1ex) + len(lab1)))
    record(s, "M1: SCFP v1.0 outcome counts and exclusion numbers in the prose", ok, "")
    te1 = [json.loads(l) for l in open(os.path.join(bench, "laptop_v1", "splits", "test.jsonl"))]
    record(s, "M1: SCFP test-split size and failure rate", N["psTestN"] == str(len(te1)) and
           N["psTestFailPct"] == f"{100 * np.mean([r['label'] for r in te1]):.1f}", "")
    tex = open(manuscript).read()
    used = set(re.findall(r"\\((?:rt|w|fw|sw|sc|sd|hp|dl|cb|ex|qs|pa|pr|ps|src|pw|nt|cal)(?:[A-Z])[A-Za-z]*)", tex)) - {"wd"}
    gen = set(N) | set(json.load(open(os.path.join(rev_tables, "numbers_v7.json"))))
    record(s, "every revision macro used in the manuscript is generated", used <= gen, f"missing {sorted(used - gen)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="path to repo/SCFP")
    ap.add_argument("--predictions", nargs="*", default=[])
    ap.add_argument("--audit-v2", action="store_true",
                    help="re-run Parts A-C: the audit of the original v2 manuscript and repo data")
    ap.add_argument("--run-dir", default=None, help="v5: benchmark run directory (Part D)")
    ap.add_argument("--tables", default=None, help="v5: paper_tables directory with numbers.json")
    ap.add_argument("--v7-dir", default=None, help="v7: scfp_runs/paper_v7 (results.json, predictions.jsonl)")
    ap.add_argument("--v7-runs", nargs="*", default=[], help="v7: natural-set run directories (traces)")
    ap.add_argument("--manuscript", default=None, help="v7: the .tex file whose macros are checked")
    ap.add_argument("--rev-tables", default=None, help="revision: paper_tables with numbers_rev.json (Part F)")
    ap.add_argument("--routing-dir", default=None, help="revision: laptop_v1/routing")
    args = ap.parse_args()

    if args.audit_v2 or not args.run_dir:
        check_confusion_matrix()
        check_routing()
        check_information_theory()
        check_intervals()
        verify_predictions(args.predictions)
        if args.repo:
            check_dataset(args.repo)
    if args.run_dir:
        check_v5(args.run_dir, args.tables)
        preds = sorted(glob.glob(os.path.join(args.run_dir, "predictions", "*.jsonl")))
        verify_predictions(preds)
    if args.v7_dir:
        check_v7(args.v7_dir, args.tables, args.v7_runs, args.manuscript)
    if args.rev_tables:
        check_revision(args.rev_tables, args.routing_dir, args.manuscript)

    width = max(len(c) for _, c, _, _ in RESULTS)
    section = None
    for sec, check, verdict, detail in RESULTS:
        if sec != section:
            print(f"\n=== {sec}")
            section = sec
        print(f"[{verdict:^14}] {check.ljust(width)}\n                 {detail}")
    tally = Counter(v for _, _, v, _ in RESULTS)
    print(f"\nSummary: {dict(tally)}")
    sys.exit(1 if tally["FAIL"] else 0)


if __name__ == "__main__":
    main()
