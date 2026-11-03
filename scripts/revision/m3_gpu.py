"""M3 (GPU part): correctness-prediction baselines on the natural set, scored from the job-A and job-B outputs.
P(True) and verbalized confidence from each generator (job A); a reward model and an LLM judge (job B, used when
present). Within-source AUC, stratified AUC, AUPRC and Brier with cluster-bootstrap CIs, per generator and pooled,
for rf_wrong and r0_wrong; tests against text and against the generator signals. See REVISION_PLAN.md (M3).

    .venv/bin/python scripts/revision/m3_gpu.py --gpua results/revision/gpuA --gpub results/revision/gpuB --out results/revision/m3
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
from gpu_generator_signals import VERBAL, traces  # noqa: E402
from m5_deconfound import ap as auprc  # noqa: E402
from m5_deconfound import within  # noqa: E402
from m7_stats import holm  # noqa: E402

GENS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b", "qwen2.5-3b"]
B, B_P, SEED = 1000, 10000, 0
INT = re.compile(r"\d+")


def load(gpua, gpub, res_dir):
    """One row per kept natural-set trace with labels, reference predictions and every available score."""
    rows = {}
    for l in open(os.path.join(res_dir, "m3", "m3_cpu_predictions.jsonl")):
        r = json.loads(l)
        rows[(r["generator"], r["trace_id"])] = {**r, "signals_rf": r["p_direct"], "signals_r0": 1 - r["p_r0_correct"]}
    for l in open(os.path.join(BENCH, "scfp_runs", "paper_v7", "predictions.jsonl")):
        r = json.loads(l)
        if r["set"] == "natural" and r["target"] == "rf_wrong" and r["predictor"] in ("text_tfidf", "signals"):
            k = (r["generator"], r["trace_id"])
            rows[k][r["predictor"] + "_saved"] = r["p"]
    n_unparsed, mass = {}, {}
    for g in GENS:
        pt = {json.loads(l)["trace_id"]: json.loads(l) for l in open(os.path.join(gpua, f"ptrue_{g}.jsonl"))}
        mass[g] = float(np.mean([v["ab_mass"] for v in pt.values()]))
        by_msg = {}
        for l in open(os.path.join(gpua, f"calls_verbal_{g}.jsonl")):
            c = json.loads(l)
            if c.get("stage") == "verbal" and c.get("ok"):
                by_msg[c["messages"][0]["content"]] = c["text"]
        miss = 0
        for t in traces(g):
            r = rows[(g, t["trace_id"])]
            r["ptrue"] = 1 - pt[t["trace_id"]]["p_true"]
            txt = by_msg[VERBAL.format(q=t["prompt_raw"], a=t["initial_response"])]
            m = INT.search(txt)
            v = int(m.group()) if m else None
            if v is None or v > 100:
                miss += 1
                r["verbal"] = np.nan
            else:
                r["verbal"] = 1 - v / 100
        n_unparsed[g] = miss
    have = ["ptrue", "verbal"]
    for name, f, key, tf in (("reward", "reward_model.jsonl", "reward", lambda x: -x), ("judge", "judge.jsonl", "p_wrong", lambda x: x)):
        p = os.path.join(gpub, f)
        if os.path.exists(p) and sum(1 for _ in open(p)) == len(rows):
            for l in open(p):
                v = json.loads(l)
                rows[(v["generator"], v["trace_id"])][name] = tf(v[key])
            have.append(name)
    return list(rows.values()), have, n_unparsed, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpua", default="results/revision/gpuA")
    ap.add_argument("--gpub", default="results/revision/gpuB")
    ap.add_argument("--out", default="results/revision/m3")
    a = ap.parse_args()
    res_dir = os.path.dirname(os.path.abspath(a.out))
    rows, have, n_unparsed, mass = load(os.path.abspath(a.gpua), os.path.abspath(a.gpub), res_dir)
    # the saved paper predictions and the M3 reference must agree
    assert max(abs(r["signals_rf"] - r["signals_saved"]) for r in rows) < 1e-9
    prob = {"ptrue", "verbal", "judge"}
    res = {"scores": have, "verbal_unparsed": n_unparsed, "ptrue_ab_mass": mass, "by": {}}
    tests = []
    rng = np.random.default_rng(SEED)
    for g in GENS + ["pooled"]:
        R = [r for r in rows if g == "pooled" or r["generator"] == g]
        strata = np.array([r["source"] + ("|" + r["generator"] if g == "pooled" else "") for r in R])
        grp = np.array([r["problem_id"] for r in R])
        cl = defaultdict(list)
        for i, x in enumerate(grp):
            cl[x].append(i)
        cl = [np.array(v) for v in cl.values()]
        draws = lambda n: [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(n)]
        D, DP = draws(B), draws(B_P)
        Y = {"rf_wrong": np.array([r["y_rf"] for r in R]), "r0_wrong": np.array([r["y_r0"] for r in R])}
        ref = {"rf_wrong": {"text": np.array([r["text_tfidf_saved"] for r in R]), "signals": np.array([r["signals_rf"] for r in R])},
               "r0_wrong": {"signals": np.array([r["signals_r0"] for r in R])}}
        out = {}
        for tgt, y in Y.items():
            out[tgt] = {}
            for s in have + list(ref[tgt]):
                p = ref[tgt][s] if s in ref[tgt] else np.array([r[s] for r in R], float)
                ok = ~np.isnan(p)
                w = lambda i: within(y[i], p[i], strata[i])[0]
                iok = np.where(ok)[0]
                v = {"n": int(ok.sum()), "within_auc": within(y[ok], p[ok], strata[ok])[0],
                     "stratified_auc": within(y[ok], p[ok], strata[ok])[1], "auprc": float(auprc(y[ok], p[ok]))}
                v["within_auc_ci95"] = list(np.percentile([w(i[ok[i]]) for i in D], [2.5, 97.5]))
                if s in prob:
                    v["brier"] = float(np.mean((p[ok] - y[ok]) ** 2))
                if s not in ref[tgt]:
                    for rn, rp in ref[tgt].items():
                        dd = lambda i: within(y[i], p[i], strata[i])[0] - within(y[i], rp[i], strata[i])[0]
                        d = np.array([dd(i[ok[i]]) for i in D])
                        v[f"diff_vs_{rn}"] = float(dd(iok))
                        v[f"diff_vs_{rn}_ci95"] = list(np.percentile(d, [2.5, 97.5]))
                        if tgt == "rf_wrong" and rn == "text":
                            dp = np.array([dd(i[ok[i]]) for i in DP])
                            v["p_vs_text"] = float(min(1.0, 2 * min(np.mean(dp <= 0), np.mean(dp >= 0))))
                            tests.append((g, s, v))
                out[tgt][s] = v
        res["by"][g] = out
        print(g, {t: {s: round(v["within_auc"], 3) for s, v in o.items()} for t, o in out.items()}, flush=True)
    for (g, s, v), h in zip(tests, holm([v["p_vs_text"] for _, _, v in tests])):
        v["p_vs_text_holm"] = h
    # decision rule: baselines whose within-source AUC CI against the signals excludes 0
    res["rule"] = {s: {g: ("above" if res["by"][g]["rf_wrong"][s]["diff_vs_signals_ci95"][0] > 0 else
                           "below" if res["by"][g]["rf_wrong"][s]["diff_vs_signals_ci95"][1] < 0 else "n.s.")
                       for g in GENS + ["pooled"]} for s in have}
    print("vs signals (rf_wrong):", res["rule"])
    json.dump(res, open(os.path.join(a.out, "m3_gpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
