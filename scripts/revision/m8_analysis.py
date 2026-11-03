"""M8 analysis of GPU job C: (1) Qwen2.5-7B (NF4) on the 600-problem subset, (2) the alternative critique/revision
template for Qwen2.5-3B and SmolLM2-1.7B, (5) the float16 vs bfloat16 check for Qwen2.5-1.5B. See REVISION_PLAN.md (M8).

    .venv/bin/python scripts/revision/m8_analysis.py --gpu results/revision/m8/gpu --out results/revision/m8
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from m5_deconfound import within  # noqa: E402
from m6_sc_generate import MODELS  # noqa: E402
from scfp_bench.answers import extract, is_correct  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import oof  # noqa: E402
from scfp_bench.experiments.signals import ATYPE, VERIFIER, grouped_folds  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

B, SEED = 1000, 0


def draws(groups, rng):
    cl = defaultdict(list)
    for i, g in enumerate(groups):
        cl[g].append(i)
    cl = [np.array(v) for v in cl.values()]
    return [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(B)]


def outcomes(rows):
    r0 = np.array([r["r0_correct"] for r in rows], bool)
    rf = np.array([r["rf_correct"] for r in rows], bool)
    return {"n": len(rows), "r0_acc": float(r0.mean()), "rf_acc": float(rf.mean()), "failure_rate": float(1 - rf.mean()),
            "repair": float(rf[~r0].mean()) if (~r0).any() else None, "harm": float((~rf[r0]).mean()) if r0.any() else None}


def models(rows, rng):
    """Text, signals without and with the verifier (LR): within-source AUC, 5-fold CV grouped by problem."""
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([r["source"] for r in rows])
    folds = grouped_folds(rows)
    text = [" [SEP] ".join([r["prompt"], r["initial_response"], r["critique"]]) for r in rows]
    P = {"text_tfidf": oof("text", {"text": text, "scalar": None, "hidden": None}, y, folds),
         "signals_noverifier": oof("scalar", {"text": None, "hidden": None,
                                              "scalar": scalar_matrix([{k: v for k, v in r.items() if k not in VERIFIER} for r in rows])}, y, folds),
         "signals": oof("scalar", {"text": None, "hidden": None, "scalar": scalar_matrix(rows)}, y, folds)}
    D = draws([r["problem_id"] for r in rows], rng)
    out = {}
    for k, p in P.items():
        out[k] = {"within_auc": within(y, p, s)[0], "ci95": list(np.percentile([within(y[i], p[i], s[i])[0] for i in D], [2.5, 97.5]))}
    d = [within(y[i], P["signals"][i], s[i])[0] - within(y[i], P["text_tfidf"][i], s[i])[0] for i in D]
    out["signals_minus_text"] = {"diff": out["signals"]["within_auc"] - out["text_tfidf"]["within_auc"],
                                 "ci95": list(np.percentile(d, [2.5, 97.5]))}
    return out


def feature_rows(traces, feats, ver):
    rows = []
    for t in traces:
        f = {k: (np.nan if v is None else v) for k, v in feats[t["trace_id"]].items()}
        v = {k: ver.get(t["trace_id"], {}).get(k, np.nan) for k in VERIFIER}
        v = {k: (np.nan if x is None else x) for k, x in v.items()}
        rows.append({**t, **f, **v})
    return rows


def q7b(gpu, rng):
    d = os.path.join(gpu, "q7b")
    tr = read_jsonl(os.path.join(d, "traces_all.jsonl"))
    kept = [t for t in tr if t["kept"]]
    feats = {r["trace_id"]: r for r in read_jsonl(os.path.join(d, "features.jsonl"))}
    ver = {r["trace_id"]: r for r in read_jsonl(os.path.join(d, "verifier.jsonl"))}
    rows = feature_rows(kept, feats, ver)
    out = {"generated": len(tr), "kept": len(kept), "outcomes": outcomes(kept),
           "outcomes_by_source": {s: outcomes([r for r in kept if r["source"] == s]) for s in ("gsm8k", "strategyqa", "logiqa")},
           "models": models(rows, rng), "versions": sorted({v for t in tr for v in t.get("model_versions", [])})}
    return out


def prompt_alt(gpu, rng, gens=("qwen2.5-3b", "smollm2-1.7b")):
    sub = json.load(open(os.path.join(gpu, "..", "subset.json")))
    ids = {p for v in sub["problems"].values() for p in v}
    res = {}
    for g in gens:
        run = MODELS[g][2]
        nat = os.path.join(BENCH, "scfp_runs", run)
        orig_all = [r for f in ("traces.jsonl", "excluded.jsonl") for r in read_jsonl(os.path.join(nat, "natural", f))
                    if r["source_model"] == g and r["problem_id"] in ids]
        alt_all = read_jsonl(os.path.join(gpu, f"alt_{g}", "traces_all.jsonl"))
        o, a = {r["trace_id"]: r for r in orig_all}, {r["trace_id"]: r for r in alt_all}
        common = sorted(set(o) & set(a))
        # paired outcomes on every subset trace (excluded ones labelled from their fallback-parsed answers)
        yo = np.array([not o[t]["rf_correct"] for t in common], float)
        ya = np.array([not a[t]["rf_correct"] for t in common], float)
        D = draws([o[t]["problem_id"] for t in common], rng)
        diff = [ya[i].mean() - yo[i].mean() for i in D]
        # models on the kept traces of each template, refit on the subset with the same CV
        kept_o = [r for r in read_jsonl(os.path.join(nat, "natural", "traces.jsonl")) if r["source_model"] == g and r["problem_id"] in ids]
        fo = {r["trace_id"]: r for r in read_jsonl(os.path.join(nat, "features", "natural.jsonl")) if r["source_model"] == g}
        vo = {r["trace_id"]: r for r in read_jsonl(os.path.join(nat, "features", "natural_verifier.jsonl"))}
        rows_o = feature_rows(kept_o, {k: {kk: vv for kk, vv in fo[k].items() if kk.startswith(("lp_", "f_", "len_", "n_tok"))}
                                       for k in fo}, vo)
        kept_a = [a[t] for t in sorted(a) if a[t]["kept"]]
        fa = {r["trace_id"]: r for r in read_jsonl(os.path.join(gpu, f"alt_{g}", "features.jsonl"))}
        va = {}
        for t in kept_a:  # same program (it depends only on P); pot_agree_f against the new critique's answer
            v = vo.get(t["trace_id"])
            if v is None or t["source"] != "gsm8k":
                continue
            fans = extract(t["critique"], "number").value
            va[t["trace_id"]] = {"pot_ran": v["pot_ran"], "pot_agree_r0": v["pot_agree_r0"],
                                 "pot_agree_f": (float(is_correct(fans, v["pot_answer"], "number")) if v["pot_answer"] is not None else None)}
        rows_a = feature_rows(kept_a, fa, va)
        res[g] = {"n_paired": len(common), "failure_original": float(yo.mean()), "failure_alt": float(ya.mean()),
                  "failure_diff": float(ya.mean() - yo.mean()), "failure_diff_ci95": list(np.percentile(diff, [2.5, 97.5])),
                  "outcomes_original": outcomes([o[t] for t in common]), "outcomes_alt": outcomes([a[t] for t in common]),
                  "kept_original": len(rows_o), "kept_alt": len(rows_a),
                  "models_original": models(rows_o, rng), "models_alt": models(rows_a, rng)}
    return res


def precision(gpu):
    d = os.path.join(gpu, "precision")
    calls = {dt: {c["messages"][0]["content"]: c for c in read_jsonl(os.path.join(d, f"calls_{dt}.jsonl")) if c.get("ok")}
             for dt in ("float16", "bfloat16")}
    run = MODELS["qwen2.5-1.5b"][2]
    orig = {}
    for f in ("traces.jsonl", "excluded.jsonl"):
        for r in read_jsonl(os.path.join(BENCH, "scfp_runs", run, "natural", f)):
            if r["source_model"] == "qwen2.5-1.5b":
                orig[r["prompt"]] = r
    from scfp_bench import prompts
    n, diff_prec, diff_text, diff_orig16, diff_origbf, matched = 0, 0, 0, 0, 0, 0
    for msg, c16 in calls["float16"].items():
        cb = calls["bfloat16"][msg]
        r = next((o for q, o in orig.items() if prompts.initial_messages(q, ATYPE[o["source"]])[0]["content"] == msg), None)
        at = ATYPE[r["source"]] if r else "number"
        a16, abf = extract(c16["text"], at, accept_boxed=True).value, extract(cb["text"], at, accept_boxed=True).value
        n += 1
        diff_text += c16["text"] != cb["text"]
        diff_prec += a16 != abf
        if r is not None:
            matched += 1
            ao = extract(r["initial_response"], at, accept_boxed=True).value
            diff_orig16 += a16 != ao
            diff_origbf += abf != ao
    return {"n": n, "matched_to_original": matched, "answers_differ_fp16_vs_bf16": diff_prec, "texts_differ_fp16_vs_bf16": diff_text,
            "fp16_answer_differs_from_original": diff_orig16, "bf16_answer_differs_from_original": diff_origbf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="results/revision/m8/gpu")
    ap.add_argument("--out", default="results/revision/m8")
    a = ap.parse_args()
    gpu, out = os.path.abspath(a.gpu), os.path.abspath(a.out)
    rng = np.random.default_rng(SEED)
    os.chdir(BENCH)
    res = {}
    if os.path.exists(os.path.join(gpu, "done_q7b.json")):
        res["q7b"] = q7b(gpu, rng)
        print("q7b:", json.dumps(res["q7b"]["outcomes"]), {k: round(v["within_auc"], 3) if "within_auc" in v else v
                                                          for k, v in res["q7b"]["models"].items()}, flush=True)
    if os.path.exists(os.path.join(gpu, "done_prompt_alt.json")):
        res["prompt_alt"] = prompt_alt(gpu, rng)
        for g, v in res["prompt_alt"].items():
            print(g, f"failure {v['failure_original']:.3f} -> {v['failure_alt']:.3f} diff {v['failure_diff']:+.3f} "
                  f"{[round(x, 3) for x in v['failure_diff_ci95']]}",
                  {k: (round(v['models_original'][k]['within_auc'], 3), round(v['models_alt'][k]['within_auc'], 3))
                   for k in ("text_tfidf", "signals_noverifier", "signals")}, flush=True)
    if os.path.exists(os.path.join(gpu, "done_precision.json")):
        res["precision"] = precision(gpu)
        print("precision:", res["precision"], flush=True)
    json.dump(res, open(os.path.join(out, "m8_gpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
