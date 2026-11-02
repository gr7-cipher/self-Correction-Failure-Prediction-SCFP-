"""A3 (round 2): numerical stability of labels. On the problems of the float16/bfloat16 check (job C, Qwen2.5-1.5B,
R0 only), a trace's R0-correctness label is stable if it is the same in float16, bfloat16 and the original logged
trace. Reports the out-of-fold signals AUC for "R0 wrong" (M3 stage-1 model, signals + verifier) and for "final
wrong" (direct model) on the stable and unstable kept traces. See REVISION_PLAN.md (Round 2, A3).

    .venv/bin/python scripts/revision/a3_precision.py --out results/revision/a3
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from m5_deconfound import auc, within  # noqa: E402
from m6_sc_generate import MODELS  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.answers import extract, is_correct  # noqa: E402
from scfp_bench.experiments.signals import ATYPE  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

B, SEED, G = 10000, 0, "qwen2.5-1.5b"
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/a3")
    a = ap.parse_args()
    d = os.path.join(ROOT, "results", "revision", "m8", "gpu", "precision")
    calls = {dt: {c["messages"][0]["content"]: c for c in read_jsonl(os.path.join(d, f"calls_{dt}.jsonl")) if c.get("ok")}
             for dt in ("float16", "bfloat16")}
    run = os.path.join(BENCH, "scfp_runs", MODELS[G][2], "natural")
    orig = {}
    for f, kept in (("traces.jsonl", True), ("excluded.jsonl", False)):
        for r in read_jsonl(os.path.join(run, f)):
            if r["source_model"] == G:
                orig[prompts.initial_messages(r["prompt"], ATYPE[r["source"]])[0]["content"]] = {**r, "kept": kept}
    preds = {p["trace_id"]: p for p in read_jsonl(os.path.join(ROOT, "results", "revision", "m3", "m3_cpu_predictions.jsonl"))
             if p["generator"] == G}
    rows, n_all, unmatched = [], 0, 0
    for msg, c16 in calls["float16"].items():
        n_all += 1
        r = orig.get(msg)
        if r is None:
            unmatched += 1
            continue
        at = ATYPE[r["source"]]
        ans = {"float16": extract(c16["text"], at, accept_boxed=True).value,
               "bfloat16": extract(calls["bfloat16"][msg]["text"], at, accept_boxed=True).value,
               "original": extract(r["initial_response"], at, accept_boxed=True).value}
        corr = {k: bool(is_correct(v, r["gold"], at)) for k, v in ans.items()}
        rows.append({"trace_id": r["trace_id"], "problem_id": r["problem_id"], "source": r["source"], "kept": r["kept"],
                     "answer_stable": len({str(v) for v in ans.values()}) == 1,
                     "label_stable": len(set(corr.values())) == 1, **{f"r0_correct_{k}": v for k, v in corr.items()}})
    kept = [r for r in rows if r["kept"] and r["trace_id"] in preds]
    rng = np.random.default_rng(SEED)
    res = {"generator": G, "n_problems": n_all, "matched": len(rows), "unmatched": unmatched,
           "answer_unstable": int(sum(not r["answer_stable"] for r in rows)),
           "label_unstable": int(sum(not r["label_stable"] for r in rows)),
           "kept_with_predictions": len(kept), "kept_label_unstable": int(sum(not r["label_stable"] for r in kept))}
    for target, key, score in (("r0_wrong", "y_r0", lambda p: 1 - p["p_r0_correct"]), ("rf_wrong", "y_rf", lambda p: p["p_direct"])):
        for name, sub in (("all", kept), ("stable", [r for r in kept if r["label_stable"]]),
                          ("unstable", [r for r in kept if not r["label_stable"]])):
            y = np.array([preds[r["trace_id"]][key] for r in sub])
            s = np.array([score(preds[r["trace_id"]]) for r in sub])
            src = np.array([r["source"] for r in sub])
            if len(sub) == 0 or y.min() == y.max():
                res[f"{target}_{name}"] = {"n": len(sub), "n_pos": int(y.sum()) if len(sub) else 0, "within_auc": None}
                continue
            v = within(y, s, src)[0]
            bs = []
            for _ in range(B):  # one trace per problem for this generator: trace bootstrap = problem bootstrap
                i = rng.integers(0, len(y), len(y))
                if y[i].min() != y[i].max():
                    x = within(y[i], s[i], src[i])[0]
                    if not np.isnan(x):
                        bs.append(x)
            res[f"{target}_{name}"] = {"n": len(sub), "n_pos": int(y.sum()), "within_auc": float(v), "pooled_auc": float(auc(y, s)),
                                       "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "a3_precision.json"), "w"), indent=1)
    with open(os.path.join(a.out, "a3_rows.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
