"""E2 analysis (round 2): self-consistency agreement as a verifier on StrategyQA and LogiQA.

Reads the k = 3 samples of every kept natural-set StrategyQA/LogiQA trace (results/revision/e2/gen/calls_<g>.jsonl) and
computes, per generator and source, the AUC of 1 - agreement for "R0 wrong" and "final wrong", and the stacked LR
(signals + SC3) against the signals, out-of-fold with grouped folds within the source. Features and model as in M6.
See REVISION_PLAN.md (Round 2, E2).

    .venv/bin/python scripts/revision/e2_analysis.py --gen results/revision/e2/gen --out results/revision/e2
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
from m5_deconfound import auc  # noqa: E402
from m6_analysis import CONFIGS, samples_by_prompt  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.answers import extract  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows, oof  # noqa: E402
from scfp_bench.experiments.signals import ATYPE, VERIFIER, grouped_folds  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

B, SEED, K = 10000, 0, 3
SOURCES = ("strategyqa", "logiqa")
SC3 = ["sc3_agree_frac", "sc3_majority_is_r0"]


def boot(fn, n, rng):
    """One trace per problem within a generator, so resampling traces is the problem-level bootstrap."""
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    v = [x for x in v if not np.isnan(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="results/revision/e2/gen")
    ap.add_argument("--out", default="results/revision/e2")
    a = ap.parse_args()
    gen, out = os.path.abspath(a.gen), os.path.abspath(a.out)
    os.chdir(BENCH)
    rng = np.random.default_rng(SEED)
    res = {}
    for g, cname in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join("configs", cname)))
        trace = {t["trace_id"]: t for t in read_jsonl(os.path.join(cfg["run_dir"], "natural", "traces.jsonl"))}
        S = samples_by_prompt(os.path.join(gen, f"calls_{g}.jsonl"))
        res[g] = {}
        for src in SOURCES:
            rows = [r for r in generator_rows(cfg, "natural", g) if r["source"] == src]
            missing, unparsed = 0, 0
            for r in rows:
                t = trace[r["trace_id"]]
                msg = prompts.initial_messages(t["prompt"], ATYPE[src])[0]["content"]
                vals = []
                for k in range(1, K + 1):
                    txt = S.get((msg, k))
                    missing += txt is None
                    v = None if txt is None else extract(txt, ATYPE[src], accept_boxed=True).value
                    unparsed += txt is not None and v is None
                    vals.append(v)
                a0 = t["r0_answer"]
                agree = [v is not None and v == a0 for v in vals]       # unparsable sample counts as disagreeing
                parsed = [v for v in vals if v is not None]
                r["sc1_agree"] = float(agree[0])
                r["sc3_agree_frac"] = float(np.mean(agree))
                r["sc3_majority_is_r0"] = float(bool(parsed) and max(set(parsed), key=parsed.count) == a0)
                r["r0_wrong"] = int(not t["r0_correct"])
            assert missing == 0, f"{g} {src}: {missing} samples missing"
            y, y0 = np.array([r["label"] for r in rows]), np.array([r["r0_wrong"] for r in rows])
            n = len(rows)
            single = {}
            for f in ("sc1_agree", "sc3_agree_frac"):
                sc = 1 - np.array([r[f] for r in rows])
                single[f] = {"r0_wrong": float(auc(y0, sc)), "r0_wrong_ci95": boot(lambda i: auc(y0[i], sc[i]), n, rng),
                             "rf_wrong": float(auc(y, sc)), "rf_wrong_ci95": boot(lambda i: auc(y[i], sc[i]), n, rng)}
            folds = grouped_folds(rows)
            base = [{k: v for k, v in r.items() if k not in VERIFIER} for r in rows]
            Xb = scalar_matrix(base)
            Xs = np.hstack([Xb, np.array([[r[c] for c in SC3] for r in rows], float)])
            P = {}
            for name, X in (("signals", Xb), ("signals_sc3", Xs)):
                P[name] = oof("scalar", {"text": None, "hidden": None, "scalar": X}, y, folds)
                P[name + "_r0"] = oof("scalar", {"text": None, "hidden": None, "scalar": X}, y0, folds)
            models = {}
            for tgt, yy, suf in (("rf_wrong", y, ""), ("r0_wrong", y0, "_r0")):
                d = boot(lambda i: auc(yy[i], P["signals_sc3" + suf][i]) - auc(yy[i], P["signals" + suf][i]), n, rng)
                models[tgt] = {"signals": float(auc(yy, P["signals" + suf])), "signals_sc3": float(auc(yy, P["signals_sc3" + suf])),
                               "diff": float(auc(yy, P["signals_sc3" + suf]) - auc(yy, P["signals" + suf])), "diff_ci95": d}
            res[g][src] = {"n": n, "r0_wrong_rate": float(y0.mean()), "failure_rate": float(y.mean()),
                           "unparsed_samples": int(unparsed), "single": single, "models": models}
            s3 = single["sc3_agree_frac"]
            print(f"{g} {src} n={n}: SC3 R0-wrong {s3['r0_wrong']:.3f} {[round(x, 3) for x in s3['r0_wrong_ci95']]} "
                  f"final {s3['rf_wrong']:.3f} | signals {models['rf_wrong']['signals']:.3f} +SC3 {models['rf_wrong']['signals_sc3']:.3f} "
                  f"diff {models['rf_wrong']['diff']:+.3f} {[round(x, 3) for x in models['rf_wrong']['diff_ci95']]} | unparsed {unparsed}", flush=True)
    # registered scope rule, per source
    rule = {}
    for src in SOURCES:
        v = [res[g][src]["single"]["sc3_agree_frac"] for g in CONFIGS]
        strong = sum(x["r0_wrong"] >= 0.80 and x["r0_wrong_ci95"][0] >= 0.70 for x in v)
        weak = sum(x["r0_wrong"] < 0.70 for x in v)
        rule[src] = {"strong_generators": strong, "weak_generators": weak,
                     "verdict": "strong" if strong >= 3 else "weak" if weak >= 3 else "mixed"}
    res["_rule"] = rule
    res["_scope"] = ("holds where agreement-based verification applies" if all(r["verdict"] == "strong" for r in rule.values())
                     else "restricted to numeric problems")
    print("rule:", rule, "->", res["_scope"], flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "e2_sc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
