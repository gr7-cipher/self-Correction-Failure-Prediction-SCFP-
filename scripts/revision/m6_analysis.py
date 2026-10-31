"""M6 analysis: what the PoT verifier adds beyond self-consistency agreement (GSM8K traces of each generator).

Reads the natural-set features and verifier features of the released runs, and the self-consistency samples
from results/revision/m6/gen/calls_<generator>.jsonl. See REVISION_PLAN.md (M6).

    .venv/bin/python scripts/revision/m6_analysis.py --gen results/revision/m6/gen --out results/revision/m6
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from m5_deconfound import ap as auprc, auc  # noqa: E402
from scfp_bench.answers import extract  # noqa: E402
from scfp_bench.experiments.improve import scalar_matrix  # noqa: E402
from scfp_bench.experiments.paper_v7 import generator_rows, oof  # noqa: E402
from scfp_bench.experiments.signals import VERIFIER, grouped_folds  # noqa: E402

CONFIGS = {"qwen2.5-0.5b": "signals_mac_big.yaml", "qwen2.5-1.5b": "signals_kaggle.yaml",
           "smollm2-1.7b": "signals_kaggle.yaml", "qwen2.5-3b": "signals_kaggle.yaml"}
B, SEED = 1000, 0
SETS = {"base": [], "pot": ["pot_ran", "pot_agree_r0", "pot_agree_f"], "sc1": ["sc1_agree"],
        "sc1_pot": ["sc1_agree", "pot_ran", "pot_agree_r0", "pot_agree_f"],
        "sc3": ["sc3_agree_frac", "sc3_majority_is_r0"],
        "sc3_pot": ["sc3_agree_frac", "sc3_majority_is_r0", "pot_ran", "pot_agree_r0", "pot_agree_f"]}
COMPARE = [("pot", "sc1"), ("sc1_pot", "sc1"), ("sc3_pot", "sc3")]


def samples_by_prompt(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("stage") == "sample" and r.get("ok"):
            out[(r["messages"][0]["content"], r["sample"])] = r["text"]
    return out


def single(y, score):
    """AUC/AUPRC of a failure score, and accuracy of 'predict failure iff score > 0.5' (binary features)."""
    return {"auc": float(auc(y, score)), "auprc": float(auprc(y, score)),
            "accuracy": float(np.mean((score > 0.5) == y)), "positive_rate": float(np.mean(score > 0.5))}


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--gen", default="results/revision/m6/gen")
    a_.add_argument("--out", default="results/revision/m6")
    a_.add_argument("--k", type=int, default=3)
    a = a_.parse_args()
    gen, out = os.path.abspath(a.gen), os.path.abspath(a.out)
    os.chdir(BENCH)
    from scfp_bench import prompts
    res, rng = {}, np.random.default_rng(SEED)
    for g, cfg_name in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join("configs", cfg_name)))
        rows = [r for r in generator_rows(cfg, "natural", g) if r["source"] == "gsm8k"]
        trace = {t["trace_id"]: t for t in [json.loads(l) for l in open(os.path.join(cfg["run_dir"], "natural", "traces.jsonl"))]}
        S = samples_by_prompt(os.path.join(gen, f"calls_{g}.jsonl"))
        missing = 0
        for r in rows:
            t = trace[r["trace_id"]]
            msg = prompts.initial_messages(t["prompt"], "number")[0]["content"]
            vals = []
            for k in range(1, a.k + 1):
                txt = S.get((msg, k))
                if txt is None:
                    missing += 1
                vals.append(None if txt is None else extract(txt, "number", accept_boxed=True).value)
            a0 = t["r0_answer"]
            agree = [v is not None and v == a0 for v in vals]     # unparsable sample counts as disagreeing
            r["sc1_agree"] = float(agree[0])
            r["sc3_agree_frac"] = float(np.mean(agree))
            parsed = [v for v in vals if v is not None]
            r["sc3_majority_is_r0"] = float(bool(parsed) and max(set(parsed), key=parsed.count) == a0)
            r["r0_wrong"] = int(not t["r0_correct"])
        assert missing == 0, f"{g}: {missing} samples missing"
        y = np.array([r["label"] for r in rows])
        y0 = np.array([r["r0_wrong"] for r in rows])
        folds = grouped_folds(rows)
        feats = {}
        for name in ("pot_ran", "pot_agree_r0", "sc1_agree", "sc3_agree_frac"):
            v = np.array([r[name] for r in rows], float)
            if name == "pot_ran":
                score = 1 - np.nan_to_num(v, nan=0.0)             # program did not run -> risk
            else:
                score = 1 - np.nan_to_num(v, nan=0.0)             # no agreement (or no program) -> risk
            feats[name] = {"rf_wrong": single(y, score), "r0_wrong": single(y0, score),
                           "coverage": float(np.mean(~np.isnan(v)))}
        base = [{k: v for k, v in r.items() if k not in VERIFIER} for r in rows]
        Xb = scalar_matrix(base)
        P = {}
        for name, cols in SETS.items():
            X = np.hstack([Xb] + [np.array([[r.get(c, np.nan) for c in cols] for r in rows], float)]) if cols else Xb
            P[name] = oof("scalar", {"text": None, "hidden": None, "scalar": X}, y, folds)
        models = {n: {"auc": float(auc(y, p)), "auprc": float(auprc(y, p))} for n, p in P.items()}
        comps = {}
        for x, ref in COMPARE:
            d = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                d.append(auc(y[i], P[x][i]) - auc(y[i], P[ref][i]))
            comps[f"{x} - {ref}"] = {"diff": models[x]["auc"] - models[ref]["auc"],
                                     "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        res[g] = {"n": len(rows), "failure_rate": float(y.mean()), "r0_wrong_rate": float(y0.mean()),
                  "single": feats, "models": models, "compare": comps}
        print(g, f"n={len(rows)}")
        for n, v in feats.items():
            print(f"  {n:15s} rf_wrong AUC {v['rf_wrong']['auc']:.3f} acc {v['rf_wrong']['accuracy']:.3f} "
                  f"auprc {v['rf_wrong']['auprc']:.3f} | r0_wrong AUC {v['r0_wrong']['auc']:.3f}")
        print("  models:", {n: round(v["auc"], 3) for n, v in models.items()})
        for n, v in comps.items():
            print(f"  {n}: {v['diff']:+.3f} {[round(x, 3) for x in v['ci95']]}")
    rule = {k: sum(res[g]["compare"][k]["ci95"][0] > 0 for g in res) for k in ("sc1_pot - sc1", "pot - sc1", "sc3_pot - sc3")}
    res["_rule_counts"] = rule
    print("generators where CI excludes 0 (positive):", rule)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "pot_vs_sc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
