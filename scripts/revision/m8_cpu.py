"""M8 (CPU part): the 600-problem subset, the answer-extraction audit with an independent second parser,
and the exclusion-sensitivity rates. See REVISION_PLAN.md (M8).

    .venv/bin/python scripts/revision/m8_cpu.py --out results/revision/m8 --audit audit
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
RUNS = {"qwen2.5-0.5b": "signals_mac_q05_big", "qwen2.5-1.5b": "signals_kaggle", "smollm2-1.7b": "signals_kaggle",
        "qwen2.5-3b": "signals_kaggle"}
SOURCES = {"gsm8k": 300, "strategyqa": 150, "logiqa": 150}


def load(run, f):
    p = os.path.join(BENCH, "scfp_runs", run, "natural", f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---------------------------------------------------------------- independent second parser
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse2(text, source):
    """Deliberately simple and independent of scfp_bench.answers: look only at the last 300 characters."""
    tail = (text or "")[-300:]
    if source == "gsm8k":
        m = NUM.findall(tail.replace("$", ""))
        if not m:
            return None
        v = m[-1].replace(",", "")
        try:
            x = float(v)
        except ValueError:
            return None
        return str(int(x)) if x == int(x) else str(x)
    if source == "strategyqa":
        m = re.findall(r"\b(yes|no)\b", tail, re.I)
        return m[-1].lower() if m else None
    m = re.findall(r"(?:\b|\(|\*)([A-D])(?:\b|\)|\*)", tail)
    return m[-1] if m else None


def same(a, b, source):
    if a is None or b is None:
        return a is None and b is None
    if source == "gsm8k":
        try:
            return abs(float(a) - float(str(b).replace(",", ""))) < 1e-6
        except ValueError:
            return False
    return str(a).strip().lower() == str(b).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m8")
    ap.add_argument("--audit", default="audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.audit, exist_ok=True)
    kept = {g: [r for r in load(run, "traces.jsonl") if r["source_model"] == g] for g, run in RUNS.items()}
    excl = {g: [r for r in load(run, "excluded.jsonl") if r["source_model"] == g] for g, run in RUNS.items()}

    # 1) the 600-problem subset (all 2,000 problems are in kept or excluded traces of Qwen2.5-0.5B)
    probs = {}
    for g in RUNS:
        for r in kept[g] + excl[g]:
            probs[r["problem_id"]] = r["source"]
    rng = random.Random(42)
    subset = {s: sorted(rng.sample(sorted(p for p, src in probs.items() if src == s), k)) for s, k in SOURCES.items()}
    json.dump({"seed": 42, "n_pool": len(probs), "per_source": SOURCES, "problems": subset},
              open(os.path.join(a.out, "subset.json"), "w"), indent=1)

    # 2) second parser: agreement with the pipeline's parsed answers on every retained trace
    agree = defaultdict(lambda: [0, 0])
    for g in RUNS:
        for r in kept[g]:
            for which, text, parsed in (("R0", r["initial_response"], r["r0_answer"]), ("Rf", r["final_response"], r["rf_answer"])):
                k = (g, r["source"], which)
                agree[k][0] += same(parse2(text, r["source"]), parsed, r["source"])
                agree[k][1] += 1
    tab = {f"{g}|{s}|{w}": {"agree": v[0], "n": v[1], "rate": v[0] / v[1]} for (g, s, w), v in sorted(agree.items())}
    # post-hoc diagnostic: where is the explicit answer line in the disagreements?
    why = defaultdict(int)
    for g in RUNS:
        for r in kept[g]:
            for text, parsed in ((r["initial_response"], r["r0_answer"]), (r["final_response"], r["rf_answer"])):
                if same(parse2(text, r["source"]), parsed, r["source"]):
                    continue
                m = list(re.finditer(r"final answer\s*[:\-]", text, re.I))
                if m and m[-1].start() < len(text) - 300:
                    why["answer line outside the last 300 characters"] += 1
                elif "boxed" in text[-300:]:
                    why["boxed answer in the tail"] += 1
                else:
                    why["other (formatting, units, extra numbers or letters after the answer)"] += 1
    tot = [sum(v[0] for v in agree.values()), sum(v[1] for v in agree.values())]
    # label agreement: would the second parser change the correctness label of Rf?
    flips, n_rf = 0, 0
    for g in RUNS:
        for r in kept[g]:
            n_rf += 1
            flips += same(parse2(r["final_response"], r["source"]), r["gold"], r["source"]) != bool(r["rf_correct"])
    # 3) audit sample: 100 retained traces, stratified over generator x source
    rng2 = random.Random(42)
    strata = [(g, s) for g in RUNS for s in SOURCES]
    quota = {st: 100 // len(strata) + (1 if i < 100 % len(strata) else 0) for i, st in enumerate(strata)}
    rows = []
    for (g, s), q in quota.items():
        pool = sorted([r for r in kept[g] if r["source"] == s], key=lambda r: r["trace_id"])
        for r in rng2.sample(pool, q):
            rows.append({"trace_id": r["trace_id"], "generator": g, "source": s,
                         "raw_text_tail": r["final_response"][-300:], "parsed_answer": r["rf_answer"], "gold": r["gold"],
                         "auto_label": "correct" if r["rf_correct"] else "wrong",
                         "second_parser": parse2(r["final_response"], s), "human_label": ""})
    with open(os.path.join(a.audit, "extraction_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace_id", "generator", "source", "raw_text_tail", "parsed_answer", "gold",
                                          "auto_label", "second_parser", "human_label"])
        w.writeheader()
        w.writerows(rows)
    # 4) exclusion sensitivity: outcome rates with exclusions counted as failures
    ex = {}
    for g in RUNS:
        k, e = kept[g], excl[g]
        rf_k = np.array([r["rf_correct"] for r in k], bool)
        ex[g] = {"kept": len(k), "excluded": len(e), "failure_rate_excluding": float(1 - rf_k.mean()),
                 "failure_rate_counting_as_failure": float((np.sum(~rf_k) + len(e)) / (len(k) + len(e))),
                 "excluded_by_source": {s: sum(r["source"] == s for r in e) for s in SOURCES}}
    v1 = os.path.join(BENCH, "scfp_runs", "laptop_v1")
    lab = [json.loads(l) for l in open(os.path.join(v1, "labeled.jsonl"))]
    exv = [json.loads(l) for l in open(os.path.join(v1, "excluded.jsonl"))]
    rfv = np.array([r["rf_correct"] for r in lab], bool)
    ex["scfp_v1"] = {"kept": len(lab), "excluded": len(exv), "failure_rate_excluding": float(1 - rfv.mean()),
                     "failure_rate_counting_as_failure": float((np.sum(~rfv) + len(exv)) / (len(lab) + len(exv)))}
    res = {"second_parser": {"by_stratum": tab, "overall_agree": tot[0], "overall_n": tot[1], "overall_rate": tot[0] / tot[1],
                             "rf_label_would_change": flips, "rf_n": n_rf},
           "audit_rows": len(rows), "exclusion_rates": ex, "disagreement_reasons_post_hoc": dict(why)}
    json.dump(res, open(os.path.join(a.out, "m8_cpu.json"), "w"), indent=1)
    print("subset sizes:", {s: len(v) for s, v in subset.items()}, "pool", len(probs))
    print("second parser agreement:", f"{tot[0]}/{tot[1]} = {tot[0] / tot[1]:.3f}", "Rf label flips:", flips, "/", n_rf)
    low = sorted(tab.items(), key=lambda kv: kv[1]["rate"])[:6]
    print("lowest strata:", [(k, round(v["rate"], 3)) for k, v in low])
    print("disagreement reasons (post hoc):", dict(why))
    print("exclusion:", {g: (round(v["failure_rate_excluding"], 3), round(v["failure_rate_counting_as_failure"], 3)) for g, v in ex.items()})


if __name__ == "__main__":
    main()
