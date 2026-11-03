"""E4 (round 2): error rates of automatic answer extraction from the completed human audit sheet.

Reads data/audit/extraction_audit_sheet.csv after the authors have filled the human columns:
  human_r0_answer / human_rf_answer    the final answer a careful reader takes from R0 / R_f ("none" if there is none)
  human_r0_correct / human_rf_correct  yes / no: is that answer correct against the gold answer
An extraction error is a trace whose automatic correctness label differs from the human one. Rates are reported
unweighted and weighted by stratum size (the sample is stratified with equal allocation), with 95% bootstrap intervals
that resample within strata (10,000 resamples, seed 0), and Wilson intervals for the unweighted rate. Refuses to run
while any human cell is blank.

    .venv/bin/python scripts/revision/e4_audit_score.py --out results/revision/e4
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
B, SEED = 10000, 0
YES, NO = {"yes", "y", "1", "true", "correct"}, {"no", "n", "0", "false", "wrong", "incorrect"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def yn(x, where):
    x = x.strip().lower()
    if x in YES:
        return True
    if x in NO:
        return False
    raise SystemExit(f"{where}: expected yes/no, got {x!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=os.path.join(ROOT, "data", "audit", "extraction_audit_sheet.csv"))
    ap.add_argument("--sample", default=os.path.join(ROOT, "data", "audit", "extraction_audit_sample.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "revision", "e4"))
    a = ap.parse_args()
    sheet = list(csv.DictReader(open(a.sheet)))
    blank = [r["audit_id"] for r in sheet if any(not r[c].strip() for c in ("human_r0_correct", "human_rf_correct"))]
    if blank:
        raise SystemExit(f"the audit is not complete: {len(blank)} of {len(sheet)} rows have blank human columns "
                         f"(first: {blank[:5]}); fill them before scoring")
    meta = {json.loads(l)["audit_id"]: json.loads(l) for l in open(a.sample)}
    rows = []
    for r in sheet:
        m = meta[r["audit_id"]]
        h0, hf = yn(r["human_r0_correct"], r["audit_id"]), yn(r["human_rf_correct"], r["audit_id"])
        rows.append({"stratum": m["stratum"], "size": m["stratum_size"], "n": m["stratum_n"], "source": m["source"],
                     "err_r0": int(h0 != bool(m["r0_correct"])), "err_rf": int(hf != bool(m["rf_correct"])),
                     "err_label": int((not hf) != bool(m["label"]))})
    w = np.array([r["size"] / r["n"] for r in rows])
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["stratum"]].append(i)
    strata = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(SEED)
    res = {"n": len(rows)}
    for k in ("err_r0", "err_rf", "err_label"):
        e = np.array([r[k] for r in rows], float)
        wt = float(np.sum(w * e) / np.sum(w))
        bs = []
        for _ in range(B):
            idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
            bs.append(np.sum(w[idx] * e[idx]) / np.sum(w[idx]))
        res[k] = {"errors": int(e.sum()), "rate": float(e.mean()), "rate_wilson95": wilson(int(e.sum()), len(e)),
                  "weighted_rate": wt, "weighted_rate_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        res[k]["by_source"] = {s: {"n": int(sum(1 for r in rows if r["source"] == s)),
                                   "errors": int(sum(r[k] for r in rows if r["source"] == s))}
                               for s in ("gsm8k", "strategyqa", "logiqa")}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "extraction_audit.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
