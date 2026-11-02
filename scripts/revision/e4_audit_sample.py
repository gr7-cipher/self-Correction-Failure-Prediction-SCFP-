"""E4 (round 2): stratified random sample of 100 natural-set traces for a human audit of answer extraction, and the
audit sheet. Strata: generator x source x (R0 correct, final correct), 48 cells; equal allocation (2 per cell) and the
remaining traces to cells drawn at random, all with a fixed seed. The human columns are left blank: they are filled by
the authors, never by a script or a model. Score the completed sheet with e4_audit_score.py.

    .venv/bin/python scripts/revision/e4_audit_sample.py
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from m6_sc_generate import MODELS  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEED, N = 42, 100
HUMAN = ["human_r0_answer", "human_r0_correct", "human_rf_answer", "human_rf_correct", "human_notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default=os.path.join(ROOT, "data", "audit", "extraction_audit_sample.jsonl"))
    ap.add_argument("--out-csv", default=os.path.join(ROOT, "data", "audit", "extraction_audit_sheet.csv"))
    a = ap.parse_args()
    rows = []
    for g, (_, _, run) in MODELS.items():
        rows += [r for r in read_jsonl(os.path.join(BENCH, "scfp_runs", run, "natural", "traces.jsonl")) if r["source_model"] == g]
    cells = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["source_model"], r["trace_id"])):
        cells[(r["source_model"], r["source"], bool(r["r0_correct"]), bool(r["rf_correct"]))].append(r)
    rng = random.Random(SEED)
    keys = sorted(cells)
    alloc = {k: min(2, len(cells[k])) for k in keys}
    while sum(alloc.values()) < N:  # remaining draws go to cells chosen at random that still have traces left
        k = rng.choice([k for k in keys if alloc[k] < len(cells[k])])
        alloc[k] += 1
    sample = []
    for k in keys:
        for r in rng.sample(cells[k], alloc[k]):
            sample.append({"stratum": "|".join(map(str, k)), "stratum_size": len(cells[k]), "stratum_n": alloc[k], **r})
    rng.shuffle(sample)  # the audit order does not reveal the stratum
    for i, r in enumerate(sample, 1):
        r["audit_id"] = f"A{i:03d}"
    os.makedirs(os.path.dirname(a.out_jsonl), exist_ok=True)
    with open(a.out_jsonl, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in sample)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "trace_id", "generator", "source", "gold", "r0_text", "r0_extracted", "rf_text", "rf_extracted"] + HUMAN)
        for r in sample:
            w.writerow([r["audit_id"], r["trace_id"], r["source_model"], r["source"], r["gold"], r["initial_response"],
                        r["r0_answer"], r["final_response"], r["rf_answer"]] + [""] * len(HUMAN))
    print(f"{len(sample)} traces from {len(rows)} kept natural-set traces, {len(keys)} strata "
          f"(cells with fewer than 2 traces: {sum(len(cells[k]) < 2 for k in keys)}); wrote {a.out_jsonl} and {a.out_csv}")


if __name__ == "__main__":
    main()
