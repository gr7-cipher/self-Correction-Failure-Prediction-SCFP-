"""M2: candidate real traces for Table 1, grouped by observable pattern only. No failure mode is assigned here;
an author fills the "author-assigned mode" field. See REVISION_PLAN.md (M2).

    .venv/bin/python scripts/revision/m2_candidates.py --out audit/taxonomy_candidates.md
"""

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from scfp_bench.experiments.signals import CORRECT_RE, ERROR_RE  # noqa: E402

HEDGE = re.compile(r"\b(not sure|unsure|might|possibly|perhaps|unclear|uncertain|i think|it seems|may be)\b", re.I)
PATTERNS = {
    "P1": ("correct -> wrong; the critique reports an error",
           lambda r: r["r0_correct"] and not r["rf_correct"] and ERROR_RE.search(r["critique"])),
    "P2": ("wrong -> wrong; answer unchanged; the critique endorses R0",
           lambda r: not r["r0_correct"] and not r["rf_correct"] and r["r0_answer"] == r["rf_answer"]
           and CORRECT_RE.search(r["critique"]) and not ERROR_RE.search(r["critique"])),
    "P3": ("wrong -> wrong; answer changed", lambda r: not r["r0_correct"] and not r["rf_correct"] and r["r0_answer"] != r["rf_answer"]),
    "P4": ("wrong -> wrong; answer unchanged; the critique reports an error",
           lambda r: not r["r0_correct"] and not r["rf_correct"] and r["r0_answer"] == r["rf_answer"] and ERROR_RE.search(r["critique"])),
    "P5": ("failed trace; the critique contains hedging words", lambda r: not r["rf_correct"] and HEDGE.search(r["critique"])),
}


def clip(t, n=700):
    t = t.strip()
    return t if len(t) <= n else t[:n // 2] + "\n[...]\n" + t[-n // 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit/taxonomy_candidates.md")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(BENCH, "scfp_runs", "laptop_v1", "labeled.jsonl"))]
    rng = random.Random(42)
    lines = ["# Candidate real traces for Table 1 (for author labelling)", "",
             "Selected by observable pattern only, from the failed SCFP v1.0 traces (seed 42). **No failure mode has "
             "been assigned.** An author should read each trace and fill in the mode (JH, CM, BA, OC, RM, FD, or "
             "Unclear) before any of them is used in the paper. Texts are shortened in the middle.", ""]
    counts = {}
    for k, (desc, f) in PATTERNS.items():
        pool = sorted([r for r in rows if f(r)], key=lambda r: r["trace_id"])
        counts[k] = len(pool)
        lines += [f"## {k}: {desc} ({len(pool)} traces match)", ""]
        for r in rng.sample(pool, min(3, len(pool))):
            lines += [f"### {r['trace_id']} ({r['source_model']}, {r['source']}; gold {r['gold']}; "
                      f"R0 answer {r['r0_answer']}; final answer {r['rf_answer']})", "",
                      "**Author-assigned mode:** ____", "", "**Problem**", "", "```", clip(r["prompt"], 500), "```", "",
                      "**Initial response (R0)**", "", "```", clip(r["initial_response"]), "```", "",
                      "**Critique (F)**", "", "```", clip(r["critique"]), "```", "",
                      "**Revised response (Rf)**", "", "```", clip(r["final_response"]), "```", ""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines))
    print(counts)


if __name__ == "__main__":
    main()
