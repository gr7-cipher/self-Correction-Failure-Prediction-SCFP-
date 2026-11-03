"""E4: blinded input for independent AI raters. Each rater gets the R0 and R_f text and the answer type only, under its own
random ids and order. The key is written to a separate directory. These raters are not human raters: their output never
goes in the human_* columns.

    .venv/bin/python scripts/revision/e4_blind_prepare.py --out <dir>
"""

import argparse
import csv
import json
import os
import random

TYPES = {"gsm8k": "a number", "strategyqa": "yes or no", "logiqa": "an option letter (A, B, C or D)"}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = list(csv.DictReader(open("data/audit/extraction_audit_sheet.csv")))
for rater, seed in (("rater1", 101), ("rater2", 202)):
    rng = random.Random(seed)
    order = rows[:]
    rng.shuffle(order)
    ids = rng.sample(range(1000, 10000), len(order))
    key = {}
    with open(os.path.join(a.out, f"{rater}_input.txt"), "w") as f:
        for r, i in zip(order, ids):
            key[f"T{i}"] = r["audit_id"]
            f.write(f"===== ITEM T{i} | expected answer type: {TYPES[r['source']]}\n--- RESPONSE 1 ---\n{r['r0_text']}\n--- RESPONSE 2 ---\n{r['rf_text']}\n\n")
    json.dump(key, open(os.path.join(a.out, "key", f"{rater}_key.json"), "w"))
