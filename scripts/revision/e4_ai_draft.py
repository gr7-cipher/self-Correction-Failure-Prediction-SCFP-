"""E4: Claude's reading of the audit sheet, kept apart from the human columns. It does not replace the human audit and
is not read by e4_audit_score.py. Rule: the answer after an explicit "Final answer" marker counts, unless the text after
it states a different final result or the revision has no final answer of its own.

    .venv/bin/python scripts/revision/e4_ai_draft.py
"""

import csv
import json

SHEET, OUT = "data/audit/extraction_audit_sheet.csv", "data/audit/extraction_audit_ai_draft.csv"
JUDGED = """YN YN NY NY YN NN NY YN NN NN NY YN NY YN NN NN YN NY YY YY NY NY YY NY NY NN NY YY NY YY NN YY YY NN YY NY YN YN YY NY
NN NY YN YN NN NY YN NY NN YN YY YY NN NN YY YN YY YY YY YN NN NY NN YN NY NY NN NN YY NN YY NN NN YY YY YY NY YY NY NN
YN YN NY YN YY YY NY YN NN NN YN NN YY NY YN YN YN YY YY NY""".split()
ANSWER = {"A001": (None, "none"), "A061": ("490", None), "A097": (None, "none"), "A100": (None, "no")}
NOTES = {
    "A001": "R_f: 'Final answer' is a sentence saying all options are consistent; no option chosen. Extractor took A.",
    "A005": "R_f: marker says Yes; the text after it says 'Not necessarily'. Marker kept.",
    "A019": "R_f: marker says Yes; the text after it says flying fish do not have good eyesight. Marker kept.",
    "A023": "R0: marker says yes; the text after it says neither film was the better investment. Marker kept.",
    "A040": "R0: reasoning concludes D, marker says C. Both wrong against gold.",
    "A044": "R0: marker says yes; the text after it argues a honey badger would not fit. Marker kept. If read as no, R0 is wrong.",
    "A047": "R_f: restated text says C, marker says D. Marker kept.",
    "A061": "R0: marker says 360, the worked text after it ends at 490. Taken as 490. Wrong against gold either way.",
    "A076": "R0: marker says $6, the worked text after it ends at $9. Taken as 9, as the extractor did. If read as 6, R0 is wrong.",
    "A097": "R_f: the revised section is cut off with no final answer; -2 appears only in the restated original.",
    "A100": "R_f: marker says Yes; the revised answer after it says No. Taken as no. This changes the label: the extractor's yes is wrong, no is right.",
}
AMBIGUOUS = {"A001", "A005", "A019", "A023", "A044", "A061", "A076", "A097", "A100"}

rows = list(csv.DictReader(open(SHEET)))
assert len(rows) == len(JUDGED) == 100
meta = {}
for line in open("data/audit/extraction_audit_sample.jsonl"):
    m = json.loads(line)
    meta[m["audit_id"]] = m
out = []
for r, j in zip(rows, JUDGED):
    a0, af = ANSWER.get(r["audit_id"], (None, None))
    out.append({"audit_id": r["audit_id"], "trace_id": r["trace_id"], "generator": r["generator"], "source": r["source"], "gold": r["gold"],
                "r0_extracted": r["r0_extracted"], "rf_extracted": r["rf_extracted"], "ai_r0_answer": a0 or r["r0_extracted"], "ai_r0_correct": "yes" if j[0] == "Y" else "no",
                "ai_rf_answer": af or r["rf_extracted"], "ai_rf_correct": "yes" if j[1] == "Y" else "no",
                "ai_ambiguous": "yes" if r["audit_id"] in AMBIGUOUS else "no", "ai_notes": NOTES.get(r["audit_id"], "")})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print("keys", sorted(next(iter(meta.values())).keys()))
d0 = [o["audit_id"] for o in out if o["ai_r0_answer"] != o["r0_extracted"]]
df = [o["audit_id"] for o in out if o["ai_rf_answer"] != o["rf_extracted"]]
print("answer differs: R0", d0, "R_f", df)
for k0, kf in (("r0_correct", "rf_correct"), ("r0_correct", "final_correct")):
    if k0 in next(iter(meta.values())) and kf in next(iter(meta.values())):
        l0 = [o["audit_id"] for o in out if (o["ai_r0_correct"] == "yes") != bool(meta[o["audit_id"]][k0])]
        lf = [o["audit_id"] for o in out if (o["ai_rf_correct"] == "yes") != bool(meta[o["audit_id"]][kf])]
        print("label differs from pipeline: R0", l0, "R_f", lf)
        break
print("ambiguous", len(AMBIGUOUS))
