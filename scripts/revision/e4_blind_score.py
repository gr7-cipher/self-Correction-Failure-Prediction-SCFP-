"""E4: score the blind AI raters. They saw only the response texts, so correctness is computed here against the gold answer.
This is an AI check, not the human audit: it writes nothing to the human_* columns and e4_audit_score.py does not read it.

    .venv/bin/python scripts/revision/e4_blind_score.py
"""

import csv
import json
import math
import os

D, OUT = "data/audit/ai_blind", "results/revision/e4/ai_blind_check.json"


def norm(v):
    v = str(v).strip().lower().replace(",", "").replace("$", "")
    try:
        return float(v)
    except ValueError:
        return v


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


sheet = {r["audit_id"]: r for r in csv.DictReader(open("data/audit/extraction_audit_sheet.csv"))}
meta = {m["audit_id"]: m for m in map(json.loads, open("data/audit/extraction_audit_sample.jsonl"))}
R = {}
for rater in ("rater1", "rater2"):
    key = json.load(open(os.path.join(D, f"{rater}_key.json")))
    rows = [json.loads(l) for l in open(os.path.join(D, f"{rater}_output.jsonl"))]
    assert len(rows) == 100 and {r["id"] for r in rows} == set(key), rater
    R[rater] = {key[r["id"]]: r for r in rows}
out, res = [], {}
for a in sorted(sheet):
    s, m, x, y = sheet[a], meta[a], R["rater1"][a], R["rater2"][a]
    g = norm(s["gold"])
    o = {"audit_id": a, "generator": s["generator"], "source": s["source"], "gold": s["gold"]}
    for k, ex, pl, ra in (("r0", "r0_extracted", "r0_correct", "r1"), ("rf", "rf_extracted", "rf_correct", "r2")):
        o[f"{k}_extracted"], o[f"{k}_pipeline_correct"] = s[ex], bool(m[pl])
        for n, r in (("rater1", x), ("rater2", y)):
            o[f"{k}_{n}_answer"], o[f"{k}_{n}_ambiguous"] = r[f"{ra}_answer"], bool(r[f"{ra}_ambiguous"])
            o[f"{k}_{n}_correct"] = norm(r[f"{ra}_answer"]) == g
    o["rater1_note"], o["rater2_note"] = x["note"], y["note"]
    out.append(o)
with open("data/audit/extraction_audit_ai_blind.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for k in ("r0", "rf"):
    e = {"n": len(out)}
    for n in ("rater1", "rater2"):
        ans = [o["audit_id"] for o in out if norm(o[f"{k}_{n}_answer"]) != norm(o[f"{k}_extracted"])]
        lab = [o["audit_id"] for o in out if o[f"{k}_{n}_correct"] != o[f"{k}_pipeline_correct"]]
        e[n] = {"answer_differs": ans, "label_differs": lab, "label_error_ci95": wilson(len(lab), len(out)),
                "ambiguous": [o["audit_id"] for o in out if o[f"{k}_{n}_ambiguous"]]}
    e["raters_answer_differ"] = [o["audit_id"] for o in out if norm(o[f"{k}_rater1_answer"]) != norm(o[f"{k}_rater2_answer"])]
    e["raters_correct_kappa"] = kappa([o[f"{k}_rater1_correct"] for o in out], [o[f"{k}_rater2_correct"] for o in out])
    both = [o["audit_id"] for o in out if o[f"{k}_rater1_correct"] == o[f"{k}_rater2_correct"] != o[f"{k}_pipeline_correct"]]
    e["both_raters_disagree_with_pipeline"] = both
    e["both_label_error_ci95"] = wilson(len(both), len(out))
    res[k] = e
fl = {o["audit_id"]: o for o in out}
res["final_label_both_disagree"] = res["rf"]["both_raters_disagree_with_pipeline"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=0))
for a in sorted(set(res["r0"]["rater1"]["answer_differs"] + res["r0"]["rater2"]["answer_differs"] + res["rf"]["rater1"]["answer_differs"] + res["rf"]["rater2"]["answer_differs"])):
    o = fl[a]
    print(a, s["gold"] if False else o["gold"], "R0", o["r0_extracted"], o["r0_rater1_answer"], o["r0_rater2_answer"], "| RF", o["rf_extracted"], o["rf_rater1_answer"], o["rf_rater2_answer"], "|", (o["rater1_note"] or "")[:110], "||", (o["rater2_note"] or "")[:110])
