"""A2 (round 2): cost per item of each verifier, from the timing run of the round-2 GPU job (e2_gpu.py timing):
100 natural-set GSM8K traces of Qwen2.5-1.5B, one T4, batch size 1, model loading excluded. See REVISION_PLAN.md (A2).

    .venv/bin/python scripts/revision/a2_cost.py --timing results/revision/e2/gen/a2 --out results/revision/a2
"""

import argparse
import json
import os

import numpy as np

KEYS = [("signals", "Generator signals (teacher-forced pass)"), ("pot", "Program check (generation)"),
        ("sc3", "Three samples (SC3)"), ("rm", "Reward model, 8B NF4"), ("judge", "LLM judge, 7B NF4")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", default="results/revision/e2/gen/a2")
    ap.add_argument("--out", default="results/revision/a2")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.timing, "timing.jsonl"))]
    meta = json.load(open(os.path.join(a.timing, "timing_meta.json")))
    res = {"n": len(rows), "meta": meta}
    for k, _ in KEYS:
        s = np.array([r[f"{k}_s"] for r in rows])
        t = np.array([r[f"{k}_tokens"] for r in rows])
        res[k] = {"gpu_s_mean": float(s.mean()), "gpu_s_median": float(np.median(s)), "gpu_s_p90": float(np.percentile(s, 90)),
                  "tokens_mean": float(t.mean())}
    e = np.array([r["pot_exec_cpu_s"] for r in rows])
    res["pot"]["exec_cpu_s_mean"] = float(e.mean())
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "verifier_cost.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k, _ in KEYS}, indent=1))


if __name__ == "__main__":
    main()
