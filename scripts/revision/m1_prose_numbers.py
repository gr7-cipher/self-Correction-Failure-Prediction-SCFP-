"""Numbers that the submitted text typed by hand about SCFP v1.0 (outcome counts, exclusion rate, test-split
failure rate, truncation at 768 tokens), recomputed from the released run so that the revised text can take them
from generated macros. See REVISION_LOG.md (M1).

    .venv/bin/python scripts/revision/m1_prose_numbers.py --out results/revision/m1
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "..", "SCFP", "benchmark"))
sys.path.insert(0, BENCH)
from scfp_bench.experiments.train_encoder import FIELDS  # noqa: E402

RUN = os.path.join(BENCH, "scfp_runs", "laptop_v1")


def rows(name):
    return [json.loads(l) for l in open(os.path.join(RUN, name))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/revision/m1")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--max-len", type=int, default=768)
    a = ap.parse_args()
    lab, ex = rows("labeled.jsonl"), rows("excluded.jsonl")
    test = rows(os.path.join("splits", "test.jsonl"))
    r0 = np.array([r["r0_correct"] for r in lab], bool)
    rf = np.array([r["rf_correct"] for r in lab], bool)
    out = {"n_kept": len(lab), "n_excluded": len(ex), "n_generated": len(lab) + len(ex),
           "excluded_share": len(ex) / (len(lab) + len(ex)),
           "r0_wrong": int((~r0).sum()), "repaired": int((rf & ~r0).sum()), "r0_right": int(r0.sum()),
           "broken": int((~rf & r0).sum()), "failure_share": float((~rf).mean()),
           "test_n": len(test), "test_failure_share": float(np.mean([r["label"] for r in test]))}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    alltr = rows(os.path.join("splits", "train.jsonl")) + rows(os.path.join("splits", "val.jsonl")) + test
    n_full = []
    for r in alltr:
        parts = [len(tok(r[FIELDS[k]], add_special_tokens=False)["input_ids"]) for k in ("P", "R0", "F")]
        n_full.append(sum(parts) + 4)  # [CLS], two [SEP] between parts, final [SEP]
    n_full = np.array(n_full)
    out.update({"n_traces_tokenized": len(n_full), "median_tokens": float(np.median(n_full)),
                "share_untruncated": float(np.mean(n_full <= a.max_len)), "max_len": a.max_len, "tokenizer": a.model})
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "prose_numbers.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
