"""E1 GPU job (round 2): routing traces on the 1,000-problem GSM8K hold-out for one generator.

Per problem: greedy R0, F, Rf and a program (the original routing paths), three self-consistency samples
(T = 0.7, same prompt as R0, as in M6), and the teacher-forced signal features of R0 and F. Every generation is
logged with CallLog and reused on resume. See REVISION_PLAN.md (Round 2, E1).

    python scripts/revision/e1_gpu.py --models qwen2.5-0.5b --out results/revision/e1/gpu
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from m6_sc_generate import MODELS  # noqa: E402  (pinned commits)
from scfp_bench import prompts  # noqa: E402
from scfp_bench.answers import extract, is_correct  # noqa: E402
from scfp_bench.experiments import signals as S  # noqa: E402
from scfp_bench.experiments.routing import PROGRAM_PROMPT, run_program, tokens  # noqa: E402
from scfp_bench.llm import LOCAL, CallLog, Client, ModelSpec  # noqa: E402
from scfp_bench.stages import read_jsonl, write_jsonl  # noqa: E402

HOLDOUT = os.path.join(BENCH, "scfp_runs", "laptop_v1", "holdout.jsonl")
BS, K, T_SC = 8, 3, 0.7


def generate(name, d, t0, deadline, limit):
    model, rev, _ = MODELS[name]
    spec = ModelSpec(name=name, provider="local", model=model, max_tokens=512, revision=rev)
    probs = sorted(read_jsonl(HOLDOUT), key=lambda p: p["problem_id"])[:limit]
    cl = Client(CallLog(os.path.join(d, "calls.jsonl")))
    rows = []
    for i in range(0, len(probs), BS):
        if deadline and time.time() - t0 > deadline:
            print("deadline reached; stopping (resume with the same command)", flush=True)
            return None
        ch = probs[i:i + BS]
        z = [0] * len(ch)
        q = [p["question"] for p in ch]
        r0 = cl.complete_batch(spec, [prompts.initial_messages(x, "number") for x in q], 0.0, z, "R0")
        f = cl.complete_batch(spec, [prompts.critique_messages(x, "number", a["text"]) for x, a in zip(q, r0)], 0.0, z, "F")
        rf = cl.complete_batch(spec, [prompts.revise_messages(x, "number", a["text"], b["text"])
                                      for x, a, b in zip(q, r0, f)], 0.0, z, "Rf")
        pg = cl.complete_batch(spec, [[{"role": "user", "content": PROGRAM_PROMPT.format(q=x)}] for x in q], 0.0, z, "program")
        sc = [cl.complete_batch(spec, [prompts.initial_messages(x, "number") for x in q], T_SC, [k] * len(ch), "sample")
              for k in range(1, K + 1)]
        for j, p in enumerate(ch):
            ex = run_program(pg[j]["text"])
            ext = extract("Final answer: " + ex["stdout"].strip().splitlines()[-1] if ex["stdout"].strip() else "", "number").value
            e0, ef = extract(r0[j]["text"], "number", accept_boxed=True), extract(rf[j]["text"], "number", accept_boxed=True)
            rows.append({"problem_id": p["problem_id"], "question": p["question"], "gold": p["gold"], "generator": name,
                         "R0": r0[j]["text"], "F": f[j]["text"], "Rf": rf[j]["text"], "program": pg[j]["text"], "exec": ex,
                         "samples": [s[j]["text"] for s in sc],
                         "call_ids": {"R0": r0[j]["call_id"], "F": f[j]["call_id"], "Rf": rf[j]["call_id"],
                                      "program": pg[j]["call_id"], "samples": [s[j]["call_id"] for s in sc]},
                         "r0_answer": e0.value, "rf_answer": ef.value,
                         "r0_parsed_from_final_line": e0.from_final_line, "rf_parsed_from_final_line": ef.from_final_line,
                         "r0_correct": is_correct(e0.value, p["gold"], "number"),
                         "int_correct": is_correct(ef.value, p["gold"], "number"),
                         "ext_answer": ext, "ext_correct": is_correct(ext, p["gold"], "number"),
                         "tok": {"R0": tokens(r0[j]), "F": tokens(f[j]), "Rf": tokens(rf[j]), "program": tokens(pg[j]),
                                 "samples": sum(tokens(s[j]) for s in sc)},
                         "lat": {"R0": r0[j]["latency_s"], "F": f[j]["latency_s"], "Rf": rf[j]["latency_s"],
                                 "program": pg[j]["latency_s"], "exec": ex["exec_s"],
                                 "samples": sum(s[j]["latency_s"] for s in sc)},
                         "model_version": r0[j]["model_version"], "batch_size": BS})
        if (i // BS) % 10 == 0:
            print(f"  {name} {min(i + BS, len(probs))}/{len(probs)} {time.time() - t0:.0f}s", flush=True)
    write_jsonl(os.path.join(d, "routing_traces.jsonl"), rows)
    return spec, rows


def features(spec, rows, path):
    """Teacher-forced log-prob features of R0 and F (as in the natural set) and the critique cues, timed."""
    import torch
    if os.path.exists(path):
        return
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    with open(path, "w") as f:
        for r in rows:
            t = {"prompt_raw": r["question"], "source": "gsm8k", "initial_response": r["R0"], "critique": r["F"],
                 "r0_answer": r["r0_answer"]}
            sync()
            t1 = time.time()
            tf = S.teacher_forced(spec, t, [])
            sync()
            lp = tf[0] if tf else {}
            f.write(json.dumps({"problem_id": r["problem_id"], "tf_latency_s": round(time.time() - t1, 4),
                                **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in lp.items()},
                                **S.verdict_features(t)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="results/revision/e1/gpu")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    t0 = time.time()
    for name in a.models:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "done.json")):
            continue
        res = generate(name, d, t0, a.deadline_s, a.limit)
        if res is None:
            return
        spec, rows = res
        features(spec, rows, os.path.join(d, "features.jsonl"))
        LOCAL.free()
        json.dump({"model": name, "n": len(rows), "elapsed_s": time.time() - t0, "batch_size": BS},
                  open(os.path.join(d, "done.json"), "w"))
        print(f"  {name} done {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
