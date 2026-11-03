"""GPU job C (M8): a 7B generator on the 600-problem subset, an alternative critique/revision template for two
generators, and a float16 vs bfloat16 check. Same labelling, verifier and teacher-forced features as the paper
(scfp_bench code); calls logged with CallLog. See REVISION_PLAN.md (M8).

    python scripts/revision/m8_gpu.py q7b prompt_alt precision --out results/revision/m8/gpu
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
from m6_sc_generate import MODELS  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.answers import extract, is_correct, normalize_number  # noqa: E402
from scfp_bench.experiments import signals as S  # noqa: E402
from scfp_bench.experiments.routing import PROGRAM_PROMPT, run_program  # noqa: E402
from scfp_bench.llm import LOCAL, CallLog, Client, ModelSpec  # noqa: E402
from scfp_bench.problems import problem_from_dict  # noqa: E402
from scfp_bench.stages import read_jsonl, trace_record, write_jsonl  # noqa: E402

Q7B = ModelSpec(name="qwen2.5-7b", provider="local", model="Qwen/Qwen2.5-7B-Instruct", max_tokens=512, quant="nf4")
ALT_CRITIQUE = ("Assume there may be an error in your previous answer. Solve the problem again independently, step by "
                "step, without relying on your previous solution. Then compare the two solutions and state clearly "
                "whether your previous answer was correct.")
ALT_REVISE = "Based on your independent solution and the comparison, give your final answer to the question."
BS = 8
LIMIT = None  # smoke tests only


def alt_critique_messages(question, answer_type, r0):
    return prompts.initial_messages(question, answer_type) + [{"role": "assistant", "content": r0},
                                                              {"role": "user", "content": ALT_CRITIQUE}]


def alt_revise_messages(question, answer_type, r0, f):
    return alt_critique_messages(question, answer_type, r0) + [
        {"role": "assistant", "content": f}, {"role": "user", "content": f"{ALT_REVISE}\n\n{prompts.FORMAT[answer_type]}"}]


def subset_problems(out_root):
    sub = json.load(open(os.path.join(out_root, "..", "subset.json")))
    ids = {p for v in sub["problems"].values() for p in v}
    allp = [problem_from_dict(d) for d in read_jsonl(os.path.join(BENCH, "scfp_runs", "signals_kaggle", "natural", "problems.jsonl"))]
    ps = sorted([p for p in allp if p.problem_id in ids], key=lambda p: p.problem_id)
    return ps[:LIMIT] if LIMIT else ps


def label(rows, probs):
    gold = {p.problem_id: p for p in probs}
    out = []
    for t in rows:
        p = gold[t["problem_id"]]
        e0 = extract(t["initial_response"], p.answer_type, accept_boxed=True)
        ef = extract(t["final_response"], p.answer_type, accept_boxed=True)
        r = {**t, "gold": p.gold, "r0_answer": e0.value, "rf_answer": ef.value,
             "r0_parsed_from_final_line": e0.from_final_line, "rf_parsed_from_final_line": ef.from_final_line,
             "r0_correct": is_correct(e0.value, p.gold, p.answer_type), "rf_correct": is_correct(ef.value, p.gold, p.answer_type)}
        r["label"] = int(not r["rf_correct"])
        r["kept"] = bool(e0.from_final_line and ef.from_final_line)
        out.append(r)
    return out


def features(spec, rows, path, alt=False):
    """Teacher-forced log-probs (with the template the trace was generated with) and critique cues."""
    if os.path.exists(path):
        return
    orig = prompts.critique_messages
    if alt:  # signals.teacher_forced rebuilds the conversation with prompts.critique_messages
        S.prompts.critique_messages = alt_critique_messages
    try:
        with open(path, "w") as f:
            for r in rows:
                r["prompt_raw"] = r["prompt"]
                tf = S.teacher_forced(spec, r, [])
                lp = tf[0] if tf else {}
                f.write(json.dumps({"trace_id": r["trace_id"], **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                                   for k, v in lp.items()}, **S.verdict_features(r)}) + "\n")
    finally:
        S.prompts.critique_messages = orig


def verifier(cl, spec, rows, path):
    if os.path.exists(path):
        return
    g = [r for r in rows if r["source"] == "gsm8k"]
    out = []
    for i in range(0, len(g), BS):
        ch = g[i:i + BS]
        gens = cl.complete_batch(spec, [[{"role": "user", "content": PROGRAM_PROMPT.format(q=r["prompt"])}] for r in ch],
                                 0.0, [0] * len(ch), "program")
        for r, x in zip(ch, gens):
            ex = run_program(x["text"])
            pot = normalize_number(ex["stdout"]) if ex["stdout"].strip() else None
            r0 = extract(r["initial_response"], "number").value
            fa = extract(r["critique"], "number").value
            out.append({"trace_id": r["trace_id"], "pot_answer": pot, "pot_error": ex["error"], "pot_ran": int(pot is not None),
                        "pot_agree_r0": float(is_correct(r0, pot, "number")) if pot is not None else None,
                        "pot_agree_f": float(is_correct(fa, pot, "number")) if pot is not None else None})
    write_jsonl(path, out)


def q7b(out, t0):
    d = os.path.join(out, "q7b")
    os.makedirs(d, exist_ok=True)
    probs = subset_problems(out)
    cl = Client(CallLog(os.path.join(d, "calls.jsonl")))
    tr_p = os.path.join(d, "traces_all.jsonl")
    if not os.path.exists(tr_p):
        traces = []
        for i in range(0, len(probs), BS):
            ch = probs[i:i + BS]
            z = [0] * len(ch)
            r0 = cl.complete_batch(Q7B, [prompts.initial_messages(p.question, p.answer_type) for p in ch], 0.0, z, "R0")
            f = cl.complete_batch(Q7B, [prompts.critique_messages(p.question, p.answer_type, a["text"]) for p, a in zip(ch, r0)], 0.0, z, "F")
            rf = cl.complete_batch(Q7B, [prompts.revise_messages(p.question, p.answer_type, a["text"], b["text"])
                                         for p, a, b in zip(ch, r0, f)], 0.0, z, "Rf")
            traces += [trace_record(p, Q7B, a, b, c) for p, a, b, c in zip(ch, r0, f, rf)]
            if (i // BS) % 10 == 0:
                print(f"  q7b {min(i + BS, len(probs))}/{len(probs)} {time.time() - t0:.0f}s", flush=True)
        write_jsonl(tr_p, label(traces, probs))
    rows = read_jsonl(tr_p)
    kept = [r for r in rows if r["kept"]]
    verifier(cl, Q7B, kept, os.path.join(d, "verifier.jsonl"))
    features(Q7B, kept, os.path.join(d, "features.jsonl"))
    LOCAL.free()
    print(f"  q7b done: {len(kept)}/{len(rows)} kept {time.time() - t0:.0f}s", flush=True)


def prompt_alt(out, t0, gens=("qwen2.5-3b", "smollm2-1.7b")):
    probs = {p.problem_id: p for p in subset_problems(out)}
    for g in gens:
        d = os.path.join(out, f"alt_{g}")
        os.makedirs(d, exist_ok=True)
        model, rev, run = MODELS[g]
        spec = ModelSpec(name=g, provider="local", model=model, max_tokens=512, revision=rev)
        cl = Client(CallLog(os.path.join(d, "calls.jsonl")))
        tr_p = os.path.join(d, "traces_all.jsonl")
        if not os.path.exists(tr_p):
            orig = []
            for f_ in ("traces.jsonl", "excluded.jsonl"):
                orig += [r for r in read_jsonl(os.path.join(BENCH, "scfp_runs", run, "natural", f_))
                         if r["source_model"] == g and r["problem_id"] in probs]
            orig.sort(key=lambda r: r["problem_id"])
            traces = []
            for i in range(0, len(orig), BS):
                ch = orig[i:i + BS]
                z = [0] * len(ch)
                pp = [probs[r["problem_id"]] for r in ch]
                f = cl.complete_batch(spec, [alt_critique_messages(p.question, p.answer_type, r["initial_response"])
                                             for p, r in zip(pp, ch)], 0.0, z, "F_alt")
                rf = cl.complete_batch(spec, [alt_revise_messages(p.question, p.answer_type, r["initial_response"], b["text"])
                                              for p, r, b in zip(pp, ch, f)], 0.0, z, "Rf_alt")
                for p, r, b, c in zip(pp, ch, f, rf):
                    t = {**{k: r[k] for k in ("trace_id", "problem_id", "source", "domain", "source_model", "prompt", "initial_response")},
                         "critique": b["text"], "final_response": c["text"], "template": "alt",
                         "call_ids": {"R0": r["call_ids"]["R0"], "F": b["call_id"], "Rf": c["call_id"]}}
                    traces.append(t)
                if (i // BS) % 10 == 0:
                    print(f"  alt {g} {min(i + BS, len(orig))}/{len(orig)} {time.time() - t0:.0f}s", flush=True)
            write_jsonl(tr_p, label(traces, list(probs.values())))
        kept = [r for r in read_jsonl(tr_p) if r["kept"]]
        features(spec, kept, os.path.join(d, "features.jsonl"), alt=True)
        LOCAL.free()
        print(f"  alt {g} done: {len(kept)} kept {time.time() - t0:.0f}s", flush=True)


def precision(out, t0, n=200):
    d = os.path.join(out, "precision")
    os.makedirs(d, exist_ok=True)
    probs = subset_problems(out)[:n]
    model, rev, _ = MODELS["qwen2.5-1.5b"]
    for dt in ("float16", "bfloat16"):
        spec = ModelSpec(name="qwen2.5-1.5b", provider="local", model=model, max_tokens=512, revision=rev, dtype=dt)
        cl = Client(CallLog(os.path.join(d, f"calls_{dt}.jsonl")))
        for i in range(0, len(probs), 16):
            ch = probs[i:i + 16]
            cl.complete_batch(spec, [prompts.initial_messages(p.question, p.answer_type) for p in ch], 0.0, [0] * len(ch), "R0")
        LOCAL.free()
        print(f"  precision {dt} done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", choices=["q7b", "prompt_alt", "precision"])
    ap.add_argument("--out", default="results/revision/m8/gpu")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--alt-gens", nargs="+", default=["qwen2.5-3b", "smollm2-1.7b"])
    a = ap.parse_args()
    global LIMIT
    LIMIT = a.limit
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for j in a.jobs:
        if j == "prompt_alt":
            prompt_alt(a.out, t0, a.alt_gens)
        else:
            {"q7b": q7b, "precision": precision}[j](a.out, t0)
        json.dump({"job": j, "elapsed_s": time.time() - t0}, open(os.path.join(a.out, f"done_{j}.json"), "w"))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
