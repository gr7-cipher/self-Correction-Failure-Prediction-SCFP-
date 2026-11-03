"""GPU job A (one model load per generator):
  M9  residual-stream states at every layer: mean over R0's answer-span tokens, and the last R0 token
  M3  P(True) (next-token probability of "A" vs "B") and verbalized confidence (0-100, greedy, logged calls)
  M8  teacher-forced log-prob features for the excluded natural-set traces (exclusion sensitivity)
See REVISION_PLAN.md (M9, M3, M8).

    python scripts/revision/gpu_generator_signals.py --models qwen2.5-3b --out results/revision/gpuA
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
from scfp_bench.experiments.signals import ATYPE, answer_span, teacher_forced, verdict_features  # noqa: E402
from scfp_bench.llm import LOCAL, CallLog, Client, ModelSpec  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

PTRUE = ("Question: {q}\n\nProposed answer: {a}\n\nIs the proposed answer correct?\n(A) True\n(B) False\n"
         "Answer with the letter only.")
VERBAL = ("Question: {q}\n\nProposed answer: {a}\n\nHow confident are you that the proposed answer is correct? "
          "Reply with a number from 0 to 100 only.")


def traces(name, f="traces.jsonl"):
    run = MODELS[name][2]
    p = os.path.join(BENCH, "scfp_runs", run, "natural", f)
    rows = [r for r in read_jsonl(p) if r["source_model"] == name]
    for r in rows:
        r["prompt_raw"] = r["prompt"]
    return sorted(rows, key=lambda r: r["trace_id"])


def hidden_all_layers(r):
    """Mean residual state over R0's answer-span tokens and the state at R0's last token, at every layer."""
    import torch
    tok, model, dev = LOCAL.tok, LOCAL.model, LOCAL.device
    msgs = prompts.initial_messages(r["prompt_raw"], ATYPE[r["source"]]) + [{"role": "assistant", "content": r["initial_response"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    a0 = text.find(r["initial_response"])
    if a0 < 0:
        return None, None
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    idx = lambda a, b: [i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s]
    i_r0 = idx(a0, a0 + len(r["initial_response"]))
    span = answer_span(r["initial_response"])
    i_ans = idx(a0 + span[0], a0 + span[1]) if span else []
    if not i_ans:  # a final \boxed{} counts as the answer, as in the labels
        b = r["initial_response"].rfind("\\boxed{")
        if b >= 0:
            e = r["initial_response"].find("}", b)
            i_ans = idx(a0 + b, a0 + (e + 1 if e > 0 else len(r["initial_response"])))
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(dev), output_hidden_states=True)
    H = torch.stack(out.hidden_states[1:])[:, 0]  # layers x tokens x d (embedding output dropped)
    last = H[:, i_r0[-1]].float().cpu().numpy().astype(np.float16) if i_r0 else None
    ans = H[:, i_ans].float().mean(1).cpu().numpy().astype(np.float16) if i_ans else None
    return ans, last


def next_token_ab(q, a):
    import torch
    tok, model, dev = LOCAL.tok, LOCAL.model, LOCAL.device
    msgs = [{"role": "user", "content": PTRUE.format(q=q, a=a)}]
    # the assistant prefix puts the next token on the option letter (without it, "(A"/"(B" tokens take the mass)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "The answer is ("
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        lg = model(input_ids=ids).logits[0, -1].float()
    ia, ib = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    p = torch.softmax(lg[[ia, ib]], -1)
    return float(p[0]), float(torch.softmax(lg, -1)[[ia, ib]].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default="results/revision/gpuA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for name in a.models:
        model, rev, _ = MODELS[name]
        spec = ModelSpec(name=name, provider="local", model=model, max_tokens=8, revision=rev)
        LOCAL._load(spec)
        rows = traces(name)[:a.limit] if a.limit else traces(name)
        done_p = os.path.join(a.out, f"hidden_{name}.npz")
        if not os.path.exists(done_p):
            ids, ans, last, has = [], [], [], []
            for i, r in enumerate(rows):
                h_ans, h_last = hidden_all_layers(r)
                ids.append(r["trace_id"]); has.append(h_ans is not None)
                ans.append(h_ans if h_ans is not None else np.zeros_like(h_last))
                last.append(h_last)
                if i % 200 == 0:
                    print(f"  {name} hidden {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
            np.savez(done_p, trace_id=np.array(ids), ans=np.stack(ans), last=np.stack(last), has_ans=np.array(has))
        pt_p = os.path.join(a.out, f"ptrue_{name}.jsonl")
        if not os.path.exists(pt_p):
            with open(pt_p, "w") as f:
                for r in rows:
                    p, mass = next_token_ab(r["prompt_raw"], r["initial_response"])
                    f.write(json.dumps({"trace_id": r["trace_id"], "p_true": p, "ab_mass": mass,
                                        "model_version": LOCAL.version}) + "\n")
            print(f"  {name} p_true done {time.time() - t0:.0f}s", flush=True)
        cl = Client(CallLog(os.path.join(a.out, f"calls_verbal_{name}.jsonl")))
        for i in range(0, len(rows), a.batch_size):
            ch = rows[i:i + a.batch_size]
            cl.complete_batch(spec, [[{"role": "user", "content": VERBAL.format(q=r["prompt_raw"], a=r["initial_response"])}]
                                     for r in ch], 0.0, [0] * len(ch), "verbal")
        print(f"  {name} verbal done {time.time() - t0:.0f}s", flush=True)
        ex_p = os.path.join(a.out, f"excluded_features_{name}.jsonl")
        if not os.path.exists(ex_p):
            ex = traces(name, "excluded.jsonl")[:a.limit] if a.limit else traces(name, "excluded.jsonl")
            lspec = ModelSpec(name=name, provider="local", model=model, max_tokens=512, revision=rev)
            with open(ex_p, "w") as f:
                for r in ex:
                    tf = teacher_forced(lspec, r, [])
                    lp = tf[0] if tf else {}
                    f.write(json.dumps({"trace_id": r["trace_id"], **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                                       for k, v in lp.items()}, **verdict_features(r)}) + "\n")
            print(f"  {name} excluded features: {len(ex)} {time.time() - t0:.0f}s", flush=True)
        LOCAL.free()
        json.dump({"model": name, "n": len(rows), "version": rev}, open(os.path.join(a.out, f"done_{name}.json"), "w"))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
