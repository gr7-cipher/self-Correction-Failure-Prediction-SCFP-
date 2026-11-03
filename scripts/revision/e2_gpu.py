"""Round-2 GPU job.

  sc      E2: k = 3 self-consistency samples for every kept natural-set StrategyQA and LogiQA trace (same settings as
          M6: R0 prompt with the source's answer type, T = 0.7, <= 512 new tokens), logged with CallLog.
  timing  A2: per-item GPU seconds and tokens of each verifier on 100 natural-set GSM8K traces of Qwen2.5-1.5B
          (seed 0), one T4, batch size 1, model loading excluded.
See REVISION_PLAN.md (Round 2, E2 and A2).

    python scripts/revision/e2_gpu.py sc --models qwen2.5-0.5b qwen2.5-1.5b --out results/revision/e2/gen
    python scripts/revision/e2_gpu.py timing --out results/revision/a2
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from m6_sc_generate import MAX_TOKENS, MODELS, T  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.experiments.signals import ATYPE  # noqa: E402
from scfp_bench.llm import LOCAL, CallLog, Client, ModelSpec  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

SOURCES = ("strategyqa", "logiqa")


def kept(name, sources):
    run = MODELS[name][2]
    rows = read_jsonl(os.path.join(BENCH, "scfp_runs", run, "natural", "traces.jsonl"))
    return sorted([r for r in rows if r["source_model"] == name and r["source"] in sources], key=lambda r: r["trace_id"])


def sc(a, t0):
    os.makedirs(a.out, exist_ok=True)
    for name in a.models:
        if os.path.exists(os.path.join(a.out, f"done_{name}.json")):
            continue
        model, rev, _ = MODELS[name]
        spec = ModelSpec(name=name, provider="local", model=model, max_tokens=MAX_TOKENS, revision=rev)
        cl = Client(CallLog(os.path.join(a.out, f"calls_{name}.jsonl")))
        rows = kept(name, SOURCES)[:a.limit] if a.limit else kept(name, SOURCES)
        jobs = [(r, k) for r in rows for k in range(1, a.k + 1)]
        print(f"{name}: {len(rows)} StrategyQA/LogiQA traces, {len(jobs)} samples", flush=True)
        for i in range(0, len(jobs), a.batch_size):
            if a.deadline_s and time.time() - t0 > a.deadline_s:
                print("deadline reached; stopping (resume with the same command)", flush=True)
                return
            ch = jobs[i:i + a.batch_size]
            cl.complete_batch(spec, [prompts.initial_messages(r["prompt"], ATYPE[r["source"]]) for r, _ in ch], T,
                              [k for _, k in ch], "sample")
            if (i // a.batch_size) % 25 == 0:
                print(f"  {name}: {min(i + a.batch_size, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        LOCAL.free()
        json.dump({"model": name, "n_traces": len(rows), "k": a.k, "done": True},
                  open(os.path.join(a.out, f"done_{name}.json"), "w"))
    print("ALL DONE", flush=True)


def timing(a, t0):
    import torch
    from gpu_external_judges import JUDGE, JUDGE_PROMPT, RM, commit_of, nf4
    from scfp_bench.experiments import signals as S
    from scfp_bench.experiments.routing import PROGRAM_PROMPT, run_program, tokens
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    g = "qwen2.5-1.5b"
    rows = kept(g, ("gsm8k",))
    rows = sorted(random.Random(0).sample(rows, a.n), key=lambda r: r["trace_id"])
    model, rev, _ = MODELS[g]
    spec = ModelSpec(name=g, provider="local", model=model, max_tokens=MAX_TOKENS, revision=rev)
    cl = Client(CallLog(os.path.join(a.out, "calls_timing.jsonl")))
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    out = {r["trace_id"]: {"trace_id": r["trace_id"]} for r in rows}
    LOCAL._load(spec)
    for r in rows:  # generator signals: one teacher-forced pass over P, R0 and F
        r["prompt_raw"] = r["prompt"]
        msgs = prompts.critique_messages(r["prompt"], "number", r["initial_response"]) + [{"role": "assistant", "content": r["critique"]}]
        n_in = len(LOCAL.tok(LOCAL.tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"])
        sync(); t1 = time.time()
        S.teacher_forced(spec, r, [])
        sync()
        out[r["trace_id"]].update({"signals_s": time.time() - t1, "signals_tokens": n_in})
    for r in rows:  # PoT: program generation (GPU) and sandboxed execution (CPU)
        sync(); t1 = time.time()
        pg = cl.complete_batch(spec, [[{"role": "user", "content": PROGRAM_PROMPT.format(q=r["prompt"])}]], 0.0, [0], "program")[0]
        sync()
        ex = run_program(pg["text"])
        out[r["trace_id"]].update({"pot_s": time.time() - t1, "pot_tokens": tokens(pg), "pot_exec_cpu_s": ex["exec_s"]})
    for r in rows:  # SC3: three samples, one at a time
        sync(); t1 = time.time()
        recs = [cl.complete_batch(spec, [prompts.initial_messages(r["prompt"], "number")], T, [k], "sample")[0] for k in (1, 2, 3)]
        sync()
        out[r["trace_id"]].update({"sc3_s": time.time() - t1, "sc3_tokens": sum(tokens(x) for x in recs)})
    gen_dtype = str(LOCAL.model.dtype)
    LOCAL.free()
    versions = {"generator": f"{model}@{rev}"}
    tok = AutoTokenizer.from_pretrained(RM)
    m = AutoModelForSequenceClassification.from_pretrained(RM, quantization_config=nf4(), device_map="cuda",
                                                           dtype=torch.float16, num_labels=1).eval()
    versions["reward_model"] = f"{RM}@{commit_of(m, RM)}"
    for r in rows:
        conv = prompts.initial_messages(r["prompt"], "number") + [{"role": "assistant", "content": r["initial_response"]}]
        ids = tok(tok.apply_chat_template(conv, tokenize=False), add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        sync(); t1 = time.time()
        with torch.no_grad():
            m(ids)
        sync()
        out[r["trace_id"]].update({"rm_s": time.time() - t1, "rm_tokens": int(ids.shape[1])})
    del m
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(JUDGE)
    m = AutoModelForCausalLM.from_pretrained(JUDGE, quantization_config=nf4(), device_map="cuda", dtype=torch.float16).eval()
    versions["judge"] = f"{JUDGE}@{commit_of(m, JUDGE)}"
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(
            q=r["prompt"], a=r["initial_response"], f=r["critique"])}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        sync(); t1 = time.time()
        with torch.no_grad():
            m(input_ids=ids)
        sync()
        out[r["trace_id"]].update({"judge_s": time.time() - t1, "judge_tokens": int(ids.shape[1])})
    with open(os.path.join(a.out, "timing.jsonl"), "w") as f:
        for v in out.values():
            f.write(json.dumps(v) + "\n")
    json.dump({"device": torch.cuda.get_device_name(0), "n": len(rows), "batch_size": 1, "versions": versions,
               "generator_dtype": gen_dtype,
               "external_quant": "nf4", "elapsed_s": time.time() - t0}, open(os.path.join(a.out, "timing_meta.json"), "w"), indent=1)
    print("TIMING DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["sc", "timing"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--deadline-s", type=float, default=None)
    a = ap.parse_args()
    {"sc": sc, "timing": timing}[a.job](a, time.time())


if __name__ == "__main__":
    main()
