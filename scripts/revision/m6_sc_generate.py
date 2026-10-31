"""M6 generation: k self-consistency samples (same initial prompt as R0, T=0.7, <=512 new tokens) for the kept
natural-set GSM8K traces of each generator. Calls are logged with the pipeline's CallLog (one file per
generator) and reused on resume. See REVISION_PLAN.md (M6).

    python scripts/revision/m6_sc_generate.py --models qwen2.5-3b --k 3 --out results/revision/m6/gen
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
from scfp_bench import prompts  # noqa: E402
from scfp_bench.llm import LOCAL, CallLog, Client, ModelSpec  # noqa: E402
from scfp_bench.stages import read_jsonl  # noqa: E402

# pinned commits of Appendix B (the ones every logged natural-set call recorded)
MODELS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "signals_mac_q05_big"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "signals_kaggle"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", "signals_kaggle"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", "signals_kaggle"),
}
T, MAX_TOKENS = 0.7, 512


def gsm8k_traces(name):
    run = MODELS[name][2]
    rows = read_jsonl(os.path.join(BENCH, "scfp_runs", run, "natural", "traces.jsonl"))
    return sorted([r for r in rows if r["source_model"] == name and r["source"] == "gsm8k"],
                  key=lambda r: r["problem_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/revision/m6/gen")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N traces per generator")
    ap.add_argument("--deadline-s", type=float, default=None, help="stop cleanly after this many seconds")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    for name in a.models:
        model, rev, _ = MODELS[name]
        spec = ModelSpec(name=name, provider="local", model=model, max_tokens=MAX_TOKENS, revision=rev)
        cl = Client(CallLog(os.path.join(a.out, f"calls_{name}.jsonl")))
        rows = gsm8k_traces(name)[:a.limit] if a.limit else gsm8k_traces(name)
        jobs = [(r, k) for r in rows for k in range(1, a.k + 1)]
        print(f"{name}: {len(rows)} GSM8K traces, {len(jobs)} samples", flush=True)
        for i in range(0, len(jobs), a.batch_size):
            if a.deadline_s and time.time() - t0 > a.deadline_s:
                print("deadline reached; stopping (resume with the same command)", flush=True)
                return
            chunk = jobs[i:i + a.batch_size]
            cl.complete_batch(spec, [prompts.initial_messages(r["prompt"], "number") for r, _ in chunk], T,
                              [k for _, k in chunk], "sample")
            if (i // a.batch_size) % 25 == 0:
                print(f"  {name}: {min(i + a.batch_size, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        LOCAL.free()
        json.dump({"model": name, "n_traces": len(rows), "k": a.k, "done": True},
                  open(os.path.join(a.out, f"done_{name}.json"), "w"))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
