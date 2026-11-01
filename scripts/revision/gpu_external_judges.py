"""GPU job B (M3): an off-the-shelf reward model and an LLM judge score every kept natural-set trace, using only
pre-revision information. 4-bit NF4 on a T4; the exact commits are resolved at load time and recorded.
See REVISION_PLAN.md (M3).

    python scripts/revision/gpu_external_judges.py --which rm judge --out results/revision/gpuB
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "SCFP", "benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
from gpu_generator_signals import traces  # noqa: E402
from m6_sc_generate import MODELS  # noqa: E402
from scfp_bench import prompts  # noqa: E402
from scfp_bench.experiments.signals import ATYPE  # noqa: E402

RM = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
JUDGE = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_PROMPT = ("Below are a question, a model's initial answer, and the model's own critique of that answer. "
                "The model will next revise its answer using the critique.\n\nQuestion:\n{q}\n\nInitial answer:\n{a}\n\n"
                "Critique:\n{f}\n\nWill the model's final answer after the revision be wrong? Answer Yes or No.")


def commit_of(model, name):
    c = getattr(model.config, "_commit_hash", None)
    if not c:
        try:
            from huggingface_hub import model_info
            c = model_info(name).sha
        except Exception:
            c = "unknown"
    return c


def nf4():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)


def all_rows(limit):
    rows = []
    for g in MODELS:
        t = traces(g)
        rows += t[:limit] if limit else t
    return rows


def run_rm(rows, out):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    p = os.path.join(out, "reward_model.jsonl")
    done = {json.loads(l)["trace_id"] + json.loads(l)["generator"] for l in open(p)} if os.path.exists(p) else set()
    tok = AutoTokenizer.from_pretrained(RM)
    m = AutoModelForSequenceClassification.from_pretrained(RM, quantization_config=nf4(), device_map="cuda",
                                                           dtype=torch.float16, num_labels=1).eval()
    ver = f"{RM}@{commit_of(m, RM)}"
    t0 = time.time()
    with open(p, "a") as f:
        for i, r in enumerate(rows):
            if r["trace_id"] + r["source_model"] in done:
                continue
            conv = prompts.initial_messages(r["prompt_raw"], ATYPE[r["source"]]) + [{"role": "assistant", "content": r["initial_response"]}]
            # render then tokenize: newer transformers return a BatchEncoding from tokenize=True
            text = tok.apply_chat_template(conv, tokenize=False)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                s = float(m(ids).logits[0][0])
            f.write(json.dumps({"trace_id": r["trace_id"], "generator": r["source_model"], "reward": s, "model_version": ver,
                                "quant": "nf4"}) + "\n")
            if i % 500 == 0:
                print(f"  rm {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    del m
    torch.cuda.empty_cache()


def run_judge(rows, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = os.path.join(out, "judge.jsonl")
    done = {json.loads(l)["trace_id"] + json.loads(l)["generator"] for l in open(p)} if os.path.exists(p) else set()
    tok = AutoTokenizer.from_pretrained(JUDGE)
    m = AutoModelForCausalLM.from_pretrained(JUDGE, quantization_config=nf4(), device_map="cuda", dtype=torch.float16).eval()
    ver = f"{JUDGE}@{commit_of(m, JUDGE)}"
    iy, inn = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    t0 = time.time()
    with open(p, "a") as f:
        for i, r in enumerate(rows):
            if r["trace_id"] + r["source_model"] in done:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(
                q=r["prompt_raw"], a=r["initial_response"], f=r["critique"])}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                lg = m(input_ids=ids).logits[0, -1].float()
            pr = torch.softmax(lg, -1)
            f.write(json.dumps({"trace_id": r["trace_id"], "generator": r["source_model"],
                                "p_wrong": float(pr[iy] / (pr[iy] + pr[inn])), "yes_no_mass": float(pr[iy] + pr[inn]),
                                "model_version": ver, "quant": "nf4"}) + "\n")
            if i % 500 == 0:
                print(f"  judge {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["rm", "judge"])
    ap.add_argument("--out", default="results/revision/gpuB")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = all_rows(a.limit)
    print(f"{len(rows)} traces", flush=True)
    if "rm" in a.which:
        run_rm(rows, a.out)
    if "judge" in a.which:
        run_judge(rows, a.out)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
