"""Routing case study on the GSM8K hold-out (Section "Dynamic Routing").

Stages:
  generate  base model answers each problem (R0), critiques it (F), revises it
            (Rf, the intrinsic path), and separately writes a Python program
            (the extrinsic path), which is screened and executed with a timeout
  score     critique log-prob (Confidence-based router) and the trained
            predictor's failure probability (Dynamic router) for every problem
  evaluate  tune thresholds on the first part of the problems, report every
            strategy on the rest, from per-problem logged costs and latencies

Paths and their cost/latency (per problem, measured unless stated):
  intrinsic  tokens and latency of R0 + F + Rf
  extrinsic  tokens and latency of R0 + F + program generation + execution
  human      tokens and latency of R0 + F, plus an ASSUMED review of
             15x the mean intrinsic cost and 45 s; the human answer is assumed
             correct (oracle). These two assumptions are reported with the results.
Routed strategies also pay the predictor's (or scorer's) measured latency.

  python -m scfp_bench.experiments.routing generate --config configs/laptop_local.yaml
  python -m scfp_bench.experiments.routing score    --config ... --predictor deberta-scfp
  python -m scfp_bench.experiments.routing evaluate --config ...
"""

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

from .. import prompts
from ..answers import extract, is_correct
from ..llm import LOCAL, CallLog, Client, ModelSpec
from ..stages import read_jsonl, write_jsonl
from .metrics import bootstrap

HUMAN_COST_X = 15.0      # assumed: human review = 15 x mean intrinsic cost
HUMAN_LATENCY_S = 45.0   # assumed: seconds per human review
W_C, W_L, W_A = 0.4, 0.2, 0.4

PROGRAM_PROMPT = (
    "Write a short Python program that computes the answer to the problem below. "
    "Use only plain Python and the math module. The program must print only the final "
    "numeric answer. Put the program in a single ```python code block.\n\nProblem: {q}")

ALLOWED_IMPORTS = {"math", "fractions", "decimal"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
                   "getattr", "setattr", "delattr", "vars", "breakpoint", "help", "exit", "quit"}


def screen(code: str):
    """Static check before running model-written code. Returns a reason or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) \
                else [(node.module or "").split(".")[0]]
            bad = [m for m in mods if m not in ALLOWED_IMPORTS]
            if bad:
                return f"import not allowed: {bad}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"name not allowed: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute not allowed: {node.attr}"
    return None


def run_program(text: str, timeout=10.0):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = m[0] if m else text
    reason = screen(code)
    if reason:
        return {"stdout": "", "error": f"rejected: {reason}", "exec_s": 0.0}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        open(p, "w").write(code)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", "-S", p], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin"})
            return {"stdout": r.stdout[-500:], "error": r.stderr[-300:] or None,
                    "exec_s": round(time.time() - t0, 3)}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "error": "timeout", "exec_s": timeout}


def tokens(rec):
    u = rec.get("usage") or {}
    return (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)


def routing_problems(cfg, n, seed):
    held = read_jsonl(os.path.join(cfg["run_dir"], "holdout.jsonl"))
    rng = random.Random(seed)
    return sorted(rng.sample(held, n), key=lambda p: p["problem_id"])


def cmd_generate(cfg, args):
    out_dir = os.path.join(cfg["run_dir"], "routing")
    os.makedirs(out_dir, exist_ok=True)
    spec = ModelSpec.from_dict(next(m for m in cfg["models"] if m["name"] == args.base_model))
    cl = Client(CallLog(os.path.join(out_dir, "calls.jsonl")))
    probs = routing_problems(cfg, args.n, args.seed)
    out_path = os.path.join(out_dir, "traces.jsonl")
    done = {r["problem_id"]: r for r in read_jsonl(out_path)} if os.path.exists(out_path) else {}
    rows = list(done.values())
    for i, p in enumerate(probs):
        if p["problem_id"] in done:
            continue
        q = p["question"]
        r0 = cl.complete(spec, prompts.initial_messages(q, "number"), 0.0, stage="R0")
        f = cl.complete(spec, prompts.critique_messages(q, "number", r0["text"]), 0.0, stage="F")
        rf = cl.complete(spec, prompts.revise_messages(q, "number", r0["text"], f["text"]), 0.0, stage="Rf")
        pg = cl.complete(spec, [{"role": "user", "content": PROGRAM_PROMPT.format(q=q)}], 0.0, stage="program")
        ex = run_program(pg["text"])
        ext_ans = extract("Final answer: " + ex["stdout"].strip().splitlines()[-1] if ex["stdout"].strip() else "", "number").value
        rows.append({"problem_id": p["problem_id"], "question": q, "gold": p["gold"],
                     "R0": r0["text"], "F": f["text"], "Rf": rf["text"], "program": pg["text"],
                     "exec": ex, "call_ids": {k: v["call_id"] for k, v in
                                              (("R0", r0), ("F", f), ("Rf", rf), ("program", pg))},
                     "int_correct": is_correct(extract(rf["text"], "number").value, p["gold"], "number"),
                     "ext_answer": ext_ans, "ext_correct": is_correct(ext_ans, p["gold"], "number"),
                     "r0_correct": is_correct(extract(r0["text"], "number").value, p["gold"], "number"),
                     "tok": {"R0": tokens(r0), "F": tokens(f), "Rf": tokens(rf), "program": tokens(pg)},
                     "lat": {"R0": r0["latency_s"], "F": f["latency_s"], "Rf": rf["latency_s"],
                             "program": pg["latency_s"], "exec": ex["exec_s"]},
                     "model_version": r0["model_version"]})
        write_jsonl(out_path, rows)
        print(f"  routing generate: {len(rows)}/{len(probs)}", flush=True)
    LOCAL.free()


def cmd_score(cfg, args):
    import torch
    from transformers import AutoTokenizer
    from .train_encoder import build, encode
    out_dir = os.path.join(cfg["run_dir"], "routing")
    rows = read_jsonl(os.path.join(out_dir, "traces.jsonl"))
    # 1) critique log-prob under the base model (Confidence-based router)
    spec = ModelSpec.from_dict(next(m for m in cfg["models"] if m["name"] == args.base_model))
    from .logprobs import score as lp_score
    as_scfp = [{"trace_id": r["problem_id"], "source_model": spec.name, "source": "gsm8k", "prompt": r["question"],
                "initial_response": r["R0"], "critique": r["F"]} for r in rows]
    t0 = time.time()
    lps = {x["trace_id"]: x["mean_logprob"] for x in lp_score(spec, as_scfp)}
    lp_lat = (time.time() - t0) / len(rows)
    LOCAL.free()
    # 2) trained predictor's failure probability (Dynamic router)
    res = json.load(open(os.path.join(cfg["run_dir"], "results", f"{args.predictor}.json")))
    a = res["args"]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a["model"])
    model = build(a["arch"], a["model"], special_attention=not a.get("no_special_attention", False))
    model.load_state_dict(torch.load(os.path.join(cfg["run_dir"], "results", f"{args.predictor}.pt"), map_location="cpu"))
    model.to(device).eval()
    pf, lat = {}, []
    for r in as_scfp:
        ids, seg = encode(tok, r, a["max_len"])
        t0 = time.time()
        with torch.no_grad():
            x = torch.tensor([ids], device=device)
            sg = torch.tensor([seg], device=device)
            if a["arch"] == "scfp":
                out = model(input_ids=x, attention_mask=torch.ones_like(x), token_type_ids=sg)["binary_logits"]
            else:
                out = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
            pf[r["trace_id"]] = float(torch.softmax(out.float(), -1)[0, 1])
        lat.append(time.time() - t0)
    for r in rows:
        r["critique_logprob"] = lps[r["problem_id"]]
        r["p_fail"] = pf[r["problem_id"]]
    write_jsonl(os.path.join(out_dir, "traces.jsonl"), rows)
    json.dump({"predictor": args.predictor, "predictor_latency_s": float(np.mean(lat)),
               "predictor_latency_median_s": float(np.median(lat)), "device": device,
               "scorer_latency_s": lp_lat}, open(os.path.join(out_dir, "score_meta.json"), "w"), indent=2)
    print(f"scored {len(rows)} problems; predictor latency {np.mean(lat) * 1000:.0f} ms on {device}")


def path_tables(rows):
    tok_int = np.array([r["tok"]["R0"] + r["tok"]["F"] + r["tok"]["Rf"] for r in rows], float)
    unit = tok_int.mean()
    C = {"int": tok_int / unit,
         "ext": np.array([r["tok"]["R0"] + r["tok"]["F"] + r["tok"]["program"] for r in rows], float) / unit,
         "hum": np.array([r["tok"]["R0"] + r["tok"]["F"] for r in rows], float) / unit + HUMAN_COST_X}
    L = {"int": np.array([r["lat"]["R0"] + r["lat"]["F"] + r["lat"]["Rf"] for r in rows]),
         "ext": np.array([r["lat"]["R0"] + r["lat"]["F"] + r["lat"]["program"] + r["lat"]["exec"] for r in rows]),
         "hum": np.array([r["lat"]["R0"] + r["lat"]["F"] for r in rows]) + HUMAN_LATENCY_S}
    A = {"int": np.array([r["int_correct"] for r in rows], float),
         "ext": np.array([r["ext_correct"] for r in rows], float), "hum": np.ones(len(rows))}
    return C, L, A, unit


def evaluate_choice(choice, C, L, A, extra_lat=0.0):
    idx = np.arange(len(choice))
    c = np.array([C[k][i] for i, k in zip(idx, choice)])
    l = np.array([L[k][i] for i, k in zip(idx, choice)]) + extra_lat
    a = np.array([A[k][i] for i, k in zip(idx, choice)])
    return c, l, a


def score_of(c, l, a, Cn, Ln):
    return 1 - (W_C * c.mean() / Cn + W_L * l.mean() / Ln + W_A * (1 - a.mean()) ** 2)


def route(s, lo, hi):
    return ["int" if v < lo else "ext" if v < hi else "hum" for v in s]


def cmd_evaluate(cfg, args):
    out_dir = os.path.join(cfg["run_dir"], "routing")
    rows = read_jsonl(os.path.join(out_dir, "traces.jsonl"))
    meta = json.load(open(os.path.join(out_dir, "score_meta.json")))
    tune, test = rows[:args.n_tune], rows[args.n_tune:]
    grid = np.round(np.arange(0.10, 0.9001, 0.05), 2)

    def fit(rows_, score_key, lat):
        C, L, A, _ = path_tables(rows_)
        Cn, Ln = C["hum"].mean(), L["hum"].mean()
        s = np.array([r[score_key] for r in rows_])
        best = None
        for lo in grid:
            for hi in grid:
                if lo < hi:
                    c, l, a = evaluate_choice(route(s, lo, hi), C, L, A, lat)
                    v = score_of(c, l, a, Cn, Ln)
                    if best is None or v > best[0]:
                        best = (v, lo, hi)
        return best[1], best[2]

    # Confidence score = rank of -logprob within the tuning set, mapped to [0, 1]
    tune_neg = np.sort([-r["critique_logprob"] for r in tune])
    for r in rows:
        r["conf_score"] = float(np.searchsorted(tune_neg, -r["critique_logprob"], side="right") / len(tune_neg))
    th_dyn = fit(tune, "p_fail", meta["predictor_latency_s"])
    th_conf = fit(tune, "conf_score", meta["scorer_latency_s"])

    C, L, A, unit = path_tables(test)
    Cn, Ln = C["hum"].mean(), L["hum"].mean()
    n = len(test)
    strategies = {}

    def add(name, choice, lat=0.0, thresholds=None):
        c, l, a = evaluate_choice(choice, C, L, A, lat)
        share = {k: sum(x == k for x in choice) / n for k in ("int", "ext", "hum")}
        lo_c = share["hum"] * C["hum"].mean() + (1 - share["hum"]) * min(C["int"].mean(), C["ext"].mean())
        strategies[name] = {"accuracy": float(a.mean()), "accuracy_ci95": bootstrap(a.astype(int), np.ones(n, int))["accuracy"],
                            "latency_s": float(l.mean()), "relative_cost": float(c.mean()),
                            "share": share, "overall_score": float(score_of(c, l, a, Cn, Ln)),
                            "thresholds": thresholds, "cost_lower_bound_check": float(lo_c)}

    add("Always Intrinsic", ["int"] * n)
    add("Always Extrinsic", ["ext"] * n)
    add("Always Human (oracle)", ["hum"] * n)
    # uniform random routing: exact expectation over the three paths
    c = (C["int"] + C["ext"] + C["hum"]) / 3
    l = (L["int"] + L["ext"] + L["hum"]) / 3
    a = (A["int"] + A["ext"] + A["hum"]) / 3
    strategies["Random (uniform, expected)"] = {
        "accuracy": float(a.mean()), "accuracy_ci95": None, "latency_s": float(l.mean()),
        "relative_cost": float(c.mean()), "share": {"int": 1 / 3, "ext": 1 / 3, "hum": 1 / 3},
        "overall_score": float(score_of(c, l, a, Cn, Ln)), "thresholds": None}
    add("Confidence-based", route([r["conf_score"] for r in test], *th_conf), meta["scorer_latency_s"], list(th_conf))
    add("Dynamic (our predictor)", route([r["p_fail"] for r in test], *th_dyn), meta["predictor_latency_s"], list(th_dyn))

    out = {"n_tune": len(tune), "n_test": n, "base_model": rows[0]["model_version"],
           "cost_unit_tokens": float(unit), "assumptions": {"human_cost_x": HUMAN_COST_X,
                                                            "human_latency_s": HUMAN_LATENCY_S,
                                                            "human_accuracy": 1.0},
           "weights": {"w_c": W_C, "w_l": W_L, "w_a": W_A}, "predictor": meta,
           "program_rejected": sum("rejected" in (r["exec"]["error"] or "") for r in rows),
           "program_errors": sum(bool(r["exec"]["error"]) for r in rows),
           "r0_accuracy_test": float(np.mean([r["r0_correct"] for r in test])),
           "strategies": strategies}
    json.dump(out, open(os.path.join(cfg["run_dir"], "results", "routing.json"), "w"), indent=2)
    for k, v in strategies.items():
        print(f"  {k:28s} acc={v['accuracy']:.3f} lat={v['latency_s']:.1f}s cost={v['relative_cost']:.2f}x "
              f"human={v['share']['hum']:.0%} score={v['overall_score']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score", "evaluate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-model", default="qwen2.5-1.5b")
    ap.add_argument("--predictor", default="deberta-scfp")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-tune", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    {"generate": cmd_generate, "score": cmd_score, "evaluate": cmd_evaluate}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
