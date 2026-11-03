"""Mean token log-probability of each critique under the model that wrote it.

This is the Confidence heuristic's score. The critique F is scored by teacher
forcing: the conversation up to the critique request is the prompt, and the
log-probabilities of F's tokens are averaged.

  python -m scfp_bench.experiments.logprobs --config configs/laptop_local.yaml
"""

import argparse
import os

import yaml

from .. import prompts
from ..llm import LOCAL, ModelSpec
from ..stages import read_jsonl, write_jsonl
from .data import load_splits


def score(spec: ModelSpec, rows):
    import torch
    LOCAL._load(spec)
    tok, model, dev = LOCAL.tok, LOCAL.model, LOCAL.device
    out = []
    for i, r in enumerate(rows):
        msgs = prompts.critique_messages(r["prompt"], _atype(r), r["initial_response"])
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        f_ids = tok(r["critique"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([list(prompt_ids) + f_ids], device=dev)
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(f_ids, device=dev)[:, None])[:, 0]
        out.append({"trace_id": r["trace_id"], "source_model": r["source_model"],
                    "mean_logprob": float(lp.mean()) if len(f_ids) else 0.0,
                    "n_tokens": len(f_ids), "scorer": LOCAL.version})
        del logits, ids
        if (i + 1) % 25 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"  {spec.name}: {i + 1}/{len(rows)}", flush=True)
    return out


def _atype(r):
    return {"gsm8k": "number", "strategyqa": "bool", "logiqa": "choice"}[r["source"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    run_dir = cfg["run_dir"]
    rows = [r for split in load_splits(run_dir).values() for r in split]
    specs = {m["name"]: ModelSpec.from_dict(m) for m in cfg["models"]}
    out_path = os.path.join(run_dir, "features", "critique_logprob.jsonl")
    done = {r["trace_id"]: r for r in read_jsonl(out_path)} if os.path.exists(out_path) else {}
    results = list(done.values())
    for name in sorted({r["source_model"] for r in rows}):
        todo = [r for r in rows if r["source_model"] == name and r["trace_id"] not in done]
        if todo:
            print(f"scoring {len(todo)} critiques with {name}", flush=True)
            results += score(specs[name], todo)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_jsonl(out_path, results)
    LOCAL.free()
    print(f"wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
