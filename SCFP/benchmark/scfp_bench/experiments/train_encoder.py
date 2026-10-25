"""Fine-tune encoder predictors on the SCFP benchmark (run on a GPU).

  --arch scfp     DeBERTaFailurePredictor from SCFP/src (segment embeddings,
                  segment-aware attention, multi-task head); three-stage
                  schedule: binary -> multi-class on failures -> joint (alpha)
  --arch plain    AutoModelForSequenceClassification (BERT, RoBERTa, plain
                  DeBERTa baselines), binary only

Inputs are the partial trace (P, R0, F) only. Each segment gets a token budget;
R0 keeps its end (where the final answer is), P and F keep their start.

Examples (Kaggle/Colab T4):
  python -m scfp_bench.experiments.train_encoder --config configs/laptop_local.yaml \\
      --arch scfp --model microsoft/deberta-v3-large --name deberta-scfp
  python -m scfp_bench.experiments.train_encoder --config configs/laptop_local.yaml \\
      --arch plain --model roberta-large --name roberta-large
  # cross-model: train on the Qwen models, test on SmolLM2
  python -m scfp_bench.experiments.train_encoder --config ... --arch scfp \\
      --model microsoft/deberta-v3-large --name deberta-scfp-xmodel \\
      --train-models qwen2.5-0.5b,qwen2.5-1.5b --test-models smollm2-1.7b
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from ..stages import FAILURE_MODES, UNCLEAR
from .data import dump_json, load_splits, write_predictions
from .metrics import bootstrap, summarize

MODES = FAILURE_MODES + [UNCLEAR]


FIELDS = {"P": "prompt", "R0": "initial_response", "F": "critique"}


def encode(tok, r, max_len, budgets=(0.25, 0.375, 0.375), inputs=("P", "R0", "F")):
    """Token ids and segment ids for (P, R0, F) within max_len.
    Components left out of `inputs` (input ablations) are encoded as empty."""
    sep = [tok.sep_token_id]
    room = max_len - 2 - 2 * len(sep)
    parts = [tok(r[FIELDS[k]], add_special_tokens=False)["input_ids"] if k in inputs else []
             for k in ("P", "R0", "F")]
    caps = [int(room * b) for b in budgets]
    spare = room - sum(min(len(p), c) for p, c in zip(parts, caps))
    for i, (p, c) in enumerate(zip(parts, caps)):  # give unused budget to longer parts
        if len(p) > c and spare > 0:
            extra = min(spare, len(p) - c)
            caps[i] += extra
            spare -= extra
    P, R0, F = parts[0][:caps[0]], parts[1][-caps[1]:] if caps[1] else [], parts[2][:caps[2]]
    ids = [tok.cls_token_id] + P + sep + R0 + sep + F + [tok.sep_token_id]
    seg = [0] * (1 + len(P) + 1) + [1] * (len(R0) + 1) + [2] * (len(F) + 1)
    return ids, seg


def batches(items, bs, shuffle, seed):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


def collate(batch, pad_id, device):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id)
    seg = torch.full((len(batch), L), 3)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n], seg[i, :n], mask[i, :n] = torch.tensor(b["ids"]), torch.tensor(b["seg"]), 1
    y = torch.tensor([b["y"] for b in batch])
    m = torch.tensor([b["mode"] for b in batch])
    return ids.to(device), seg.to(device), mask.to(device), y.to(device), m.to(device)


def build(arch, model_name, special_attention=True):
    if arch == "scfp":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
        from scfp.models.deberta import DeBERTaFailurePredictor
        return DeBERTaFailurePredictor(model_name=model_name, num_failure_modes=len(MODES),
                                       use_specialized_attention=special_attention)
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def forward(model, arch, ids, seg, mask, use_segments=True):
    if arch == "scfp":
        # Without segment embeddings every token gets segment 0, which also turns
        # the segment-aware attention mask into ordinary full attention.
        out = model(input_ids=ids, attention_mask=mask,
                    token_type_ids=seg if use_segments else torch.zeros_like(seg))
        return out["binary_logits"], out["multiclass_logits"]
    return model(input_ids=ids, attention_mask=mask).logits, None


@torch.no_grad()
def predict(model, arch, data, bs, pad_id, device, use_segments=True):
    model.eval()
    pb, pm = [], []
    for b in batches(data, bs, False, 0):
        ids, seg, mask, _, _ = collate(b, pad_id, device)
        lb, lm = forward(model, arch, ids, seg, mask, use_segments)
        pb.append(torch.softmax(lb.float(), -1)[:, 1].cpu())
        if lm is not None:
            pm.append(lm.float().argmax(-1).cpu())
    model.train()
    return torch.cat(pb).numpy(), (torch.cat(pm).numpy() if pm else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", choices=["scfp", "plain"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs-binary", type=int, default=10)
    ap.add_argument("--epochs-multi", type=int, default=15)
    ap.add_argument("--epochs-joint", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-models", default=None, help="comma list; default all")
    ap.add_argument("--test-models", default=None, help="comma list; default all")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap examples per split")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (auto if unset)")
    ap.add_argument("--inputs", default="P,R0,F", help="input ablation, e.g. P,F")
    ap.add_argument("--no-segments", action="store_true", help="ablation: all tokens in segment 0")
    ap.add_argument("--no-special-attention", action="store_true", help="ablation: drop the extra layer")
    ap.add_argument("--binary-only", action="store_true", help="ablation: skip multi-class and joint stages")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_dir = cfg["run_dir"]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    S = load_splits(run_dir)
    keep = lambda rows, ms: [r for r in rows if ms is None or r["source_model"] in ms.split(",")]
    S = {"train": keep(S["train"], args.train_models), "val": keep(S["val"], args.train_models),
         "test": keep(S["test"], args.test_models)}
    if args.limit:
        S = {k: v[:args.limit] for k, v in S.items()}
    has_modes = all(r.get("failure_type") not in (None, "UNLABELED") for v in S.values() for r in v if r["label"] == 1)
    inputs = tuple(args.inputs.split(","))
    use_seg = not args.no_segments
    data = {k: [{"ids": e[0], "seg": e[1], "y": r["label"],
                 "mode": MODES.index(r["failure_type"]) if has_modes and r["label"] == 1 else -100,
                 "row": r} for r in v for e in [encode(tok, r, args.max_len, inputs=inputs)]] for k, v in S.items()}

    model = build(args.arch, args.model, special_attention=not args.no_special_attention).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pad = tok.pad_token_id
    ytr = np.array([d["y"] for d in data["train"]])
    w_bin = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(1)), dtype=torch.float, device=device)
    if has_modes:
        mtr = np.array([d["mode"] for d in data["train"] if d["mode"] >= 0])
        cnt = np.bincount(mtr, minlength=len(MODES)).clip(1)
        w_mode = torch.tensor(len(mtr) / (len(MODES) * cnt), dtype=torch.float, device=device)

    stages_ = [("binary", args.epochs_binary)]
    if args.arch == "scfp" and has_modes and not args.binary_only:
        stages_ += [("multi", args.epochs_multi), ("joint", args.epochs_joint)]
    elif args.arch == "scfp":
        print("failure-mode labels missing (annotation not finished): training the binary stage only")
    log, t0, best_state, best_val = [], time.time(), None, -1
    for stage, epochs in stages_:
        bad = 0
        for ep in range(epochs):
            train = data["train"] if stage != "multi" else [d for d in data["train"] if d["mode"] >= 0]
            for b in batches(train, args.batch_size, True, args.seed + ep):
                ids, seg, mask, y, m = collate(b, pad, device)
                lb, lm = forward(model, args.arch, ids, seg, mask, use_seg)
                l_bin = torch.nn.functional.cross_entropy(lb.float(), y, weight=w_bin)
                if stage == "binary":
                    loss = l_bin
                else:
                    l_multi = torch.nn.functional.cross_entropy(lm.float(), m, weight=w_mode, ignore_index=-100)
                    loss = l_multi if stage == "multi" else args.alpha * l_bin + (1 - args.alpha) * l_multi
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pv, _ = predict(model, args.arch, data["val"], args.batch_size, pad, device, use_seg)
            yv = np.array([d["y"] for d in data["val"]])
            f1 = f1_score(yv, (pv >= 0.5).astype(int), average="macro")
            log.append({"stage": stage, "epoch": ep + 1, "val_binary_macro_f1": float(f1)})
            print(f"  {stage} epoch {ep + 1}: val macro-F1 {f1:.4f}", flush=True)
            if f1 > best_val:
                best_val, bad = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
    model.load_state_dict(best_state)

    pt, pmode = predict(model, args.arch, data["test"], args.batch_size, pad, device, use_seg)
    yt = np.array([d["y"] for d in data["test"]])
    yp = (pt >= 0.5).astype(int)
    rows = [{"trace_id": d["row"]["trace_id"], "model": args.name, "task": "binary", "split": "test",
             "source_model": d["row"]["source_model"], "source": d["row"]["source"],
             "y_true": int(t), "y_pred": int(a), "p_fail": float(p)}
            for d, t, a, p in zip(data["test"], yt, yp, pt)]
    write_predictions(run_dir, args.name, "binary", rows)
    result = {"args": vars(args), "device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "wall_clock_s": round(time.time() - t0, 1), "best_val_macro_f1": best_val,
              "log": log, "test": {**summarize(yt, yp, pt), "ci95": bootstrap(yt, yp, pt)}}
    if pmode is not None and has_modes:
        fails = [i for i, d in enumerate(data["test"]) if d["mode"] >= 0]
        mrows = [{"trace_id": data["test"][i]["row"]["trace_id"], "model": args.name, "task": "multiclass",
                  "split": "test", "y_true": int(data["test"][i]["mode"]), "y_pred": int(pmode[i])} for i in fails]
        write_predictions(run_dir, args.name, "multiclass", mrows)
        result["test_multiclass"] = summarize([r["y_true"] for r in mrows], [r["y_pred"] for r in mrows])
    dump_json(run_dir, f"{args.name}.json", result)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
