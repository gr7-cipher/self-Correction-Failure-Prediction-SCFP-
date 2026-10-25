"""Figures and analyses computed from the trained predictor and prediction files.

  ROC curves      empirical ROC of every binary predictor (step functions)
  confusion       binary confusion matrix of our model on the test split
  latency         measured single-trace inference latency of our model
  attention       [CLS] attention mass on prompt / R0 / critique tokens in the
                  last backbone layer, correct vs. incorrect test predictions,
                  Welch t-test

  python -m scfp_bench.experiments.analysis --config configs/laptop_local.yaml --predictor deberta-scfp
"""

import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from ..stages import read_jsonl  # noqa: E402
from .data import dump_json, load_splits  # noqa: E402
from .make_tables import NAMES, ENCODERS  # noqa: E402

LABELS = dict(NAMES, **dict(ENCODERS))


def roc_figure(rd, out):
    plt.figure(figsize=(5.2, 4.6))
    for f in sorted(glob.glob(os.path.join(rd, "predictions", "*__binary.jsonl"))):
        rows = read_jsonl(f)
        name = rows[0]["model"]
        if name.startswith(("abl-", "xmodel-", "smoke")) or name in ("majority",):
            continue
        y = [r["y_true"] for r in rows]
        p = [r["p_fail"] for r in rows]
        fpr, tpr, _ = roc_curve(y, p)
        plt.step(fpr, tpr, where="post", lw=1.6, label=LABELS.get(name, name))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_roc.pdf")); plt.close()


def confusion_figure(rd, out, name):
    rows = read_jsonl(os.path.join(rd, "predictions", f"{name}__binary.jsonl"))
    y = [r["y_true"] for r in rows]; yp = [r["y_pred"] for r in rows]
    cm = confusion_matrix(y, yp, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["success", "failure"]); ax.set_yticks([0, 1], ["success", "failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_confusion.pdf")); plt.close()
    return {"matrix": cm.tolist(), "n": int(cm.sum()), "accuracy": float(np.trace(cm) / cm.sum())}


def model_analyses(rd, name):
    import torch
    from transformers import AutoTokenizer
    from .train_encoder import build, encode
    res = json.load(open(os.path.join(rd, "results", f"{name}.json")))
    a = res["args"]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a["model"])
    model = build(a["arch"], a["model"], special_attention=not a.get("no_special_attention", False))
    model.load_state_dict(torch.load(os.path.join(rd, "results", f"{name}.pt"), map_location="cpu"))
    model.to(device).eval()
    preds = {r["trace_id"]: r for r in read_jsonl(os.path.join(rd, "predictions", f"{name}__binary.jsonl"))}
    test = load_splits(rd)["test"]
    lat, att = [], {"correct": [], "incorrect": []}
    for r in test:
        ids, seg = encode(tok, r, a["max_len"])
        x = torch.tensor([ids], device=device); s = torch.tensor([seg], device=device)
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=torch.ones_like(x), token_type_ids=s,
                        return_attention_weights=True)
        if device == "mps":
            torch.mps.synchronize()
        lat.append(time.time() - t0)
        A = out["attention_weights"][-1][0].float().mean(0)[0].cpu().numpy()  # [CLS] row, mean over heads
        sg = np.array(seg)
        shares = {k: float(A[sg == v].sum()) for k, v in (("prompt", 0), ("R0", 1), ("critique", 2))}
        p = preds[r["trace_id"]]
        att["correct" if p["y_pred"] == p["y_true"] else "incorrect"].append(shares)
    lat = np.array(lat[3:])  # drop warm-up calls
    fc = [s["critique"] for s in att["correct"]]; fi = [s["critique"] for s in att["incorrect"]]
    t = stats.ttest_ind(fc, fi, equal_var=False) if len(fc) > 1 and len(fi) > 1 else None
    attention = {"n_correct": len(fc), "n_incorrect": len(fi),
                 "critique_share_correct_mean": float(np.mean(fc)), "critique_share_correct_sd": float(np.std(fc, ddof=1)),
                 "critique_share_incorrect_mean": float(np.mean(fi)), "critique_share_incorrect_sd": float(np.std(fi, ddof=1)),
                 "welch_t": float(t.statistic) if t else None, "welch_df": float(t.df) if t and hasattr(t, "df") else None,
                 "p_value": float(t.pvalue) if t else None,
                 "mean_shares_all": {k: float(np.mean([s[k] for g in att.values() for s in g])) for k in ("prompt", "R0", "critique")}}
    latency = {"device": device, "hardware": "Apple M1 (8 GB)" if device == "mps" else device,
               "median_ms": float(np.median(lat) * 1000), "mean_ms": float(np.mean(lat) * 1000),
               "p90_ms": float(np.percentile(lat, 90) * 1000), "n": int(len(lat)), "batch_size": 1}
    return attention, latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictor", default="deberta-scfp")
    ap.add_argument("--out", default="../../../paper_tables")
    args = ap.parse_args()
    rd = yaml.safe_load(open(args.config))["run_dir"]
    os.makedirs(args.out, exist_ok=True)
    roc_figure(rd, args.out)
    cm = confusion_figure(rd, args.out, args.predictor)
    attention, latency = model_analyses(rd, args.predictor)
    dump_json(rd, "analysis.json", {"confusion": cm, "attention": attention, "latency": latency})
    print(json.dumps({"confusion": cm, "attention": attention, "latency": latency}, indent=2))


if __name__ == "__main__":
    main()
