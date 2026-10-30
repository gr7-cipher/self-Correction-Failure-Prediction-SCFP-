"""Generator-side signals for predicting self-correction failure.

The text-only predictors read (P, R0, F). The generator knows more about its own
uncertainty than its text shows, so this experiment adds signals that are all
available BEFORE the revision Rf is generated:

  self-consistency  K extra answers sampled at T=0.7; agreement with R0,
                    number of distinct answers, answer entropy
  log-probs         mean token log-prob of R0, of R0's final-answer span, of F
                    (teacher forcing in the generator itself)
  critique verdict  does F claim an error / claim correctness, and does F
                    contain a different answer than R0
  hidden states     generator activations at the last token of R0 and of F,
                    at the middle and the last layer

Two datasets:
  natural  problems sampled WITHOUT the difficulty filter (the filter keeps only
           problems the model is unsure about, which removes the easiest-to-
           predict cases); disjoint from SCFP v1 and from the routing hold-out
  v1       the SCFP v1.0 traces of the paper (only samples and features are new)

Stages (all resumable; generation calls are logged and cached):
  python -m scfp_bench.experiments.signals build    --config configs/signals.yaml
  python -m scfp_bench.experiments.signals features --config configs/signals.yaml
  python -m scfp_bench.experiments.signals evaluate --config configs/signals.yaml
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np
import yaml

from .. import prompts
from ..answers import extract, is_correct
from ..llm import LOCAL, CallLog, Client, ModelSpec
from ..problems import load_sources, problem_from_dict
from ..stages import read_jsonl, trace_record, write_jsonl

ATYPE = {"gsm8k": "number", "strategyqa": "bool", "logiqa": "choice"}
ERROR_RE = re.compile(r"\b(error|errors|mistake|mistakes|incorrect|wrong|miscalculat\w*|overlook\w*|"
                      r"should (?:be|have)|not correct|inaccura\w*|misread|misunderst\w*)\b", re.I)
CORRECT_RE = re.compile(r"\b(is correct|are correct|was correct|correctly|accurate|verified|holds|"
                        r"no errors?|no mistakes?|is valid|is right|confirm\w*)\b", re.I)


# ---------------------------------------------------------------- helpers
def D(cfg, *p):
    out = os.path.join(cfg["run_dir"], *p)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


def specs(cfg):
    return {m["name"]: ModelSpec.from_dict(m) for m in cfg["models"]}


# ---------------------------------------------------------------- build
def natural_problems(cfg):
    """Sample problems per source with no difficulty filter, disjoint from v1 and the hold-out."""
    sc = cfg["signals"]
    pool = load_sources(sc["sources"])
    excluded = set()
    for f in sc.get("exclude_problem_files", []):
        if os.path.exists(f):
            excluded |= {json.loads(l)["problem_id"] for l in open(f)}
    rng = random.Random(cfg["seed"])
    out = []
    for src, n in sc["natural_per_source"].items():
        cand = [p for p in pool if p.source == src and p.problem_id not in excluded]
        rng.shuffle(cand)
        out += cand[:n]
    return out


def cmd_build(cfg):
    sc = cfg["signals"]
    cl = Client(CallLog(D(cfg, "calls.jsonl")))
    S = specs(cfg)
    gen_models = [S[n] for n in sc["models"]]
    bs = cfg.get("batch_size", 8)

    # natural set: greedy traces
    nat_p = D(cfg, "natural", "problems.jsonl")
    if not os.path.exists(nat_p):
        write_jsonl(nat_p, [p.to_dict() for p in natural_problems(cfg)])
    probs = [problem_from_dict(d) for d in read_jsonl(nat_p)]
    print(f"natural set: {len(probs)} problems x {len(gen_models)} models", flush=True)
    traces = []
    for spec in gen_models:
        for i in range(0, len(probs), bs):
            chunk = probs[i:i + bs]
            z = [0] * len(chunk)
            r0 = cl.complete_batch(spec, [prompts.initial_messages(p.question, p.answer_type) for p in chunk], 0.0, z, "R0")
            f = cl.complete_batch(spec, [prompts.critique_messages(p.question, p.answer_type, a["text"])
                                         for p, a in zip(chunk, r0)], 0.0, z, "F")
            rf = cl.complete_batch(spec, [prompts.revise_messages(p.question, p.answer_type, a["text"], b["text"])
                                          for p, a, b in zip(chunk, r0, f)], 0.0, z, "Rf")
            traces += [trace_record(p, spec, a, b, c) for p, a, b, c in zip(chunk, r0, f, rf)]
            print(f"  natural {spec.name}: {min(i + bs, len(probs))}/{len(probs)}", flush=True)
        LOCAL.free()
    gold = {p.problem_id: p for p in probs}
    rows = []
    for t in traces:
        p = gold[t["problem_id"]]
        ab = sc.get("accept_boxed", False)
        e0 = extract(t["initial_response"], p.answer_type, accept_boxed=ab)
        ef = extract(t["final_response"], p.answer_type, accept_boxed=ab)
        rows.append({**t, "gold": p.gold, "r0_answer": e0.value, "rf_answer": ef.value,
                     "r0_parsed_from_final_line": e0.from_final_line, "rf_parsed_from_final_line": ef.from_final_line,
                     "r0_correct": is_correct(e0.value, p.gold, p.answer_type),
                     "rf_correct": is_correct(ef.value, p.gold, p.answer_type)})
    for r in rows:
        r["label"] = int(not r["rf_correct"])
        r["outcome_cell"] = f"{'right' if r['r0_correct'] else 'wrong'}->{'right' if r['rf_correct'] else 'wrong'}"
    keep = [r for r in rows if r["r0_parsed_from_final_line"] and r["rf_parsed_from_final_line"]]
    write_jsonl(D(cfg, "natural", "traces.jsonl"), keep)
    write_jsonl(D(cfg, "natural", "excluded.jsonl"), [r for r in rows if r not in keep])
    print(f"natural set: {len(keep)} traces kept, {len(rows) - len(keep)} without a final-answer line", flush=True)

    # samples for self-consistency, for both sets
    K, T = sc.get("samples", 0), sc.get("sample_temperature", 0.7)
    sample_spec = {n: ModelSpec(**{**S[n].__dict__, "max_tokens": sc.get("sample_max_tokens", S[n].max_tokens)})
                   for n in sc["models"]}
    for set_name in ("natural", "v1"):
        rows = load_set(cfg, set_name)
        for name in sc["models"]:
            mine = sorted({r["problem_id"]: r for r in rows if r["source_model"] == name}.values(),
                          key=lambda r: r["problem_id"])
            jobs = [(r, k) for r in mine for k in range(1, K + 1)]
            for i in range(0, len(jobs), bs):
                chunk = jobs[i:i + bs]
                cl.complete_batch(sample_spec[name], [prompts.initial_messages(r["prompt_raw"], ATYPE[r["source"]])
                                                      for r, _ in chunk], T, [k for _, k in chunk], "sample")
                if (i // bs) % 10 == 0:
                    print(f"  samples {set_name} {name}: {min(i + bs, len(jobs))}/{len(jobs)}", flush=True)
            LOCAL.free()


def load_set(cfg, name):
    """Traces of a set, with the raw problem text used for generation."""
    if name == "natural":
        p = D(cfg, "natural", "traces.jsonl")
        rows = read_jsonl(p) if os.path.exists(p) else []
    else:
        rows = [r for s in ("train", "val", "test")
                for r in read_jsonl(os.path.join(cfg["signals"]["v1_run_dir"], "splits", f"{s}.jsonl"))]
        lim = cfg["signals"].get("v1_limit")  # smoke tests only
        if lim:
            rows = [r for m in cfg["signals"]["models"] for r in [x for x in rows if x["source_model"] == m][:lim]]
    for r in rows:
        r["prompt_raw"] = r["prompt"]
        r["set"] = name
    return rows


# ---------------------------------------------------------------- features
def answer_span(text):
    """Character span of the value on the last 'Final answer:' line, or None."""
    m = list(re.finditer(r"final answer\s*[:\-]\s*(.+)", text, re.I))
    return (m[-1].start(1), m[-1].end(1)) if m else None


def teacher_forced(spec, r, layers_frac):
    """Log-probs of R0 / its answer / F and hidden states at the ends of R0 and F."""
    import torch
    LOCAL._load(spec)
    tok, model, dev = LOCAL.tok, LOCAL.model, LOCAL.device
    msgs = prompts.critique_messages(r["prompt_raw"], ATYPE[r["source"]], r["initial_response"]) + \
        [{"role": "assistant", "content": r["critique"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    r0_start = text.find(r["initial_response"])
    f_start = text.rfind(r["critique"])
    if r0_start < 0 or f_start < 0:
        return None
    r0_end, f_end = r0_start + len(r["initial_response"]), f_start + len(r["critique"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(dev)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    logp = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
    tok_lp = np.concatenate([[np.nan], logp])  # log-prob of token i given tokens < i

    def idx(a, b):
        return [i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s]
    i_r0, i_f = idx(r0_start, r0_end), idx(f_start, f_end)
    span = answer_span(r["initial_response"])
    i_ans = idx(r0_start + span[0], r0_start + span[1]) if span else []
    n_layers = len(out.hidden_states) - 1
    hs = {}
    for frac in layers_frac:
        L = max(1, round(frac * n_layers))
        H = out.hidden_states[L][0]
        hs[f"L{L}_R0"] = H[i_r0[-1]].float().cpu().numpy().astype(np.float16) if i_r0 else None
        hs[f"L{L}_F"] = H[i_f[-1]].float().cpu().numpy().astype(np.float16) if i_f else None
    res = {"lp_r0_mean": float(np.nanmean(tok_lp[i_r0])) if i_r0 else np.nan,
           "lp_r0_min": float(np.nanmin(tok_lp[i_r0])) if i_r0 else np.nan,
           "lp_answer_mean": float(np.nanmean(tok_lp[i_ans])) if i_ans else np.nan,
           "lp_f_mean": float(np.nanmean(tok_lp[i_f])) if i_f else np.nan,
           "lp_f_min": float(np.nanmin(tok_lp[i_f])) if i_f else np.nan,
           "n_tok_r0": len(i_r0), "n_tok_f": len(i_f)}
    del out
    return res, hs


def verdict_features(r):
    f = r["critique"]
    other = extract(f, ATYPE[r["source"]]).value
    return {"f_claims_error": int(bool(ERROR_RE.search(f))),
            "f_claims_correct": int(bool(CORRECT_RE.search(f))),
            "f_n_error_words": len(ERROR_RE.findall(f)),
            "f_proposes_other_answer": int(other is not None and r.get("r0_answer") is not None and other != r["r0_answer"]),
            "len_r0_words": len(r["initial_response"].split()), "len_f_words": len(f.split())}


def consistency_features(r, samples):
    vals = [s for s in samples if s is not None]
    a0 = r.get("r0_answer")
    if not vals:
        return {"sc_agree": np.nan, "sc_distinct": np.nan, "sc_entropy": np.nan, "sc_majority_is_r0": np.nan,
                "sc_parsed": 0}
    c = Counter(vals)
    p = np.array(list(c.values()), float) / len(vals)
    return {"sc_agree": float(np.mean([v == a0 for v in vals])),
            "sc_distinct": float(len(set(vals + [a0]))),
            "sc_entropy": float(-(p * np.log(p)).sum()),
            "sc_majority_is_r0": float(c.most_common(1)[0][0] == a0),
            "sc_parsed": len(vals)}


def cmd_features(cfg):
    import torch
    sc = cfg["signals"]
    S = specs(cfg)
    log = CallLog(D(cfg, "calls.jsonl"))
    by_stage = {}
    for rec in log.index.values():
        if rec["stage"] == "sample":
            q = rec["messages"][0]["content"]
            by_stage.setdefault((rec["model_name"], q), []).append(rec)
    for set_name in ("natural", "v1"):
        rows = load_set(cfg, set_name)
        out_p = D(cfg, "features", f"{set_name}.jsonl")
        hs_p = D(cfg, "features", f"{set_name}_hidden.npz")
        if os.path.exists(out_p) and os.path.exists(hs_p):
            print(f"features {set_name}: done, skipping")
            continue
        feats, hidden, skipped = [], {}, 0
        for name in sc["models"]:
            mine = [r for r in rows if r["source_model"] == name]
            for i, r in enumerate(mine):
                q = prompts.initial_messages(r["prompt_raw"], ATYPE[r["source"]])[0]["content"]
                samples = [extract(x["text"], ATYPE[r["source"]]).value for x in by_stage.get((name, q), [])]
                tf = teacher_forced(S[name], r, sc["layers_frac"])
                if tf is None:
                    skipped += 1
                    continue
                lp, hs = tf
                feats.append({"trace_id": r["trace_id"], "problem_id": r["problem_id"], "source": r["source"],
                              "source_model": name, "set": set_name, "label": r["label"],
                              "r0_correct": bool(r["r0_correct"]), "outcome_cell": r.get("outcome_cell"),
                              **lp, **verdict_features(r), **consistency_features(r, samples)})
                for k, v in hs.items():
                    if v is not None:
                        hidden[f"{r['trace_id']}|{k}"] = v
                if (i + 1) % 50 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    print(f"  features {set_name} {name}: {i + 1}/{len(mine)}", flush=True)
            LOCAL.free()
        write_jsonl(out_p, feats)
        np.savez_compressed(hs_p, **hidden)
        print(f"features {set_name}: {len(feats)} traces, {len(hidden)} hidden vectors, "
              f"{skipped} skipped (text not found in chat template)", flush=True)


# ---------------------------------------------------------------- evaluate
SCALARS = ["lp_r0_mean", "lp_r0_min", "lp_answer_mean", "lp_f_mean", "lp_f_min", "f_claims_error",
           "f_claims_correct", "f_n_error_words", "f_proposes_other_answer", "len_r0_words", "len_f_words",
           "sc_agree", "sc_distinct", "sc_entropy", "sc_majority_is_r0"]


def grouped_folds(rows, k=5, seed=42):
    rng = random.Random(seed)
    pids = sorted({r["problem_id"] for r in rows})
    rng.shuffle(pids)
    f = {p: i % k for i, p in enumerate(pids)}
    return np.array([f[r["problem_id"]] for r in rows])


def oof_predict(X, y, folds, kind):
    """Out-of-fold failure probabilities. kind: 'scalar' | 'hidden' | 'text'."""
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        if kind == "text":
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                                  LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=5000, class_weight="balanced"))
            Xtr, Xte = [X[i] for i in np.where(tr)[0]], [X[i] for i in np.where(te)[0]]
        else:
            steps = [SimpleImputer(strategy="median"), StandardScaler()]
            if kind == "hidden":
                steps.append(PCA(n_components=min(64, int(tr.sum()) - 1), random_state=0))
            model = make_pipeline(*steps, LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=3, max_iter=5000,
                                                               class_weight="balanced"))
            Xtr, Xte = X[tr], X[te]
        if len(set(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        model.fit(Xtr, y[tr])
        p[te] = model.predict_proba(Xte)[:, 1]
    return p


def cmd_evaluate(cfg):
    from .metrics import bootstrap, mcnemar, summarize
    sc = cfg["signals"]
    results, lines = {}, []
    for set_name in ("natural", "v1"):
        fp = D(cfg, "features", f"{set_name}.jsonl")
        if not os.path.exists(fp):
            continue
        feats = read_jsonl(fp)
        H = dict(np.load(D(cfg, "features", f"{set_name}_hidden.npz")))
        text = {r["trace_id"]: r for r in load_set(cfg, set_name)}
        for target in ("rf_wrong", "r0_wrong"):
            pooled = {}
            for name in sc["models"] + ["all"]:
                rows = [r for r in feats if name == "all" or r["source_model"] == name]
                if name == "all":
                    # pooled = concatenation of per-generator out-of-fold predictions
                    rows = [r for r in feats if r["source_model"] in sc["models"]]
                if len(rows) < 30:
                    continue
                y = np.array([r["label"] if target == "rf_wrong" else int(not r["r0_correct"]) for r in rows])
                if name != "all":
                    folds = grouped_folds(rows)
                    preds = {
                        "majority": np.full(len(y), np.nan),
                        "text_tfidf": oof_predict([" [SEP] ".join([text[r["trace_id"]]["prompt"], text[r["trace_id"]]["initial_response"],
                                                                   text[r["trace_id"]]["critique"]]) for r in rows], y, folds, "text"),
                        "signals_scalar": oof_predict(np.array([[r[s] for s in SCALARS] for r in rows], float), y, folds, "scalar"),
                    }
                    # this generator's layers, named mid/last so generators of different depth pool
                    ids = {r["trace_id"] for r in rows}
                    hk = sorted({k.split("|")[1] for k in H if k.split("|")[0] in ids}, key=lambda k: (int(k[1:].split("_")[0]), k))
                    dim = next(H[k] for k in H if k.split("|")[0] in ids).shape if hk else None
                    hx = {}
                    for k in hk:
                        tag = ("last" if k.split("_")[0] == hk[-1].split("_")[0] else "mid") + "_" + k.split("_", 1)[1]
                        hx[tag] = np.array([H.get(f"{r['trace_id']}|{k}", np.full(dim, np.nan)) for r in rows], float)
                        preds[f"hidden_{tag}"] = oof_predict(hx[tag], y, folds, "hidden")
                    if "mid_F" in hx:
                        Xc = np.hstack([np.array([[r[s] for s in SCALARS] for r in rows], float), hx["mid_F"]])
                        preds["combined"] = oof_predict(Xc, y, folds, "hidden")
                    for k, v in preds.items():
                        pooled.setdefault(k, ([], []))
                        pooled[k][0].extend(y.tolist()); pooled[k][1].extend(v.tolist())
                else:
                    if not pooled:  # no generator had enough traces in this set
                        continue
                    preds = {k: np.array(v[1]) for k, v in pooled.items()}
                    y = np.array(next(iter(pooled.values()))[0])
                maj = int(np.nanmean(y) >= 0.5)
                ymaj = np.full(len(y), maj)
                ref = (np.asarray(preds["text_tfidf"]) >= 0.5).astype(int)
                for k, p in preds.items():
                    if k == "majority":
                        yp, pf = ymaj, np.full(len(y), np.nanmean(y))
                    else:
                        pf = np.asarray(p, float); yp = (pf >= 0.5).astype(int)
                    s = summarize(y, yp, pf); ci = bootstrap(y, yp, pf)
                    s.update({"ci95": ci, "failure_rate": float(y.mean())})
                    if k != "majority":
                        s["p_vs_majority"] = mcnemar(y, ymaj, yp)["p_value"]
                    if k not in ("majority", "text_tfidf"):
                        s["p_vs_tfidf"] = mcnemar(y, ref, yp)["p_value"]
                    results.setdefault(set_name, {}).setdefault(target, {}).setdefault(name, {})[k] = s
    json.dump(results, open(D(cfg, "results", "signals_summary.json"), "w"), indent=2, default=float)
    for set_name, T in results.items():
        for target, M in T.items():
            lines.append(f"\n## {set_name} set, target: {'final answer wrong (paper label)' if target == 'rf_wrong' else 'initial answer wrong (diagnostic)'}\n")
            lines.append("| Generator | Predictor | n | AUC [95% CI] | Accuracy | Macro F1 | p vs majority | p vs TF-IDF |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for name, P in M.items():
                for k, s in P.items():
                    auc = s.get("auc"); ci = s["ci95"].get("auc")
                    auct = f"{auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if auc is not None and ci else "--"
                    f = lambda x: f"{s[x]:.3f}" if x in s else "--"
                    lines.append(f"| {name} | {k} | {s['n']} | {auct} | {100 * s['accuracy']:.1f}% | {s['macro_f1']:.3f} | {f('p_vs_majority')} | {f('p_vs_tfidf')} |")
    md = "# Generator-side signals: out-of-fold results (5-fold, grouped by problem)\n" + "\n".join(lines) + "\n"
    open(D(cfg, "results", "signals_summary.md"), "w").write(md)
    print(md)


# ---------------------------------------------------------------- verifier (GSM8K)
VERIFIER = ["pot_ran", "pot_agree_r0", "pot_agree_f"]


def cmd_verify(cfg):
    """Program-of-thought check of the initial answer, before any revision: the generator writes a
    program for the problem, it is screened and run in an isolated subprocess, and its printed
    number is compared with the initial answer and with the answer the critique proposes."""
    from ..answers import normalize_number
    from .routing import PROGRAM_PROMPT, run_program
    sc = cfg["signals"]
    cl = Client(CallLog(D(cfg, "calls.jsonl")))
    S = specs(cfg)
    bs = cfg.get("batch_size", 8)
    for set_name in ("natural", "v1"):
        out = []
        for name in sc["models"]:
            rows = [r for r in load_set(cfg, set_name) if r["source"] == "gsm8k" and r["source_model"] == name]
            for i in range(0, len(rows), bs):
                chunk = rows[i:i + bs]
                gens = cl.complete_batch(S[name], [[{"role": "user", "content": PROGRAM_PROMPT.format(q=r["prompt_raw"])}]
                                                   for r in chunk], 0.0, [0] * len(chunk), "program")
                for r, g in zip(chunk, gens):
                    ex = run_program(g["text"])
                    pot = normalize_number(ex["stdout"]) if ex["stdout"].strip() else None
                    r0 = extract(r["initial_response"], "number").value
                    fa = extract(r["critique"], "number").value
                    out.append({"trace_id": r["trace_id"], "pot_answer": pot, "pot_error": ex["error"],
                                "pot_ran": int(pot is not None),
                                "pot_agree_r0": float(is_correct(r0, pot, "number")) if pot is not None else np.nan,
                                "pot_agree_f": float(is_correct(fa, pot, "number")) if pot is not None else np.nan})
                if (i // bs) % max(1, 25 // bs) == 0:
                    print(f"  verify {set_name} {name}: {min(i + bs, len(rows))}/{len(rows)}", flush=True)
            LOCAL.free()
        write_jsonl(D(cfg, "features", f"{set_name}_verifier.jsonl"), out)
        print(f"verify {set_name}: {len(out)} GSM8K traces, program ran for {sum(o['pot_ran'] for o in out)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "features", "evaluate", "verify"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    {"build": cmd_build, "features": cmd_features, "evaluate": cmd_evaluate, "verify": cmd_verify}[a.cmd](cfg)


if __name__ == "__main__":
    main()
