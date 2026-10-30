"""Can text and generator-side signals be combined into a better failure predictor?

Uses the features built by signals.py (no new generation). Every predictor is evaluated with the
same 5-fold grouped cross-validation as signals.py; stacking is nested (the meta-model is fit on
inner out-of-fold predictions of the training folds only), so no test-fold label leaks.

    python -m scfp_bench.experiments.improve --config configs/signals_mac.yaml
"""
import argparse
import json
import os

import numpy as np
import yaml
from scipy import sparse

from ..stages import read_jsonl
from .metrics import bootstrap, mcnemar, summarize
from .signals import D, SCALARS, VERIFIER, grouped_folds, load_set

SOURCES = ["gsm8k", "strategyqa", "logiqa"]
HIDDEN = "last_R0"  # last layer at the end of R0 (L24 for Qwen2.5-0.5B, L28 for 1.5B)
NO_VERIFIER = False  # --no-verifier: ignore verifier features (to reproduce earlier tables)


def _lr(Cs):
    from sklearn.linear_model import LogisticRegressionCV
    return LogisticRegressionCV(Cs=Cs, cv=3, max_iter=5000, class_weight="balanced")


def scalar_matrix(rows):
    cols = SCALARS + [v for v in VERIFIER if any(v in r for r in rows)]  # verifier only if run
    X = np.array([[r.get(s, np.nan) for s in cols] for r in rows], float)
    src = np.array([[r["source"] == s for s in SOURCES] for r in rows], float)
    return np.hstack([X, src])


class Base:
    """One base predictor: fit(idx) on training rows, then proba(idx)."""

    def __init__(self, kind, data):
        self.kind, self.data = kind, data

    def fit(self, idx, y):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        d = self.data
        if self.kind == "text":
            m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                              _lr([0.01, 0.1, 1, 10]))
            m.fit([d["text"][i] for i in idx], y[idx])
        elif self.kind == "concat":  # TF-IDF and standardized scalars in one linear model
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
            self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            Xt = self.vec.fit_transform([d["text"][i] for i in idx])
            Xs = sparse.csr_matrix(self.pre.fit_transform(d["scalar"][idx]))
            m = _lr([0.01, 0.1, 1, 10]).fit(sparse.hstack([Xt, Xs]).tocsr(), y[idx])
        elif self.kind == "scalar":
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _lr([0.001, 0.01, 0.1, 1]))
            m.fit(d["scalar"][idx], y[idx])
        elif self.kind == "gbm":
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                               class_weight="balanced", random_state=0)
            m.fit(d["scalar"][idx], y[idx])
        elif self.kind == "hidden":
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              PCA(n_components=min(64, len(idx) - 1), random_state=0), _lr([0.001, 0.01, 0.1, 1]))
            m.fit(d["hidden"][idx], y[idx])
        self.m = m
        return self

    def proba(self, idx):
        d = self.data
        if self.kind == "text":
            return self.m.predict_proba([d["text"][i] for i in idx])[:, 1]
        if self.kind == "concat":
            X = sparse.hstack([self.vec.transform([d["text"][i] for i in idx]),
                               sparse.csr_matrix(self.pre.transform(d["scalar"][idx]))]).tocsr()
            return self.m.predict_proba(X)[:, 1]
        key = "hidden" if self.kind == "hidden" else "scalar"
        return self.m.predict_proba(d[key][idx])[:, 1]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_fit_predict(kinds, data, y, tr, te, groups):
    """Nested stacking: inner grouped OOF on tr -> meta LR; bases refit on all of tr -> predict te."""
    inner = grouped_folds([{"problem_id": groups[i]} for i in tr], k=5, seed=7)
    Z = np.zeros((len(tr), len(kinds)))
    for k in range(5):
        a, b = tr[inner != k], tr[inner == k]
        for j, kind in enumerate(kinds):
            Z[inner == k, j] = Base(kind, data).fit(a, y).proba(b)
    meta = _lr([0.01, 0.1, 1, 10]).fit(_logit(Z), y[tr])
    Zte = np.column_stack([Base(kind, data).fit(tr, y).proba(te) for kind in kinds])
    return meta.predict_proba(_logit(Zte))[:, 1]


PREDICTORS = {
    "text_tfidf": ("base", "text"),
    "signals_scalar": ("base", "scalar"),
    "signals_gbm": ("base", "gbm"),
    f"hidden_{HIDDEN}": ("base", "hidden"),
    "concat_text_scalar": ("base", "concat"),
    "stack_text_scalar": ("stack", ["text", "scalar"]),
    "stack_text_scalar_hidden": ("stack", ["text", "scalar", "hidden"]),
    "stack_all": ("stack", ["text", "scalar", "gbm", "hidden"]),
}


def usable(data, names=PREDICTORS):
    """Predictors whose inputs exist (no hidden-state probe when pooling generators of different sizes)."""
    kinds = lambda n: [PREDICTORS[n][1]] if PREDICTORS[n][0] == "base" else PREDICTORS[n][1]
    return [n for n in names if data["hidden"] is not None or "hidden" not in kinds(n)]


def predict(name, data, y, tr, te, groups):
    how, what = PREDICTORS[name]
    if how == "base":
        return Base(what, data).fit(tr, y).proba(te)
    return stack_fit_predict(what, data, y, tr, te, groups)


def load(cfg, set_name, target, model=None):
    feats = read_jsonl(D(cfg, "features", f"{set_name}.jsonl"))
    feats = [r for r in feats if r["source_model"] in cfg["signals"]["models"] and model in (None, r["source_model"])]
    if len(feats) < 30:  # too few traces for 5-fold CV (e.g. a generator absent from SCFP v1)
        return feats, None, np.zeros(len(feats))
    vp = D(cfg, "features", f"{set_name}_verifier.jsonl")
    if os.path.exists(vp) and not NO_VERIFIER:
        ver = {v["trace_id"]: v for v in read_jsonl(vp)}
        feats = [{**r, **{k: ver.get(r["trace_id"], {}).get(k, np.nan) for k in VERIFIER}} for r in feats]
    H = dict(np.load(D(cfg, "features", f"{set_name}_hidden.npz")))
    text = {r["trace_id"]: r for r in load_set(cfg, set_name)}
    dim = H[next(iter(H))].shape
    last = {}  # trace id -> key of its last-layer R0 vector (layer count differs per generator)
    for k in H:
        tid, _, lay = k.rpartition("|")
        if lay.endswith("_R0") and int(lay[1:-3]) > int(last.get(tid, "|L0_R0").rpartition("|")[2][1:-3]):
            last[tid] = k
    y = np.array([r["label"] if target == "rf_wrong" else int(not r["r0_correct"]) for r in feats])
    data = {"text": [" [SEP] ".join([text[r["trace_id"]][k] for k in ("prompt", "initial_response", "critique")])
                     for r in feats],
            "scalar": scalar_matrix(feats),
            "hidden": None}
    vecs = [H[last[r["trace_id"]]] if r["trace_id"] in last else np.full(dim, np.nan) for r in feats]
    if len({v.shape for v in vecs}) == 1:  # generators with different hidden sizes cannot share one probe
        data["hidden"] = np.array(vecs, float)
    return feats, data, y


def evaluate(cfg, set_name, target, model=None):
    feats, data, y = load(cfg, set_name, target, model)
    if len(y) < 30:
        return y, None
    folds = grouped_folds(feats)
    groups = [r["problem_id"] for r in feats]
    out = {}
    for name in usable(data):
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            p[te] = predict(name, data, y, tr, te, groups)
        out[name] = p
        print(f"  {set_name} {target} {model or ''} {name}: AUC {summarize(y, (p >= .5).astype(int), p)['auc']:.3f}", flush=True)
    return y, out


def learning_curve(cfg, set_name, target, names, fracs=(0.25, 0.5, 0.75, 1.0), seeds=(0, 1, 2)):
    """AUC when each training fold keeps only a fraction of its problems (test folds unchanged)."""
    from sklearn.metrics import roc_auc_score
    feats, data, y = load(cfg, set_name, target)
    folds = grouped_folds(feats)
    groups = np.array([r["problem_id"] for r in feats])
    names = [n if usable(data, [n]) else "stack_text_scalar" for n in names]  # pooled: no shared probe
    res = {n: {} for n in names}
    for frac in fracs:
        for n in names:
            aucs = []
            for s in (seeds if frac < 1 else seeds[:1]):
                rng = np.random.default_rng(s)
                p = np.zeros(len(y))
                for k in range(5):
                    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
                    pids = np.unique(groups[tr])
                    keep = set(rng.choice(pids, int(round(frac * len(pids))), replace=False))
                    tr = np.array([i for i in tr if groups[i] in keep])
                    p[te] = predict(n, data, y, tr, te, list(groups))
                aucs.append(roc_auc_score(y, p))
            res[n][frac] = (float(np.mean(aucs)), float(np.std(aucs)), int(round(frac * 0.8 * len(y))))
            print(f"  curve {n} frac {frac}: AUC {np.mean(aucs):.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--out", default="improve_summary")
    args = ap.parse_args()
    global NO_VERIFIER
    NO_VERIFIER = args.no_verifier
    cfg = yaml.safe_load(open(args.config))
    lines, results = [], {}
    gens = cfg["signals"]["models"]
    # one generator: as before; several: each generator alone, then one model pooled over all
    groups = [None] if len(gens) == 1 else gens + [None]
    for set_name, target, gen in [(s, t, g) for s in ("natural", "v1") for t in ("rf_wrong", "r0_wrong") for g in groups]:
        y, preds = evaluate(cfg, set_name, target, gen)
        if preds is None:
            continue
        tag = "" if len(gens) == 1 else f", generator: {gen or 'pooled (one model over all generators)'}"
        base = (preds["text_tfidf"] >= .5).astype(int)
        maj = np.full(len(y), int(y.mean() >= .5))
        lines += [f"\n## {set_name} set, target: {'final answer wrong (paper label)' if target == 'rf_wrong' else 'initial answer wrong (diagnostic)'}{tag}\n",
                  f"Majority class: {max(y.mean(), 1 - y.mean()):.1%} (n = {len(y)})\n",
                  "| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |",
                  "|---|---|---|---|---|---|"]
        for name, p in preds.items():
            yp = (p >= .5).astype(int)
            s, ci = summarize(y, yp, p), bootstrap(y, yp, p)
            pm = mcnemar(y, yp, maj)["p_value"]
            pt = "--" if name == "text_tfidf" else f"{mcnemar(y, yp, base)['p_value']:.3f}"
            lines.append(f"| {name} | {s['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}] | "
                         f"{s['accuracy']:.1%} [{ci['accuracy'][0]:.1%}, {ci['accuracy'][1]:.1%}] | "
                         f"{s['macro_f1']:.3f} | {pm:.3f} | {pt} |")
            results[f"{set_name}|{target}|{name}" + ("" if len(gens) == 1 else f"|{gen or 'pooled'}")] = {**s, "ci": ci, "p_vs_majority": pm}
    if not args.no_curve:
        lc = learning_curve(cfg, "natural", "rf_wrong", ["text_tfidf", "stack_text_scalar_hidden"])
        names = list(lc)
        lines += ["\n## Learning curve (natural set, paper label): out-of-fold AUC vs. training traces per fold\n",
                  "| Predictor | " + " | ".join(f"{int(f * 100)}% (~{v[2]} traces)" for f, v in lc[names[0]].items()) + " |",
                  "|---|" + "---|" * len(lc[names[0]])]
        for n in names:
            lines.append(f"| {n} | " + " | ".join(f"{m:.3f} ± {sd:.3f}" for m, sd, _ in lc[n].values()) + " |")
        results["learning_curve"] = {n: {str(f): v for f, v in d.items()} for n, d in lc.items()}
    os.makedirs(D(cfg, "results"), exist_ok=True)
    md = ("# Combining text and generator-side signals (5-fold grouped CV, nested stacking)\n\n"
          "Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: " + HIDDEN + ".\n" + "\n".join(lines) + "\n")
    open(D(cfg, "results", args.out + ".md"), "w").write(md)
    json.dump(results, open(D(cfg, "results", args.out + ".json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
