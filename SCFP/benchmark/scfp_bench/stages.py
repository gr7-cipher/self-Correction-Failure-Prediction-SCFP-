"""Pipeline stages. Each stage reads and writes JSONL files under cfg["run_dir"]."""

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import prompts
from .answers import extract, is_correct
from .llm import CallLog, Client, ModelSpec
from .problems import load_sources, problem_from_dict

FAILURE_MODES = ["Justification Hallucination", "Confidence Miscalibration",
                 "Bias Amplification", "Over-correction", "Reasoning Myopia"]
UNCLEAR = "Unclear / none of these"


# ---------------------------------------------------------------- helpers
def path(cfg, *parts):
    p = os.path.join(cfg["run_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def models(cfg, names):
    by_name = {m["name"]: ModelSpec.from_dict(m) for m in cfg["models"]}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"models not defined in config: {missing}")
    return [by_name[n] for n in names]


def client(cfg):
    return Client(CallLog(path(cfg, "calls.jsonl")))


def run_parallel(cfg, fn, items, desc, specs=()):
    """Run fn over items. Local models run one call at a time, grouped by model."""
    out, done = [], 0
    workers = 1 if any(s.provider == "local" for s in specs) else cfg.get("concurrency", 8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futs):
            out.append(fut.result())
            done += 1
            if done % (200 if workers > 1 else 10) == 0 or done == len(items):
                print(f"  {desc}: {done}/{len(items)}", flush=True)
    return out


# ---------------------------------------------------------------- stages
def stage_holdout(cfg):
    """Pool problems and reserve the routing hold-out before anything else runs."""
    problems = load_sources(cfg["sources"])
    write_jsonl(path(cfg, "problems.jsonl"), [p.to_dict() for p in problems])
    h = cfg["routing_holdout"]
    pool = [p for p in problems if p.source == h["source"] and p.origin_split == h["origin_split"]]
    rng = random.Random(cfg["seed"])
    held = sorted(rng.sample(pool, h["n"]), key=lambda p: p.problem_id)
    write_jsonl(path(cfg, "holdout.jsonl"), [p.to_dict() for p in held])
    print(f"pooled {len(problems)} problems; reserved {len(held)} {h['source']} "
          f"{h['origin_split']} problems for routing")


def stage_filter(cfg):
    """Estimate per-problem accuracy; keep problems inside the difficulty band."""
    fc = cfg["filter"]
    held = {p["problem_id"] for p in read_jsonl(path(cfg, "holdout.jsonl"))}
    problems = [problem_from_dict(d) for d in read_jsonl(path(cfg, "problems.jsonl"))]
    rng = random.Random(cfg["seed"])
    candidates = []
    for source, cap in fc["candidates_per_source"].items():
        pool = [p for p in problems if p.source == source and p.problem_id not in held]
        rng.shuffle(pool)
        candidates.extend(pool[:cap])
    specs = models(cfg, fc["models"])
    cl = client(cfg)
    jobs = [(p, s, k) for s in specs for p in candidates for k in range(fc["samples_per_problem"])]
    print(f"filter: {len(candidates)} candidates x {len(specs)} models x "
          f"{fc['samples_per_problem']} samples = {len(jobs)} calls")

    def one(job):
        p, spec, k = job
        rec = cl.complete(spec, prompts.initial_messages(p.question, p.answer_type),
                          fc["temperature"], sample=k, stage="filter")
        ex_ = extract(rec["text"], p.answer_type, len(p.options) or 4)
        return p.problem_id, rec["call_id"], is_correct(ex_.value, p.gold, p.answer_type)

    if any(s.provider == "local" for s in specs):
        results = batched_filter(cfg, cl, jobs, fc["temperature"])
    else:
        results = run_parallel(cfg, one, jobs, "filter", specs)
    per = defaultdict(lambda: {"n": 0, "correct": 0, "call_ids": []})
    for pid, cid, ok in results:
        per[pid]["n"] += 1
        per[pid]["correct"] += int(ok)
        per[pid]["call_ids"].append(cid)
    lo, hi = fc["band"]
    rows, selected = [], []
    for p in candidates:
        r = per[p.problem_id]
        acc = r["correct"] / r["n"]
        keep = lo <= acc <= hi
        rows.append({"problem_id": p.problem_id, "source": p.source, "accuracy": acc,
                     "n_samples": r["n"], "in_band": keep, "call_ids": r["call_ids"]})
        if keep:
            selected.append(p)
    write_jsonl(path(cfg, "filter.jsonl"), rows)

    final, report = [], {}
    for source, target in fc["target_per_source"].items():
        pool = [p for p in selected if p.source == source]
        rng.shuffle(pool)
        take = pool[:target]
        final.extend(take)
        report[source] = {"candidates": sum(1 for p in candidates if p.source == source),
                          "in_band": len(pool), "target": target, "selected": len(take)}
        if len(take) < target:
            print(f"  WARNING {source}: only {len(take)} of {target} target problems are in "
                  f"the band. Raise candidates_per_source or report the smaller count.")
    write_jsonl(path(cfg, "selected.jsonl"), [p.to_dict() for p in final])
    json.dump(report, open(path(cfg, "filter_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


def stage_generate(cfg):
    """Collect one (R0, F, Rf) trace per selected problem and generator model."""
    gc = cfg["generate"]
    problems = [problem_from_dict(d) for d in read_jsonl(path(cfg, "selected.jsonl"))]
    specs = models(cfg, gc["models"])
    cl = client(cfg)
    T = gc["temperature"]
    jobs = [(p, s) for s in specs for p in problems]
    print(f"generate: {len(problems)} problems x {len(specs)} models = {len(jobs)} traces "
          f"({3 * len(jobs)} calls)")

    def one(job):
        p, spec = job
        r0 = cl.complete(spec, prompts.initial_messages(p.question, p.answer_type), T, stage="R0")
        f = cl.complete(spec, prompts.critique_messages(p.question, p.answer_type, r0["text"]),
                        T, stage="F")
        rf = cl.complete(spec, prompts.revise_messages(p.question, p.answer_type, r0["text"],
                                                       f["text"]), T, stage="Rf")
        return trace_record(p, spec, r0, f, rf)

    if any(s.provider == "local" for s in specs):
        traces = batched_generate(cfg, cl, jobs, T)
    else:
        traces = run_parallel(cfg, one, jobs, "generate", specs)
    traces = sorted(traces, key=lambda t: t["trace_id"])
    write_jsonl(path(cfg, "traces.jsonl"), traces)


def trace_record(p, spec, r0, f, rf):
    tid = hashlib.sha256(f"{p.problem_id}|{spec.name}".encode()).hexdigest()[:16]
    return {"trace_id": tid, "problem_id": p.problem_id, "source": p.source,
            "domain": p.domain, "source_model": spec.name, "prompt": p.question,
            "initial_response": r0["text"], "critique": f["text"],
            "final_response": rf["text"],
            "call_ids": {"R0": r0["call_id"], "F": f["call_id"], "Rf": rf["call_id"]},
            "model_versions": sorted({r0["model_version"], f["model_version"],
                                      rf["model_version"]}),
            "providers": sorted({r0["provider"], f["provider"], rf["provider"]})}


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def batched_filter(cfg, cl, jobs, temperature):
    bs, out = cfg.get("batch_size", 1), []
    for spec in {s.name: s for _, s, _ in jobs}.values():
        mine = [j for j in jobs if j[1].name == spec.name]
        for k, chunk in enumerate(_batches(mine, bs)):
            recs = cl.complete_batch(spec, [prompts.initial_messages(p.question, p.answer_type)
                                            for p, _, _ in chunk], temperature,
                                     [s for _, _, s in chunk], stage="filter")
            for (p, _, _), rec in zip(chunk, recs):
                ex_ = extract(rec["text"], p.answer_type, len(p.options) or 4)
                out.append((p.problem_id, rec["call_id"], is_correct(ex_.value, p.gold, p.answer_type)))
            print(f"  filter {spec.name}: {min((k + 1) * bs, len(mine))}/{len(mine)}", flush=True)
    return out


def batched_generate(cfg, cl, jobs, T):
    bs, out = cfg.get("batch_size", 1), []
    for spec in {s.name: s for _, s in jobs}.values():
        mine = [p for p, s in jobs if s.name == spec.name]
        for k, chunk in enumerate(_batches(mine, bs)):
            zeros = [0] * len(chunk)
            r0 = cl.complete_batch(spec, [prompts.initial_messages(p.question, p.answer_type)
                                          for p in chunk], T, zeros, stage="R0")
            f = cl.complete_batch(spec, [prompts.critique_messages(p.question, p.answer_type, a["text"])
                                         for p, a in zip(chunk, r0)], T, zeros, stage="F")
            rf = cl.complete_batch(spec, [prompts.revise_messages(p.question, p.answer_type,
                                                                  a["text"], b["text"])
                                          for p, a, b in zip(chunk, r0, f)], T, zeros, stage="Rf")
            out += [trace_record(p, spec, a, b, c) for p, a, b, c in zip(chunk, r0, f, rf)]
            print(f"  generate {spec.name}: {min((k + 1) * bs, len(mine))}/{len(mine)}", flush=True)
    return out


def stage_label(cfg):
    """Score R0 and Rf automatically and derive the binary outcome label."""
    probs = {d["problem_id"]: problem_from_dict(d) for d in read_jsonl(path(cfg, "selected.jsonl"))}
    definition = cfg["label"]["failure_definition"]
    rows, cells = [], Counter()
    for t in read_jsonl(path(cfg, "traces.jsonl")):
        p = probs[t["problem_id"]]
        n_opt = len(p.options) or 4
        e0 = extract(t["initial_response"], p.answer_type, n_opt)
        ef = extract(t["final_response"], p.answer_type, n_opt)
        c0, cf = is_correct(e0.value, p.gold, p.answer_type), is_correct(ef.value, p.gold, p.answer_type)
        cell = f"{'right' if c0 else 'wrong'}->{'right' if cf else 'wrong'}"
        if definition == "final_incorrect":
            label = int(not cf)
        elif definition == "degraded_only":
            label = int(c0 and not cf)
        else:
            raise ValueError(f"unknown failure_definition {definition}")
        cells[cell] += 1
        rows.append({**t, "gold": p.gold, "r0_answer": e0.value, "rf_answer": ef.value,
                     "r0_parsed_from_final_line": e0.from_final_line,
                     "rf_parsed_from_final_line": ef.from_final_line,
                     "r0_correct": c0, "rf_correct": cf, "outcome_cell": cell,
                     "label": label, "failure_definition": definition})
    write_jsonl(path(cfg, "labeled.jsonl"), rows)
    unparsed = sum(not (r["r0_parsed_from_final_line"] and r["rf_parsed_from_final_line"]) for r in rows)
    summary = {"traces": len(rows), "outcome_cells": dict(cells),
               "failure_rate": sum(r["label"] for r in rows) / max(len(rows), 1),
               "traces_without_final_answer_line": unparsed}
    json.dump(summary, open(path(cfg, "label_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def stage_split(cfg):
    """Split by problem (never by trace) and write a hash manifest."""
    rows = read_jsonl(path(cfg, "labeled.jsonl"))
    held = {p["problem_id"] for p in read_jsonl(path(cfg, "holdout.jsonl"))}
    leaked = [r["trace_id"] for r in rows if r["problem_id"] in held]
    if leaked:
        raise RuntimeError(f"{len(leaked)} traces use routing hold-out problems")
    rng = random.Random(cfg["seed"])
    split_of = {}
    for source in sorted({r["source"] for r in rows}):
        pids = sorted({r["problem_id"] for r in rows if r["source"] == source})
        rng.shuffle(pids)
        a, b, _ = cfg["split"]["ratios"]
        n1, n2 = round(len(pids) * a), round(len(pids) * (a + b))
        for i, pid in enumerate(pids):
            split_of[pid] = "train" if i < n1 else "val" if i < n2 else "test"
    manifest = {"seed": cfg["seed"], "ratios": cfg["split"]["ratios"], "files": {}}
    for name in ("train", "val", "test"):
        part = [r for r in rows if split_of[r["problem_id"]] == name]
        p = path(cfg, "splits", f"{name}.jsonl")
        write_jsonl(p, part)
        manifest["files"][name] = {
            "traces": len(part), "problems": len({r["problem_id"] for r in part}),
            "failure_rate": round(sum(r["label"] for r in part) / max(len(part), 1), 4),
            "by_source": dict(Counter(r["source"] for r in part)),
            "by_model": dict(Counter(r["source_model"] for r in part)),
            "sha256": sha256_file(p)}
    json.dump(manifest, open(path(cfg, "splits", "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))


LABEL_CONFIG = """<View>
  <Header value="Problem"/>
  <Text name="prompt" value="$prompt"/>
  <Header value="Gold answer"/>
  <Text name="gold" value="$gold"/>
  <Header value="Initial response (R0)"/>
  <Text name="r0" value="$initial_response"/>
  <Header value="Self-critique (F)"/>
  <Text name="f" value="$critique"/>
  <Header value="Final response (Rf)"/>
  <Text name="rf" value="$final_response"/>
  <Header value="Primary failure mode (pick exactly one)"/>
  <Choices name="failure_mode" toName="f" choice="single" required="true">
{choices}
  </Choices>
  <TextArea name="rationale" toName="f" placeholder="One sentence: why this mode?" rows="2"/>
</View>
"""


def stage_export(cfg):
    """Write failed traces as Label Studio tasks and assign annotators."""
    ac = cfg["annotation"]
    rows = []
    for name in ("train", "val", "test"):
        rows += [dict(r, split=name) for r in read_jsonl(path(cfg, "splits", f"{name}.jsonl"))]
    failed = [r for r in rows if r["label"] == 1]
    rng = random.Random(cfg["seed"])
    rng.shuffle(failed)
    n_overlap = round(len(failed) * ac["agreement_fraction"])
    tasks, assign = [], []
    annotators = ac["annotators"]
    for i, r in enumerate(failed):
        k = ac["agreement_raters"] if i < n_overlap else ac["raters_per_trace"]
        who = rng.sample(annotators, k)
        assign.append({"trace_id": r["trace_id"], "annotators": who, "agreement_set": i < n_overlap})
        tasks.append({"data": {k_: r[k_] for k_ in ("trace_id", "prompt", "gold",
                                                    "initial_response", "critique",
                                                    "final_response")}})
    json.dump(tasks, open(path(cfg, "annotation", "tasks.json"), "w"), ensure_ascii=False, indent=1)
    write_jsonl(path(cfg, "annotation", "assignments.jsonl"), assign)
    choices = "\n".join(f'    <Choice value="{c}"/>' for c in FAILURE_MODES + [UNCLEAR])
    open(path(cfg, "annotation", "label_config.xml"), "w").write(LABEL_CONFIG.format(choices=choices))
    print(f"exported {len(tasks)} failed traces; {n_overlap} in the agreement set "
          f"({ac['agreement_raters']} raters), rest {ac['raters_per_trace']} rater(s)")


def parse_label_studio(export_path):
    """Label Studio JSON export -> rows of {trace_id, annotator, mode}."""
    out = []
    for task in json.load(open(export_path)):
        tid = task["data"]["trace_id"]
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = str(ann.get("completed_by", {}).get("email", ann.get("completed_by")))
            for res in ann.get("result", []):
                if res.get("from_name") == "failure_mode":
                    out.append({"trace_id": tid, "annotator": who,
                                "mode": res["value"]["choices"][0],
                                "created_at": ann.get("created_at")})
    return out


def fleiss_kappa(table):
    """table: list of per-item category counts, each row summing to the same n."""
    import numpy as np
    m = np.asarray(table, dtype=float)
    n = m.sum(1)
    if len(set(n)) != 1:
        raise ValueError("Fleiss' kappa needs the same number of raters per item")
    n = n[0]
    p_j = m.sum(0) / m.sum()
    P_i = ((m * m).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    if P_e >= 1.0:  # every rating in one category: kappa is undefined
        return None
    return float((P_bar - P_e) / (1 - P_e))


def stage_agree(cfg, export_path):
    """Compute agreement on the agreement set and build final failure-mode labels."""
    anns = parse_label_studio(export_path) if export_path.endswith(".json") else read_jsonl(export_path)
    write_jsonl(path(cfg, "annotation", "annotations.jsonl"), anns)
    assign = {a["trace_id"]: a for a in read_jsonl(path(cfg, "annotation", "assignments.jsonl"))}
    by_trace = defaultdict(list)
    for a in anns:
        by_trace[a["trace_id"]].append(a["mode"])
    cats = FAILURE_MODES + [UNCLEAR]
    k = cfg["annotation"]["agreement_raters"]
    agree_items = [v for t, v in by_trace.items() if assign.get(t, {}).get("agreement_set") and len(v) == k]
    kappa = fleiss_kappa([[v.count(c) for c in cats] for v in agree_items]) if agree_items else None
    adjud = {}
    adj_path = path(cfg, "annotation", "adjudications.jsonl")
    if os.path.exists(adj_path):
        adjud = {a["trace_id"]: a["mode"] for a in read_jsonl(adj_path)}
    final, pending = {}, []
    for t, votes in by_trace.items():
        if t in adjud:
            final[t] = adjud[t]
            continue
        top = Counter(votes).most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            final[t] = top[0][0]
        else:
            pending.append({"trace_id": t, "votes": votes})
    write_jsonl(path(cfg, "annotation", "needs_adjudication.jsonl"), pending)
    report = {"annotations": len(anns), "traces_annotated": len(by_trace),
              "agreement_items": len(agree_items), "raters_per_agreement_item": k,
              "fleiss_kappa": kappa, "final_labels": len(final),
              "needs_adjudication": len(pending),
              "mode_counts": dict(Counter(final.values()))}
    json.dump(report, open(path(cfg, "annotation", "agreement.json"), "w"), indent=2)
    for name in ("train", "val", "test"):
        rows = read_jsonl(path(cfg, "splits", f"{name}.jsonl"))
        for r in rows:
            r["failure_type"] = "N/A" if r["label"] == 0 else final.get(r["trace_id"], "UNLABELED")
        write_jsonl(path(cfg, "final", f"{name}.jsonl"), rows)
    print(json.dumps(report, indent=2))


def stage_validate(cfg, allow_mock=False):
    """Refuse to release unless every trace is backed by logged, real API calls."""
    log = CallLog(path(cfg, "calls.jsonl"))
    problems = []
    use_final = os.path.exists(os.path.join(cfg["run_dir"], "final", "train.jsonl"))
    src = "final" if use_final else "splits"
    held = {p["problem_id"] for p in read_jsonl(path(cfg, "holdout.jsonl"))}
    split_of, errors = {}, []
    for name in ("train", "val", "test"):
        for r in read_jsonl(path(cfg, src, f"{name}.jsonl")):
            if split_of.setdefault(r["problem_id"], name) != name:
                errors.append(f"problem {r['problem_id']} appears in {split_of[r['problem_id']]} and {name}")
            if r["problem_id"] in held:
                errors.append(f"trace {r['trace_id']} uses a routing hold-out problem")
            for stage_name, cid in r["call_ids"].items():
                rec = log.get(cid)
                if rec is None:
                    errors.append(f"trace {r['trace_id']}: {stage_name} call {cid[:12]} not in calls.jsonl")
                elif rec["provider"] == "mock" and not allow_mock:
                    errors.append(f"trace {r['trace_id']}: {stage_name} came from the MOCK provider")
            if use_final and r["label"] == 1 and r["failure_type"] in ("UNLABELED",):
                errors.append(f"trace {r['trace_id']}: failure without a failure-mode label")
            problems.append(r)
    if not use_final:
        man = json.load(open(path(cfg, "splits", "manifest.json")))
        for name, info in man["files"].items():
            if sha256_file(path(cfg, "splits", f"{name}.jsonl")) != info["sha256"]:
                errors.append(f"splits/{name}.jsonl changed after the manifest was written")
    print(f"validated {len(problems)} traces from {src}/: {len(errors)} problems found")
    for e in errors[:20]:
        print("  -", e)
    if errors:
        raise SystemExit(1)
