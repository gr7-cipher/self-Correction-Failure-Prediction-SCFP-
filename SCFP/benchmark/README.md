# SCFP benchmark builder

Builds the SCFP correction-trace benchmark from real model calls and real
annotations. Every trace in the output points to the logged API calls that
produced it, and `validate` refuses to release a dataset where that is not true.

This replaces `SCFP/data/scfp_v1.jsonl`, which was template-generated (13
critique templates, label fully determined by template) and must not be used
for any reported result.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install "openai>=1.40" "anthropic>=0.34" "datasets>=2.19" pyyaml pytest numpy scipy scikit-learn torch transformers accelerate sentencepiece protobuf
cd SCFP/benchmark
export PYTHONPATH=.
```

Export the API keys named in the config (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`TOGETHER_API_KEY`, ...). Keys are read from the environment and never logged.

## Without API keys (free)

The `local` provider runs open-weight Hugging Face models in-process (Apple
MPS, CUDA, or CPU) and logs each call exactly like an API call, with the model
version recorded as `<repo>@<commit sha>`.

| Config | Where | Models | Size | Time |
|---|---|---|---|---|
| `configs/laptop_local.yaml` | This Mac (M1, 8 GB) | Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B | 300 problems, 900 traces, ~4,500 calls | ~8–12 h (measured ~5 s/call for 0.5B unbatched) |
| `configs/free_gpu.yaml` | Kaggle or Colab free T4 | Qwen2.5-3B, Phi-3.5-mini, Granite-3.1-2B, SmolLM2-1.7B | 1,500 problems, 6,000 traces, ~25,000 calls | Not measured; time the first 100 calls |

On the Mac, keep `batch_size: 1`: batching was slower on MPS (18 s/call at
batch 8). On CUDA, batching is what makes the GPU run fast.

Runs are resumable. If a run stops (laptop sleeps, Kaggle session ends),
re-run the same command with the same `run_dir`, and every call already in
`calls.jsonl` is reused.

Small models self-correct differently from frontier models. The paper must
name the models actually used and must not describe them as GPT-4 or Claude traces.

## Run

1. **Smoke test (free, fake model):** `python -m scfp_bench all-auto --config configs/smoke_mock.yaml`.
   Output is tagged `provider: mock`; `validate` rejects it.
2. **Pilot (a few hundred real calls):** fill in the `REPLACE_ME` model ids in
   `configs/pilot.yaml`, then `python -m scfp_bench all-auto --config configs/pilot.yaml`.
   Check `label_summary.json` (outcome cells, unparsed answers) and the token
   usage in `calls.jsonl` before scaling up.
3. **Full build:** same with `configs/scfp_v1.yaml`. Stages are resumable: a
   re-run reuses every call already in `calls.jsonl`.
4. **Annotation:** `python -m scfp_bench export --config ...` writes
   `annotation/tasks.json` and `annotation/label_config.xml` for Label Studio,
   plus `assignments.jsonl` (who labels what; the agreement subset gets 3 raters).
5. **Agreement and final labels:** `python -m scfp_bench agree --config ... --annotations <label-studio-export.json>`.
   This writes `annotation/agreement.json` (Fleiss' kappa) and lists ties in
   `needs_adjudication.jsonl`. Resolve them in `annotation/adjudications.jsonl`
   (`{"trace_id": ..., "mode": ...}`) and run `agree` again.
6. **Release check:** `python -m scfp_bench validate --config ...` must pass.

## What each stage writes (under `run_dir`)

| File | Contents |
|---|---|
| `calls.jsonl` | Every API call: request, response, model version, usage, latency, timestamp |
| `problems.jsonl`, `holdout.jsonl` | Pooled problems; the 1,000 GSM8K routing problems reserved first |
| `filter.jsonl`, `filter_report.json` | Per-problem accuracy; shortfalls against targets are reported, never padded |
| `traces.jsonl`, `labeled.jsonl` | (R0, F, Rf) traces; automatic correctness and outcome cell |
| `splits/*.jsonl`, `splits/manifest.json` | Split by problem, with counts and SHA-256 hashes |
| `annotation/*` | Tasks, assignments, raw annotations, agreement report |
| `final/*.jsonl` | Released data with `failure_type` |

## Decisions to make before the full run

- **Models:** choose dated model ids and record them. GPT-4, Claude 3 Opus and
  Gemini 1.0 Pro, used in the old draft, are retired or close to it.
- **Failure definition:** `final_incorrect` (final answer wrong) or
  `degraded_only` (right became wrong). The paper must state which.
- **Prompts:** `scfp_bench/prompts.py` is the source of truth. The paper's
  Appendix A must quote it exactly, including the added "Final answer:" format line.
- **Difficulty band:** if few GSM8K problems land in [0.2, 0.8] for current
  models, report the smaller count or add a harder math source. Do not widen
  the band silently.

## Rough cost

Calls = filter (candidates × filter models × samples) + generation (problems ×
models × 3). With `scfp_v1.yaml` that is about 75,000 calls. At roughly 1,000
input+output tokens per call, that is ~75M tokens. Measure the real per-call
usage in the pilot and multiply.
