# Reproducing the paper

This file covers the current manuscript (`paper/main.tex`, branch `revision-round2`). Sections of `README.md` below its top note describe an earlier version of the framework.

## What is where

| Path | Contents |
|---|---|
| `paper/` | Manuscript, bibliography, generated tables and figures, PDF and diff |
| `SCFP/benchmark/` | Benchmark builder, experiment code, configs |
| `SCFP/benchmark/scfp_runs/` | Logged model calls, traces, splits, features and predictions of every run |
| `scripts/revision/` | One script per experiment in the paper |
| `scripts/kaggle/` | The Kaggle GPU jobs as they were run (`k_e1_*`, `k_e2*`, `k_gpu*`, `k_revision_m`, `k_signals`, `kernel_seed*`), queue scripts and the earlier GPU package |
| `results/revision/` | Output of every analysis, including raw round-2 GPU outputs and Kaggle logs (`kaggle_logs/`) |
| `data/audit/` | Extraction audit: sample, author sheet, and AI checks kept separately |
| `verification/` | Independent re-check of the numbers |
| `response/` | Response letter, agent report, title options |

## Files too large for git

These are assets of the [`round2-artifacts` release](https://github.com/gr7-cipher/self-Correction-Failure-Prediction-SCFP-/releases/tag/round2-artifacts). `SHA256SUMS.txt` there has their checksums.

| Asset | Put it at | Needed for |
|---|---|---|
| `deberta-scfp.pt` | `SCFP/benchmark/scfp_runs/laptop_v1/results/` | DeBERTa scoring without retraining |
| `hidden_*.npz` (4 files) | `results/revision/gpuA/` | Rerunning the hidden-state probes |
| `scfp_revision_package.zip` | Kaggle dataset `scfp-revision-package` | Rerunning the Kaggle jobs |

No table needs them to be rebuilt from the saved results.

## Environment

Python 3.10 for the analyses. `requirements.txt` lists direct dependencies and `requirements-lock.txt` the exact versions used on the Apple M1. GPU jobs install `scripts/kaggle/gpu_package/requirements_gpu.txt`. LaTeX builds use `tectonic`.

```bash
python3.10 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-lock.txt
```

## Rebuild the paper and re-check the numbers

No GPU is needed.

```bash
make tables
```

```bash
make pdf
```

```bash
make diff2
```

```bash
bash scripts/revision/verify.sh
```

## Rerun the analyses

Each reads saved outputs and writes to `results/revision/`.

```bash
.venv/bin/python scripts/revision/e1_analysis.py --gpu results/revision/e1/gpu --out results/revision/e1
```

```bash
.venv/bin/python scripts/revision/e2_analysis.py --gen results/revision/e2/gen --out results/revision/e2
```

```bash
.venv/bin/python scripts/revision/a2_cost.py --timing results/revision/e2/gen/a2 --out results/revision/a2
```

```bash
.venv/bin/python scripts/revision/e4_audit_score.py --out results/revision/e4
```

The other experiments have one script each (`a1_delta.py`, `a3_precision.py`, `t7_nat_main.py`, `m6_analysis.py`) or a runner (`run_M3.sh` to `run_M9.sh`).

## Rerun the GPU jobs

The jobs ran on one Kaggle NVIDIA T4. Attach `scfp_revision_package.zip` as the dataset and push a job folder with the Kaggle CLI. Round-2 run times are in `response/agent_report.md`. Greedy outputs differ across precision and hardware, so labels are properties of the logged traces and regenerated traces will not match exactly.

## Not in this repository

Base models and source datasets come from Hugging Face, with commits pinned in `scripts/revision/m6_sc_generate.py`. Human failure-mode labels are still being collected.
