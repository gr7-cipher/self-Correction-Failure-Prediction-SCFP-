# SCFP: GPU experiments

This folder holds the code, data and one-command scripts for two experiments on a free
16 GB GPU (Kaggle T4 or P100, or a Colab T4):

| Script | Experiment | Time on a T4 | Results |
|---|---|---|---|
| `run_signals.sh` | **Stronger generators on the natural set (recommended first).** Is self-correction failure of 1.5B–3B models more predictable than that of the 0.5B model? | about 8–11 h (resumable; may need 2 Kaggle sessions) | `scfp_signals_results.zip` |
| `run_gpu.sh` | **Larger models.** DeBERTa-v3-large, RoBERTa-large and DeBERTa-v3-small with 5-fold cross-validation | about 5–7 h | `scfp_gpu_results.zip` |

In the notebook `SCFP_GPU.ipynb`, set `EXPERIMENT = 'signals'` or `'larger_models'` in the
second cell.

## Experiment 1: stronger generators on the natural set (`run_signals.sh`)

**What the Mac already showed** (Qwen2.5-0.5B, 1,718 unfiltered traces; full tables in
`benchmark/scfp_runs/scfp_v1/reference_mac_results/signals_q05_big/`):

- On SCFP v1 (difficulty-filtered) every predictor is at chance (AUC about 0.5).
- On the unfiltered natural set, TF-IDF reaches AUC 0.611 and has plateaued (the learning curve
  is flat from about 700 training traces).
- Nested stacking of text, generator signals and hidden-state probes reaches **AUC 0.641,
  62.2% accuracy** (majority 56.6%), significantly better than TF-IDF (McNemar p = 0.031).
- Whether the revision changes the answer is easy to predict (AUC about 0.89); whether the
  initial answer is right is not (about 0.70). The second is the bottleneck.

**What this run does.** The same 2,000 unfiltered problems (same seed and exclusions, so traces
can be compared per problem) with three stronger generators: Qwen2.5-1.5B, SmolLM2-1.7B and
Qwen2.5-3B.

| Stage | What |
|---|---|
| `build` | Greedy traces (initial answer, critique, revision) for each problem and generator. A final `\boxed{}` counts as an explicit answer |
| `features` | Log-probabilities, critique verdicts and hidden states at the middle and last layer (teacher forcing) |
| `verify` | Program-of-thought check of the initial answer on GSM8K: the generator writes a program, it is screened and run in an isolated subprocess, and its output is compared with the initial answer and with the critique's answer |
| `evaluate` | 5-fold CV grouped by problem: TF-IDF vs. scalar signals vs. probes (`signals_summary.md`) |
| stacking | The same plus nested stacking, per generator and pooled (`improve_summary.md`) |

**Reading the result** (`scfp_runs/signals_v2/results/improve_summary.md`):
- **Stronger generators are more predictable** if their AUC on the paper label is clearly above
  the 0.5B model's 0.64, with a McNemar p < 0.05 against TF-IDF.
- **The verifier helps** if the stacked predictors gain over the Mac run, mainly on GSM8K.
- If neither happens, the plateau at AUC 0.6–0.65 is a robust finding across model sizes.

**Settings.** Edit `benchmark/configs/signals.yaml` to change the generators or problem counts.
Removing `qwen2.5-3b` cuts about 40% of the time.

## Experiment 2: larger models (`run_gpu.sh`)


## What it runs

| Step | What | Why |
|---|---|---|
| 1 | Majority, random, length, confidence, TF-IDF + logistic regression, with 5-fold grouped cross-validation | Reference baselines on exactly the same folds |
| 2 | **DeBERTa-v3-large** (segment-aware, the paper's model), DeBERTa-v3-large (plain), **RoBERTa-large**, DeBERTa-v3-small (segment-aware), each with 5-fold grouped CV | The question: does a larger model help? |
| 3 | DeBERTa-v3-large on the paper's fixed split, seeds 42, 43, 44 | Directly comparable with the paper's tables, plus seed variance |
| 4 | Summary tables and a results zip | |

### Why cross-validation instead of the paper's test split

The fixed test split has only 108 traces, so its 95% interval on accuracy is about ±9 points.
With 5-fold cross-validation, every one of the 716 traces is predicted exactly once by a
model that never saw its problem; the folds are grouped by problem and stratified by source.
That narrows the interval to about ±3.5 points and gives paired McNemar tests on 716 traces.

The single split was also optimistic. Under cross-validation, TF-IDF drops from 63.0% to
**57.3% [53.5, 60.8]**, which is not significantly above the majority class (52.9%, p = 0.14).
See `benchmark/scfp_runs/scfp_v1/reference_mac_results/cv_summary_cpu_baselines.md`.

### Settings

| Setting | Value |
|---|---|
| Sequence length | 768 tokens (RoBERTa is capped at its 512-token limit) |
| Batch size | 4 × 4 gradient-accumulation steps = effective 16 |
| Precision | fp16 autocast with fp32 weights, gradient checkpointing |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Epochs | up to 10, early stopping on validation macro-F1 (patience 5) |

Unlike the Mac runs, the word embeddings are **not** frozen.

## How to run

### Kaggle (recommended: 30 free GPU hours per week)

1. Zip this folder as `SCFP_GPU_package.zip`.
2. On kaggle.com, go to *Datasets* → *New Dataset* and upload the zip.
3. Create a notebook. Choose *File* → *Import Notebook* and select `SCFP_GPU.ipynb`.
4. In the right panel:
   - *Add Input*: your dataset.
   - *Accelerator*: *GPU T4 x1* (or *P100*).
   - *Internet*: On. Models are downloaded from Hugging Face. Kaggle requires a
     phone-verified account for GPU and Internet.
5. Choose *Save Version* → *Save & Run All (Commit)*. The run continues after you close the
   browser.
6. When it finishes, download `scfp_gpu_results.zip` from the version's *Output*.

### Google Colab

1. Open `SCFP_GPU.ipynb` in Colab and choose *Runtime* → *Change runtime type* → *T4 GPU*.
2. Run all cells and upload `SCFP_GPU_package.zip` when asked.
3. The results zip downloads at the end.

Free Colab sessions can stop after a few hours. To resume, re-run all cells and upload the
previous `scfp_gpu_results.zip` next to the package; finished folds are skipped.

### Any machine with a GPU

```bash
pip install -r requirements_gpu.txt
bash run_gpu.sh
```

To check that everything works before using GPU time, run a quick CPU test with a tiny model
(15–20 minutes; its numbers are meaningless):

```bash
QUICK=1 bash run_gpu.sh
```

## Expected time on a T4

| Step | Time |
|---|---|
| CV, DeBERTa-v3-large (×2 configurations) | about 1–1.5 h each |
| CV, RoBERTa-large | about 1 h |
| CV, DeBERTa-v3-small | about 30 min |
| Fixed split × 3 seeds | about 45 min |
| **Total** | **about 5–7 h**, within one Kaggle session (12 h limit) |

## What you get back

`scfp_gpu_results.zip` contains `benchmark/scfp_runs/scfp_v1/` with:

| File | Contents |
|---|---|
| `results/cv_summary.md` / `.json` | Accuracy [95% CI], macro-F1, AUC, McNemar p vs TF-IDF and vs majority, all on 716 traces |
| `results/fixed_split_summary.md` | DeBERTa-v3-large on the paper's 108-trace test split, 3 seeds |
| `cv/fold*/predictions/*.jsonl` and `predictions/*.jsonl` | Per-example predictions; every number can be recomputed from these |
| `results/<run>.json` | Training logs, GPU name, wall-clock time |

Model weights are not included in the zip.

## How to read the result

- **A larger model helps** only if its cross-validated accuracy is significantly above
  TF-IDF's (McNemar p < 0.05 in `cv_summary.md`), not merely a higher point estimate.
- **If no model beats TF-IDF or the majority class**, that is itself a well-supported finding
  with 716 traces: on this benchmark, the partial trace carries little signal about whether
  self-correction will fail.

## Folder layout

| Path | Contents |
|---|---|
| `run_gpu.sh` | One-command run |
| `SCFP_GPU.ipynb` | Kaggle/Colab notebook |
| `requirements_gpu.txt` | Python dependencies |
| `benchmark/scfp_bench/` | Pipeline and experiment code. CV runner: `experiments/cv.py`; training: `experiments/train_encoder.py` |
| `benchmark/configs/gpu.yaml` | Configuration |
| `benchmark/scfp_runs/scfp_v1/splits/` | SCFP v1.0: 716 traces (train / val / test) |
| `benchmark/scfp_runs/scfp_v1/features/` | Critique log-probabilities (Confidence baseline) |
| `benchmark/scfp_runs/scfp_v1/reference_mac_results/` | The Mac results reported in the paper (v6) |
| `src/scfp/` | Model code: segment-aware DeBERTa predictor |
