# Self-Correction Failure Prediction (SCFP)

Code, data and results for the paper *Predicting Self-Correction Failure on Numeric Problems Reduces to Verifying the First Answer: Evidence from Generator Signals on Unfiltered Problems*. The paper is a manuscript under revision.

In intrinsic self-correction, a language model critiques and revises its own answer. We ask whether the outcome can be predicted before the revision is generated, for small (at most 7B parameters) open-weight generators.

## Main results

The numbers below come from `paper/main.pdf`, where each one has its table and interval.

| Question | Result |
|---|---|
| Can failure be predicted on unfiltered problems? | Yes, within limits. On 2,000 problems and four generators, signals computed before revising reach a within-source AUC of 0.634 to 0.734, significantly better than text features for every generator. |
| What makes it predictable? | Mainly whether the first answer can be verified. The gain is concentrated on arithmetic problems that a program can check, and larger external verifiers match or exceed the signals. |
| Does this hold on non-numeric problems? | Untested. Agreement-based verification is also weak on StrategyQA and LogiQA (AUC 0.485 to 0.670 for a wrong first answer), so we have no verifier good enough to test the claim there. |
| Is the helped/harmed change easier to predict? | No, once the answer change is not given away. For Qwen2.5-3B, helped against harmed is at chance. |
| What about a difficulty-filtered benchmark? | On SCFP v1.0 (716 traces), failure is close to unpredictable and fine-tuned encoders do not beat bag-of-words features. The filter reduces within-dataset signal for all three generators tested. |
| Does routing on predicted failure help? | Only partly. On 700 held-out GSM8K problems and three generators, routers built on these predictors have the best combined score in at most 115 of 594 cost settings, and fixed policies in at least 474. They raise accuracy mainly by sending problems to a human, whose accuracy and cost are assumed. |

Answer extraction was audited on a stratified sample of 100 traces and showed no correctness-label errors (95% upper bound 3.7%). One author did the reading, and it was not blind, as the paper states. The failure-mode annotation scheme is a proposal: its human labels are still being collected, and no result depends on them.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript, bibliography, generated tables and figures, PDF and diff |
| `SCFP/benchmark/` | Benchmark builder, experiment code, configs and the logged runs (`scfp_runs/`) |
| `SCFP/src/` | The segment-aware DeBERTa predictor and routing code |
| `scripts/revision/` | One script per experiment in the paper |
| `scripts/kaggle/` | The Kaggle GPU jobs as they were run |
| `results/revision/` | Output of every analysis, with raw GPU outputs and job logs |
| `data/audit/` | Extraction audit sample, author sheet and separate AI checks |
| `verification/` | Independent re-check of the paper's numbers |
| `response/` | Response letter and report for the current revision |

## Quick start

Rebuilding the paper's tables and checking its numbers needs no GPU. You need Python 3.10, and [tectonic](https://tectonic-typesetting.github.io/) to build the PDF.

```bash
git clone -b revision-round2 https://github.com/gr7-cipher/self-Correction-Failure-Prediction-SCFP-.git
```

```bash
cd self-Correction-Failure-Prediction-SCFP-
```

```bash
python3.10 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-lock.txt
```

```bash
make tables
```

```bash
make pdf
```

```bash
bash scripts/revision/verify.sh
```

[REPRODUCE.md](REPRODUCE.md) explains how to rerun each analysis and the GPU jobs. Regenerated traces will not match the logged ones exactly, because greedy outputs differ across precision and hardware, so the labels belong to the logged traces.

## Large files

The trained DeBERTa weights, the residual-stream arrays and the Kaggle dataset package are too large for git. They are in the [`round2-artifacts` release](https://github.com/gr7-cipher/self-Correction-Failure-Prediction-SCFP-/releases/tag/round2-artifacts), with checksums. Base models and source datasets come from Hugging Face.

## Tests

```bash
.venv/bin/python -m pytest SCFP/benchmark/tests SCFP/tests
```

Both suites pass.

## License

The project is dual-licensed: see [LICENSE](LICENSE) (Apache 2.0) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
