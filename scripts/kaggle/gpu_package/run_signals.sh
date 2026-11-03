#!/usr/bin/env bash
# Natural-set failure prediction with stronger generators on one GPU (Kaggle T4/P100, 16 GB).
#
#   bash run_signals.sh     # about 8-11 hours on a T4 (resumable; may need 2 Kaggle sessions)
#
# 1. build     2,000 unfiltered problems x 3 generators (Qwen2.5-1.5B, SmolLM2-1.7B, Qwen2.5-3B)
# 2. features  log-probs, critique verdicts and hidden states (teacher forcing)
# 3. verify    program-of-thought check of the initial answer (GSM8K)
# 4. evaluate  5-fold grouped CV: TF-IDF vs. signals vs. probes vs. nested stacking, per generator
#
# Every generation call is logged in scfp_runs/signals_v2/calls.jsonl and reused on re-runs, so a
# stopped session resumes where it left off. Results: scfp_runs/signals_v2/results/
set -uo pipefail
cd "$(dirname "$0")/benchmark"
export PYTHONPATH=.
PY=${PY:-python}
CFG=${CFG:-configs/signals.yaml}
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "warning: no GPU found; this will be very slow"

for stage in build features verify evaluate; do
  echo "=== $stage  $(date '+%F %T')"
  $PY -u -m scfp_bench.experiments.signals $stage --config $CFG || { echo "!!! $stage failed"; exit 1; }
done
echo "=== stacking  $(date '+%F %T')"
$PY -W ignore -u -m scfp_bench.experiments.improve --config $CFG || { echo "!!! stacking failed"; exit 1; }

cd ..
rm -f scfp_signals_results.zip
zip -qr scfp_signals_results.zip benchmark/scfp_runs/signals_v2 -x "*.npz"
echo ">>> Done. Send back: $(pwd)/scfp_signals_results.zip"
echo "    (hidden-state .npz files are excluded from the zip; they can be regenerated)"
