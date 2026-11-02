#!/usr/bin/env bash
# Independent re-check of every number in the revised paper (Parts D, E, F of verify_metrics.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
B=SCFP/benchmark/scfp_runs
.venv/bin/python verification/verify_metrics.py --run-dir $B/laptop_v1 --tables paper/paper_tables \
  --v7-dir $B/paper_v7 --v7-runs $B/signals_mac_q05_big $B/signals_kaggle \
  --rev-tables paper/paper_tables --routing-dir $B/laptop_v1/routing --manuscript paper/main.tex --round2 results/revision \
  > results/revision/verify_output.txt
tail -1 results/revision/verify_output.txt
