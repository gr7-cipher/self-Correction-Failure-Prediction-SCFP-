#!/usr/bin/env bash
# M5: deconfounded natural-set metrics (within-source AUC, identity baseline, AUPRC, Brier, cluster bootstrap),
# filter effect and sub-tasks within source, and the post-hoc source x fold check. ~2.5 min on an M1, no GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m5_deconfound.py --out results/revision/m5
.venv/bin/python scripts/revision/m5b_filter_subtasks.py --out results/revision/m5
.venv/bin/python scripts/revision/m5c_fold_strata.py --out results/revision/m5
make tables
