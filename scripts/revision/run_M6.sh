#!/usr/bin/env bash
# M6: the PoT verifier against self-consistency agreement on the natural-set GSM8K traces.
# Generation (GPU, ~3.8 h on one Kaggle T4; run once, logged calls in results/revision/m6/gen):
#   python scripts/revision/m6_sc_generate.py --out results/revision/m6/gen
# Analysis (CPU, a few minutes) and tables:
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m6_analysis.py --gen results/revision/m6/gen --out results/revision/m6
make tables
