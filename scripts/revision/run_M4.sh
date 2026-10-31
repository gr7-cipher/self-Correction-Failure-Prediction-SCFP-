#!/usr/bin/env bash
# M4: routing baselines without self-correction, Pareto sweep, risk-coverage, Overall-Score sensitivity.
# Uses only the existing routing log (no generation). ~15 s on an M1.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m4_routing.py --out results/revision/m4
make tables
