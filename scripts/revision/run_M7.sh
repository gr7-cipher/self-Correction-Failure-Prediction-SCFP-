#!/usr/bin/env bash
# M7: statistical rigor. Seed retraining (GPU; 4.7 GPU-h in total on Kaggle T4s, one private job per seed):
#   bash scripts/revision/m7_train_seeds.sh <seed>   for seed in 1 2 3 4; outputs copied to results/revision/m7/seeds
# CPU part (cluster bootstrap, permutation tests, Holm, power, calibration) and the seed aggregation:
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m7_stats.py --out results/revision/m7
.venv/bin/python scripts/revision/m7_seeds.py --seeds results/revision/m7/seeds --out results/revision/m7
make tables
