#!/usr/bin/env bash
# M8: external validity. CPU: subset, second parser and audit sample, exclusion sensitivity. GPU job C (Kaggle T4,
# 4.7 GPU-h): python scripts/revision/m8_gpu.py q7b prompt_alt precision --out results/revision/m8/gpu
# The exclusion analysis needs the job-A excluded-trace features in results/revision/gpuA.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m8_cpu.py --out results/revision/m8 --audit audit
.venv/bin/python scripts/revision/m8_exclusion.py --gpua results/revision/gpuA --out results/revision/m8
.venv/bin/python scripts/revision/m8_analysis.py --gpu results/revision/m8/gpu --out results/revision/m8
make tables
