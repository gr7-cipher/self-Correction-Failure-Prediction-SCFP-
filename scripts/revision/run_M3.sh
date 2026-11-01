#!/usr/bin/env bash
# M3: the Delta-label, the deployable two-stage gate and the answer-span log-prob (CPU), then the correctness-prediction
# baselines from the GPU jobs: P(True) and verbalized confidence (job A, gpu_generator_signals.py), reward model and
# LLM judge (job B, gpu_external_judges.py). Outputs of the GPU jobs are copied to results/revision/gpuA and gpuB.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/revision/m3_cpu.py --out results/revision/m3
.venv/bin/python scripts/revision/m3_gpu.py --gpua results/revision/gpuA --gpub results/revision/gpuB --out results/revision/m3
make tables
