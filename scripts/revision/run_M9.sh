#!/usr/bin/env bash
# M9: answer-token probes at every layer and nested stacking. The residual-stream arrays come from GPU job A
# (gpu_generator_signals.py on a Kaggle T4, copied to results/revision/gpuA; checksums in SHA256SUMS). ~1 h on an M1.
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd results/revision/gpuA && shasum -a 256 -c SHA256SUMS)
.venv/bin/python scripts/revision/m9_probe.py --hidden results/revision/gpuA --out results/revision/m9
make tables
