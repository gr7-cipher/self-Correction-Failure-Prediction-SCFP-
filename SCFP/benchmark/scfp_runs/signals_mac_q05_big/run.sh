#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
for s in build features evaluate; do
  echo "=== $s  $(date '+%F %T')"
  ../../.venv/bin/python -u -m scfp_bench.experiments.signals $s --config configs/signals_mac_big.yaml || { echo "!!! $s failed"; exit 1; }
done
echo "=== all done  $(date '+%F %T')"
