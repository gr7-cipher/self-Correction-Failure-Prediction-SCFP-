#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
../../.venv/bin/python -u -m scfp_bench.experiments.signals verify --config configs/signals_mac_big.yaml || { echo "!!! verify failed"; exit 1; }
../../.venv/bin/python -W ignore -u -m scfp_bench.experiments.improve --config configs/signals_mac_big.yaml --no-curve --out improve_verifier || { echo "!!! evaluate failed"; exit 1; }
echo "=== all done  $(date '+%F %T')"
