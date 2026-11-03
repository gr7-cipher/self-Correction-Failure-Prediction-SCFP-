#!/usr/bin/env bash
# Push queued private Kaggle jobs whenever fewer than 2 GPU sessions are running (Kaggle's limit).
cd "$(dirname "$0")"
KG() { ~/.local/bin/uv tool run --python 3.12 --from kaggle kaggle "$@" 2>&1; }
ALL="scfp-revision scfp-rev-seed1 scfp-rev-seed2 scfp-rev-seed3 scfp-rev-seed4 scfp-rev-gpua scfp-rev-gpub scfp-rev-gpuc"
QUEUE=(train/kernel_seed2 train/kernel_seed3 train/kernel_seed4 k_gpua k_gpub k_gpuc)
pushed=()
while true; do
  running=0; line=""
  for k in $ALL; do
    s=$(KG kernels status gaith7/$k | tail -1)
    case "$s" in *RUNNING*|*QUEUED*) running=$((running+1)); line="$line $k:run";; *COMPLETE*) line="$line $k:done";; *ERROR*) line="$line $k:ERROR";; esac
  done
  echo "$(date +%H:%M) running=$running$line"
  while [ $running -lt 2 ] && [ ${#QUEUE[@]} -gt 0 ]; do
    d=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
    out=$(KG kernels push -p $d --accelerator NvidiaTeslaT4 | tail -1)
    echo "  push $d: $out"
    case "$out" in *successfully*) running=$((running+1)); pushed+=("$d");; *) QUEUE=("$d" "${QUEUE[@]}"); break;; esac
  done
  if [ ${#QUEUE[@]} -eq 0 ] && [ $running -eq 0 ]; then echo "QUEUE EMPTY AND ALL FINISHED"; break; fi
  sleep 600
done
