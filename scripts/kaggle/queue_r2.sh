#!/usr/bin/env bash
# Round 2: push queued private Kaggle jobs whenever fewer than 2 GPU sessions are running; download outputs when done.
cd "$(dirname "$0")"
KG() { ~/.local/bin/uv tool run --python 3.12 --with six --from kaggle kaggle "$@" 2>&1; }
QUEUE=(k_e1_q05 k_e1_sm k_e2b k_e2a k_e1_q15)
ID() { case $1 in k_e1_q05) echo scfp-r2-e1-q05;; k_e1_sm) echo scfp-r2-e1-sm;; k_e1_q15) echo scfp-r2-e1-q15;; k_e2a) echo scfp-r2-e2a;; k_e2b) echo scfp-r2-e2b;; esac; }
pushed=(); fetched=()
sleep 5
while true; do
  running=0; line=""
  for d in ${pushed[@]+"${pushed[@]}"}; do
    k=$(ID $d); s=$(KG kernels status gaith7/$k | tail -1)
    case "$s" in
      *RUNNING*|*QUEUED*) running=$((running+1)); line="$line $k:run";;
      *COMPLETE*|*ERROR*|*CANCEL*) line="$line $k:$(echo $s | grep -oE 'COMPLETE|ERROR|CANCEL')"
        if [[ ! " ${fetched[*]-} " == *" $k "* ]]; then mkdir -p out_$k && KG kernels output gaith7/$k -p out_$k >/dev/null; fetched+=("$k"); echo "  fetched $k"; fi;;
      *) line="$line $k:?";;
    esac
  done
  echo "$(date +%H:%M) running=$running$line"
  while [ $running -lt 2 ] && [ ${#QUEUE[@]} -gt 0 ]; do
    d=${QUEUE[0]}
    out=$(KG kernels push -p $d --accelerator NvidiaTeslaT4 | tail -1)
    echo "  push $d: $out"
    case "$out" in *successfully*) QUEUE=("${QUEUE[@]:1}"); running=$((running+1)); pushed+=("$d");; *) break;; esac
  done
  if [ ${#QUEUE[@]} -eq 0 ] && [ $running -eq 0 ]; then echo "QUEUE EMPTY AND ALL FINISHED"; break; fi
  sleep 600
done
