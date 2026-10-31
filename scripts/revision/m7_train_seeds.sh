#!/usr/bin/env bash
# M7: retrain every Table 8 encoder and Table 9 ablation with one extra seed (REVISION_PLAN.md, M7).
# Identical to run_mac_overnight.sh except --seed, --device cuda and --no-amp (float32, as on the M1).
#   bash scripts/revision/m7_train_seeds.sh 1
set -uo pipefail
SEED=$1
cd "$(dirname "$0")/../../SCFP/benchmark"
export PYTHONPATH=.
PY=${PY:-python}
CFG=configs/laptop_local.yaml
RD=scfp_runs/laptop_v1
SMALL=microsoft/deberta-v3-small
DEV=${DEV:-cuda}
COMMON="--max-len 768 --batch-size 2 --grad-accum 8 --freeze-word-embeddings --device $DEV --no-amp --seed $SEED"
run() {
  local name="$1__seed$SEED"; shift
  if [ -f "$RD/results/$name.json" ]; then echo "=== $name: done, skipping"; return; fi
  echo "=== $name  $(date '+%F %T')"
  $PY -m scfp_bench.experiments.train_encoder --config $CFG --name "$name" $COMMON "$@" || echo "!!! $name failed"
}
run deberta-scfp        --arch scfp  --model $SMALL
run deberta-plain       --arch plain --model $SMALL
run roberta-base        --arch plain --model FacebookAI/roberta-base
run bert-base           --arch plain --model google-bert/bert-base-uncased
run abl-no-segments     --arch scfp  --model $SMALL --no-segments
run abl-no-attention    --arch scfp  --model $SMALL --no-special-attention
run abl-inputs-P-R0     --arch scfp  --model $SMALL --inputs P,R0
run abl-inputs-P-F      --arch scfp  --model $SMALL --inputs P,F
run abl-inputs-R0-F     --arch scfp  --model $SMALL --inputs R0,F
run abl-deberta-xsmall  --arch scfp  --model microsoft/deberta-v3-xsmall
echo "=== seed $SEED done $(date '+%F %T')"
