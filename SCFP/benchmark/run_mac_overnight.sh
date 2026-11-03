#!/usr/bin/env bash
# Every measured result of the paper, run on an Apple M1 (8 GB) in one go.
# Close memory-heavy apps first (browsers, Acrobat): training needs most of the RAM.
# Re-running skips finished runs (results/<name>.json exists) and resumes routing.
#
#   caffeinate -i bash run_mac_overnight.sh 2>&1 | tee -a scfp_runs/laptop_v1/overnight.log
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=../../.venv/bin/python
CFG=configs/laptop_local.yaml
RD=scfp_runs/laptop_v1
SMALL=microsoft/deberta-v3-small
# DeBERTa-v3-large does not fit in 8 GB; every encoder uses the same memory settings.
COMMON="--max-len 768 --batch-size 2 --grad-accum 8 --freeze-word-embeddings --device mps"

run() {  # name, then train_encoder arguments
  local name=$1; shift
  if [ -f "$RD/results/$name.json" ]; then echo "=== $name: done, skipping"; return; fi
  echo "=== $name  $(date '+%F %T')"
  $PY -m scfp_bench.experiments.train_encoder --config $CFG --name "$name" $COMMON "$@" \
    || echo "!!! $name failed"
}

run deberta-scfp        --arch scfp  --model $SMALL --save
run deberta-plain       --arch plain --model $SMALL
run roberta-base        --arch plain --model FacebookAI/roberta-base
run bert-base           --arch plain --model google-bert/bert-base-uncased
run abl-no-segments     --arch scfp  --model $SMALL --no-segments
run abl-no-attention    --arch scfp  --model $SMALL --no-special-attention
run abl-inputs-P-R0     --arch scfp  --model $SMALL --inputs P,R0
run abl-inputs-P-F      --arch scfp  --model $SMALL --inputs P,F
run abl-inputs-R0-F     --arch scfp  --model $SMALL --inputs R0,F
run abl-deberta-xsmall  --arch scfp  --model microsoft/deberta-v3-xsmall
run xmodel-qwen-to-all  --arch scfp  --model $SMALL \
    --train-models qwen2.5-0.5b,qwen2.5-1.5b --test-models qwen2.5-0.5b,qwen2.5-1.5b,smollm2-1.7b

echo "=== routing  $(date '+%F %T')"
$PY -m scfp_bench.experiments.routing generate --config $CFG --n 200
$PY -m scfp_bench.experiments.routing score    --config $CFG --predictor deberta-scfp
$PY -m scfp_bench.experiments.routing evaluate --config $CFG --n-tune 50
$PY -m scfp_bench.experiments.analysis --config $CFG --predictor deberta-scfp || echo "!!! analysis failed"
$PY -m scfp_bench.experiments.make_tables --config $CFG --out ../../../paper_tables
echo "=== all done  $(date '+%F %T')"
