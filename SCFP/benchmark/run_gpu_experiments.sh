#!/usr/bin/env bash
# Runs every fine-tuned experiment of the paper on one GPU (Kaggle/Colab T4, 16 GB).
# Defaults: batch 4 x gradient accumulation 4 = effective batch 16, fp16 autocast and
# gradient checkpointing. Each run writes predictions/<name>__binary.jsonl
# and results/<name>.json under the run directory; make_tables picks them up.
#
#   bash run_gpu_experiments.sh                 # everything
#   ONLY="deberta-scfp roberta-large" bash run_gpu_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
CFG=${CFG:-configs/laptop_local.yaml}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}   # effective batch 16, as in the paper
LARGE=microsoft/deberta-v3-large

run() {  # name, then train_encoder arguments
  local name=$1; shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi
  echo "=== $name"
  python -m scfp_bench.experiments.train_encoder --config "$CFG" --name "$name" --batch-size "$BATCH" --grad-accum "$ACCUM" "$@"
}

# Main model and fine-tuned baselines (Tables main_results / encoder_results)
run deberta-scfp        --arch scfp  --model $LARGE
run deberta-plain       --arch plain --model $LARGE
run roberta-large       --arch plain --model FacebookAI/roberta-large
run bert-base           --arch plain --model google-bert/bert-base-uncased

# Ablations (Table ablation_studies)
run abl-no-segments     --arch scfp --model $LARGE --no-segments
run abl-no-attention    --arch scfp --model $LARGE --no-special-attention
run abl-binary-only     --arch scfp --model $LARGE --binary-only
run abl-inputs-P-R0     --arch scfp --model $LARGE --inputs P,R0
run abl-inputs-P-F      --arch scfp --model $LARGE --inputs P,F
run abl-inputs-R0-F     --arch scfp --model $LARGE --inputs R0,F
run abl-deberta-base    --arch scfp --model microsoft/deberta-v3-base

# Cross-model generalization (Table cross_model_generalization): train on Qwen2.5 only,
# test on all generators; split the prediction file by source_model for the table.
run xmodel-qwen-to-all  --arch scfp --model $LARGE \
    --train-models qwen2.5-0.5b,qwen2.5-1.5b --test-models qwen2.5-0.5b,qwen2.5-1.5b,smollm2-1.7b

python -m scfp_bench.experiments.make_tables --config "$CFG" --out ../../../paper_tables
echo "done: copy the run directory's predictions/ and results/ back, then re-run verify_metrics.py"
