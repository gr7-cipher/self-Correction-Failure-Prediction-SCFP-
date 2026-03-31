#!/bin/bash
# Minimal debug run script for SCFP framework
set -e

echo "Starting SCFP Debug Pass..."

# Step 1: Preprocess a small subset of real data
echo "Step 1: Preprocessing 100 traces for debugging..."
python3.10 scripts/preprocess_data.py \
    --input data/scfp_v1.jsonl \
    --output data/processed_debug \
    --limit 100 \
    --seed 42

# Step 2: Run 3-Stage Training (Minimal epochs for debug)
echo ""
echo "Step 2: Training 3-Stage DeBERTa (1 epoch each) on CPU..."
python3.10 scripts/train_deberta.py \
    --data-dir data/processed_debug \
    --output-dir models/deberta_debug \
    --three-stage \
    --epochs-binary 1 \
    --epochs-multiclass 1 \
    --epochs-joint 1 \
    --batch-size 2 \
    --max-length 128 \
    --device cpu \
    --no-fp16 \
    --seed 42

# Step 3: Run Evaluation
echo ""
echo "Step 3: Running evaluation on debug model..."
python3.10 scripts/evaluate_all.py \
    --models-dir models/deberta_debug \
    --data-dir data/processed_debug \
    --output results_debug
