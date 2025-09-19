#!/bin/bash

# SCFP Framework - Complete Reproduction Script
# This script reproduces all experiments from the paper

set -e  # Exit on any error

echo "======================================================"
echo "SCFP Framework - Complete Reproduction"
echo "======================================================"

# Configuration
DATA_SIZE=12000
SEED=42
BATCH_SIZE=16
MAX_LENGTH=1024
EPOCHS=5

# Create directories
echo "Creating directories..."
mkdir -p data/raw data/processed models/deberta models/baselines results logs

# Step 1: Generate synthetic dataset
echo ""
echo "Step 1: Generating synthetic dataset..."
echo "------------------------------------------------------"
python scripts/generate_synthetic_data.py \
    --output data/raw/scfp_synthetic.json \
    --size $DATA_SIZE \
    --success-rate 0.6 \
    --seed $SEED \
    --jh-rate 0.25 \
    --cm-rate 0.20 \
    --ba-rate 0.20 \
    --oc-rate 0.20 \
    --rm-rate 0.15

# Step 2: Preprocess data
echo ""
echo "Step 2: Preprocessing data..."
echo "------------------------------------------------------"
python scripts/preprocess_data.py \
    --input data/raw/scfp_synthetic.json \
    --output data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --stratify failure_mode \
    --seed $SEED

# Step 3: Train DeBERTa model
echo ""
echo "Step 3: Training DeBERTa-v3 model..."
echo "------------------------------------------------------"
python scripts/train_deberta.py \
    --data-dir data/processed \
    --output-dir models/deberta \
    --model-name microsoft/deberta-v3-base \
    --batch-size $BATCH_SIZE \
    --learning-rate 2e-5 \
    --num-epochs $EPOCHS \
    --max-length $MAX_LENGTH \
    --seed $SEED \
    2>&1 | tee logs/deberta_training.log

# Step 4: Train baseline models
echo ""
echo "Step 4: Training baseline models..."
echo "------------------------------------------------------"
python scripts/train_baselines.py \
    --data-dir data/processed \
    --output-dir models/baselines \
    --models all \
    --batch-size $BATCH_SIZE \
    --max-length 512 \
    --seed $SEED \
    2>&1 | tee logs/baselines_training.log

# Step 5: Comprehensive evaluation
echo ""
echo "Step 5: Running comprehensive evaluation..."
echo "------------------------------------------------------"
python scripts/evaluate_all.py \
    --models-dir models \
    --data-dir data/processed \
    --output results \
    --ablation \
    --generate-plots \
    2>&1 | tee logs/evaluation.log

# Step 6: Demo routing system
echo ""
echo "Step 6: Demonstrating routing system..."
echo "------------------------------------------------------"
python scripts/demo_routing.py \
    --model models/deberta/final_model \
    --output results/routing_demo.json \
    2>&1 | tee logs/routing_demo.log

# Step 7: Generate final report
echo ""
echo "Step 7: Generating final report..."
echo "------------------------------------------------------"
python scripts/generate_results.py \
    --results-dir results \
    --output results/summary \
    2>&1 | tee logs/report_generation.log

# Step 8: Run unit tests
echo ""
echo "Step 8: Running unit tests..."
echo "------------------------------------------------------"
python -m pytest tests/ -v --tb=short 2>&1 | tee logs/tests.log

echo ""
echo "======================================================"
echo "REPRODUCTION COMPLETE!"
echo "======================================================"
echo ""
echo "Results are available in the following locations:"
echo "  - Synthetic dataset: data/raw/scfp_synthetic.json"
echo "  - Processed data: data/processed/"
echo "  - Trained models: models/"
echo "  - Evaluation results: results/"
echo "  - Logs: logs/"
echo ""
echo "Key files:"
echo "  - Main results: results/comprehensive_evaluation.json"
echo "  - Model comparison plots: results/model_comparison.png"
echo "  - Routing demo: results/routing_demo.json"
echo "  - Final report: results/summary/REPORT.md"
echo ""
echo "To view the main results:"
echo "  cat results/comprehensive_evaluation.json | jq '.summary'"
echo ""
echo "To run individual components:"
echo "  ./scripts/generate_synthetic_data.py --help"
echo "  ./scripts/train_deberta.py --help"
echo "  ./scripts/evaluate_all.py --help"
echo "  ./scripts/demo_routing.py --help"
echo ""
echo "For interactive routing demo:"
echo "  python scripts/demo_routing.py --interactive"
echo ""
