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

# Step 1: Prepare dataset
echo ""
if [ ! -f "data/scfp_v1.jsonl" ]; then
    echo "ERROR: Real benchmark dataset (data/scfp_v1.jsonl) not found!"
    echo "This repository requires the real SCFP v1.0 benchmark for correct implementation."
    exit 1
fi

echo "Step 1: Using REAL benchmark dataset (scfp_v1.jsonl)..."
INPUT_DATA="data/scfp_v1.jsonl"

# Step 2: Preprocess data
echo ""
echo "Step 2: Preprocessing data..."
echo "------------------------------------------------------"
python scripts/preprocess_data.py \
    --input $INPUT_DATA \
    --output data/processed \
    --seed $SEED

# Step 3: Train DeBERTa model (3-Stage)
echo ""
echo "Step 3: Training DeBERTa-v3 model (3-STAGE PROCEDURE)..."
echo "------------------------------------------------------"
python scripts/train_deberta.py \
    --data-dir data/processed \
    --output-dir models/deberta \
    --three-stage \
    --epochs-binary 10 \
    --epochs-multiclass 15 \
    --epochs-joint 20 \
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
