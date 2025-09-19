#!/usr/bin/env python3
"""
Train baseline models for SCFP comparison.

This script trains all baseline models including BERT, RoBERTa,
and evaluates heuristic baselines.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.dataset import SCFPDataset, CorrectionTrace
from scfp.data.loaders import create_dataloaders
from scfp.models.baselines import (
    BaselineModels, FineTunedBERTBaseline, FineTunedRoBERTaBaseline
)
from scfp.training.trainer import Trainer, TrainingConfig
from scfp.training.metrics import EvaluationMetrics


def evaluate_heuristic_baseline(baseline, test_traces, baseline_name):
    """Evaluate a heuristic baseline model."""
    print(f"\nEvaluating {baseline_name}...")
    
    # Convert traces to text format
    trace_texts = []
    binary_labels = []
    multiclass_labels = []
    
    for trace in test_traces:
        text = f"Prompt: {trace.prompt} [SEP] Initial Response: {trace.initial_response} [SEP] Critique: {trace.critique}"
        trace_texts.append(text)
        binary_labels.append(1 if trace.is_success else 0)
        
        # Map failure mode to index
        mode_to_idx = {
            "success": 0, "jh": 1, "cm": 2, "ba": 3, "oc": 4, "rm": 5
        }
        multiclass_labels.append(mode_to_idx[trace.failure_mode.value])
    
    # Get predictions
    binary_probs = baseline.predict_failure_probability(trace_texts)
    binary_preds = (binary_probs < 0.5).astype(int)  # Failure prob < 0.5 means success
    
    multiclass_preds, multiclass_probs = baseline.predict_failure_mode(trace_texts)
    
    # Calculate metrics
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds, average='macro')
    
    # For binary AUC, we need success probabilities
    success_probs = 1 - binary_probs
    binary_auc = roc_auc_score(binary_labels, success_probs)
    
    multiclass_accuracy = accuracy_score(multiclass_labels, multiclass_preds)
    multiclass_f1 = f1_score(multiclass_labels, multiclass_preds, average='macro', zero_division=0)
    
    results = {
        "binary_accuracy": binary_accuracy,
        "binary_f1_macro": binary_f1,
        "binary_auc_roc": binary_auc,
        "multiclass_accuracy": multiclass_accuracy,
        "multiclass_f1_macro": multiclass_f1,
        "macro_f1": binary_f1,  # For compatibility
        "auc_roc": binary_auc
    }
    
    print(f"  Binary Accuracy: {binary_accuracy:.4f}")
    print(f"  Binary F1 (Macro): {binary_f1:.4f}")
    print(f"  Binary AUC-ROC: {binary_auc:.4f}")
    print(f"  Multiclass Accuracy: {multiclass_accuracy:.4f}")
    print(f"  Multiclass F1 (Macro): {multiclass_f1:.4f}")
    
    return results


def train_neural_baseline(model, model_name, train_loader, val_loader, test_loader, device, output_dir):
    """Train a neural baseline model."""
    print(f"\nTraining {model_name}...")
    
    # Create training config
    config = TrainingConfig(
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=16,
        output_dir=os.path.join(output_dir, model_name.lower().replace("-", "_")),
        early_stopping_patience=2,
        save_steps=1000,
        eval_steps=500
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        device=device
    )
    
    # Train
    training_history = trainer.train()
    
    # Evaluate on test set
    trainer.eval_dataloader = test_loader
    test_metrics = trainer.evaluate()
    
    print(f"  Binary Accuracy: {test_metrics['binary_accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # Save model
    final_model_dir = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    
    return test_metrics, training_history


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory with splits")
    parser.add_argument("--output-dir", type=str, default="./models/baselines", help="Output directory")
    parser.add_argument("--models", nargs="+", 
                       choices=["random", "confidence", "length", "gpt4o", "bert", "roberta", "all"],
                       default=["all"], help="Models to train/evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for neural models")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data for heuristic baselines
    test_traces_path = os.path.join(args.data_dir, "test.json")
    with open(test_traces_path, 'r') as f:
        test_data = json.load(f)
    test_traces = [CorrectionTrace.from_dict(item) for item in test_data]
    
    print(f"Loaded {len(test_traces)} test traces")
    
    # Determine which models to evaluate
    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["random", "confidence", "length", "gpt4o", "bert", "roberta"]
    
    results = {}
    
    # Evaluate heuristic baselines
    heuristic_models = {
        "random": ("Random Baseline", BaselineModels.get_random_baseline(args.seed)),
        "confidence": ("Confidence Heuristic", BaselineModels.get_confidence_heuristic()),
        "length": ("Length Heuristic", BaselineModels.get_length_heuristic()),
        "gpt4o": ("GPT-4o Judge (Simulated)", BaselineModels.get_gpt4o_judge(args.seed))
    }
    
    for model_key, (model_name, model) in heuristic_models.items():
        if model_key in models_to_run:
            results[model_key] = evaluate_heuristic_baseline(model, test_traces, model_name)
    
    # Train neural baselines
    neural_models = {}
    if "bert" in models_to_run:
        neural_models["bert"] = ("BERT-base", "bert-base-uncased", BertTokenizer, FineTunedBERTBaseline)
    if "roberta" in models_to_run:
        neural_models["roberta"] = ("RoBERTa-large", "roberta-large", RobertaTokenizer, FineTunedRoBERTaBaseline)
    
    for model_key, (model_name, model_path, tokenizer_class, model_class) in neural_models.items():
        print(f"\nPreparing {model_name}...")
        
        # Load tokenizer
        tokenizer = tokenizer_class.from_pretrained(model_path)
        
        # Load datasets
        train_dataset = SCFPDataset.from_json(
            os.path.join(args.data_dir, "train.json"),
            tokenizer=tokenizer,
            max_length=args.max_length,
            include_final_response=False
        )
        
        val_dataset = SCFPDataset.from_json(
            os.path.join(args.data_dir, "val.json"),
            tokenizer=tokenizer,
            max_length=args.max_length,
            include_final_response=False
        )
        
        test_dataset = SCFPDataset.from_json(
            os.path.join(args.data_dir, "test.json"),
            tokenizer=tokenizer,
            max_length=args.max_length,
            include_final_response=False
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            use_weighted_sampling=True
        )
        
        # Initialize model
        model = model_class(model_name=model_path)
        
        # Train and evaluate
        test_metrics, training_history = train_neural_baseline(
            model, model_name, train_loader, val_loader, test_loader, device, args.output_dir
        )
        
        results[model_key] = test_metrics
    
    # Save all results
    print("\n" + "="*60)
    print("BASELINE COMPARISON RESULTS")
    print("="*60)
    
    # Create comparison table
    comparison_metrics = ["binary_accuracy", "macro_f1", "auc_roc"]
    
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'AUC-ROC':<10}")
    print("-" * 60)
    
    for model_key, model_results in results.items():
        model_names = {
            "random": "Random",
            "confidence": "Confidence",
            "length": "Length",
            "gpt4o": "GPT-4o Judge",
            "bert": "BERT-base",
            "roberta": "RoBERTa-large"
        }
        
        name = model_names.get(model_key, model_key)
        accuracy = model_results.get("binary_accuracy", 0)
        f1 = model_results.get("macro_f1", 0)
        auc = model_results.get("auc_roc", 0)
        
        print(f"{name:<20} {accuracy:<10.4f} {f1:<10.4f} {auc:<10.4f}")
    
    # Save detailed results
    detailed_results = {
        "experiment_config": {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "models_evaluated": models_to_run,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device)
        },
        "results": results,
        "test_set_size": len(test_traces)
    }
    
    results_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print("Baseline evaluation complete!")


if __name__ == "__main__":
    main()
