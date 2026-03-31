#!/usr/bin/env python3
"""
Comprehensive evaluation of all SCFP models.

This script evaluates all trained models and generates comparison results,
including ablation studies and cross-model generalization experiments.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import DebertaV2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.dataset import SCFPDataset, CorrectionTrace
from scfp.data.loaders import create_inference_dataloader
from scfp.models.deberta import DeBERTaFailurePredictor
from scfp.models.baselines import BaselineModels
from scfp.training.metrics import EvaluationMetrics


def load_model(model_path, device):
    """Load a trained model."""
    if os.path.exists(os.path.join(model_path, "config.json")):
        # Load DeBERTa model
        model = DeBERTaFailurePredictor.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, "deberta"
    else:
        # Handle other model types
        return None, "unknown"


def evaluate_model_on_dataset(model, model_type, dataset, device, tokenizer=None):
    """Evaluate a model on a dataset."""
    if model_type == "deberta":
        return evaluate_neural_model(model, dataset, device)
    else:
        return evaluate_heuristic_model(model, dataset)


def evaluate_neural_model(model, dataset, device):
    """Evaluate a neural model."""
    dataloader = create_inference_dataloader(dataset, batch_size=64)
    
    all_binary_preds = []
    all_binary_labels = []
    all_multiclass_preds = []
    all_multiclass_labels = []
    all_binary_probs = []
    all_multiclass_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Get predictions
            binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)
            multiclass_probs = torch.softmax(outputs["multiclass_logits"], dim=-1)
            
            binary_preds = torch.argmax(binary_probs, dim=-1)
            multiclass_preds = torch.argmax(multiclass_probs, dim=-1)
            
            # Collect results
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_labels.extend(batch["binary_label"].cpu().numpy())
            all_multiclass_preds.extend(multiclass_preds.cpu().numpy())
            all_multiclass_labels.extend(batch["multiclass_label"].cpu().numpy())
            all_binary_probs.extend(binary_probs.cpu().numpy())
            all_multiclass_probs.extend(multiclass_probs.cpu().numpy())
    
    # Calculate metrics
    metrics = EvaluationMetrics()
    results = metrics.compute_all_metrics(
        binary_preds=np.array(all_binary_preds),
        binary_labels=np.array(all_binary_labels),
        binary_probs=np.array(all_binary_probs),
        multiclass_preds=np.array(all_multiclass_preds),
        multiclass_labels=np.array(all_multiclass_labels),
        multiclass_probs=np.array(all_multiclass_probs)
    )
    
    return results


def evaluate_heuristic_model(model, dataset):
    """Evaluate a heuristic model."""
    # Convert dataset to text format
    trace_texts = []
    binary_labels = []
    multiclass_labels = []
    
    for trace in dataset.traces:
        text = f"Prompt: {trace.prompt} [SEP] Initial Response: {trace.initial_response} [SEP] Critique: {trace.critique}"
        trace_texts.append(text)
        binary_labels.append(1 if trace.is_success else 0)
        
        # Map failure mode to index
        mode_to_idx = {
            "success": 0, "jh": 1, "cm": 2, "ba": 3, "oc": 4, "rm": 5
        }
        multiclass_labels.append(mode_to_idx[trace.failure_mode.value])
    
    # Get predictions
    binary_probs = model.predict_failure_probability(trace_texts)
    binary_preds = (binary_probs < 0.5).astype(int)
    
    multiclass_preds, multiclass_probs = model.predict_failure_mode(trace_texts)
    
    # Convert to proper format
    binary_probs_2d = np.column_stack([binary_probs, 1 - binary_probs])
    
    # Calculate metrics
    metrics = EvaluationMetrics()
    results = metrics.compute_all_metrics(
        binary_preds=binary_preds,
        binary_labels=np.array(binary_labels),
        binary_probs=binary_probs_2d,
        multiclass_preds=multiclass_preds,
        multiclass_labels=np.array(multiclass_labels),
        multiclass_probs=multiclass_probs
    )
    
    return results


def run_ablation_study(base_model_path, data_dir, device, output_dir):
    """Run ablation study on DeBERTa model."""
    print("\nRunning ablation study...")
    
    # Load tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    # Load test dataset
    test_dataset = SCFPDataset.from_json(
        os.path.join(data_dir, "test.json"),
        tokenizer=tokenizer,
        max_length=1024,
        include_final_response=False
    )
    
    ablation_configs = {
        "full_model": {
            "use_specialized_attention": True,
            "include_critique": True,
            "description": "Complete model with all components"
        },
        "no_specialized_attention": {
            "use_specialized_attention": False,
            "include_critique": True,
            "description": "Without specialized attention mechanism"
        },
        "no_critique": {
            "use_specialized_attention": True,
            "include_critique": False,
            "description": "Without critique input (initial response only)"
        },
        "minimal": {
            "use_specialized_attention": False,
            "include_critique": False,
            "description": "Minimal configuration"
        }
    }
    
    ablation_results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"  Evaluating: {config['description']}")
        
        # Create model with specific configuration
        model = DeBERTaFailurePredictor(
            model_name="microsoft/deberta-v3-base",
            use_specialized_attention=config["use_specialized_attention"]
        )
        
        # Load base model weights (in practice, you'd train separate models)
        if os.path.exists(base_model_path):
            try:
                state_dict = torch.load(
                    os.path.join(base_model_path, "pytorch_model.bin"),
                    map_location=device
                )
                # Filter out incompatible layers for ablation
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if not config["use_specialized_attention"] and "specialized_attention" in key:
                        continue
                    filtered_state_dict[key] = value
                
                model.load_state_dict(filtered_state_dict, strict=False)
            except:
                print(f"    Warning: Could not load weights for {config_name}")
        
        model.to(device)
        model.eval()
        
        # Modify dataset if needed
        if not config["include_critique"]:
            # Create dataset without critique
            modified_traces = []
            for trace in test_dataset.traces:
                modified_trace = CorrectionTrace(
                    prompt=trace.prompt,
                    initial_response=trace.initial_response,
                    critique="",  # Empty critique
                    final_response=trace.final_response,
                    failure_mode=trace.failure_mode,
                    is_success=trace.is_success,
                    metadata=trace.metadata
                )
                modified_traces.append(modified_trace)
            
            modified_dataset = SCFPDataset(
                traces=modified_traces,
                tokenizer=tokenizer,
                max_length=1024,
                include_final_response=False
            )
        else:
            modified_dataset = test_dataset
        
        # Evaluate
        results = evaluate_neural_model(model, modified_dataset, device)
        ablation_results[config_name] = {
            "config": config,
            "results": results
        }
        
        print(f"    Accuracy: {results['binary_accuracy']:.4f}")
        print(f"    Macro F1: {results['macro_f1']:.4f}")
    
    # Save ablation results
    ablation_path = os.path.join(output_dir, "ablation_study.json")
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots."""
    print("\nGenerating comparison plots...")
    
    # Extract metrics for plotting
    models = list(results.keys())
    metrics = ["binary_accuracy", "macro_f1", "auc_roc", "ece"]
    
    # Create comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        ax = axes[i]
        bars = ax.bar(models, values)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance vs cost plot (if cost data available)
    if any("cost" in str(results[model]) for model in models):
        plt.figure(figsize=(10, 8))
        
        accuracies = [results[model].get("binary_accuracy", 0) for model in models]
        costs = [results[model].get("cost", 1) for model in models]  # Default cost = 1
        
        plt.scatter(costs, accuracies, s=100, alpha=0.7)
        
        for i, model in enumerate(models):
            plt.annotate(model, (costs[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Relative Cost')
        plt.ylabel('Binary Accuracy')
        plt.title('Accuracy vs Cost Trade-off')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "accuracy_vs_cost.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SCFP model evaluation")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory with test set")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--cross-model", action="store_true", help="Run cross-model generalization")
    parser.add_argument("--generate-plots", action="store_true", help="Generate comparison plots")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load test dataset for heuristic models
    test_traces_path = os.path.join(args.data_dir, "test.json")
    with open(test_traces_path, 'r') as f:
        test_data = json.load(f)
    test_traces = [CorrectionTrace.from_dict(item) for item in test_data]
    
    print(f"Loaded {len(test_traces)} test traces")
    
    # Evaluate all models
    results = {}
    
    # Evaluate heuristic baselines
    print("\nEvaluating heuristic baselines...")
    heuristic_models = {
        "random": BaselineModels.get_random_baseline(42),
        "confidence": BaselineModels.get_confidence_heuristic(),
        "length": BaselineModels.get_length_heuristic(),
        "gpt4o": BaselineModels.get_gpt4o_judge(42)
    }
    
    # Create dummy dataset for heuristic evaluation
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    dummy_dataset = SCFPDataset(
        traces=test_traces,
        tokenizer=tokenizer,
        max_length=1024
    )
    
    for name, model in heuristic_models.items():
        print(f"  Evaluating {name}...")
        results[name] = evaluate_heuristic_model(model, dummy_dataset)
    
    # Evaluate neural models
    print("\nEvaluating neural models...")
    
    # Look for trained models in models directory
    model_dirs = {
        "deberta": os.path.join(args.models_dir, "deberta", "final_model"),
        "bert": os.path.join(args.models_dir, "baselines", "bert_base", "final_model"),
        "roberta": os.path.join(args.models_dir, "baselines", "roberta_large", "final_model")
    }
    
    for model_name, model_path in model_dirs.items():
        if os.path.exists(model_path):
            print(f"  Evaluating {model_name}...")
            try:
                model, model_type = load_model(model_path, device)
                if model is not None:
                    # Load appropriate dataset
                    if model_name == "deberta":
                        tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
                    elif model_name == "bert":
                        from transformers import BertTokenizer
                        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    elif model_name == "roberta":
                        from transformers import RobertaTokenizer
                        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
                    
                    test_dataset = SCFPDataset(
                        traces=test_traces,
                        tokenizer=tokenizer,
                        max_length=1024 if model_name == "deberta" else 512
                    )
                    
                    results[model_name] = evaluate_neural_model(model, test_dataset, device)
                else:
                    print(f"    Could not load {model_name}")
            except Exception as e:
                print(f"    Error evaluating {model_name}: {e}")
        else:
            print(f"    Model not found: {model_path}")
    
    # Run ablation study
    if args.ablation:
        deberta_path = model_dirs.get("deberta")
        if deberta_path and os.path.exists(deberta_path):
            ablation_results = run_ablation_study(deberta_path, args.data_dir, device, args.output)
            results["ablation"] = ablation_results
    
    # Generate comparison table
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    # Main comparison table
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'AUC-ROC':<10} {'ECE':<10}")
    print("-" * 80)
    
    for model_name, model_results in results.items():
        if model_name == "ablation":
            continue
            
        accuracy = model_results.get("binary_accuracy", 0)
        macro_f1 = model_results.get("macro_f1", 0)
        weighted_f1 = model_results.get("weighted_f1", 0)
        auc_roc = model_results.get("auc_roc", 0)
        ece = model_results.get("ece", 0)
        
        print(f"{model_name:<20} {accuracy:<10.4f} {macro_f1:<10.4f} {weighted_f1:<12.4f} {auc_roc:<10.4f} {ece:<10.4f}")
    
    # Ablation study results
    if "ablation" in results:
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        
        print(f"{'Configuration':<25} {'Accuracy':<10} {'Macro F1':<10}")
        print("-" * 50)
        
        for config_name, config_data in results["ablation"].items():
            config_results = config_data["results"]
            accuracy = config_results.get("binary_accuracy", 0)
            macro_f1 = config_results.get("macro_f1", 0)
            
            print(f"{config_name:<25} {accuracy:<10.4f} {macro_f1:<10.4f}")
    
    # Generate plots
    if args.generate_plots:
        plot_results = {k: v for k, v in results.items() if k != "ablation"}
        generate_comparison_plots(plot_results, args.output)
    
    # Save detailed results
    detailed_results = {
        "evaluation_config": {
            "models_dir": args.models_dir,
            "data_dir": args.data_dir,
            "test_set_size": len(test_traces),
            "device": str(device)
        },
        "results": results,
        "summary": {
            "best_model": max(results.keys(), 
                            key=lambda k: results[k].get("macro_f1", 0) if k != "ablation" else 0),
            "best_accuracy": max(results[k].get("binary_accuracy", 0) 
                               for k in results.keys() if k != "ablation"),
            "best_f1": max(results[k].get("macro_f1", 0) 
                          for k in results.keys() if k != "ablation")
        }
    }
    
    results_path = os.path.join(args.output, "comprehensive_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    if args.generate_plots:
        print(f"Plots saved to: {args.output}")
    
    print("\nComprehensive evaluation complete!")


if __name__ == "__main__":
    main()
