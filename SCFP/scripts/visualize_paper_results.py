#!/usr/bin/env python3
"""
Generate paper-quality results for the SCFP framework.
Produces Table 3 (Main), Table 4 (Ablation), Figure 3 (ROC), and Figure 4 (Calibration).
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pathlib import Path

def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

def generate_roc_curves(results, output_dir):
    """Figure 3: ROC Curves for all benchmarks."""
    plt.figure(figsize=(10, 8))
    
    for model_name in results.keys():
        if "_raw" in model_name: continue
        if model_name == "ablation": continue
        
        raw_key = f"{model_name}_raw"
        if raw_key not in results: continue
        
        raw_data = results[raw_key]
        y_true = np.array(raw_data["binary_labels"])
        y_prob = np.array(raw_data["binary_probs"])[:, 1]  # Success prob
        
        # We want to plot failure prediction ROC, so flip labels/probs
        y_true_failure = 1 - y_true
        y_prob_failure = 1 - y_prob
        
        fpr, tpr, _ = roc_curve(y_true_failure, y_prob_failure)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 3: ROC Curves for Failure Prediction')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "figure3_roc_curves.png"), dpi=300)
    print(f"Saved Figure 3 to {output_dir}")

def generate_calibration_plots(results, output_dir):
    """Figure 4: Reliability Diagrams."""
    plt.figure(figsize=(10, 8))
    
    # Focus on DeBERTa (Ours) vs Baselines
    target_models = ["deberta", "roberta", "bert", "gpt4o"]
    
    for model_name in target_models:
        raw_key = f"{model_name}_raw"
        if raw_key not in results: continue
        
        raw_data = results[raw_key]
        y_true = 1 - np.array(raw_data["binary_labels"]) # Failure label
        y_prob = 1 - np.array(raw_data["binary_probs"])[:, 1] # Failure prob
        
        from scfp.training.metrics import reliability_diagram_data
        diag = reliability_diagram_data(y_true, y_prob, n_bins=10)
        
        plt.plot(diag["bin_confidences"], diag["bin_accuracies"], marker='o', label=model_name)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Figure 4: Reliability Diagram (Calibration)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "figure4_calibration.png"), dpi=300)
    print(f"Saved Figure 4 to {output_dir}")

def generate_main_table(results):
    """Table 3: Main Performance Comparison."""
    data = []
    for model_name in results.keys():
        if "_raw" in model_name: continue
        if model_name == "ablation": continue
        
        res = results[model_name]
        data.append({
            "Model": model_name,
            "Accuracy": f"{res['binary_accuracy']*100:.1f}%",
            "Macro F1": f"{res['macro_f1']:.3f}",
            "AUC-ROC": f"{res['auc_roc']:.3f}",
            "ECE": f"{res['ece']:.3f}"
        })
    
    df = pd.DataFrame(data)
    print("\nTable 3: Main Performance Comparison")
    print(df.to_markdown(index=False))
    return df

def generate_ablation_table(results):
    """Table 4: Ablation Study."""
    if "ablation" not in results:
        return
    
    data = []
    for config, config_data in results["ablation"].items():
        res = config_data["results"]
        data.append({
            "Configuration": config,
            "Accuracy": f"{res['binary_accuracy']*100:.1f}%",
            "Macro F1": f"{res['macro_f1']:.3f}"
        })
        
    df = pd.DataFrame(data)
    print("\nTable 4: Ablation Study")
    print(df.to_markdown(index=False))
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/paper")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plotting_style()
    
    with open(args.results_json, 'r') as f:
        full_data = json.load(f)
    
    results = full_data["results"]
    
    # Generate everything
    generate_main_table(results)
    generate_ablation_table(results)
    generate_roc_curves(results, args.output_dir)
    generate_calibration_plots(results, args.output_dir)

if __name__ == "__main__":
    main()
