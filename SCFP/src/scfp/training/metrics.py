"""
Evaluation metrics for SCFP framework.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
import torch


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Calculate data for reliability diagram.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins
    
    Returns:
        Dictionary with bin data for plotting
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            count_in_bin = in_bin.sum()
        else:
            accuracy_in_bin = 0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
            count_in_bin = 0
        
        bin_accuracies.append(accuracy_in_bin)
        bin_confidences.append(avg_confidence_in_bin)
        bin_counts.append(count_in_bin)
    
    return {
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),
        "bin_boundaries": bin_boundaries
    }


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for SCFP models.
    """
    
    def __init__(self):
        self.failure_mode_names = [
            "Success", "JH", "CM", "BA", "OC", "RM"
        ]
    
    def compute_binary_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute binary classification metrics.
        
        Args:
            y_true: True binary labels (0=failure, 1=success)
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities [N, 2]
        
        Returns:
            Dictionary of binary metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # AUC-ROC
        if y_prob.shape[1] == 2:
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_roc = roc_auc_score(y_true, y_prob)
        
        # Calibration metrics
        if y_prob.shape[1] == 2:
            ece = expected_calibration_error(y_true, y_prob[:, 1])
        else:
            ece = expected_calibration_error(y_true, y_prob)
        
        return {
            "binary_accuracy": accuracy,
            "binary_precision": precision,
            "binary_recall": recall,
            "binary_f1": f1,
            "binary_precision_macro": precision_macro,
            "binary_recall_macro": recall_macro,
            "binary_f1_macro": f1_macro,
            "binary_precision_weighted": precision_weighted,
            "binary_recall_weighted": recall_weighted,
            "binary_f1_weighted": f1_weighted,
            "auc_roc": auc_roc,
            "ece": ece
        }
    
    def compute_multiclass_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute multiclass classification metrics.
        
        Args:
            y_true: True multiclass labels
            y_pred: Predicted multiclass labels
            y_prob: Predicted probabilities [N, num_classes]
        
        Returns:
            Dictionary of multiclass metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Multiclass AUC-ROC (one-vs-rest)
        try:
            auc_roc_multiclass = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            # Handle case where some classes are missing
            auc_roc_multiclass = 0.0
        
        metrics = {
            "multiclass_accuracy": accuracy,
            "multiclass_precision_macro": precision_macro,
            "multiclass_recall_macro": recall_macro,
            "multiclass_f1_macro": f1_macro,
            "multiclass_precision_weighted": precision_weighted,
            "multiclass_recall_weighted": recall_weighted,
            "multiclass_f1_weighted": f1_weighted,
            "multiclass_auc_roc": auc_roc_multiclass
        }
        
        # Add per-class metrics
        for i, mode_name in enumerate(self.failure_mode_names):
            if i < len(precision_per_class):
                metrics[f"{mode_name.lower()}_precision"] = precision_per_class[i]
                metrics[f"{mode_name.lower()}_recall"] = recall_per_class[i]
                metrics[f"{mode_name.lower()}_f1"] = f1_per_class[i]
                metrics[f"{mode_name.lower()}_support"] = support_per_class[i]
        
        return metrics
    
    def compute_confusion_matrices(
        self, 
        binary_true: np.ndarray, 
        binary_pred: np.ndarray,
        multiclass_true: np.ndarray, 
        multiclass_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrices.
        
        Returns:
            Dictionary with binary and multiclass confusion matrices
        """
        binary_cm = confusion_matrix(binary_true, binary_pred)
        multiclass_cm = confusion_matrix(multiclass_true, multiclass_pred)
        
        return {
            "binary_confusion_matrix": binary_cm,
            "multiclass_confusion_matrix": multiclass_cm
        }
    
    def compute_all_metrics(
        self,
        binary_preds: np.ndarray,
        binary_labels: np.ndarray,
        binary_probs: np.ndarray,
        multiclass_preds: np.ndarray,
        multiclass_labels: np.ndarray,
        multiclass_probs: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            binary_preds: Binary predictions
            binary_labels: Binary true labels
            binary_probs: Binary prediction probabilities
            multiclass_preds: Multiclass predictions
            multiclass_labels: Multiclass true labels
            multiclass_probs: Multiclass prediction probabilities
        
        Returns:
            Dictionary with all metrics
        """
        # Binary metrics
        binary_metrics = self.compute_binary_metrics(
            binary_labels, binary_preds, binary_probs
        )
        
        # Multiclass metrics
        multiclass_metrics = self.compute_multiclass_metrics(
            multiclass_labels, multiclass_preds, multiclass_probs
        )
        
        # Confusion matrices
        confusion_matrices = self.compute_confusion_matrices(
            binary_labels, binary_preds, multiclass_labels, multiclass_preds
        )
        
        # Combine all metrics
        all_metrics = {
            **binary_metrics,
            **multiclass_metrics
        }
        
        # Add main metrics with standard names for compatibility
        all_metrics.update({
            "accuracy": binary_metrics["binary_accuracy"],
            "macro_f1": binary_metrics["binary_f1_macro"],
            "weighted_f1": binary_metrics["binary_f1_weighted"],
        })
        
        return all_metrics
    
    def generate_classification_report(
        self,
        binary_true: np.ndarray,
        binary_pred: np.ndarray,
        multiclass_true: np.ndarray,
        multiclass_pred: np.ndarray
    ) -> Dict[str, str]:
        """
        Generate detailed classification reports.
        
        Returns:
            Dictionary with binary and multiclass classification reports
        """
        binary_report = classification_report(
            binary_true, binary_pred,
            target_names=["Failure", "Success"]
        )
        
        multiclass_report = classification_report(
            multiclass_true, multiclass_pred,
            target_names=self.failure_mode_names,
            zero_division=0
        )
        
        return {
            "binary_report": binary_report,
            "multiclass_report": multiclass_report
        }
    
    def compute_calibration_data(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        n_bins: int = 10
    ) -> Dict:
        """
        Compute calibration data for reliability diagrams.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            n_bins: Number of bins
        
        Returns:
            Calibration data dictionary
        """
        return reliability_diagram_data(y_true, y_prob, n_bins)
    
    def compute_failure_mode_analysis(
        self,
        true_modes: np.ndarray,
        pred_modes: np.ndarray,
        mode_probs: np.ndarray
    ) -> Dict[str, any]:
        """
        Analyze failure mode prediction performance.
        
        Args:
            true_modes: True failure mode labels
            pred_modes: Predicted failure mode labels
            mode_probs: Prediction probabilities for each mode
        
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Overall accuracy by failure mode
        for i, mode_name in enumerate(self.failure_mode_names):
            mask = true_modes == i
            if mask.sum() > 0:
                mode_accuracy = (pred_modes[mask] == i).mean()
                analysis[f"{mode_name.lower()}_detection_accuracy"] = mode_accuracy
        
        # Confusion between failure modes
        cm = confusion_matrix(true_modes, pred_modes)
        analysis["mode_confusion_matrix"] = cm
        
        # Most confused pairs
        confused_pairs = []
        for i in range(len(self.failure_mode_names)):
            for j in range(len(self.failure_mode_names)):
                if i != j and cm[i, j] > 0:
                    confusion_rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                    confused_pairs.append({
                        "true_mode": self.failure_mode_names[i],
                        "pred_mode": self.failure_mode_names[j],
                        "confusion_rate": confusion_rate,
                        "count": cm[i, j]
                    })
        
        # Sort by confusion rate
        confused_pairs.sort(key=lambda x: x["confusion_rate"], reverse=True)
        analysis["most_confused_pairs"] = confused_pairs[:10]  # Top 10
        
        # Confidence analysis by mode
        for i, mode_name in enumerate(self.failure_mode_names):
            mask = true_modes == i
            if mask.sum() > 0:
                mode_confidences = mode_probs[mask, i]
                analysis[f"{mode_name.lower()}_avg_confidence"] = mode_confidences.mean()
                analysis[f"{mode_name.lower()}_confidence_std"] = mode_confidences.std()
        
        return analysis
