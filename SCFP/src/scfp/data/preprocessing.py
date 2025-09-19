"""
Data preprocessing utilities for SCFP framework.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import numpy as np

from .dataset import CorrectionTrace, FailureMode


class DataPreprocessor:
    """
    Handles preprocessing of correction traces for training and evaluation.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def split_dataset(
        self,
        traces: List[CorrectionTrace],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "failure_mode"
    ) -> Tuple[List[CorrectionTrace], List[CorrectionTrace], List[CorrectionTrace]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            traces: List of correction traces
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by: Strategy for stratification ('failure_mode', 'success', or None)
        
        Returns:
            Tuple of (train_traces, val_traces, test_traces)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Prepare stratification labels
        if stratify_by == "failure_mode":
            stratify_labels = [trace.failure_mode.value for trace in traces]
        elif stratify_by == "success":
            stratify_labels = [trace.is_success for trace in traces]
        else:
            stratify_labels = None
        
        # First split: train vs (val + test)
        train_traces, temp_traces = train_test_split(
            traces,
            test_size=(val_ratio + test_ratio),
            random_state=self.seed,
            stratify=stratify_labels
        )
        
        # Second split: val vs test
        if stratify_labels is not None:
            # Get stratify labels for temp_traces
            temp_indices = [i for i, trace in enumerate(traces) if trace in temp_traces]
            temp_stratify = [stratify_labels[i] for i in temp_indices]
        else:
            temp_stratify = None
        
        val_traces, test_traces = train_test_split(
            temp_traces,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=self.seed,
            stratify=temp_stratify
        )
        
        return train_traces, val_traces, test_traces
    
    def balance_dataset(
        self,
        traces: List[CorrectionTrace],
        balance_type: str = "undersample"
    ) -> List[CorrectionTrace]:
        """
        Balance the dataset to handle class imbalance.
        
        Args:
            traces: List of correction traces
            balance_type: 'undersample', 'oversample', or 'none'
        
        Returns:
            Balanced list of traces
        """
        if balance_type == "none":
            return traces
        
        # Group traces by failure mode
        mode_groups = {}
        for trace in traces:
            mode = trace.failure_mode
            if mode not in mode_groups:
                mode_groups[mode] = []
            mode_groups[mode].append(trace)
        
        if balance_type == "undersample":
            # Find minimum class size
            min_size = min(len(group) for group in mode_groups.values())
            
            balanced_traces = []
            for mode, group in mode_groups.items():
                # Randomly sample min_size traces from each group
                sampled = np.random.choice(group, size=min_size, replace=False)
                balanced_traces.extend(sampled)
        
        elif balance_type == "oversample":
            # Find maximum class size
            max_size = max(len(group) for group in mode_groups.values())
            
            balanced_traces = []
            for mode, group in mode_groups.items():
                # Oversample to max_size
                if len(group) < max_size:
                    # Sample with replacement
                    sampled = np.random.choice(group, size=max_size, replace=True)
                else:
                    sampled = group
                balanced_traces.extend(sampled)
        
        # Shuffle the balanced dataset
        np.random.shuffle(balanced_traces)
        return balanced_traces
    
    def filter_traces(
        self,
        traces: List[CorrectionTrace],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        domains: Optional[List[str]] = None,
        failure_modes: Optional[List[FailureMode]] = None
    ) -> List[CorrectionTrace]:
        """
        Filter traces based on various criteria.
        
        Args:
            traces: List of correction traces
            min_length: Minimum total text length
            max_length: Maximum total text length
            domains: List of allowed domains
            failure_modes: List of allowed failure modes
        
        Returns:
            Filtered list of traces
        """
        filtered_traces = []
        
        for trace in traces:
            # Check length constraints
            if min_length is not None or max_length is not None:
                total_length = len(trace.prompt) + len(trace.initial_response) + len(trace.critique)
                
                if min_length is not None and total_length < min_length:
                    continue
                if max_length is not None and total_length > max_length:
                    continue
            
            # Check domain constraints
            if domains is not None:
                trace_domain = trace.metadata.get("domain") if trace.metadata else None
                if trace_domain not in domains:
                    continue
            
            # Check failure mode constraints
            if failure_modes is not None and trace.failure_mode not in failure_modes:
                continue
            
            filtered_traces.append(trace)
        
        return filtered_traces
    
    def augment_traces(
        self,
        traces: List[CorrectionTrace],
        augmentation_factor: float = 1.5
    ) -> List[CorrectionTrace]:
        """
        Augment traces through various techniques.
        
        Args:
            traces: List of correction traces
            augmentation_factor: Factor by which to increase dataset size
        
        Returns:
            Augmented list of traces
        """
        augmented_traces = traces.copy()
        target_size = int(len(traces) * augmentation_factor)
        
        while len(augmented_traces) < target_size:
            # Select a random trace to augment
            original_trace = np.random.choice(traces)
            
            # Apply augmentation (simplified version)
            augmented_trace = self._augment_single_trace(original_trace)
            augmented_traces.append(augmented_trace)
        
        return augmented_traces
    
    def _augment_single_trace(self, trace: CorrectionTrace) -> CorrectionTrace:
        """
        Augment a single trace through text modifications.
        
        This is a simplified version - in practice, you might use
        more sophisticated augmentation techniques.
        """
        # Simple paraphrasing by adding variation phrases
        variation_phrases = [
            "In other words, ",
            "To put it differently, ",
            "That is to say, ",
            "More specifically, "
        ]
        
        # Randomly add variation to one of the text fields
        field_to_augment = np.random.choice(["prompt", "initial_response", "critique"])
        variation = np.random.choice(variation_phrases)
        
        if field_to_augment == "prompt":
            new_prompt = variation + trace.prompt
            new_initial = trace.initial_response
            new_critique = trace.critique
        elif field_to_augment == "initial_response":
            new_prompt = trace.prompt
            new_initial = variation + trace.initial_response
            new_critique = trace.critique
        else:
            new_prompt = trace.prompt
            new_initial = trace.initial_response
            new_critique = variation + trace.critique
        
        return CorrectionTrace(
            prompt=new_prompt,
            initial_response=new_initial,
            critique=new_critique,
            final_response=trace.final_response,
            failure_mode=trace.failure_mode,
            is_success=trace.is_success,
            metadata={**(trace.metadata or {}), "augmented": True}
        )
    
    def create_cross_validation_splits(
        self,
        traces: List[CorrectionTrace],
        n_folds: int = 5
    ) -> List[Tuple[List[CorrectionTrace], List[CorrectionTrace]]]:
        """
        Create cross-validation splits.
        
        Args:
            traces: List of correction traces
            n_folds: Number of folds
        
        Returns:
            List of (train, val) splits
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Use failure modes for stratification
        stratify_labels = [trace.failure_mode.value for trace in traces]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        splits = []
        for train_idx, val_idx in skf.split(traces, stratify_labels):
            train_traces = [traces[i] for i in train_idx]
            val_traces = [traces[i] for i in val_idx]
            splits.append((train_traces, val_traces))
        
        return splits
    
    def save_splits(
        self,
        train_traces: List[CorrectionTrace],
        val_traces: List[CorrectionTrace],
        test_traces: List[CorrectionTrace],
        output_dir: str
    ):
        """Save train/val/test splits to separate files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split
        splits = {
            "train": train_traces,
            "val": val_traces,
            "test": test_traces
        }
        
        for split_name, traces in splits.items():
            output_path = os.path.join(output_dir, f"{split_name}.json")
            data = [trace.to_dict() for trace in traces]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save split statistics
        stats = {
            "train_size": len(train_traces),
            "val_size": len(val_traces),
            "test_size": len(test_traces),
            "total_size": len(train_traces) + len(val_traces) + len(test_traces)
        }
        
        # Add failure mode distribution for each split
        for split_name, traces in splits.items():
            mode_counts = {}
            for trace in traces:
                mode = trace.failure_mode.value
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            stats[f"{split_name}_distribution"] = mode_counts
        
        stats_path = os.path.join(output_dir, "split_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def load_splits(self, input_dir: str) -> Tuple[List[CorrectionTrace], List[CorrectionTrace], List[CorrectionTrace]]:
        """Load train/val/test splits from files."""
        splits = {}
        
        for split_name in ["train", "val", "test"]:
            file_path = os.path.join(input_dir, f"{split_name}.json")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            traces = [CorrectionTrace.from_dict(item) for item in data]
            splits[split_name] = traces
        
        return splits["train"], splits["val"], splits["test"]
