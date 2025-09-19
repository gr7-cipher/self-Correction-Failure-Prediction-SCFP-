"""
Core dataset classes for SCFP framework.
"""

import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import numpy as np


class FailureMode(Enum):
    """Enumeration of self-correction failure modes."""
    SUCCESS = "success"
    JUSTIFICATION_HALLUCINATION = "jh"  # JH
    CONFIDENCE_MISCALIBRATION = "cm"    # CM
    BIAS_AMPLIFICATION = "ba"           # BA
    OVER_CORRECTION = "oc"              # OC
    REASONING_MYOPIA = "rm"             # RM


@dataclass
class CorrectionTrace:
    """
    Represents a complete self-correction trace.
    
    Attributes:
        prompt: Original input prompt
        initial_response: Model's initial response
        critique: Self-generated critique/feedback
        final_response: Revised response after correction
        failure_mode: Type of failure (if any)
        is_success: Whether correction was successful
        metadata: Additional metadata (confidence scores, etc.)
    """
    prompt: str
    initial_response: str
    critique: str
    final_response: str
    failure_mode: FailureMode
    is_success: bool
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert trace to dictionary format."""
        return {
            "prompt": self.prompt,
            "initial_response": self.initial_response,
            "critique": self.critique,
            "final_response": self.final_response,
            "failure_mode": self.failure_mode.value,
            "is_success": self.is_success,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CorrectionTrace":
        """Create trace from dictionary."""
        return cls(
            prompt=data["prompt"],
            initial_response=data["initial_response"],
            critique=data["critique"],
            final_response=data["final_response"],
            failure_mode=FailureMode(data["failure_mode"]),
            is_success=data["is_success"],
            metadata=data.get("metadata", {})
        )


class SCFPDataset(Dataset):
    """
    PyTorch Dataset for Self-Correction Failure Prediction.
    
    This dataset handles the loading and preprocessing of correction traces
    for training failure prediction models.
    """
    
    def __init__(
        self,
        traces: List[CorrectionTrace],
        tokenizer,
        max_length: int = 1024,
        include_final_response: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            traces: List of correction traces
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            include_final_response: Whether to include final response in input
        """
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_final_response = include_final_response
        
        # Create label mappings
        self.failure_modes = list(FailureMode)
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(self.failure_modes)}
        self.idx_to_mode = {idx: mode for mode, idx in self.mode_to_idx.items()}
        
    def __len__(self) -> int:
        return len(self.traces)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        trace = self.traces[idx]
        
        # Construct input text
        input_parts = [
            f"Prompt: {trace.prompt}",
            f"Initial Response: {trace.initial_response}",
            f"Critique: {trace.critique}"
        ]
        
        if self.include_final_response:
            input_parts.append(f"Final Response: {trace.final_response}")
        
        input_text = " [SEP] ".join(input_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels
        binary_label = 1 if trace.is_success else 0
        multiclass_label = self.mode_to_idx[trace.failure_mode]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "multiclass_label": torch.tensor(multiclass_label, dtype=torch.long),
            "trace_id": torch.tensor(idx, dtype=torch.long)
        }
    
    @classmethod
    def from_json(
        cls,
        json_path: str,
        tokenizer,
        max_length: int = 1024,
        include_final_response: bool = False
    ) -> "SCFPDataset":
        """Load dataset from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        traces = [CorrectionTrace.from_dict(item) for item in data]
        
        return cls(
            traces=traces,
            tokenizer=tokenizer,
            max_length=max_length,
            include_final_response=include_final_response
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        # Binary class weights
        success_count = sum(1 for trace in self.traces if trace.is_success)
        failure_count = len(self.traces) - success_count
        
        binary_weights = torch.tensor([
            len(self.traces) / (2 * failure_count),  # failure weight
            len(self.traces) / (2 * success_count)   # success weight
        ])
        
        return binary_weights
    
    def get_multiclass_weights(self) -> torch.Tensor:
        """Calculate multiclass weights for failure modes."""
        mode_counts = {}
        for trace in self.traces:
            mode = trace.failure_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        weights = []
        for mode in self.failure_modes:
            count = mode_counts.get(mode, 1)  # Avoid division by zero
            weight = len(self.traces) / (len(self.failure_modes) * count)
            weights.append(weight)
        
        return torch.tensor(weights)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_traces": len(self.traces),
            "success_rate": sum(1 for t in self.traces if t.is_success) / len(self.traces),
            "failure_mode_distribution": {},
            "avg_prompt_length": np.mean([len(t.prompt.split()) for t in self.traces]),
            "avg_initial_response_length": np.mean([len(t.initial_response.split()) for t in self.traces]),
            "avg_critique_length": np.mean([len(t.critique.split()) for t in self.traces]),
        }
        
        for mode in self.failure_modes:
            count = sum(1 for t in self.traces if t.failure_mode == mode)
            stats["failure_mode_distribution"][mode.value] = count
        
        return stats


class PartialTrace:
    """
    Represents a partial correction trace (before final revision).
    Used for prediction during inference.
    """
    
    def __init__(self, prompt: str, initial_response: str, critique: str):
        self.prompt = prompt
        self.initial_response = initial_response
        self.critique = critique
    
    def to_input_text(self) -> str:
        """Convert to input text format for model."""
        return " [SEP] ".join([
            f"Prompt: {self.prompt}",
            f"Initial Response: {self.initial_response}",
            f"Critique: {self.critique}"
        ])
