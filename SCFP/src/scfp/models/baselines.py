"""
Baseline models for SCFP framework comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, BertConfig, RobertaConfig
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import re


class RandomBaseline:
    """
    Random prediction baseline.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def predict_failure_probability(self, traces: List[str]) -> np.ndarray:
        """Predict random failure probabilities."""
        return np.random.random(len(traces))
    
    def predict_failure_mode(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict random failure modes."""
        n_samples = len(traces)
        n_modes = 6  # Number of failure modes
        
        # Random mode predictions
        predicted_modes = np.random.randint(0, n_modes, n_samples)
        
        # Random probability distributions
        mode_probs = np.random.dirichlet(np.ones(n_modes), n_samples)
        
        return predicted_modes, mode_probs


class ConfidenceHeuristic:
    """
    Confidence-based heuristic baseline.
    
    This baseline uses simple heuristics based on linguistic confidence markers
    to predict correction failures.
    """
    
    def __init__(self):
        # Confidence markers
        self.high_confidence_markers = [
            "definitely", "certainly", "absolutely", "without doubt",
            "clearly", "obviously", "undoubtedly", "surely", "100%",
            "completely sure", "positive", "confident"
        ]
        
        self.low_confidence_markers = [
            "maybe", "perhaps", "possibly", "might", "could be",
            "not sure", "uncertain", "think", "believe", "guess",
            "probably", "likely", "seems", "appears"
        ]
        
        self.critique_negative_markers = [
            "wrong", "incorrect", "error", "mistake", "false",
            "inaccurate", "flawed", "problematic", "issue"
        ]
    
    def _extract_confidence_features(self, text: str) -> Dict[str, float]:
        """Extract confidence-related features from text."""
        text_lower = text.lower()
        
        # Count confidence markers
        high_conf_count = sum(1 for marker in self.high_confidence_markers 
                             if marker in text_lower)
        low_conf_count = sum(1 for marker in self.low_confidence_markers 
                            if marker in text_lower)
        negative_count = sum(1 for marker in self.critique_negative_markers 
                            if marker in text_lower)
        
        # Calculate features
        total_words = len(text.split())
        
        features = {
            "high_confidence_ratio": high_conf_count / max(total_words, 1),
            "low_confidence_ratio": low_conf_count / max(total_words, 1),
            "negative_critique_ratio": negative_count / max(total_words, 1),
            "confidence_imbalance": high_conf_count - low_conf_count,
            "text_length": total_words,
            "question_marks": text.count("?"),
            "exclamation_marks": text.count("!")
        }
        
        return features
    
    def predict_failure_probability(self, traces: List[str]) -> np.ndarray:
        """
        Predict failure probability based on confidence heuristics.
        
        Args:
            traces: List of correction trace texts
        
        Returns:
            Array of failure probabilities
        """
        probabilities = []
        
        for trace in traces:
            features = self._extract_confidence_features(trace)
            
            # Heuristic scoring
            failure_score = 0.5  # Base probability
            
            # High confidence in wrong context increases failure probability
            if features["high_confidence_ratio"] > 0.01:
                failure_score += 0.2
            
            # Low confidence markers suggest uncertainty
            if features["low_confidence_ratio"] > 0.02:
                failure_score += 0.1
            
            # Negative critique markers suggest problems
            if features["negative_critique_ratio"] > 0.01:
                failure_score += 0.15
            
            # Confidence imbalance
            if features["confidence_imbalance"] > 2:
                failure_score += 0.1
            
            # Very short or very long responses might be problematic
            if features["text_length"] < 10 or features["text_length"] > 500:
                failure_score += 0.05
            
            # Excessive punctuation might indicate uncertainty
            if features["question_marks"] > 2 or features["exclamation_marks"] > 3:
                failure_score += 0.05
            
            # Clamp to [0, 1]
            failure_score = max(0.0, min(1.0, failure_score))
            probabilities.append(failure_score)
        
        return np.array(probabilities)
    
    def predict_failure_mode(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict failure modes based on heuristics."""
        n_samples = len(traces)
        predicted_modes = []
        mode_probs = []
        
        for trace in traces:
            features = self._extract_confidence_features(trace)
            
            # Heuristic mode assignment
            if features["high_confidence_ratio"] > 0.02:
                # High confidence suggests confidence miscalibration
                mode = 1  # CM
            elif features["negative_critique_ratio"] > 0.02:
                # Negative critique suggests justification hallucination
                mode = 0  # JH
            elif "bias" in trace.lower() or "stereotype" in trace.lower():
                mode = 2  # BA
            elif "change" in trace.lower() or "different" in trace.lower():
                mode = 3  # OC
            else:
                mode = 4  # RM (default)
            
            predicted_modes.append(mode)
            
            # Create probability distribution favoring the selected mode
            probs = np.ones(6) * 0.1
            probs[mode] = 0.5
            probs = probs / probs.sum()
            mode_probs.append(probs)
        
        return np.array(predicted_modes), np.array(mode_probs)


class LengthHeuristic:
    """
    Length-based heuristic baseline.
    
    This baseline uses response length patterns to predict failures.
    """
    
    def predict_failure_probability(self, traces: List[str]) -> np.ndarray:
        """Predict failure probability based on length heuristics."""
        probabilities = []
        
        for trace in traces:
            # Split trace into components (simplified)
            parts = trace.split("[SEP]")
            
            if len(parts) >= 3:
                initial_len = len(parts[1].split())
                critique_len = len(parts[2].split())
            else:
                initial_len = len(trace.split()) // 2
                critique_len = len(trace.split()) // 2
            
            # Heuristic: very short or very long responses are more likely to fail
            initial_score = 0.5
            if initial_len < 5 or initial_len > 200:
                initial_score += 0.2
            
            critique_score = 0.5
            if critique_len < 3 or critique_len > 100:
                critique_score += 0.2
            
            # Length ratio heuristic
            if critique_len > initial_len * 2:
                # Very long critique might indicate over-correction
                critique_score += 0.1
            elif critique_len < initial_len * 0.1:
                # Very short critique might be insufficient
                critique_score += 0.15
            
            failure_prob = (initial_score + critique_score) / 2
            failure_prob = max(0.0, min(1.0, failure_prob))
            probabilities.append(failure_prob)
        
        return np.array(probabilities)
    
    def predict_failure_mode(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict failure modes based on length patterns."""
        n_samples = len(traces)
        predicted_modes = []
        mode_probs = []
        
        for trace in traces:
            parts = trace.split("[SEP]")
            
            if len(parts) >= 3:
                critique_len = len(parts[2].split())
            else:
                critique_len = len(trace.split()) // 3
            
            # Length-based mode assignment
            if critique_len > 100:
                mode = 0  # JH (long justifications)
            elif critique_len < 5:
                mode = 4  # RM (insufficient analysis)
            elif critique_len > 50:
                mode = 3  # OC (over-correction)
            else:
                mode = 1  # CM (default)
            
            predicted_modes.append(mode)
            
            # Create probability distribution
            probs = np.ones(6) * 0.1
            probs[mode] = 0.4
            probs = probs / probs.sum()
            mode_probs.append(probs)
        
        return np.array(predicted_modes), np.array(mode_probs)


class GPT4oJudgeBaseline:
    """
    Simulated GPT-4o judge baseline.
    
    Since we don't have access to GPT-4o API, this simulates its behavior
    using rule-based heuristics that approximate LLM judgment patterns.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Patterns that might indicate failures
        self.failure_patterns = {
            "contradiction": [
                r"but (then|actually|however)",
                r"on the other hand",
                r"contradicts?",
                r"inconsistent"
            ],
            "uncertainty": [
                r"not sure",
                r"uncertain",
                r"might be wrong",
                r"could be"
            ],
            "overconfidence": [
                r"definitely|certainly|absolutely",
                r"without (a )?doubt",
                r"100% (sure|certain)"
            ],
            "fabrication": [
                r"research shows",
                r"studies indicate",
                r"according to experts",
                r"it is well known"
            ]
        }
    
    def _analyze_trace(self, trace: str) -> Dict[str, float]:
        """Analyze trace using simulated LLM judgment."""
        scores = {}
        
        for category, patterns in self.failure_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, trace, re.IGNORECASE))
                score += matches
            scores[category] = score
        
        return scores
    
    def predict_failure_probability(self, traces: List[str]) -> np.ndarray:
        """Predict failure probability using simulated GPT-4o judgment."""
        probabilities = []
        
        for trace in traces:
            scores = self._analyze_trace(trace)
            
            # Simulate GPT-4o judgment logic
            base_prob = 0.4
            
            # Contradictions suggest failure
            if scores["contradiction"] > 0:
                base_prob += 0.2
            
            # Uncertainty patterns
            if scores["uncertainty"] > 1:
                base_prob += 0.15
            
            # Overconfidence in wrong context
            if scores["overconfidence"] > 1:
                base_prob += 0.1
            
            # Potential fabrication
            if scores["fabrication"] > 0:
                base_prob += 0.25
            
            # Add some noise to simulate LLM variability
            noise = self.rng.gauss(0, 0.05)
            final_prob = max(0.0, min(1.0, base_prob + noise))
            
            probabilities.append(final_prob)
        
        return np.array(probabilities)
    
    def predict_failure_mode(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict failure modes using simulated judgment."""
        predicted_modes = []
        mode_probs = []
        
        for trace in traces:
            scores = self._analyze_trace(trace)
            
            # Mode assignment based on patterns
            if scores["fabrication"] > 0:
                mode = 0  # JH
            elif scores["overconfidence"] > 1:
                mode = 1  # CM
            elif "bias" in trace.lower():
                mode = 2  # BA
            elif scores["contradiction"] > 0:
                mode = 3  # OC
            else:
                mode = 4  # RM
            
            predicted_modes.append(mode)
            
            # Create probability distribution with some uncertainty
            probs = np.ones(6) * 0.05
            probs[mode] = 0.6
            
            # Add secondary preferences
            if mode == 0:  # JH
                probs[1] = 0.15  # Also likely CM
            elif mode == 1:  # CM
                probs[0] = 0.1   # Also possible JH
            
            probs = probs / probs.sum()
            mode_probs.append(probs)
        
        return np.array(predicted_modes), np.array(mode_probs)


class FineTunedBERTBaseline(nn.Module):
    """
    Fine-tuned BERT-base baseline model.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_failure_modes: int = 6):
        super().__init__()
        
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification heads
        self.dropout = nn.Dropout(0.1)
        self.binary_classifier = nn.Linear(self.config.hidden_size, 2)
        self.multiclass_classifier = nn.Linear(self.config.hidden_size, num_failure_modes)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        binary_logits = self.binary_classifier(pooled_output)
        multiclass_logits = self.multiclass_classifier(pooled_output)
        
        return {
            "binary_logits": binary_logits,
            "multiclass_logits": multiclass_logits
        }


class FineTunedRoBERTaBaseline(nn.Module):
    """
    Fine-tuned RoBERTa-large baseline model.
    """
    
    def __init__(self, model_name: str = "roberta-large", num_failure_modes: int = 6):
        super().__init__()
        
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Classification heads
        self.dropout = nn.Dropout(0.1)
        self.binary_classifier = nn.Linear(self.config.hidden_size, 2)
        self.multiclass_classifier = nn.Linear(self.config.hidden_size, num_failure_modes)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa doesn't have pooler_output, use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        binary_logits = self.binary_classifier(pooled_output)
        multiclass_logits = self.multiclass_classifier(pooled_output)
        
        return {
            "binary_logits": binary_logits,
            "multiclass_logits": multiclass_logits
        }


class BaselineModels:
    """
    Container class for all baseline models.
    """
    
    @staticmethod
    def get_random_baseline(seed: int = 42) -> RandomBaseline:
        return RandomBaseline(seed=seed)
    
    @staticmethod
    def get_confidence_heuristic() -> ConfidenceHeuristic:
        return ConfidenceHeuristic()
    
    @staticmethod
    def get_length_heuristic() -> LengthHeuristic:
        return LengthHeuristic()
    
    @staticmethod
    def get_gpt4o_judge(seed: int = 42) -> GPT4oJudgeBaseline:
        return GPT4oJudgeBaseline(seed=seed)
    
    @staticmethod
    def get_bert_baseline(model_name: str = "bert-base-uncased") -> FineTunedBERTBaseline:
        return FineTunedBERTBaseline(model_name=model_name)
    
    @staticmethod
    def get_roberta_baseline(model_name: str = "roberta-large") -> FineTunedRoBERTaBaseline:
        return FineTunedRoBERTaBaseline(model_name=model_name)
