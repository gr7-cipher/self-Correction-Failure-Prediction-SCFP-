"""
Dynamic routing system for intelligent correction strategy selection.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .cost_model import CostModel
from ..data.dataset import PartialTrace, FailureMode


class RoutingStrategy(Enum):
    """Available correction strategies."""
    INTRINSIC = "intrinsic"      # Self-correction only
    EXTERNAL = "external"        # External tools/APIs
    HUMAN = "human"              # Human oversight
    HYBRID = "hybrid"            # Combination approach


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    strategy: RoutingStrategy
    confidence: float
    failure_probability: float
    predicted_failure_mode: Optional[FailureMode]
    reasoning: str
    cost_estimate: float
    expected_accuracy: float


class DynamicRouter:
    """
    Dynamic router that intelligently selects correction strategies
    based on failure prediction and cost-benefit analysis.
    """
    
    def __init__(
        self,
        failure_predictor,
        cost_model: CostModel,
        thresholds: Dict[str, float] = None,
        strategy_preferences: Dict[str, float] = None
    ):
        """
        Initialize dynamic router.
        
        Args:
            failure_predictor: Trained failure prediction model
            cost_model: Cost model for different strategies
            thresholds: Thresholds for routing decisions
            strategy_preferences: Preference weights for strategies
        """
        self.failure_predictor = failure_predictor
        self.cost_model = cost_model
        
        # Default thresholds
        self.thresholds = thresholds or {
            "intrinsic_max_failure_prob": 0.3,    # Use intrinsic if failure prob < 0.3
            "external_max_failure_prob": 0.7,     # Use external if failure prob < 0.7
            "human_min_failure_prob": 0.7,        # Use human if failure prob >= 0.7
            "confidence_threshold": 0.8,          # Minimum confidence for decisions
            "cost_sensitivity": 1.0               # How much to weight cost vs accuracy
        }
        
        # Strategy preferences (higher = more preferred)
        self.strategy_preferences = strategy_preferences or {
            "intrinsic": 1.0,
            "external": 0.8,
            "human": 0.6,
            "hybrid": 0.9
        }
        
        # Performance estimates for each strategy
        self.strategy_performance = {
            RoutingStrategy.INTRINSIC: {"accuracy": 0.75, "variance": 0.15},
            RoutingStrategy.EXTERNAL: {"accuracy": 0.85, "variance": 0.10},
            RoutingStrategy.HUMAN: {"accuracy": 0.95, "variance": 0.05},
            RoutingStrategy.HYBRID: {"accuracy": 0.90, "variance": 0.08}
        }
    
    def route(
        self,
        prompt: str,
        initial_response: str,
        critique: str,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """
        Make routing decision for a correction trace.
        
        Args:
            prompt: Original prompt
            initial_response: Model's initial response
            critique: Self-generated critique
            context: Additional context (domain, urgency, etc.)
        
        Returns:
            Routing decision with strategy and reasoning
        """
        # Create partial trace
        partial_trace = PartialTrace(prompt, initial_response, critique)
        
        # Get failure prediction
        failure_prob, failure_mode, model_confidence = self._predict_failure(partial_trace)
        
        # Analyze context
        context_factors = self._analyze_context(context or {})
        
        # Evaluate strategies
        strategy_scores = self._evaluate_strategies(
            failure_prob, failure_mode, model_confidence, context_factors
        )
        
        # Select best strategy
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s]["score"])
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            failure_prob, failure_mode, model_confidence, 
            context_factors, strategy_scores, best_strategy
        )
        
        return RoutingDecision(
            strategy=best_strategy,
            confidence=model_confidence,
            failure_probability=failure_prob,
            predicted_failure_mode=failure_mode,
            reasoning=reasoning,
            cost_estimate=strategy_scores[best_strategy]["cost"],
            expected_accuracy=strategy_scores[best_strategy]["expected_accuracy"]
        )
    
    def _predict_failure(self, partial_trace: PartialTrace) -> Tuple[float, FailureMode, float]:
        """
        Predict failure probability and mode.
        
        Returns:
            Tuple of (failure_probability, failure_mode, model_confidence)
        """
        # Convert trace to input format
        input_text = partial_trace.to_input_text()
        
        # Tokenize (this would use the actual tokenizer)
        # For now, we'll simulate the prediction
        
        # Simulate failure prediction
        failure_prob = np.random.beta(2, 3)  # Biased toward lower failure rates
        failure_mode = FailureMode.JUSTIFICATION_HALLUCINATION  # Most common
        model_confidence = np.random.beta(5, 2)  # Biased toward higher confidence
        
        return failure_prob, failure_mode, model_confidence
    
    def _analyze_context(self, context: Dict) -> Dict[str, float]:
        """
        Analyze contextual factors that influence routing.
        
        Args:
            context: Context dictionary
        
        Returns:
            Dictionary of normalized context factors
        """
        factors = {
            "urgency": 0.5,      # How urgent is the response (0=low, 1=high)
            "domain_complexity": 0.5,  # Domain complexity (0=simple, 1=complex)
            "stakes": 0.5,       # Stakes of getting it wrong (0=low, 1=high)
            "cost_sensitivity": 0.5,   # How cost-sensitive is the user
            "accuracy_requirement": 0.5,  # Required accuracy level
        }
        
        # Update with provided context
        for key, value in context.items():
            if key in factors:
                factors[key] = max(0.0, min(1.0, float(value)))
        
        # Infer some factors from others
        if "domain" in context:
            domain = context["domain"].lower()
            if domain in ["medical", "legal", "financial"]:
                factors["stakes"] = 0.9
                factors["accuracy_requirement"] = 0.9
            elif domain in ["creative", "casual"]:
                factors["stakes"] = 0.2
                factors["accuracy_requirement"] = 0.6
        
        return factors
    
    def _evaluate_strategies(
        self,
        failure_prob: float,
        failure_mode: FailureMode,
        model_confidence: float,
        context_factors: Dict[str, float]
    ) -> Dict[RoutingStrategy, Dict[str, float]]:
        """
        Evaluate all available strategies.
        
        Returns:
            Dictionary mapping strategies to their evaluation scores
        """
        strategy_scores = {}
        
        for strategy in RoutingStrategy:
            # Get base cost and accuracy
            base_cost = self.cost_model.get_cost(strategy)
            base_accuracy = self.strategy_performance[strategy]["accuracy"]
            accuracy_variance = self.strategy_performance[strategy]["variance"]
            
            # Adjust accuracy based on failure probability
            if strategy == RoutingStrategy.INTRINSIC:
                # Intrinsic correction is less effective when failure prob is high
                adjusted_accuracy = base_accuracy * (1 - failure_prob * 0.5)
            elif strategy == RoutingStrategy.EXTERNAL:
                # External tools are more consistent
                adjusted_accuracy = base_accuracy * (1 - failure_prob * 0.2)
            elif strategy == RoutingStrategy.HUMAN:
                # Human oversight is most reliable
                adjusted_accuracy = base_accuracy * (1 - failure_prob * 0.1)
            else:  # HYBRID
                # Hybrid combines benefits
                adjusted_accuracy = base_accuracy * (1 - failure_prob * 0.15)
            
            # Adjust for context factors
            context_multiplier = self._calculate_context_multiplier(strategy, context_factors)
            final_accuracy = adjusted_accuracy * context_multiplier
            
            # Calculate utility score (accuracy - cost_penalty)
            cost_penalty = base_cost * context_factors["cost_sensitivity"] * self.thresholds["cost_sensitivity"]
            preference_bonus = self.strategy_preferences.get(strategy.value, 1.0) * 0.1
            
            utility_score = final_accuracy - cost_penalty + preference_bonus
            
            strategy_scores[strategy] = {
                "score": utility_score,
                "cost": base_cost,
                "expected_accuracy": final_accuracy,
                "base_accuracy": base_accuracy,
                "context_multiplier": context_multiplier
            }
        
        return strategy_scores
    
    def _calculate_context_multiplier(
        self, 
        strategy: RoutingStrategy, 
        context_factors: Dict[str, float]
    ) -> float:
        """
        Calculate context-based multiplier for strategy effectiveness.
        
        Args:
            strategy: The routing strategy
            context_factors: Context factors
        
        Returns:
            Multiplier for strategy effectiveness (typically 0.8-1.2)
        """
        multiplier = 1.0
        
        if strategy == RoutingStrategy.INTRINSIC:
            # Intrinsic works better for simple, low-stakes scenarios
            multiplier *= (1 + (1 - context_factors["domain_complexity"]) * 0.1)
            multiplier *= (1 + (1 - context_factors["stakes"]) * 0.1)
            
        elif strategy == RoutingStrategy.EXTERNAL:
            # External tools work better for complex, structured problems
            multiplier *= (1 + context_factors["domain_complexity"] * 0.15)
            
        elif strategy == RoutingStrategy.HUMAN:
            # Human oversight is most valuable for high-stakes scenarios
            multiplier *= (1 + context_factors["stakes"] * 0.2)
            multiplier *= (1 + context_factors["accuracy_requirement"] * 0.15)
            
        elif strategy == RoutingStrategy.HYBRID:
            # Hybrid is generally robust across contexts
            multiplier *= (1 + np.mean(list(context_factors.values())) * 0.1)
        
        # Urgency affects all strategies differently
        urgency = context_factors["urgency"]
        if strategy in [RoutingStrategy.HUMAN, RoutingStrategy.HYBRID]:
            # Human/hybrid strategies are slower
            multiplier *= (1 - urgency * 0.1)
        else:
            # Automated strategies benefit from urgency
            multiplier *= (1 + urgency * 0.05)
        
        return max(0.8, min(1.2, multiplier))
    
    def _generate_reasoning(
        self,
        failure_prob: float,
        failure_mode: FailureMode,
        model_confidence: float,
        context_factors: Dict[str, float],
        strategy_scores: Dict[RoutingStrategy, Dict[str, float]],
        selected_strategy: RoutingStrategy
    ) -> str:
        """
        Generate human-readable reasoning for the routing decision.
        
        Returns:
            Reasoning string explaining the decision
        """
        reasoning_parts = []
        
        # Failure probability analysis
        if failure_prob < 0.3:
            reasoning_parts.append(f"Low failure probability ({failure_prob:.2f}) suggests intrinsic correction may succeed.")
        elif failure_prob < 0.7:
            reasoning_parts.append(f"Moderate failure probability ({failure_prob:.2f}) indicates potential issues with self-correction.")
        else:
            reasoning_parts.append(f"High failure probability ({failure_prob:.2f}) suggests intrinsic correction likely to fail.")
        
        # Failure mode analysis
        if failure_mode == FailureMode.JUSTIFICATION_HALLUCINATION:
            reasoning_parts.append("Predicted failure mode is Justification Hallucination, which benefits from external verification.")
        elif failure_mode == FailureMode.CONFIDENCE_MISCALIBRATION:
            reasoning_parts.append("Predicted failure mode is Confidence Miscalibration, requiring careful oversight.")
        elif failure_mode == FailureMode.OVER_CORRECTION:
            reasoning_parts.append("Predicted failure mode is Over-correction, suggesting need for conservative approach.")
        
        # Context analysis
        high_stakes = context_factors["stakes"] > 0.7
        high_complexity = context_factors["domain_complexity"] > 0.7
        high_urgency = context_factors["urgency"] > 0.7
        
        if high_stakes:
            reasoning_parts.append("High-stakes scenario requires maximum accuracy.")
        if high_complexity:
            reasoning_parts.append("Complex domain benefits from specialized tools or expertise.")
        if high_urgency:
            reasoning_parts.append("Urgent request favors faster automated approaches.")
        
        # Strategy justification
        selected_score = strategy_scores[selected_strategy]
        reasoning_parts.append(
            f"Selected {selected_strategy.value} strategy with utility score {selected_score['score']:.3f} "
            f"(expected accuracy: {selected_score['expected_accuracy']:.2f}, cost: {selected_score['cost']:.2f})."
        )
        
        # Alternative comparison
        alternatives = sorted(
            [(s, scores["score"]) for s, scores in strategy_scores.items() if s != selected_strategy],
            key=lambda x: x[1], reverse=True
        )
        
        if alternatives:
            best_alt, best_alt_score = alternatives[0]
            score_diff = selected_score["score"] - best_alt_score
            if score_diff < 0.1:
                reasoning_parts.append(f"Close decision vs {best_alt.value} (score difference: {score_diff:.3f}).")
        
        return " ".join(reasoning_parts)
    
    def batch_route(
        self,
        traces: List[Tuple[str, str, str]],
        contexts: Optional[List[Dict]] = None
    ) -> List[RoutingDecision]:
        """
        Route multiple traces in batch.
        
        Args:
            traces: List of (prompt, initial_response, critique) tuples
            contexts: Optional list of context dictionaries
        
        Returns:
            List of routing decisions
        """
        if contexts is None:
            contexts = [{}] * len(traces)
        
        decisions = []
        for i, (prompt, initial_response, critique) in enumerate(traces):
            context = contexts[i] if i < len(contexts) else {}
            decision = self.route(prompt, initial_response, critique, context)
            decisions.append(decision)
        
        return decisions
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update routing thresholds."""
        self.thresholds.update(new_thresholds)
    
    def update_strategy_preferences(self, new_preferences: Dict[str, float]):
        """Update strategy preferences."""
        self.strategy_preferences.update(new_preferences)
    
    def get_routing_statistics(self, decisions: List[RoutingDecision]) -> Dict[str, any]:
        """
        Calculate statistics from routing decisions.
        
        Args:
            decisions: List of routing decisions
        
        Returns:
            Statistics dictionary
        """
        if not decisions:
            return {}
        
        stats = {
            "total_decisions": len(decisions),
            "strategy_distribution": {},
            "avg_failure_probability": np.mean([d.failure_probability for d in decisions]),
            "avg_confidence": np.mean([d.confidence for d in decisions]),
            "avg_cost": np.mean([d.cost_estimate for d in decisions]),
            "avg_expected_accuracy": np.mean([d.expected_accuracy for d in decisions])
        }
        
        # Strategy distribution
        for strategy in RoutingStrategy:
            count = sum(1 for d in decisions if d.strategy == strategy)
            stats["strategy_distribution"][strategy.value] = {
                "count": count,
                "percentage": count / len(decisions) * 100
            }
        
        # Failure mode distribution
        failure_modes = {}
        for decision in decisions:
            if decision.predicted_failure_mode:
                mode = decision.predicted_failure_mode.value
                failure_modes[mode] = failure_modes.get(mode, 0) + 1
        
        stats["failure_mode_distribution"] = failure_modes
        
        return stats
