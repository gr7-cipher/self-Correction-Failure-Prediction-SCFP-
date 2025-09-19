"""
Cost model for different correction strategies.
"""

from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np


class CostType(Enum):
    """Types of costs to consider."""
    COMPUTATIONAL = "computational"  # CPU/GPU time
    MONETARY = "monetary"           # API calls, human time
    LATENCY = "latency"            # Response time
    QUALITY = "quality"            # Potential accuracy loss


@dataclass
class CostProfile:
    """Cost profile for a correction strategy."""
    computational: float  # Relative computational cost (1.0 = baseline)
    monetary: float      # Monetary cost in arbitrary units
    latency: float       # Expected latency in seconds
    quality_risk: float  # Risk of quality degradation (0-1)


class CostModel:
    """
    Model for estimating costs of different correction strategies.
    
    This model considers multiple cost dimensions including computational
    resources, monetary costs, latency, and quality risks.
    """
    
    def __init__(self, cost_weights: Optional[Dict[str, float]] = None):
        """
        Initialize cost model.
        
        Args:
            cost_weights: Weights for different cost types
        """
        # Default cost weights
        self.cost_weights = cost_weights or {
            "computational": 0.3,
            "monetary": 0.4,
            "latency": 0.2,
            "quality": 0.1
        }
        
        # Cost profiles for each strategy
        self._initialize_cost_profiles()
    
    def _initialize_cost_profiles(self):
        """Initialize cost profiles for each strategy."""
        from .router import RoutingStrategy
        
        self.cost_profiles = {
            RoutingStrategy.INTRINSIC: CostProfile(
                computational=1.0,    # Baseline computational cost
                monetary=0.0,         # No external costs
                latency=2.0,          # Fast response
                quality_risk=0.3      # Moderate quality risk
            ),
            
            RoutingStrategy.EXTERNAL: CostProfile(
                computational=0.5,    # Less local computation
                monetary=5.0,         # API costs
                latency=5.0,          # Network latency
                quality_risk=0.15     # Lower quality risk
            ),
            
            RoutingStrategy.HUMAN: CostProfile(
                computational=0.1,    # Minimal computation
                monetary=50.0,        # Human expert time
                latency=300.0,        # 5 minutes average
                quality_risk=0.05     # Lowest quality risk
            ),
            
            RoutingStrategy.HYBRID: CostProfile(
                computational=1.5,    # More computation for coordination
                monetary=10.0,        # Combined costs
                latency=8.0,          # Moderate latency
                quality_risk=0.1      # Low quality risk
            )
        }
    
    def get_cost(self, strategy, context: Optional[Dict] = None) -> float:
        """
        Calculate total cost for a strategy.
        
        Args:
            strategy: Routing strategy
            context: Optional context for cost adjustment
        
        Returns:
            Total weighted cost
        """
        profile = self.cost_profiles[strategy]
        context = context or {}
        
        # Base costs
        costs = {
            "computational": profile.computational,
            "monetary": profile.monetary,
            "latency": profile.latency,
            "quality": profile.quality_risk
        }
        
        # Apply context adjustments
        costs = self._apply_context_adjustments(costs, context)
        
        # Calculate weighted total
        total_cost = sum(
            costs[cost_type] * self.cost_weights[cost_type]
            for cost_type in costs
        )
        
        return total_cost
    
    def _apply_context_adjustments(
        self, 
        costs: Dict[str, float], 
        context: Dict
    ) -> Dict[str, float]:
        """
        Apply context-based cost adjustments.
        
        Args:
            costs: Base costs dictionary
            context: Context information
        
        Returns:
            Adjusted costs dictionary
        """
        adjusted_costs = costs.copy()
        
        # Urgency affects latency costs
        urgency = context.get("urgency", 0.5)
        if urgency > 0.7:
            adjusted_costs["latency"] *= 2.0  # High urgency makes latency more costly
        
        # Domain complexity affects computational costs
        complexity = context.get("domain_complexity", 0.5)
        adjusted_costs["computational"] *= (1 + complexity * 0.5)
        
        # Budget constraints affect monetary costs
        budget_constraint = context.get("budget_constraint", 0.5)
        if budget_constraint > 0.7:
            adjusted_costs["monetary"] *= 1.5  # Budget constraints make money more costly
        
        # Quality requirements affect quality risk costs
        quality_req = context.get("accuracy_requirement", 0.5)
        adjusted_costs["quality"] *= (1 + quality_req)
        
        return adjusted_costs
    
    def compare_strategies(
        self, 
        strategies: list, 
        context: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare costs across multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            context: Optional context
        
        Returns:
            Dictionary with cost breakdown for each strategy
        """
        comparison = {}
        
        for strategy in strategies:
            profile = self.cost_profiles[strategy]
            total_cost = self.get_cost(strategy, context)
            
            comparison[strategy.value] = {
                "total_cost": total_cost,
                "computational": profile.computational,
                "monetary": profile.monetary,
                "latency": profile.latency,
                "quality_risk": profile.quality_risk
            }
        
        return comparison
    
    def get_cost_breakdown(self, strategy, context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get detailed cost breakdown for a strategy.
        
        Args:
            strategy: Routing strategy
            context: Optional context
        
        Returns:
            Detailed cost breakdown
        """
        profile = self.cost_profiles[strategy]
        context = context or {}
        
        # Get adjusted costs
        costs = {
            "computational": profile.computational,
            "monetary": profile.monetary,
            "latency": profile.latency,
            "quality": profile.quality_risk
        }
        
        adjusted_costs = self._apply_context_adjustments(costs, context)
        
        # Calculate weighted contributions
        breakdown = {}
        for cost_type, cost_value in adjusted_costs.items():
            weighted_cost = cost_value * self.cost_weights[cost_type]
            breakdown[f"{cost_type}_raw"] = cost_value
            breakdown[f"{cost_type}_weighted"] = weighted_cost
            breakdown[f"{cost_type}_weight"] = self.cost_weights[cost_type]
        
        breakdown["total_cost"] = sum(
            breakdown[f"{ct}_weighted"] for ct in adjusted_costs.keys()
        )
        
        return breakdown
    
    def optimize_cost_weights(
        self, 
        historical_decisions: list, 
        outcomes: list,
        learning_rate: float = 0.01
    ):
        """
        Optimize cost weights based on historical performance.
        
        Args:
            historical_decisions: List of past routing decisions
            outcomes: List of actual outcomes (success/failure)
            learning_rate: Learning rate for weight updates
        """
        # This is a simplified optimization - in practice, you might use
        # more sophisticated methods like gradient descent or Bayesian optimization
        
        # Calculate performance by strategy
        strategy_performance = {}
        for decision, outcome in zip(historical_decisions, outcomes):
            strategy = decision.strategy
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"successes": 0, "total": 0}
            
            strategy_performance[strategy]["total"] += 1
            if outcome:  # Success
                strategy_performance[strategy]["successes"] += 1
        
        # Adjust weights based on performance
        for strategy, perf in strategy_performance.items():
            success_rate = perf["successes"] / perf["total"] if perf["total"] > 0 else 0
            
            # If a strategy is performing well, reduce its perceived cost
            # If performing poorly, increase its perceived cost
            cost_adjustment = (0.5 - success_rate) * learning_rate
            
            profile = self.cost_profiles[strategy]
            profile.quality_risk = max(0.01, min(0.99, profile.quality_risk + cost_adjustment))
    
    def estimate_roi(
        self, 
        strategy, 
        expected_accuracy: float, 
        baseline_accuracy: float = 0.7,
        value_per_correct: float = 1.0,
        context: Optional[Dict] = None
    ) -> float:
        """
        Estimate return on investment for a strategy.
        
        Args:
            strategy: Routing strategy
            expected_accuracy: Expected accuracy with this strategy
            baseline_accuracy: Baseline accuracy without intervention
            value_per_correct: Value of getting a correct answer
            context: Optional context
        
        Returns:
            Estimated ROI
        """
        cost = self.get_cost(strategy, context)
        
        # Calculate expected benefit
        accuracy_improvement = expected_accuracy - baseline_accuracy
        expected_benefit = accuracy_improvement * value_per_correct
        
        # ROI = (Benefit - Cost) / Cost
        if cost > 0:
            roi = (expected_benefit - cost) / cost
        else:
            roi = float('inf') if expected_benefit > 0 else 0
        
        return roi
    
    def get_cost_efficiency_ranking(
        self, 
        strategies: list, 
        expected_accuracies: Dict,
        context: Optional[Dict] = None
    ) -> list:
        """
        Rank strategies by cost efficiency.
        
        Args:
            strategies: List of strategies to rank
            expected_accuracies: Dictionary mapping strategies to expected accuracies
            context: Optional context
        
        Returns:
            List of strategies ranked by cost efficiency (best first)
        """
        efficiency_scores = []
        
        for strategy in strategies:
            cost = self.get_cost(strategy, context)
            accuracy = expected_accuracies.get(strategy, 0.5)
            
            # Efficiency = Accuracy / Cost (higher is better)
            efficiency = accuracy / max(cost, 0.001)  # Avoid division by zero
            
            efficiency_scores.append((strategy, efficiency))
        
        # Sort by efficiency (descending)
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [strategy for strategy, _ in efficiency_scores]
    
    def update_cost_weights(self, new_weights: Dict[str, float]):
        """Update cost weights."""
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.cost_weights = {k: v / total_weight for k, v in new_weights.items()}
    
    def update_cost_profile(self, strategy, new_profile: CostProfile):
        """Update cost profile for a strategy."""
        self.cost_profiles[strategy] = new_profile
    
    def get_cost_summary(self) -> Dict[str, any]:
        """Get summary of cost model configuration."""
        from .router import RoutingStrategy
        
        summary = {
            "cost_weights": self.cost_weights,
            "strategy_profiles": {}
        }
        
        for strategy in RoutingStrategy:
            profile = self.cost_profiles[strategy]
            summary["strategy_profiles"][strategy.value] = {
                "computational": profile.computational,
                "monetary": profile.monetary,
                "latency": profile.latency,
                "quality_risk": profile.quality_risk,
                "total_cost": self.get_cost(strategy)
            }
        
        return summary
