"""
Unit tests for SCFP routing system.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.routing.router import DynamicRouter, RoutingStrategy, RoutingDecision
from scfp.routing.cost_model import CostModel, CostProfile


class TestCostModel:
    """Test CostModel class."""
    
    def test_initialization(self):
        """Test cost model initialization."""
        cost_model = CostModel()
        
        # Check default weights
        assert "computational" in cost_model.cost_weights
        assert "monetary" in cost_model.cost_weights
        assert "latency" in cost_model.cost_weights
        assert "quality" in cost_model.cost_weights
        
        # Check cost profiles exist for all strategies
        for strategy in RoutingStrategy:
            assert strategy in cost_model.cost_profiles
    
    def test_get_cost(self):
        """Test cost calculation."""
        cost_model = CostModel()
        
        # Test basic cost calculation
        intrinsic_cost = cost_model.get_cost(RoutingStrategy.INTRINSIC)
        human_cost = cost_model.get_cost(RoutingStrategy.HUMAN)
        
        # Human should be more expensive than intrinsic
        assert human_cost > intrinsic_cost
        
        # Test with context
        high_urgency_context = {"urgency": 0.9}
        cost_with_urgency = cost_model.get_cost(RoutingStrategy.HUMAN, high_urgency_context)
        
        # Cost should be affected by context
        assert isinstance(cost_with_urgency, float)
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        cost_model = CostModel()
        
        strategies = [RoutingStrategy.INTRINSIC, RoutingStrategy.EXTERNAL, RoutingStrategy.HUMAN]
        comparison = cost_model.compare_strategies(strategies)
        
        assert len(comparison) == 3
        for strategy in strategies:
            assert strategy.value in comparison
            assert "total_cost" in comparison[strategy.value]
    
    def test_cost_breakdown(self):
        """Test detailed cost breakdown."""
        cost_model = CostModel()
        
        breakdown = cost_model.get_cost_breakdown(RoutingStrategy.INTRINSIC)
        
        # Check all cost components are present
        expected_components = ["computational", "monetary", "latency", "quality"]
        for component in expected_components:
            assert f"{component}_raw" in breakdown
            assert f"{component}_weighted" in breakdown
            assert f"{component}_weight" in breakdown
        
        assert "total_cost" in breakdown


class TestDynamicRouter:
    """Test DynamicRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create router for testing."""
        cost_model = CostModel()
        return DynamicRouter(
            failure_predictor=None,  # Using dummy predictor for tests
            cost_model=cost_model
        )
    
    def test_initialization(self, router):
        """Test router initialization."""
        assert router.cost_model is not None
        assert router.thresholds is not None
        assert router.strategy_preferences is not None
    
    def test_route_decision(self, router):
        """Test routing decision making."""
        decision = router.route(
            prompt="What is 2+2?",
            initial_response="5",
            critique="That's wrong, it should be 4",
            context={"domain": "math", "urgency": 0.5}
        )
        
        # Check decision structure
        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.strategy, RoutingStrategy)
        assert 0 <= decision.confidence <= 1
        assert 0 <= decision.failure_probability <= 1
        assert decision.cost_estimate >= 0
        assert decision.expected_accuracy >= 0
        assert isinstance(decision.reasoning, str)
    
    def test_batch_routing(self, router):
        """Test batch routing."""
        traces = [
            ("Prompt 1", "Response 1", "Critique 1"),
            ("Prompt 2", "Response 2", "Critique 2"),
            ("Prompt 3", "Response 3", "Critique 3")
        ]
        
        decisions = router.batch_route(traces)
        
        assert len(decisions) == 3
        for decision in decisions:
            assert isinstance(decision, RoutingDecision)
    
    def test_context_effects(self, router):
        """Test how context affects routing decisions."""
        base_decision = router.route(
            prompt="Test prompt",
            initial_response="Test response",
            critique="Test critique"
        )
        
        high_stakes_decision = router.route(
            prompt="Test prompt",
            initial_response="Test response",
            critique="Test critique",
            context={"stakes": 0.9, "accuracy_requirement": 0.9}
        )
        
        # High stakes should generally prefer more reliable strategies
        # (though exact behavior depends on failure prediction)
        assert isinstance(base_decision.strategy, RoutingStrategy)
        assert isinstance(high_stakes_decision.strategy, RoutingStrategy)
    
    def test_threshold_updates(self, router):
        """Test threshold updates."""
        original_threshold = router.thresholds["intrinsic_max_failure_prob"]
        
        new_thresholds = {"intrinsic_max_failure_prob": 0.5}
        router.update_thresholds(new_thresholds)
        
        assert router.thresholds["intrinsic_max_failure_prob"] == 0.5
        assert router.thresholds["intrinsic_max_failure_prob"] != original_threshold
    
    def test_routing_statistics(self, router):
        """Test routing statistics calculation."""
        # Create some dummy decisions
        decisions = []
        for i in range(5):
            decision = router.route(
                prompt=f"Prompt {i}",
                initial_response=f"Response {i}",
                critique=f"Critique {i}"
            )
            decisions.append(decision)
        
        stats = router.get_routing_statistics(decisions)
        
        # Check statistics structure
        assert "total_decisions" in stats
        assert "strategy_distribution" in stats
        assert "avg_failure_probability" in stats
        assert "avg_confidence" in stats
        assert "avg_cost" in stats
        assert "avg_expected_accuracy" in stats
        
        assert stats["total_decisions"] == 5


class TestRoutingStrategy:
    """Test RoutingStrategy enum."""
    
    def test_all_strategies_defined(self):
        """Test all routing strategies are defined."""
        expected_strategies = ["INTRINSIC", "EXTERNAL", "HUMAN", "HYBRID"]
        
        for strategy_name in expected_strategies:
            assert hasattr(RoutingStrategy, strategy_name)
    
    def test_strategy_values(self):
        """Test strategy values."""
        assert RoutingStrategy.INTRINSIC.value == "intrinsic"
        assert RoutingStrategy.EXTERNAL.value == "external"
        assert RoutingStrategy.HUMAN.value == "human"
        assert RoutingStrategy.HYBRID.value == "hybrid"


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""
    
    def test_decision_creation(self):
        """Test routing decision creation."""
        decision = RoutingDecision(
            strategy=RoutingStrategy.INTRINSIC,
            confidence=0.8,
            failure_probability=0.3,
            predicted_failure_mode=None,
            reasoning="Test reasoning",
            cost_estimate=1.5,
            expected_accuracy=0.75
        )
        
        assert decision.strategy == RoutingStrategy.INTRINSIC
        assert decision.confidence == 0.8
        assert decision.failure_probability == 0.3
        assert decision.predicted_failure_mode is None
        assert decision.reasoning == "Test reasoning"
        assert decision.cost_estimate == 1.5
        assert decision.expected_accuracy == 0.75


if __name__ == "__main__":
    pytest.main([__file__])
