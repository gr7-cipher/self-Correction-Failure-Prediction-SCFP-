#!/usr/bin/env python3
"""
Demo script for the dynamic routing system.

This script demonstrates how the SCFP framework can be used to make
intelligent routing decisions for correction strategies.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.routing.router import DynamicRouter, RoutingStrategy
from scfp.routing.cost_model import CostModel
from scfp.data.dataset import PartialTrace


def create_demo_traces():
    """Create demo correction traces for testing."""
    traces = [
        {
            "prompt": "What is the capital of France?",
            "initial_response": "The capital of France is London.",
            "critique": "Wait, that's not right. London is the capital of the UK, not France.",
            "context": {"domain": "geography", "urgency": 0.3, "stakes": 0.4}
        },
        {
            "prompt": "Calculate the derivative of x^2 + 3x + 2",
            "initial_response": "The derivative is 2x + 3.",
            "critique": "Let me double-check this calculation. The derivative of x^2 is 2x, the derivative of 3x is 3, and the derivative of a constant is 0. So yes, 2x + 3 is correct.",
            "context": {"domain": "math", "urgency": 0.7, "stakes": 0.8}
        },
        {
            "prompt": "Explain the process of photosynthesis",
            "initial_response": "Photosynthesis is when plants eat sunlight to grow bigger.",
            "critique": "This explanation is too simplified and not scientifically accurate. I should provide a more detailed and precise explanation.",
            "context": {"domain": "science", "urgency": 0.2, "stakes": 0.9}
        },
        {
            "prompt": "What are the side effects of aspirin?",
            "initial_response": "Aspirin is completely safe and has no side effects.",
            "critique": "This is definitely wrong. Aspirin can have several side effects including stomach irritation, bleeding, and allergic reactions.",
            "context": {"domain": "medical", "urgency": 0.9, "stakes": 0.95}
        },
        {
            "prompt": "Write a creative story about a dragon",
            "initial_response": "Once upon a time, there was a dragon named Fred who liked to knit sweaters.",
            "critique": "This is a good start for a creative story. Maybe I could add more details about Fred's adventures.",
            "context": {"domain": "creative", "urgency": 0.1, "stakes": 0.2}
        }
    ]
    
    return traces


def demonstrate_routing_decision(router, trace_data, trace_idx):
    """Demonstrate routing decision for a single trace."""
    print(f"\n{'='*60}")
    print(f"TRACE {trace_idx + 1}")
    print(f"{'='*60}")
    
    print(f"Prompt: {trace_data['prompt']}")
    print(f"Initial Response: {trace_data['initial_response']}")
    print(f"Critique: {trace_data['critique']}")
    print(f"Context: {trace_data['context']}")
    
    # Make routing decision
    decision = router.route(
        prompt=trace_data['prompt'],
        initial_response=trace_data['initial_response'],
        critique=trace_data['critique'],
        context=trace_data['context']
    )
    
    print(f"\nROUTING DECISION:")
    print(f"  Strategy: {decision.strategy.value.upper()}")
    print(f"  Confidence: {decision.confidence:.3f}")
    print(f"  Failure Probability: {decision.failure_probability:.3f}")
    print(f"  Predicted Failure Mode: {decision.predicted_failure_mode.value if decision.predicted_failure_mode else 'N/A'}")
    print(f"  Cost Estimate: {decision.cost_estimate:.3f}")
    print(f"  Expected Accuracy: {decision.expected_accuracy:.3f}")
    print(f"\nReasoning: {decision.reasoning}")
    
    return decision


def analyze_routing_patterns(decisions):
    """Analyze patterns in routing decisions."""
    print(f"\n{'='*60}")
    print("ROUTING ANALYSIS")
    print(f"{'='*60}")
    
    # Strategy distribution
    strategy_counts = {}
    for decision in decisions:
        strategy = decision.strategy.value
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print("\nStrategy Distribution:")
    for strategy, count in strategy_counts.items():
        percentage = count / len(decisions) * 100
        print(f"  {strategy.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Average metrics by strategy
    strategy_metrics = {}
    for decision in decisions:
        strategy = decision.strategy.value
        if strategy not in strategy_metrics:
            strategy_metrics[strategy] = {
                "failure_probs": [],
                "confidences": [],
                "costs": [],
                "accuracies": []
            }
        
        strategy_metrics[strategy]["failure_probs"].append(decision.failure_probability)
        strategy_metrics[strategy]["confidences"].append(decision.confidence)
        strategy_metrics[strategy]["costs"].append(decision.cost_estimate)
        strategy_metrics[strategy]["accuracies"].append(decision.expected_accuracy)
    
    print("\nAverage Metrics by Strategy:")
    print(f"{'Strategy':<12} {'Fail Prob':<10} {'Confidence':<10} {'Cost':<8} {'Accuracy':<8}")
    print("-" * 55)
    
    for strategy, metrics in strategy_metrics.items():
        avg_fail_prob = sum(metrics["failure_probs"]) / len(metrics["failure_probs"])
        avg_confidence = sum(metrics["confidences"]) / len(metrics["confidences"])
        avg_cost = sum(metrics["costs"]) / len(metrics["costs"])
        avg_accuracy = sum(metrics["accuracies"]) / len(metrics["accuracies"])
        
        print(f"{strategy.capitalize():<12} {avg_fail_prob:<10.3f} {avg_confidence:<10.3f} {avg_cost:<8.3f} {avg_accuracy:<8.3f}")
    
    # Context analysis
    print("\nContext Factor Analysis:")
    
    # Group by domain
    domain_strategies = {}
    for i, decision in enumerate(decisions):
        # Get domain from demo traces (simplified)
        domains = ["geography", "math", "science", "medical", "creative"]
        domain = domains[i] if i < len(domains) else "unknown"
        
        if domain not in domain_strategies:
            domain_strategies[domain] = []
        domain_strategies[domain].append(decision.strategy.value)
    
    print("\nStrategy by Domain:")
    for domain, strategies in domain_strategies.items():
        strategy_dist = {}
        for strategy in strategies:
            strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
        
        print(f"  {domain.capitalize()}:")
        for strategy, count in strategy_dist.items():
            print(f"    {strategy}: {count}")


def demonstrate_cost_model(cost_model):
    """Demonstrate cost model functionality."""
    print(f"\n{'='*60}")
    print("COST MODEL ANALYSIS")
    print(f"{'='*60}")
    
    # Show cost profiles
    print("\nCost Profiles:")
    summary = cost_model.get_cost_summary()
    
    print(f"{'Strategy':<12} {'Computational':<12} {'Monetary':<10} {'Latency':<8} {'Quality Risk':<12} {'Total':<8}")
    print("-" * 70)
    
    for strategy, profile in summary["strategy_profiles"].items():
        print(f"{strategy.capitalize():<12} {profile['computational']:<12.1f} {profile['monetary']:<10.1f} {profile['latency']:<8.1f} {profile['quality_risk']:<12.3f} {profile['total_cost']:<8.3f}")
    
    # Show cost weights
    print(f"\nCost Weights:")
    for cost_type, weight in summary["cost_weights"].items():
        print(f"  {cost_type.capitalize()}: {weight:.3f}")
    
    # Demonstrate context effects
    print(f"\nContext Effects on Costs:")
    
    contexts = [
        {"urgency": 0.9, "stakes": 0.9, "description": "High urgency, high stakes"},
        {"urgency": 0.1, "stakes": 0.1, "description": "Low urgency, low stakes"},
        {"domain_complexity": 0.9, "description": "High complexity domain"},
        {"budget_constraint": 0.9, "description": "High budget constraint"}
    ]
    
    from scfp.routing.router import RoutingStrategy
    
    for context in contexts:
        print(f"\n  {context['description']}:")
        for strategy in RoutingStrategy:
            cost = cost_model.get_cost(strategy, context)
            print(f"    {strategy.value}: {cost:.3f}")


def interactive_mode(router):
    """Interactive mode for testing custom traces."""
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print(f"{'='*60}")
    print("Enter your own correction traces to see routing decisions.")
    print("Type 'quit' to exit interactive mode.\n")
    
    while True:
        try:
            print("\nEnter trace information:")
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            initial_response = input("Initial Response: ").strip()
            if initial_response.lower() == 'quit':
                break
            
            critique = input("Critique: ").strip()
            if critique.lower() == 'quit':
                break
            
            # Optional context
            print("\nOptional context (press Enter to skip):")
            domain = input("Domain: ").strip() or "general"
            urgency = input("Urgency (0-1): ").strip()
            stakes = input("Stakes (0-1): ").strip()
            
            # Parse context
            context = {"domain": domain}
            if urgency:
                try:
                    context["urgency"] = float(urgency)
                except ValueError:
                    context["urgency"] = 0.5
            if stakes:
                try:
                    context["stakes"] = float(stakes)
                except ValueError:
                    context["stakes"] = 0.5
            
            # Make routing decision
            decision = router.route(prompt, initial_response, critique, context)
            
            print(f"\nROUTING DECISION:")
            print(f"  Strategy: {decision.strategy.value.upper()}")
            print(f"  Failure Probability: {decision.failure_probability:.3f}")
            print(f"  Expected Accuracy: {decision.expected_accuracy:.3f}")
            print(f"  Cost: {decision.cost_estimate:.3f}")
            print(f"  Reasoning: {decision.reasoning}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Demo SCFP dynamic routing system")
    parser.add_argument("--model", type=str, help="Path to trained failure prediction model")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--config", type=str, help="Router configuration file")
    parser.add_argument("--output", type=str, help="Save results to file")
    
    args = parser.parse_args()
    
    print("SCFP Dynamic Routing System Demo")
    print("="*50)
    
    # Initialize cost model
    cost_model = CostModel()
    
    # Initialize router (with dummy predictor for demo)
    # In practice, you would load a trained model
    router = DynamicRouter(
        failure_predictor=None,  # Using simulated predictions for demo
        cost_model=cost_model
    )
    
    print("Router initialized with default configuration.")
    
    if args.config:
        print(f"Loading router configuration from: {args.config}")
        # Load custom configuration if provided
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if "thresholds" in config:
            router.update_thresholds(config["thresholds"])
        if "strategy_preferences" in config:
            router.update_strategy_preferences(config["strategy_preferences"])
    
    # Demonstrate cost model
    demonstrate_cost_model(cost_model)
    
    # Create demo traces
    demo_traces = create_demo_traces()
    
    # Process each demo trace
    decisions = []
    for i, trace_data in enumerate(demo_traces):
        decision = demonstrate_routing_decision(router, trace_data, i)
        decisions.append(decision)
    
    # Analyze routing patterns
    analyze_routing_patterns(decisions)
    
    # Get routing statistics
    stats = router.get_routing_statistics(decisions)
    
    print(f"\n{'='*60}")
    print("ROUTING STATISTICS")
    print(f"{'='*60}")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Average failure probability: {stats['avg_failure_probability']:.3f}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Average cost: {stats['avg_cost']:.3f}")
    print(f"Average expected accuracy: {stats['avg_expected_accuracy']:.3f}")
    
    # Save results if requested
    if args.output:
        results = {
            "demo_traces": demo_traces,
            "routing_decisions": [
                {
                    "strategy": d.strategy.value,
                    "confidence": d.confidence,
                    "failure_probability": d.failure_probability,
                    "cost_estimate": d.cost_estimate,
                    "expected_accuracy": d.expected_accuracy,
                    "reasoning": d.reasoning
                }
                for d in decisions
            ],
            "statistics": stats,
            "cost_model_summary": cost_model.get_cost_summary()
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(router)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
