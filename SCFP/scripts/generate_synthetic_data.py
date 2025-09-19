#!/usr/bin/env python3
"""
Generate synthetic SCFP dataset.

This script generates a synthetic dataset for training and evaluation
since the original SCFP benchmark may not be available.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.synthetic import SyntheticDataGenerator, SyntheticConfig


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SCFP dataset")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--size", type=int, default=12000, help="Total number of samples")
    parser.add_argument("--success-rate", type=float, default=0.6, help="Success rate (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to include")
    
    # Failure mode distribution
    parser.add_argument("--jh-rate", type=float, default=0.25, help="Justification Hallucination rate")
    parser.add_argument("--cm-rate", type=float, default=0.20, help="Confidence Miscalibration rate")
    parser.add_argument("--ba-rate", type=float, default=0.20, help="Bias Amplification rate")
    parser.add_argument("--oc-rate", type=float, default=0.20, help="Over-correction rate")
    parser.add_argument("--rm-rate", type=float, default=0.15, help="Reasoning Myopia rate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure failure distribution
    failure_distribution = {
        "jh": args.jh_rate,
        "cm": args.cm_rate,
        "ba": args.ba_rate,
        "oc": args.oc_rate,
        "rm": args.rm_rate
    }
    
    # Normalize failure distribution
    total_failure_rate = sum(failure_distribution.values())
    if total_failure_rate > 0:
        failure_distribution = {k: v / total_failure_rate for k, v in failure_distribution.items()}
    
    # Create configuration
    config = SyntheticConfig(
        total_samples=args.size,
        success_rate=args.success_rate,
        failure_distribution=failure_distribution,
        domains=args.domains
    )
    
    print(f"Generating synthetic dataset with {args.size} samples...")
    print(f"Success rate: {args.success_rate:.2f}")
    print(f"Failure distribution: {failure_distribution}")
    print(f"Output: {args.output}")
    
    # Generate dataset
    generator = SyntheticDataGenerator(config=config, seed=args.seed)
    traces = generator.generate_and_save(args.output)
    
    print(f"Generated {len(traces)} traces")
    
    # Print statistics
    success_count = sum(1 for trace in traces if trace.is_success)
    failure_count = len(traces) - success_count
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(traces)}")
    print(f"  Successful corrections: {success_count} ({success_count/len(traces)*100:.1f}%)")
    print(f"  Failed corrections: {failure_count} ({failure_count/len(traces)*100:.1f}%)")
    
    # Failure mode breakdown
    from scfp.data.dataset import FailureMode
    mode_counts = {}
    for trace in traces:
        mode = trace.failure_mode.value
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    print(f"\nFailure Mode Distribution:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} ({count/len(traces)*100:.1f}%)")
    
    # Domain distribution
    domain_counts = {}
    for trace in traces:
        domain = trace.metadata.get("domain", "unknown") if trace.metadata else "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nDomain Distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} ({count/len(traces)*100:.1f}%)")
    
    print(f"\nDataset saved to: {args.output}")


if __name__ == "__main__":
    main()
