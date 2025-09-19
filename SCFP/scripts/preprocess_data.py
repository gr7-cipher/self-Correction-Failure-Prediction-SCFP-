#!/usr/bin/env python3
"""
Preprocess SCFP dataset for training.

This script loads the raw dataset and creates train/val/test splits
with appropriate preprocessing.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.dataset import CorrectionTrace, SCFPDataset
from scfp.data.preprocessing import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess SCFP dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--stratify", type=str, default="failure_mode", 
                       choices=["failure_mode", "success", "none"],
                       help="Stratification strategy")
    parser.add_argument("--balance", type=str, default="none",
                       choices=["undersample", "oversample", "none"],
                       help="Balancing strategy")
    parser.add_argument("--min-length", type=int, default=None, help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum text length")
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    print(f"Loading dataset from: {args.input}")
    
    # Load raw data
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert to traces
    traces = [CorrectionTrace.from_dict(item) for item in raw_data]
    print(f"Loaded {len(traces)} traces")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(seed=args.seed)
    
    # Apply filters
    if args.min_length or args.max_length or args.domains:
        print("Applying filters...")
        original_count = len(traces)
        traces = preprocessor.filter_traces(
            traces,
            min_length=args.min_length,
            max_length=args.max_length,
            domains=args.domains
        )
        print(f"Filtered: {original_count} -> {len(traces)} traces")
    
    # Apply balancing
    if args.balance != "none":
        print(f"Applying {args.balance} balancing...")
        original_count = len(traces)
        traces = preprocessor.balance_dataset(traces, balance_type=args.balance)
        print(f"Balanced: {original_count} -> {len(traces)} traces")
    
    # Create splits
    print("Creating train/val/test splits...")
    train_traces, val_traces, test_traces = preprocessor.split_dataset(
        traces,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=args.stratify if args.stratify != "none" else None
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_traces)}")
    print(f"  Validation: {len(val_traces)}")
    print(f"  Test: {len(test_traces)}")
    
    # Save splits
    print(f"Saving splits to: {args.output}")
    preprocessor.save_splits(train_traces, val_traces, test_traces, args.output)
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    splits = {
        "train": train_traces,
        "val": val_traces,
        "test": test_traces
    }
    
    for split_name, split_traces in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Size: {len(split_traces)}")
        
        # Success/failure distribution
        success_count = sum(1 for t in split_traces if t.is_success)
        failure_count = len(split_traces) - success_count
        print(f"  Success: {success_count} ({success_count/len(split_traces)*100:.1f}%)")
        print(f"  Failure: {failure_count} ({failure_count/len(split_traces)*100:.1f}%)")
        
        # Failure mode distribution
        from scfp.data.dataset import FailureMode
        mode_counts = {}
        for trace in split_traces:
            mode = trace.failure_mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"  Failure modes:")
        for mode, count in sorted(mode_counts.items()):
            print(f"    {mode}: {count} ({count/len(split_traces)*100:.1f}%)")
        
        # Domain distribution
        domain_counts = {}
        for trace in split_traces:
            domain = trace.metadata.get("domain", "unknown") if trace.metadata else "unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print(f"  Domains:")
        for domain, count in sorted(domain_counts.items()):
            print(f"    {domain}: {count} ({count/len(split_traces)*100:.1f}%)")
    
    # Save preprocessing configuration
    config = {
        "input_file": args.input,
        "output_dir": args.output,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "stratify": args.stratify,
        "balance": args.balance,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "domains": args.domains,
        "seed": args.seed,
        "original_size": len(raw_data),
        "final_size": len(traces),
        "train_size": len(train_traces),
        "val_size": len(val_traces),
        "test_size": len(test_traces)
    }
    
    config_path = os.path.join(args.output, "preprocessing_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
