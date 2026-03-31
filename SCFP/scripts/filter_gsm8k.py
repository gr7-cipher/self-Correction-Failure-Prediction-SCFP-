#!/usr/bin/env python3
"""
ID-based filtering script to ensure zero overlap between the GSM8K routing study
and the SCFP training data, as required by the SCFP framework specifications.
"""

import json
import argparse
import os
from typing import List, Set

def filter_dataset(input_path: str, output_path: str, exclude_ids: Set[str] = None, exclude_domain: str = "Mathematical Reasoning (GSM8K)"):
    """
    Filter the dataset by excluding specific IDs or a whole domain.
    """
    print(f"Filtering dataset: {input_path}")
    
    filtered_data = []
    excluded_count = 0
    
    # In JSONL format
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # Check for ID overlap
            item_id = str(item.get("id", ""))
            if exclude_ids and item_id in exclude_ids:
                excluded_count += 1
                continue
            
            # Check for domain-based exclusion (if specified)
            domain = item.get("metadata", {}).get("domain", "")
            if exclude_domain and domain == exclude_domain:
                # Note: The paper suggests a strict ID-based filter for the *case study*
                # but might allow other GSM8K for training. 
                # If exclude_ids is provided, we prioritize that.
                if not exclude_ids:
                    excluded_count += 1
                    continue
            
            filtered_data.append(item)
    
    print(f"Excluded {excluded_count} items.")
    print(f"Remaining items: {len(filtered_data)}")
    
    # Save filtered data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Filtered dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SCFP dataset to avoid GSM8K overlap.")
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl file")
    parser.add_argument("--exclude_ids", type=str, help="Comma-separated list of IDs to exclude")
    parser.add_argument("--exclude_domain", type=str, default="Mathematical Reasoning (GSM8K)", help="Domain to exclude if no IDs provided")
    
    args = parser.parse_args()
    
    exclude_ids = set(args.exclude_ids.split(",")) if args.exclude_ids else None
    
    filter_dataset(args.input, args.output, exclude_ids, args.exclude_domain)
