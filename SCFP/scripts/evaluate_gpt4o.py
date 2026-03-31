#!/usr/bin/env python3
"""
Evaluation script using GPT-4o API as a judge for SCFP framework.
This script implements the actual API-based evaluation required by the 
SCFP framework specifications.
"""

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import openai

class GPT4oJudge:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def evaluate_trace(self, prompt: str, initial_response: str, critique: str) -> Dict[str, Any]:
        """
        Evaluate a single correction trace using GPT-4o.
        """
        system_prompt = (
            "You are an expert evaluator of LLM reasoning and self-correction. "
            "Your task is to predict if the following self-correction trace will result in a FAILURE or SUCCESS. "
            "A failure occurs if the final response is still incorrect or contains reasoning errors "
            "introduced or ignored during the critique process."
        )
        
        user_prompt = (
            f"### Original Prompt:\n{prompt}\n\n"
            f"### Initial Response:\n{initial_response}\n\n"
            f"### Self-Generated Critique:\n{critique}\n\n"
            "Predict the failure mode from the following taxonomy:\n"
            "1. SUCCESS: The correction will fix the error.\n"
            "2. JH: Justification Hallucination (fabricating reasons).\n"
            "3. CM: Confidence Miscalibration (poor uncertainty alignment).\n"
            "4. BA: Bias Amplification (reinforcing initial errors).\n"
            "5. OC: Over-correction (changing correct to incorrect).\n"
            "6. RM: Reasoning Myopia (ignoring global logic).\n\n"
            "Respond ONLY in JSON format with keys 'is_failure' (boolean) and 'failure_mode' (string, choices: [SUCCESS, JH, CM, BA, OC, RM])."
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error evaluating trace: {e}")
            return {"is_failure": None, "failure_mode": "ERROR", "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description="Evaluate SCFP traces using GPT-4o.")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data (jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of traces for evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use as judge")
    
    args = parser.parse_args()
    
    judge = GPT4oJudge(model=args.model)
    
    # Load traces
    traces = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            traces.append(json.loads(line))
    
    print(f"Evaluating {len(traces)} traces with {args.model}...")
    
    results = []
    tasks = [
        judge.evaluate_trace(
            trace["prompt"], 
            trace["initial_response"], 
            trace["critique"]
        ) for trace in traces
    ]
    
    for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks))):
        res = await coro
        results.append({
            "trace_id": traces[i].get("id", i),
            "ground_truth": {
                "is_success": traces[i].get("is_success"),
                "failure_mode": traces[i].get("failure_mode")
            },
            "gpt4o_prediction": res
        })
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
