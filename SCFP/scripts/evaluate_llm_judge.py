#!/usr/bin/env python3
"""
Evaluation script for high-end LLM judges (Llama-3-70B, Qwen-Max) using CoT.
Supports vLLM (local) and API-based (Groq/Togther) inference.
"""

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import openai  # Compatible with vLLM and most providers

class LLMJudge:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(
            base_url=api_base,
            api_key=api_key or "EMPTY"
        )
        self.model = model

    async def evaluate_trace(self, prompt: str, initial_response: str, critique: str) -> Dict[str, Any]:
        """
        Evaluate a single correction trace using CoT prompting.
        """
        from scfp.models.llm_baselines import CoTJudge
        
        system_prompt = CoTJudge.get_cot_system_prompt()
        user_prompt = CoTJudge.get_cot_user_prompt(prompt, initial_response, critique)
        
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
            return {"is_failure": None, "failure_mode": "ERROR", "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description="Evaluate SCFP traces using Llama-3-70B / Qwen Judge.")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data (jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="vLLM or provider API base")
    parser.add_argument("--api-key", type=str, default=None, help="API key if required")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., meta-llama/Meta-Llama-3-70B-Instruct)")
    parser.add_argument("--limit", type=int, default=100)
    
    args = parser.parse_args()
    
    judge = LLMJudge(api_base=args.api_base, api_key=args.api_key, model=args.model)
    
    # Load traces
    traces = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            traces.append(json.loads(line))
    
    print(f"Evaluating {len(traces)} traces with {args.model} via {args.api_base}...")
    
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
            "judge_prediction": res
        })
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
