"""
Advanced LLM-based comparative models for SCFP framework.
Includes SFT (Supervised Fine-Tuned) Llama-3 and Qwen implementation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Union
from peft import PeftModel, PeftConfig

class SFTFailurePredictor:
    """
    Wrapper for fine-tuned LLM predictors (Llama-3, Qwen).
    Processes correction traces as instructions to predict failure.
    """
    
    def __init__(
        self, 
        model_id: str, 
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_4bit=load_in_4bit if device == "cuda" else False,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load LoRA adapter if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
        self.device = device
        self.model.eval()

    def create_prompt(self, prompt: str, initial_response: str, critique: str) -> str:
        """
        Format trace into an instruct-style prompt for the SFT model.
        """
        return (
            "### Instruction: Analyze the following self-correction trace and predict "
            "if the final response will be a SUCCESS or a FAILURE. If FAILURE, identify the mode.\n\n"
            f"### Original Prompt:\n{prompt}\n\n"
            f"### Initial Response:\n{initial_response}\n\n"
            f"### Self-Critique:\n{critique}\n\n"
            "### Response Format:\n"
            "Result: [SUCCESS/FAILURE]\n"
            "Mode: [JH/CM/BA/OC/RM/NONE]\n\n"
            "### Analysis:\nResult:"
        )

    @torch.no_grad()
    def predict(self, prompt: str, initial_response: str, critique: str) -> Dict[str, Union[bool, str]]:
        """
        Predict failure for a single trace.
        """
        full_prompt = self.create_prompt(prompt, initial_response, critique)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
            do_sample=False
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_sft_response(response)

    def _parse_sft_response(self, text: str) -> Dict[str, Union[bool, str]]:
        """
        Robustly parse the LLM output.
        """
        text = text.upper()
        is_failure = "FAILURE" in text
        
        failure_mode = "NONE"
        for mode in ["JH", "CM", "BA", "OC", "RM"]:
            if mode in text:
                failure_mode = mode
                break
                
        return {
            "is_failure": is_failure,
            "failure_mode": failure_mode if is_failure else "SUCCESS"
        }

class CoTJudge:
    """
    Expert judge implementation using Chain-of-Thought (CoT) prompting.
    Used for GPT-4o+CoT and Llama-3-70B+CoT benchmarks.
    """
    
    @staticmethod
    def get_cot_system_prompt() -> str:
        return (
            "You are an expert meta-reasoner. Your task is to evaluate whether an LLM's "
            "self-correction process successfully fixes an error or introduces a failure. "
            "You must first provide a detailed Chain-of-Thought (CoT) analysis of the "
            "critique's logic before making a final prediction."
        )

    @staticmethod
    def get_cot_user_prompt(prompt: str, initial_response: str, critique: str) -> str:
        return (
            f"### Task:\nPredict the failure mode of the following correction trace.\n\n"
            f"### Context:\nPrompt: {prompt}\nResponse: {initial_response}\nCritique: {critique}\n\n"
            "### Step-by-Step Reasoning:\n"
            "1. Evaluate if the critique identifies a real error.\n"
            "2. Determine if the proposed fix is logically sound.\n"
            "3. Identify any justification hallucinations or bias.\n\n"
            "### Prediction:\n"
            "Please provide your prediction in valid JSON format at the end of your response "
            "with keys 'reasoning', 'is_failure' (bool), and 'failure_mode' (string: SUCCESS, JH, CM, BA, OC, RM)."
        )
