#!/usr/bin/env python3
"""
Training script for Supervised Fine-Tuning (SFT) of Llama-3 and Qwen baselines.
Uses QLoRA for efficient 4-bit fine-tuning.
"""

import os
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse
import json
from pathlib import Path

def format_instruction(sample):
    """Format single sample into SFT instruction template."""
    return f"### Instruction: Analyze the following self-correction trace and predict if the final response will be a SUCCESS or a FAILURE. If FAILURE, identify the mode.\n\n### Original Prompt:\n{sample['prompt']}\n\n### Initial Response:\n{sample['initial_response']}\n\n### Self-Critique:\n{sample['critique']}\n\n### Response Format:\nResult: [SUCCESS/FAILURE]\nMode: [JH/CM/BA/OC/RM/NONE]\n\n### Analysis:\nResult: {'FAILURE' if not sample['is_success'] else 'SUCCESS'}\nMode: {sample['failure_mode'].upper() if not sample['is_success'] else 'NONE'}"

def train_sft(args):
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Load and process data
    dataset = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    if args.limit:
        dataset = dataset[:args.limit]
        
    sft_dataset = Dataset.from_list([{'text': format_instruction(s)} for s in dataset])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=100,
        evaluation_strategy="no",
        fp16=True,
        report_to="none"
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config
    )
    
    # Train
    print(f"Starting SFT fine-tuning for {args.model_id}...")
    trainer.train()
    
    # Save model
    trainer.save_model(os.path.join(args.output_dir, "final_adapter"))
    print(f"Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT training for SCFP.")
    parser.add_argument("--model_id", type=str, required=True, help="Model (llama-3-8b, qwen-7b, etc.)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to jsonl benchmark")
    parser.add_argument("--output_dir", type=str, default="./models/sft_output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    train_sft(args)

if __name__ == "__main__":
    main()
