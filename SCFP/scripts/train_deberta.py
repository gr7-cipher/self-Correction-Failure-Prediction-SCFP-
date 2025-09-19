#!/usr/bin/env python3
"""
Train DeBERTa-v3 failure prediction model.

This script trains the main SCFP model using the DeBERTa-v3 architecture
with specialized attention mechanisms.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from transformers import DebertaV2Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.dataset import SCFPDataset
from scfp.data.loaders import create_dataloaders
from scfp.models.deberta import DeBERTaFailurePredictor
from scfp.training.trainer import Trainer, TrainingConfig


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return TrainingConfig.from_dict(config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa failure prediction model")
    parser.add_argument("--config", type=str, help="Training configuration file")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory with splits")
    parser.add_argument("--output-dir", type=str, default="./models/deberta", help="Output directory")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base", help="Base model name")
    
    # Training parameters (can override config)
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Model parameters
    parser.add_argument("--no-specialized-attention", action="store_true", 
                       help="Disable specialized attention")
    parser.add_argument("--dropout-rate", type=float, help="Dropout rate")
    
    # Hardware
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = TrainingConfig()
        print("Using default configuration")
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.max_length:
        config.max_length = args.max_length
    if args.seed:
        config.seed = args.seed
    if args.model_name:
        config.model_name = args.model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_specialized_attention:
        config.use_specialized_attention = False
    if args.dropout_rate:
        config.dropout_rate = args.dropout_rate
    if args.no_fp16:
        config.fp16 = False
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config.model_name)
    
    # Load datasets
    print(f"Loading datasets from: {args.data_dir}")
    
    train_dataset = SCFPDataset.from_json(
        os.path.join(args.data_dir, "train.json"),
        tokenizer=tokenizer,
        max_length=config.max_length,
        include_final_response=False
    )
    
    val_dataset = SCFPDataset.from_json(
        os.path.join(args.data_dir, "val.json"),
        tokenizer=tokenizer,
        max_length=config.max_length,
        include_final_response=False
    )
    
    test_dataset = SCFPDataset.from_json(
        os.path.join(args.data_dir, "test.json"),
        tokenizer=tokenizer,
        max_length=config.max_length,
        include_final_response=False
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        use_weighted_sampling=config.use_class_weights
    )
    
    # Initialize model
    print(f"Initializing model: {config.model_name}")
    model = DeBERTaFailurePredictor(
        model_name=config.model_name,
        num_failure_modes=6,
        use_specialized_attention=config.use_specialized_attention,
        dropout_rate=config.dropout_rate
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        device=device
    )
    
    # Print training configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Train model
    print("Starting training...")
    training_history = trainer.train()
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    trainer.model.eval()
    
    # Switch to test loader for final evaluation
    trainer.eval_dataloader = test_loader
    test_metrics = trainer.evaluate()
    
    print("\nTest Set Results:")
    print(f"  Binary Accuracy: {test_metrics['binary_accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  ECE: {test_metrics['ece']:.4f}")
    
    # Save final model
    final_model_dir = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    
    # Save test results
    results = {
        "training_config": config.to_dict(),
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_info": {
            "model_name": config.model_name,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "use_specialized_attention": config.use_specialized_attention
        }
    }
    
    results_path = os.path.join(config.output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {final_model_dir}")
    print(f"Results saved to: {results_path}")
    
    # Calibrate temperature on validation set
    print("\nCalibrating temperature...")
    trainer.model.calibrate_temperature(val_loader, device)
    
    # Save calibrated model
    calibrated_model_dir = os.path.join(config.output_dir, "calibrated_model")
    trainer.save_model(calibrated_model_dir)
    print(f"Calibrated model saved to: {calibrated_model_dir}")


if __name__ == "__main__":
    main()
