"""
Training utilities for SCFP models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from pathlib import Path

from .metrics import EvaluationMetrics
from .losses import MultiTaskLoss


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model parameters
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 1024
    use_specialized_attention: bool = True
    
    # Training parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Loss parameters
    binary_weight: float = 1.0
    multiclass_weight: float = 1.0
    confidence_weight: float = 0.5
    use_class_weights: bool = True
    
    # Evaluation parameters
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_macro_f1"
    
    # Scheduler parameters
    scheduler_type: str = "linear"  # "linear", "cosine", "constant"
    
    # Output parameters
    output_dir: str = "./models"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class Trainer:
    """
    Trainer for SCFP models with multi-task learning support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_metrics()
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = -float('inf')
        self.early_stopping_counter = 0
        self.training_history = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        if self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        else:  # constant
            self.scheduler = None
    
    def _setup_loss_function(self):
        """Setup multi-task loss function."""
        # Get class weights if needed
        binary_weights = None
        multiclass_weights = None
        
        if self.config.use_class_weights:
            # This would typically be computed from the dataset
            # For now, we'll use balanced weights
            binary_weights = torch.tensor([1.0, 1.0]).to(self.device)
            multiclass_weights = torch.tensor([1.0] * 6).to(self.device)
        
        self.loss_fn = MultiTaskLoss(
            binary_weight=self.config.binary_weight,
            multiclass_weight=self.config.multiclass_weight,
            confidence_weight=self.config.confidence_weight,
            binary_class_weights=binary_weights,
            multiclass_class_weights=multiclass_weights
        )
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        self.metrics = EvaluationMetrics()
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training on {self.device}")
        self.logger.info(f"Number of training examples: {len(self.train_dataloader.dataset)}")
        self.logger.info(f"Number of epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            # Log metrics
            self._log_metrics(train_metrics, eval_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint(epoch, eval_metrics)
            
            # Early stopping check
            current_metric = eval_metrics.get(self.config.metric_for_best_model, 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.early_stopping_counter = 0
                self._save_best_model(eval_metrics)
            else:
                self.early_stopping_counter += 1
                
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Load best model if requested
        if self.config.load_best_model_at_end:
            self._load_best_model()
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0
        total_binary_loss = 0
        total_multiclass_loss = 0
        total_confidence_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                binary_logits=outputs["binary_logits"],
                multiclass_logits=outputs["multiclass_logits"],
                confidence=outputs.get("confidence"),
                binary_labels=batch["binary_label"],
                multiclass_labels=batch["multiclass_label"]
            )
            
            loss = loss_dict["total_loss"]
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.fp16:
                # Use automatic mixed precision if available
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    with autocast():
                        loss.backward()
                except ImportError:
                    loss.backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            total_binary_loss += loss_dict["binary_loss"].item()
            total_multiclass_loss += loss_dict["multiclass_loss"].item()
            if "confidence_loss" in loss_dict:
                total_confidence_loss += loss_dict["confidence_loss"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # Calculate average losses
        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_binary_loss": total_binary_loss / num_batches,
            "train_multiclass_loss": total_multiclass_loss / num_batches,
            "train_confidence_loss": total_confidence_loss / num_batches if total_confidence_loss > 0 else 0
        }
        
        return avg_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        self.model.eval()
        
        all_binary_preds = []
        all_binary_labels = []
        all_multiclass_preds = []
        all_multiclass_labels = []
        all_binary_probs = []
        all_multiclass_probs = []
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    binary_logits=outputs["binary_logits"],
                    multiclass_logits=outputs["multiclass_logits"],
                    confidence=outputs.get("confidence"),
                    binary_labels=batch["binary_label"],
                    multiclass_labels=batch["multiclass_label"]
                )
                
                total_loss += loss_dict["total_loss"].item()
                num_batches += 1
                
                # Get predictions
                binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)
                multiclass_probs = torch.softmax(outputs["multiclass_logits"], dim=-1)
                
                binary_preds = torch.argmax(binary_probs, dim=-1)
                multiclass_preds = torch.argmax(multiclass_probs, dim=-1)
                
                # Collect predictions and labels
                all_binary_preds.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(batch["binary_label"].cpu().numpy())
                all_multiclass_preds.extend(multiclass_preds.cpu().numpy())
                all_multiclass_labels.extend(batch["multiclass_label"].cpu().numpy())
                all_binary_probs.extend(binary_probs.cpu().numpy())
                all_multiclass_probs.extend(multiclass_probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics.compute_all_metrics(
            binary_preds=np.array(all_binary_preds),
            binary_labels=np.array(all_binary_labels),
            binary_probs=np.array(all_binary_probs),
            multiclass_preds=np.array(all_multiclass_preds),
            multiclass_labels=np.array(all_multiclass_labels),
            multiclass_probs=np.array(all_multiclass_probs)
        )
        
        # Add loss
        metrics["eval_loss"] = total_loss / num_batches
        
        self.model.train()
        return metrics
    
    def _log_metrics(self, train_metrics: Dict[str, float], eval_metrics: Dict[str, float], epoch: int):
        """Log training and evaluation metrics."""
        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics, "epoch": epoch}
        
        # Add to history
        self.training_history.append(all_metrics)
        
        # Log to console
        self.logger.info(f"Epoch {epoch + 1} Results:")
        self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        self.logger.info(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
        self.logger.info(f"  Binary Accuracy: {eval_metrics['binary_accuracy']:.4f}")
        self.logger.info(f"  Macro F1: {eval_metrics['macro_f1']:.4f}")
        self.logger.info(f"  AUC-ROC: {eval_metrics['auc_roc']:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.training_history, f, indent=2)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config.to_dict(),
            "metrics": metrics
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.bin"))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_best_model(self, metrics: Dict[str, float]):
        """Save the best model."""
        best_model_dir = os.path.join(self.config.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
        
        # Save config and metrics
        with open(os.path.join(best_model_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        with open(os.path.join(best_model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Best model saved with {self.config.metric_for_best_model}: {self.best_metric:.4f}")
    
    def _load_best_model(self):
        """Load the best model."""
        best_model_path = os.path.join(self.config.output_dir, "best_model", "pytorch_model.bin")
        
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info("Best model loaded")
        else:
            self.logger.warning("Best model not found, using current model")
    
    def save_model(self, output_dir: str):
        """Save the final trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model from checkpoint."""
        # Load model state
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Load training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.bin")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.best_metric = training_state["best_metric"]
            
            if "optimizer_state_dict" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            
            if "scheduler_state_dict" in training_state and self.scheduler:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_dir}")
