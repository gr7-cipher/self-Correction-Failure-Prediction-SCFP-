"""
DeBERTa-v3 based failure prediction model for SCFP framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2Config
from typing import Dict, Optional, Tuple
import numpy as np


class SpecializedAttention(nn.Module):
    """
    Specialized attention mechanism for analyzing correction traces.
    
    This attention layer is designed to focus on relationships between
    the initial response and critique components of correction traces.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of specialized attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Attended hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, seq_len, seq_len
            )
            # Apply mask (set masked positions to large negative value)
            attention_scores = attention_scores.masked_fill(
                extended_attention_mask == 0, -1e9
            )
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Residual connection and layer norm
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for binary and multi-class classification.
    """
    
    def __init__(self, hidden_size: int, num_failure_modes: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_failure_modes = num_failure_modes
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Binary classification head (success/failure)
        self.binary_head = nn.Linear(hidden_size // 4, 2)
        
        # Multi-class classification head (failure modes)
        self.multiclass_head = nn.Linear(hidden_size // 4, num_failure_modes)
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(hidden_size // 4, 1)
        
    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of multi-task head.
        
        Args:
            pooled_output: Pooled output from encoder [batch_size, hidden_size]
        
        Returns:
            Dictionary with binary_logits, multiclass_logits, and confidence
        """
        # Shared representation
        shared_repr = self.shared_layer(pooled_output)
        
        # Binary classification
        binary_logits = self.binary_head(shared_repr)
        
        # Multi-class classification
        multiclass_logits = self.multiclass_head(shared_repr)
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(shared_repr))
        
        return {
            "binary_logits": binary_logits,
            "multiclass_logits": multiclass_logits,
            "confidence": confidence
        }


class DeBERTaFailurePredictor(nn.Module):
    """
    DeBERTa-v3 based model for predicting self-correction failures.
    
    This model analyzes correction traces (prompt, initial response, critique)
    to predict both the likelihood and type of correction failure.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_failure_modes: int = 6,
        use_specialized_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_failure_modes = num_failure_modes
        self.use_specialized_attention = use_specialized_attention
        
        # Load DeBERTa configuration and model
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=self.config)
        
        # Specialized attention layer
        if use_specialized_attention:
            self.specialized_attention = SpecializedAttention(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads
            )
        
        # Pooling layer
        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh()
        )
        
        # Multi-task prediction head
        self.prediction_head = MultiTaskHead(
            hidden_size=self.config.hidden_size,
            num_failure_modes=num_failure_modes
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention_weights: Whether to return attention weights
        
        Returns:
            Dictionary with predictions and optional attention weights
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention_weights
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply specialized attention if enabled
        if self.use_specialized_attention:
            sequence_output = self.specialized_attention(
                sequence_output, attention_mask
            )
        
        # Pool the sequence output (use [CLS] token)
        pooled_output = self.pooler(sequence_output[:, 0])  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions from multi-task head
        predictions = self.prediction_head(pooled_output)
        
        # Apply temperature scaling for calibration
        predictions["binary_logits_calibrated"] = predictions["binary_logits"] / self.temperature
        predictions["multiclass_logits_calibrated"] = predictions["multiclass_logits"] / self.temperature
        
        # Add attention weights if requested
        if return_attention_weights:
            predictions["attention_weights"] = outputs.attentions
        
        return predictions
    
    def predict_failure_probability(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_calibrated: bool = True
    ) -> torch.Tensor:
        """
        Predict failure probability for correction traces.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            use_calibrated: Whether to use temperature-calibrated logits
        
        Returns:
            Failure probabilities [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            if use_calibrated:
                logits = outputs["binary_logits_calibrated"]
            else:
                logits = outputs["binary_logits"]
            
            # Convert to probabilities (failure is class 0, success is class 1)
            probs = F.softmax(logits, dim=-1)
            failure_probs = probs[:, 0]  # Probability of failure
            
            return failure_probs
    
    def predict_failure_mode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_calibrated: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict failure mode for correction traces.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            use_calibrated: Whether to use temperature-calibrated logits
        
        Returns:
            Tuple of (predicted_modes, mode_probabilities)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            if use_calibrated:
                logits = outputs["multiclass_logits_calibrated"]
            else:
                logits = outputs["multiclass_logits"]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            predicted_modes = torch.argmax(probs, dim=-1)
            
            return predicted_modes, probs
    
    def get_model_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get model confidence scores.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Confidence scores [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            confidence = outputs["confidence"].squeeze(-1)
            return confidence
    
    def calibrate_temperature(
        self,
        val_loader,
        device: torch.device,
        max_iter: int = 50
    ):
        """
        Calibrate temperature parameter using validation set.
        
        Args:
            val_loader: Validation data loader
            device: Device to run calibration on
            max_iter: Maximum optimization iterations
        """
        self.eval()
        
        # Collect validation predictions and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                binary_labels = batch["binary_label"].to(device)
                
                outputs = self.forward(input_ids, attention_mask)
                all_logits.append(outputs["binary_logits"])
                all_labels.append(binary_labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            loss = F.cross_entropy(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Calibrated temperature: {self.temperature.item():.4f}")
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        config_dict = {
            "model_name": self.model_name,
            "num_failure_modes": self.num_failure_modes,
            "use_specialized_attention": self.use_specialized_attention,
            "hidden_size": self.config.hidden_size,
            "temperature": self.temperature.item()
        }
        
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory."""
        import json
        import os
        
        # Load configuration
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(
            model_name=config_dict["model_name"],
            num_failure_modes=config_dict["num_failure_modes"],
            use_specialized_attention=config_dict["use_specialized_attention"]
        )
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        # Set temperature
        model.temperature.data.fill_(config_dict.get("temperature", 1.0))
        
        return model
