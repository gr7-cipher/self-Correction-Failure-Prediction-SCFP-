"""
Loss functions for SCFP multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for SCFP framework.
    
    Combines binary classification loss (success/failure), multiclass 
    classification loss (failure modes), and optional confidence loss.
    """
    
    def __init__(
        self,
        binary_weight: float = 1.0,
        multiclass_weight: float = 1.0,
        confidence_weight: float = 0.5,
        binary_class_weights: Optional[torch.Tensor] = None,
        multiclass_class_weights: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ):
        """
        Initialize multi-task loss.
        
        Args:
            binary_weight: Weight for binary classification loss
            multiclass_weight: Weight for multiclass classification loss
            confidence_weight: Weight for confidence loss
            binary_class_weights: Class weights for binary classification
            multiclass_class_weights: Class weights for multiclass classification
            temperature: Temperature for loss scaling
        """
        super().__init__()
        
        self.binary_weight = binary_weight
        self.multiclass_weight = multiclass_weight
        self.confidence_weight = confidence_weight
        self.temperature = temperature
        
        # Binary classification loss
        self.binary_loss_fn = nn.CrossEntropyLoss(weight=binary_class_weights)
        
        # Multiclass classification loss
        self.multiclass_loss_fn = nn.CrossEntropyLoss(weight=multiclass_class_weights)
        
        # Confidence loss (MSE between predicted confidence and accuracy)
        self.confidence_loss_fn = nn.MSELoss()
    
    def forward(
        self,
        binary_logits: torch.Tensor,
        multiclass_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        multiclass_labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            binary_logits: Binary classification logits [batch_size, 2]
            multiclass_logits: Multiclass classification logits [batch_size, num_classes]
            binary_labels: Binary labels [batch_size]
            multiclass_labels: Multiclass labels [batch_size]
            confidence: Predicted confidence scores [batch_size, 1] (optional)
        
        Returns:
            Dictionary with individual and total losses
        """
        # Scale logits by temperature
        binary_logits_scaled = binary_logits / self.temperature
        multiclass_logits_scaled = multiclass_logits / self.temperature
        
        # Binary classification loss
        binary_loss = self.binary_loss_fn(binary_logits_scaled, binary_labels)
        
        # Multiclass classification loss
        multiclass_loss = self.multiclass_loss_fn(multiclass_logits_scaled, multiclass_labels)
        
        # Initialize total loss
        total_loss = (
            self.binary_weight * binary_loss + 
            self.multiclass_weight * multiclass_loss
        )
        
        loss_dict = {
            "binary_loss": binary_loss,
            "multiclass_loss": multiclass_loss,
            "total_loss": total_loss
        }
        
        # Add confidence loss if confidence predictions are provided
        if confidence is not None:
            # Calculate target confidence as binary prediction accuracy
            with torch.no_grad():
                binary_probs = F.softmax(binary_logits_scaled, dim=-1)
                binary_preds = torch.argmax(binary_probs, dim=-1)
                target_confidence = (binary_preds == binary_labels).float().unsqueeze(-1)
            
            confidence_loss = self.confidence_loss_fn(confidence, target_confidence)
            total_loss += self.confidence_weight * confidence_loss
            
            loss_dict["confidence_loss"] = confidence_loss
            loss_dict["total_loss"] = total_loss
        
        return loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    where pt is the model's estimated probability for the true class.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]
        
        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss to prevent overconfidence.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (default: 0.1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]
        
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), self.confidence)
        targets_one_hot += self.smoothing / self.num_classes
        
        # Compute loss
        loss = -torch.sum(targets_one_hot * log_probs, dim=-1)
        
        return loss.mean()


class UncertaintyLoss(nn.Module):
    """
    Uncertainty-aware loss that incorporates prediction uncertainty.
    """
    
    def __init__(self, base_loss_fn: nn.Module, uncertainty_weight: float = 1.0):
        """
        Initialize uncertainty loss.
        
        Args:
            base_loss_fn: Base loss function (e.g., CrossEntropyLoss)
            uncertainty_weight: Weight for uncertainty regularization
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            uncertainty: Predicted uncertainty [batch_size, 1]
        
        Returns:
            Uncertainty-aware loss
        """
        # Base loss
        base_loss = self.base_loss_fn(inputs, targets)
        
        # Uncertainty regularization
        # Encourage low uncertainty for correct predictions
        probs = F.softmax(inputs, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)
        correct_mask = (pred_labels == targets).float()
        
        # Uncertainty should be low when prediction is correct
        uncertainty_reg = torch.mean(uncertainty.squeeze() * correct_mask)
        
        # Total loss
        total_loss = base_loss + self.uncertainty_weight * uncertainty_reg
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better representations.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            temperature: Temperature for scaling
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Labels [batch_size]
        
        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create label mask
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute positive and negative similarities
        pos_similarity = similarity_matrix * mask
        neg_similarity = similarity_matrix * (1 - mask)
        
        # Compute loss
        pos_loss = -torch.log(torch.exp(pos_similarity).sum(dim=1) + 1e-8)
        neg_loss = torch.log(torch.exp(neg_similarity).sum(dim=1) + 1e-8)
        
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.
    """
    
    def __init__(
        self,
        binary_loss_fn: nn.Module,
        multiclass_loss_fn: nn.Module,
        initial_binary_weight: float = 1.0,
        initial_multiclass_weight: float = 1.0,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize adaptive loss.
        
        Args:
            binary_loss_fn: Binary classification loss function
            multiclass_loss_fn: Multiclass classification loss function
            initial_binary_weight: Initial weight for binary loss
            initial_multiclass_weight: Initial weight for multiclass loss
            adaptation_rate: Rate of weight adaptation
        """
        super().__init__()
        
        self.binary_loss_fn = binary_loss_fn
        self.multiclass_loss_fn = multiclass_loss_fn
        self.adaptation_rate = adaptation_rate
        
        # Learnable weights
        self.binary_weight = nn.Parameter(torch.tensor(initial_binary_weight))
        self.multiclass_weight = nn.Parameter(torch.tensor(initial_multiclass_weight))
    
    def forward(
        self,
        binary_logits: torch.Tensor,
        multiclass_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        multiclass_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive multi-task loss.
        
        Args:
            binary_logits: Binary classification logits
            multiclass_logits: Multiclass classification logits
            binary_labels: Binary labels
            multiclass_labels: Multiclass labels
        
        Returns:
            Dictionary with losses
        """
        # Compute individual losses
        binary_loss = self.binary_loss_fn(binary_logits, binary_labels)
        multiclass_loss = self.multiclass_loss_fn(multiclass_logits, multiclass_labels)
        
        # Apply adaptive weights
        weighted_binary_loss = torch.exp(-self.binary_weight) * binary_loss + self.binary_weight
        weighted_multiclass_loss = torch.exp(-self.multiclass_weight) * multiclass_loss + self.multiclass_weight
        
        total_loss = weighted_binary_loss + weighted_multiclass_loss
        
        return {
            "binary_loss": binary_loss,
            "multiclass_loss": multiclass_loss,
            "weighted_binary_loss": weighted_binary_loss,
            "weighted_multiclass_loss": weighted_multiclass_loss,
            "total_loss": total_loss,
            "binary_weight": self.binary_weight,
            "multiclass_weight": self.multiclass_weight
        }
