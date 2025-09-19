"""
Data loading utilities for SCFP framework.
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Tuple, Optional
import torch

from .dataset import SCFPDataset


def create_dataloaders(
    train_dataset: SCFPDataset,
    val_dataset: SCFPDataset,
    test_dataset: Optional[SCFPDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        use_weighted_sampling: Whether to use weighted sampling for training
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        # Calculate sample weights based on class distribution
        class_weights = train_dataset.get_class_weights()
        sample_weights = []
        
        for i in range(len(train_dataset)):
            trace = train_dataset.traces[i]
            weight = class_weights[1] if trace.is_success else class_weights[0]
            sample_weights.append(weight)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    # Create validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Create test loader if test dataset provided
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader


def create_inference_dataloader(
    dataset: SCFPDataset,
    batch_size: int = 64,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for inference/prediction.
    
    Args:
        dataset: Dataset for inference
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        DataLoader for inference
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
