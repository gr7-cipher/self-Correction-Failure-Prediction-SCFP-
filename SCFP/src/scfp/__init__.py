"""
Self-Correction Failure Prediction (SCFP) Framework

A comprehensive framework for predicting when and how Large Language Models
will fail during intrinsic self-correction processes.
"""

__version__ = "1.0.0"
__author__ = "Shahed Almobydeen, Gaith Rjoub, Jamal Bentahar, Ahmad Irjoob"
__email__ = "salmobydeen@aut.edu.jo"

from .models import (
    DeBERTaFailurePredictor,
    BaselineModels,
    FailureTaxonomy
)

from .data import (
    SCFPDataset,
    SyntheticDataGenerator,
    CorrectionTrace
)

from .training import (
    Trainer,
    TrainingConfig,
    EvaluationMetrics
)

from .routing import (
    DynamicRouter,
    CostModel,
    RoutingStrategy
)

__all__ = [
    # Models
    "DeBERTaFailurePredictor",
    "BaselineModels", 
    "FailureTaxonomy",
    
    # Data
    "SCFPDataset",
    "SyntheticDataGenerator",
    "CorrectionTrace",
    
    # Training
    "Trainer",
    "TrainingConfig",
    "EvaluationMetrics",
    
    # Routing
    "DynamicRouter",
    "CostModel",
    "RoutingStrategy",
]
