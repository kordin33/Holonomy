"""
Training package - Training loops, losses, and utilities
"""

from .trainer import Trainer, TrainingMetrics
from .losses import DeepfakeLoss, FocalLoss, LabelSmoothingLoss
from .optimizers import get_optimizer, get_scheduler

__all__ = [
    "Trainer",
    "TrainingMetrics",
    "DeepfakeLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "get_optimizer",
    "get_scheduler",
]
