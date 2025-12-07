"""
DermaLens - AI-powered dog skin lesion classification
"""

__version__ = "1.0.0"
__author__ = "DermaLens Team"

from src.model import DermaLensModel
from src.data_loader import DermaLensDataset, create_dataloaders
from src.train import Trainer, train_model
from src.utils import evaluate_model, get_device

__all__ = [
    "DermaLensModel",
    "DermaLensDataset",
    "create_dataloaders",
    "Trainer",
    "train_model",
    "evaluate_model",
    "get_device"
]
