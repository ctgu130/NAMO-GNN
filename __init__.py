"""Neighborhood-Aware Multi-Order Graph Neural Network for grid-node risk classification."""

from .config import ModelConfig, TrainConfig
from .model import NAMOGNN

__all__ = ["ModelConfig", "TrainConfig", "NAMOGNN"]
__version__ = "1.0.0"
