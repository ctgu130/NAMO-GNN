from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and feature-construction settings."""

    num_basic_features: int = 7
    num_input_features: int = 10
    neighbor_limit: int = 6
    hidden_width: int = 64
    aggregation_orders: Tuple[int, ...] = (1, 2, 3)
    graph_layers: int = 2
    se_reduction: int = 4
    dropout: float = 0.2
    num_classes: int = 3
    quantiles: Tuple[float, float] = (0.33, 0.66)
    epsilon: float = 1.0e-8
    risk_directions: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)
    entropy_prior: Tuple[float, ...] = (
        0.3021,
        0.2322,
        0.0988,
        0.0769,
        0.0634,
        0.0460,
        0.0450,
        0.0395,
        0.0679,
        0.0384,
    )

    def validate(self) -> None:
        if self.num_basic_features != 7:
            raise ValueError("The method expects seven basic node indicators.")
        if self.num_input_features != 10:
            raise ValueError("The method expects ten input indicators.")
        if len(self.risk_directions) != self.num_basic_features:
            raise ValueError("risk_directions must contain seven entries.")
        if any(v not in (-1, 1) for v in self.risk_directions):
            raise ValueError("Each risk direction must be either +1 or -1.")
        if len(self.entropy_prior) != self.num_input_features:
            raise ValueError("entropy_prior must contain ten entries.")
        if min(self.aggregation_orders) < 1:
            raise ValueError("Aggregation orders must be positive integers.")
        if self.graph_layers < 1:
            raise ValueError("graph_layers must be at least one.")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, partitioning, and reproducibility settings."""

    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 20
    gradient_clip_norm: float = 5.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 2026
    num_workers: int = 0
    device: str = "auto"
