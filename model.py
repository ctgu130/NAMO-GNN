from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class EntropyPriorScaler(nn.Module):
    """Identity-preserving input scaling initialized from the reported entropy records."""

    def __init__(self, prior: tuple[float, ...]) -> None:
        super().__init__()
        initial = torch.tensor(prior, dtype=torch.float32)
        self.register_buffer("prior", initial)
        self.log_correction = nn.Parameter(torch.zeros_like(initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.prior * torch.exp(self.log_correction)
        return x * scale


def normalize_adjacency(adjacency: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Add valid-node self loops and apply symmetric degree normalization."""

    valid = mask.to(adjacency.dtype)
    adjacency = adjacency * valid[:, :, None] * valid[:, None, :]
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency + eye[None, :, :] * valid[:, :, None]
    degree = adjacency.sum(dim=-1).clamp_min(epsilon)
    inv_sqrt = degree.rsqrt()
    normalized = adjacency * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
    return normalized * valid[:, :, None] * valid[:, None, :]


def matrix_power_batch(matrix: torch.Tensor, order: int) -> torch.Tensor:
    """Compute a positive integer power for each matrix in a batch."""

    if order == 1:
        return matrix
    result = matrix
    for _ in range(1, order):
        result = torch.bmm(result, matrix)
    return result


class MultiOrderBranch(nn.Module):
    """Order-specific propagation followed by local first-order refinement."""

    def __init__(self, in_dim: int, hidden_dim: int, order: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.order = order
        self.layers = layers
        self.input_map = nn.Linear(in_dim, hidden_dim, bias=False)
        self.refine = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(max(0, layers - 1))
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        order_adj = matrix_power_batch(adjacency, self.order)
        h = torch.bmm(order_adj, x)
        h = self.dropout(self.activation(self.input_map(h)))
        for layer in self.refine:
            h = torch.bmm(adjacency, h)
            h = self.dropout(self.activation(layer(h)))
        return h


class MaskedSqueezeExcitation(nn.Module):
    """Channel recalibration using descriptors pooled over valid local nodes."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.to(x.dtype).unsqueeze(-1)
        descriptor = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weights = self.gate(self.expand(self.activation(self.reduce(descriptor))))
        return x * weights.unsqueeze(1)


class NAMOGNN(nn.Module):
    """Neighborhood-aware multi-order graph classifier for relative node risk."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_scaler = EntropyPriorScaler(config.entropy_prior)
        self.branches = nn.ModuleList(
            MultiOrderBranch(
                config.num_input_features,
                config.hidden_width,
                order,
                config.graph_layers,
                config.dropout,
            )
            for order in config.aggregation_orders
        )
        channels = config.hidden_width * len(config.aggregation_orders)
        self.recalibration = MaskedSqueezeExcitation(channels, config.se_reduction)
        self.classifier = nn.Linear(channels, config.num_classes)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = normalize_adjacency(adjacency, mask, self.config.epsilon)
        scaled = self.input_scaler(features)
        branch_features = [branch(scaled, normalized) for branch in self.branches]
        fused = torch.cat(branch_features, dim=-1)
        recalibrated = self.recalibration(fused, mask)
        center_vector = recalibrated[:, 0, :]
        return self.classifier(center_vector)
