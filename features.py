from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ModelConfig
from .data import GridData, LocalGraphDataset


@dataclass(frozen=True)
class FeatureState:
    """Statistics fitted exclusively on training scenarios."""

    basic_min: np.ndarray
    basic_max: np.ndarray
    supplemental_min: np.ndarray
    supplemental_max: np.ndarray
    variability: np.ndarray
    thresholds: np.ndarray
    retained_neighbors: tuple[tuple[int, ...], ...]


def _safe_minmax(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, epsilon: float) -> np.ndarray:
    scaled = (values - lower) / np.maximum(upper - lower, epsilon)
    return np.clip(scaled, 0.0, 1.0)


def _align_risk_direction(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    aligned = values.copy()
    inverse = directions < 0
    aligned[..., inverse] = 1.0 - aligned[..., inverse]
    return aligned


def _retained_neighbors(adjacency: np.ndarray, limit: int) -> tuple[tuple[int, ...], ...]:
    degree = adjacency.sum(axis=1)
    neighborhoods: list[tuple[int, ...]] = []
    for node in range(adjacency.shape[0]):
        neighbors = np.flatnonzero(adjacency[node] > 0)
        order = sorted(neighbors.tolist(), key=lambda j: (-float(degree[j]), int(j)))
        neighborhoods.append(tuple(order[:limit]))
    return tuple(neighborhoods)


def _supplemental_features(
    basic_scaled: np.ndarray,
    retained: tuple[tuple[int, ...], ...],
    variability: np.ndarray,
) -> np.ndarray:
    scenarios, nodes, _ = basic_scaled.shape
    state_summary = basic_scaled.mean(axis=-1)
    concentration = np.empty((scenarios, nodes), dtype=np.float32)
    disparity = np.empty((scenarios, nodes), dtype=np.float32)
    for node, neighbors in enumerate(retained):
        if len(neighbors) == 0:
            concentration[:, node] = state_summary[:, node]
            disparity[:, node] = 0.0
            continue
        local = state_summary[:, list(neighbors)]
        concentration[:, node] = local.mean(axis=1)
        disparity[:, node] = np.abs(local - state_summary[:, node, None]).mean(axis=1)
    variability_block = np.broadcast_to(variability[None, :], (scenarios, nodes))
    return np.stack([concentration, disparity, variability_block], axis=-1).astype(np.float32)


def fit_feature_state(data: GridData, train_scenarios: Sequence[int], config: ModelConfig) -> FeatureState:
    """Fit normalization, historical variability, and class thresholds on training scenarios."""

    config.validate()
    train_idx = np.asarray(train_scenarios, dtype=np.int64)
    train_basic = data.basic_features[train_idx]
    basic_min = train_basic.min(axis=(0, 1))
    basic_max = train_basic.max(axis=(0, 1))
    directions = np.asarray(config.risk_directions, dtype=np.int64)

    all_basic = _safe_minmax(data.basic_features, basic_min, basic_max, config.epsilon)
    all_basic = _align_risk_direction(all_basic, directions)
    train_scaled = all_basic[train_idx]
    train_state_summary = train_scaled.mean(axis=-1)
    mean = train_state_summary.mean(axis=0)
    std = train_state_summary.std(axis=0, ddof=0)
    variability = std / (np.abs(mean) + config.epsilon)

    retained = _retained_neighbors(data.adjacency, config.neighbor_limit)
    all_supplemental = _supplemental_features(all_basic, retained, variability)
    train_supplemental = all_supplemental[train_idx]
    supplemental_min = train_supplemental.min(axis=(0, 1))
    supplemental_max = train_supplemental.max(axis=(0, 1))
    all_supplemental_scaled = _safe_minmax(
        all_supplemental, supplemental_min, supplemental_max, config.epsilon
    )
    all_features = np.concatenate([all_basic, all_supplemental_scaled], axis=-1)

    weights = np.asarray(config.entropy_prior, dtype=np.float32)
    score = np.tensordot(all_features, weights, axes=([-1], [0])) / weights.sum()
    train_scores = score[train_idx].reshape(-1)
    thresholds = np.quantile(train_scores, np.asarray(config.quantiles, dtype=np.float64)).astype(np.float32)

    return FeatureState(
        basic_min=basic_min.astype(np.float32),
        basic_max=basic_max.astype(np.float32),
        supplemental_min=supplemental_min.astype(np.float32),
        supplemental_max=supplemental_max.astype(np.float32),
        variability=variability.astype(np.float32),
        thresholds=thresholds,
        retained_neighbors=retained,
    )


def transform_features(data: GridData, state: FeatureState, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Apply fitted statistics and return ten-dimensional features and class labels."""

    directions = np.asarray(config.risk_directions, dtype=np.int64)
    basic = _safe_minmax(data.basic_features, state.basic_min, state.basic_max, config.epsilon)
    basic = _align_risk_direction(basic, directions)
    supplemental = _supplemental_features(basic, state.retained_neighbors, state.variability)
    supplemental = _safe_minmax(
        supplemental, state.supplemental_min, state.supplemental_max, config.epsilon
    )
    features = np.concatenate([basic, supplemental], axis=-1).astype(np.float32)
    weights = np.asarray(config.entropy_prior, dtype=np.float32)
    score = np.tensordot(features, weights, axes=([-1], [0])) / weights.sum()
    labels = np.digitize(score, state.thresholds, right=False).astype(np.int64)
    return features, labels


def build_local_graph_dataset(data: GridData, state: FeatureState, config: ModelConfig) -> LocalGraphDataset:
    """Construct padded induced subgraphs with the central node at local index zero."""

    features, labels = transform_features(data, state, config)
    scenarios, nodes, channels = features.shape
    width = config.neighbor_limit + 1
    total = scenarios * nodes

    local_x = np.zeros((total, width, channels), dtype=np.float32)
    local_a = np.zeros((total, width, width), dtype=np.float32)
    local_mask = np.zeros((total, width), dtype=bool)
    local_y = np.zeros(total, dtype=np.int64)
    scenario_index = np.zeros(total, dtype=np.int64)
    center_node = np.zeros(total, dtype=np.int64)

    row = 0
    for s in range(scenarios):
        for center in range(nodes):
            local_nodes = (center,) + state.retained_neighbors[center]
            n_local = len(local_nodes)
            local_x[row, :n_local] = features[s, list(local_nodes)]
            induced = data.adjacency[np.ix_(local_nodes, local_nodes)]
            local_a[row, :n_local, :n_local] = induced
            local_mask[row, :n_local] = True
            local_y[row] = labels[s, center]
            scenario_index[row] = s
            center_node[row] = center
            row += 1

    return LocalGraphDataset(local_x, local_a, local_mask, local_y, scenario_index, center_node)
