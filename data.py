from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GridData:
    """Scenario-level node indicators and a fixed transmission-grid topology."""

    basic_features: np.ndarray
    adjacency: np.ndarray
    scenario_ids: np.ndarray

    @property
    def num_scenarios(self) -> int:
        return int(self.basic_features.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.basic_features.shape[1])

    def validate(self) -> None:
        if self.basic_features.ndim != 3 or self.basic_features.shape[2] != 7:
            raise ValueError("basic_features must have shape [scenario, node, 7].")
        if self.adjacency.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("adjacency must have shape [node, node].")
        if self.scenario_ids.shape != (self.num_scenarios,):
            raise ValueError("scenario_ids must have one entry per scenario.")
        if not np.all(np.isfinite(self.basic_features)):
            raise ValueError("basic_features contains non-finite values.")
        if not np.all(np.isfinite(self.adjacency)):
            raise ValueError("adjacency contains non-finite values.")
        if not np.allclose(self.adjacency, self.adjacency.T):
            raise ValueError("adjacency must be symmetric for the stated transmission-grid formulation.")
        if np.any(self.adjacency < 0):
            raise ValueError("adjacency entries must be non-negative.")


def load_grid_data(path: str | Path) -> GridData:
    """Load a compressed NumPy archive with basic_features and adjacency arrays."""

    archive = np.load(Path(path), allow_pickle=False)
    if "basic_features" not in archive or "adjacency" not in archive:
        raise KeyError("The archive must contain basic_features and adjacency arrays.")
    basic_features = np.asarray(archive["basic_features"], dtype=np.float32)
    adjacency = np.asarray(archive["adjacency"], dtype=np.float32)
    scenario_ids = (
        np.asarray(archive["scenario_ids"], dtype=np.int64)
        if "scenario_ids" in archive
        else np.arange(basic_features.shape[0], dtype=np.int64)
    )
    data = GridData(basic_features=basic_features, adjacency=adjacency, scenario_ids=scenario_ids)
    data.validate()
    return data


class LocalGraphDataset(Dataset):
    """Padded local-subgraph samples centered on individual grid nodes."""

    def __init__(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        scenario_index: np.ndarray,
        center_node: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.scenario_index = torch.as_tensor(scenario_index, dtype=torch.long)
        self.center_node = torch.as_tensor(center_node, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "adjacency": self.adjacency[index],
            "mask": self.mask[index],
            "label": self.labels[index],
            "scenario_index": self.scenario_index[index],
            "center_node": self.center_node[index],
        }


def select_scenarios(dataset: LocalGraphDataset, scenario_indices: Sequence[int]) -> LocalGraphDataset:
    """Return samples whose scenario index belongs to the requested subset."""

    wanted = torch.as_tensor(list(scenario_indices), dtype=torch.long)
    keep = (dataset.scenario_index[:, None] == wanted[None, :]).any(dim=1)
    ids = torch.nonzero(keep, as_tuple=False).flatten()
    return LocalGraphDataset(
        dataset.features[ids].numpy(),
        dataset.adjacency[ids].numpy(),
        dataset.mask[ids].numpy(),
        dataset.labels[ids].numpy(),
        dataset.scenario_index[ids].numpy(),
        dataset.center_node[ids].numpy(),
    )
