from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import load_grid_data, select_scenarios
from .features import build_local_graph_dataset, fit_feature_state
from .train import evaluate_model, resolve_device, train_model


def _train_val_partition(scenarios: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(scenarios)
    n_val = max(1, int(round(val_fraction * len(order))))
    n_val = min(n_val, max(1, len(order) - 1))
    return order[n_val:], order[:n_val]


def cross_validate(
    data_path: str | Path,
    folds: int = 5,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, object]:
    """Scenario-grouped K-fold evaluation with fold-local feature statistics."""

    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    grid = load_grid_data(data_path)
    scenario_indices = np.arange(grid.num_scenarios)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=train_config.seed)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_val_pos, test_pos) in enumerate(splitter.split(scenario_indices), start=1):
        train_val = scenario_indices[train_val_pos]
        test_ids = scenario_indices[test_pos]
        train_ids, val_ids = _train_val_partition(
            train_val,
            train_config.val_fraction / (train_config.train_fraction + train_config.val_fraction),
            train_config.seed + fold,
        )
        state = fit_feature_state(grid, train_ids, model_config)
        full_set = build_local_graph_dataset(grid, state, model_config)
        train_set = select_scenarios(full_set, train_ids)
        val_set = select_scenarios(full_set, val_ids)
        test_set = select_scenarios(full_set, test_ids)
        fold_train_config = replace(train_config, seed=train_config.seed + fold)
        model, _ = train_model(train_set, val_set, model_config, fold_train_config)
        loader = DataLoader(test_set, batch_size=train_config.batch_size, shuffle=False)
        metrics = evaluate_model(model, loader, resolve_device(train_config.device))
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)

    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_metrics])),
            "std": float(np.std([row[name] for row in fold_metrics], ddof=1)),
        }
        for name in metric_names
    }
    return {"folds": fold_metrics, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate NAMO-GNN by operating scenario.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="auto", type=str)
    args = parser.parse_args()

    result = cross_validate(
        args.data,
        folds=args.folds,
        train_config=TrainConfig(seed=args.seed, device=args.device),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
