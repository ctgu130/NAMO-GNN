from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import LocalGraphDataset, load_grid_data, select_scenarios
from .features import build_local_graph_dataset, fit_feature_state
from .model import NAMOGNN


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    weights = labels.numel() / (num_classes * counts)
    return weights / weights.mean()


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["features"].to(device),
                batch["adjacency"].to(device),
                batch["mask"].to(device),
            )
            truth.append(batch["label"].numpy())
            prediction.append(logits.argmax(dim=-1).cpu().numpy())
    return metrics_from_arrays(np.concatenate(truth), np.concatenate(prediction))


def train_model(
    train_set: LocalGraphDataset,
    val_set: LocalGraphDataset,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> tuple[NAMOGNN, dict[str, float]]:
    """Optimize one model for the fixed epoch budget reported by the method."""

    set_reproducibility(train_config.seed)
    device = resolve_device(train_config.device)
    model = NAMOGNN(model_config).to(device)
    weights = class_weights(train_set.labels, model_config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    for _ in range(train_config.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["features"].to(device),
                batch["adjacency"].to(device),
                batch["mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()

    validation = evaluate_model(model, val_loader, device)
    return model, validation


def scenario_split(num_scenarios: int, config: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(num_scenarios)
    n_train = max(1, int(round(config.train_fraction * num_scenarios)))
    n_val = max(1, int(round(config.val_fraction * num_scenarios)))
    if n_train + n_val >= num_scenarios:
        n_train = max(1, num_scenarios - 2)
        n_val = 1
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def save_checkpoint(
    path: Path,
    model: NAMOGNN,
    model_config: ModelConfig,
    train_config: TrainConfig,
    feature_state,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "feature_state": {
            "basic_min": feature_state.basic_min,
            "basic_max": feature_state.basic_max,
            "supplemental_min": feature_state.supplemental_min,
            "supplemental_max": feature_state.supplemental_max,
            "variability": feature_state.variability,
            "thresholds": feature_state.thresholds,
            "retained_neighbors": feature_state.retained_neighbors,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAMO-GNN on transmission-grid node indicators.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default=Path("checkpoints/namognn.pt"), type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="auto", type=str)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig(seed=args.seed, device=args.device)
    grid = load_grid_data(args.data)
    train_ids, val_ids, test_ids = scenario_split(grid.num_scenarios, train_config)
    state = fit_feature_state(grid, train_ids, model_config)
    full_set = build_local_graph_dataset(grid, state, model_config)
    train_set = select_scenarios(full_set, train_ids)
    val_set = select_scenarios(full_set, val_ids)
    test_set = select_scenarios(full_set, test_ids)

    model, validation = train_model(train_set, val_set, model_config, train_config)
    device = resolve_device(train_config.device)
    test_loader = DataLoader(test_set, batch_size=train_config.batch_size, shuffle=False)
    test_metrics = evaluate_model(model, test_loader, device)
    save_checkpoint(args.output, model, model_config, train_config, state)
    print(json.dumps({"validation": validation, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
