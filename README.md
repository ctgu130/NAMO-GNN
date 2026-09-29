# NAMO-GNN

> **Neighborhood-Aware Multi-Order Graph Neural Network for Relative Transmission-Grid Node Risk Classification**

<p align="center">
  <b>Scenario-conditioned node modeling · Local multi-order graph learning · Hidden-channel recalibration</b>
</p>

---

## 🧭 Overview

NAMO-GNN implements the transmission-grid node risk classification pipeline described in the accompanying manuscript. Each learning sample is an **operating-scenario–central-node pair**. Seven operational and structural node indicators are complemented by two neighborhood statistics and one cross-scenario variability statistic. A bounded local subgraph is then processed by first-, second-, and third-order graph propagation, hidden-channel recalibration, central-node readout, and three-class prediction.

The implementation is organized to keep all statistics that depend on the data distribution inside the training partition. This includes feature scaling, node variability, and the two score quantiles used to define the low/medium/high classes.

## 🧠 Method correspondence

| Manuscript component | Implementation |
|---|---|
| Seven node indicators | `basic_features[..., 7]` |
| Neighborhood concentration | Mean state summary over retained direct neighbors |
| Neighborhood disparity | Mean absolute central–neighbor state difference |
| Historical variability | Coefficient of variation over training scenarios |
| Local domain | Central node + at most six degree-ranked direct neighbors |
| Graph orders | 1, 2, and 3 within the fixed induced subgraph |
| Input scaling | Identity-preserving entropy-record initialization with a trainable multiplicative correction |
| Hidden recalibration | Mask-aware squeeze-and-excitation block |
| Readout | Central node at local index 0 |
| Labels | 0.33 and 0.66 quantiles of the ten-indicator composite score |
| Objective | Class-weighted cross-entropy |

The manuscript reports two graph-convolutional layers and gives the order-specific operator in Eq. (7). The executable implementation applies the stated order-specific propagation in the first layer and a first-order local refinement in the second layer while keeping the selected local node set fixed. The squeeze-and-excitation reduction ratio is set to 4 because the manuscript does not state a ratio explicitly.

## 📁 Project structure

```text
namognn/
├── __init__.py
├── config.py      # Architecture, feature, and optimization settings
├── data.py        # Dataset validation and local-graph sample containers
├── features.py    # Fold-local statistics, ten-indicator construction, labels
├── model.py       # Multi-order graph network and channel recalibration
├── train.py       # Optimization, fixed-epoch training, checkpoint export
└── evaluate.py    # Scenario-grouped K-fold evaluation
```


## 📈 Reported manuscript results

The manuscript reports the following performance at **40% renewable penetration under strong intermittency**:

| Model | Accuracy (%) | Macro-Precision (%) | Macro-Recall (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| CNN | 86.4 | 85.1 | 84.6 | 84.8 |
| LSTM | 87.1 | 86.2 | 85.7 | 85.9 |
| GCN | 90.1 | 89.2 | 88.9 | 89.0 |
| GAT | 91.3 | 90.5 | 90.1 | 90.3 |
| MS-GCN | 92.4 | 91.6 | 91.2 | 91.4 |
| **NAMO-GNN** | **94.6** | **94.1** | **93.8** | **93.9** |

These values are manuscript-reported results. Reproducing them requires the corresponding converged scenario archive and the exact indicator-direction configuration used in the study.

## 📦 Data interface

Store the converged operating scenarios in a compressed NumPy archive:

```python
np.savez_compressed(
    "grid_data.npz",
    basic_features=basic_features,  # float32, [S, N, 7]
    adjacency=adjacency,            # float32, [N, N]
    scenario_ids=scenario_ids,      # int64, [S], optional
)
```

`basic_features` must follow one fixed seven-indicator order across all scenarios and nodes. The default entropy records assume the manuscript order:

1. line loading ratio;
2. power-limit violation risk index;
3. power-flow transfer impact;
4. voltage-limit violation risk index;
5. improved voltage stability index;
6. node degree centrality;
7. electrical betweenness.

All seven channels must be assigned a common risk direction before model fitting. `ModelConfig.risk_directions` uses `+1` when a larger normalized value denotes higher risk and `-1` when the direction must be inverted. The manuscript does not state the complete direction vector, so this field should be set to the indicator definitions used by the experiment.

## ⚙️ Installation

A minimal environment requires Python 3.10+ with PyTorch, NumPy, and scikit-learn.

```bash
pip install torch numpy scikit-learn
```

## 🚀 Training

```bash
python -m namognn.train \
  --data /path/to/grid_data.npz \
  --output checkpoints/namognn.pt \
  --seed 2026
```

Default settings follow the manuscript where explicitly reported: learning rate `1e-3`, batch size `512`, 20 epochs, two graph-convolutional layers, branch width `64`, dropout `0.2`, graph orders `{1,2,3}`, and neighbor limit `L=6`.

## 📊 Scenario-grouped cross-validation

```bash
python -m namognn.evaluate \
  --data /path/to/grid_data.npz \
  --folds 5 \
  --seed 2026
```

Evaluation is grouped at the **scenario level**. Every fold refits normalization bounds, historical variability, supplemental-feature scaling, and score thresholds using training scenarios only. This prevents statistics from the held-out scenarios from entering feature construction or label calibration.

The output reports fold-wise and aggregate values for:

- accuracy;
- macro-precision;
- macro-recall;
- macro-F1.

## 🔬 Reproducibility protocol

- Fixed random seeds control NumPy and PyTorch initialization.
- Scenario indices, rather than individual node samples, define validation and test partitions in cross-validation.
- Historical variability is computed only from the training scenarios of each split.
- Quantile thresholds are fitted only on training composite scores.
- Class weights are computed from the training labels of each split.
- Training uses the fixed epoch budget reported in the manuscript; validation metrics are reported without replacing the final state.
- Gradient clipping limits unstable updates without changing the classification objective.

## 🧪 Engineering verification

The release was checked in three consecutive implementation-review cycles. Each cycle included Python compilation, tensor-shape checks, fold-local preprocessing checks, finite-value checks, and scenario-grouped cross-validation on a controlled numerical testbed. The successive revisions addressed data-partition leakage, padded-node handling, fold-level class weighting, fixed-epoch state handling, and deterministic split behavior. The reported manuscript metrics are **not** reproduced by these checks because the manuscript simulation archive is not included in this package.

## 🧩 Checkpoint contents

A training checkpoint stores:

- model parameters;
- architecture configuration;
- optimization configuration;
- training-fitted feature bounds;
- node variability coefficients;
- class thresholds;
- retained local neighborhoods.

This keeps inference-time preprocessing aligned with the fitted model.

## 📌 Scope

The code implements relative three-class node risk prediction under the scoring-rule supervision described in the manuscript. The classes remain relative to the training distribution and should not be interpreted as direct replacements for power-flow constraints, N−1 security assessment, or optimal-power-flow feasibility checks.

---

### Citation

When the manuscript becomes publicly citable, replace this section with its final bibliographic entry and archival identifier.
