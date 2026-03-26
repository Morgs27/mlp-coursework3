"""Publication-style plotting helpers for evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#17202A",
    "steel": "#3F566B",
    "blue": "#2E86AB",
    "teal": "#2A9D8F",
    "sand": "#E9C46A",
    "coral": "#E76F51",
    "paper": "#F7F4EA",
}


def _setup_axes() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = PALETTE["paper"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = PALETTE["steel"]
    plt.rcParams["axes.labelcolor"] = PALETTE["ink"]
    plt.rcParams["xtick.color"] = PALETTE["ink"]
    plt.rcParams["ytick.color"] = PALETTE["ink"]


def save_dataset_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_axes()
    output_path = Path(output_path)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    match_counts = df["sport_event_id"].value_counts().sort_index()
    axes[0].bar(match_counts.index, match_counts.values, color=PALETTE["blue"])
    axes[0].set_title("Samples Per Match")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=30)

    segment_counts = df["segment_label"].value_counts().sort_values(ascending=False)
    axes[1].bar(segment_counts.index, segment_counts.values, color=PALETTE["teal"])
    axes[1].set_title("Samples Per Segment")
    axes[1].tick_params(axis="x", rotation=45)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    _setup_axes()
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, y_pred, s=48, color=PALETTE["blue"], edgecolors=PALETTE["ink"], alpha=0.85)
    min_value = float(min(np.min(y_true), np.min(y_pred)))
    max_value = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([min_value, max_value], [min_value, max_value], color=PALETTE["coral"], linewidth=2, linestyle="--")
    axis.set_title("Predicted vs Actual Score")
    axis.set_xlabel("Actual Score")
    axis.set_ylabel("Predicted Score")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: str | Path) -> None:
    _setup_axes()
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="YlGnBu")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted Segment")
    axis.set_ylabel("True Segment")
    axis.set_title("Segment Classification Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center", color=PALETTE["ink"])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
