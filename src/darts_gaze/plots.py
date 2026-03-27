"""Publication-style plotting helpers for evaluation outputs."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


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


def _save_figure(figure: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    alt_path = output_path.with_suffix(".png" if output_path.suffix.lower() != ".png" else ".pdf")
    figure.savefig(alt_path, bbox_inches="tight", dpi=200)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_dataset_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_axes()
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
    _save_figure(figure, output_path)
    plt.close(figure)


def save_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    _setup_axes()
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, y_pred, s=48, color=PALETTE["blue"], edgecolors=PALETTE["ink"], alpha=0.85)
    min_value = float(min(np.min(y_true), np.min(y_pred)))
    max_value = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([min_value, max_value], [min_value, max_value], color=PALETTE["coral"], linewidth=2, linestyle="--")
    axis.set_title("Predicted vs Actual Score")
    axis.set_xlabel("Actual Score")
    axis.set_ylabel("Predicted Score")
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: str | Path) -> None:
    _setup_axes()
    axis_size = max(8, min(20, len(labels) * 0.6))
    figure, axis = plt.subplots(figsize=(axis_size, axis_size * 0.8))
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
    _save_figure(figure, output_path)
    plt.close(figure)


def save_valid_face_rate_by_match(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_axes()
    match_df = df[df["sport_event_id"].notna()].copy()
    if match_df.empty:
        return
    if match_df["valid_face"].dtype != bool:
        lowered = match_df["valid_face"].astype(str).str.lower()
        match_df["valid_face"] = lowered.isin({"1", "true", "t", "yes"})

    rates = (
        match_df.groupby("sport_event_id", dropna=False)["valid_face"]
        .mean()
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(rates.index, rates.values, color=PALETTE["coral"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Valid-Face Rate")
    axis.set_title("Valid-Face Rate by Match")
    axis.tick_params(axis="x", rotation=30)
    for idx, value in enumerate(rates.values):
        axis.text(idx, min(1.02, value + 0.03), f"{value:.2f}", ha="center", va="bottom", color=PALETTE["ink"])
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_segment_imbalance(df: pd.DataFrame, output_path: str | Path, top_n: int = 10) -> None:
    _setup_axes()
    if "segment_label" not in df.columns or df.empty:
        return

    counts = df["segment_label"].value_counts()
    top_counts = counts.head(top_n)
    other_count = int(counts.iloc[top_n:].sum())
    labels = top_counts.index.tolist()
    values = top_counts.values.tolist()
    if other_count:
        labels.append("OTHER")
        values.append(other_count)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color=PALETTE["sand"])
    axis.set_title("Top Segment Frequency (Top 10 + Other)")
    axis.set_ylabel("Samples")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_gaze_trend_scatter(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str,
    title: str,
    top_n: int = 6,
) -> None:
    _setup_axes()
    required_columns = {"average_gaze_x", "average_gaze_y", "player_name", label_column}
    if df.empty or not required_columns.issubset(df.columns):
        return

    plot_df = df.copy()
    plot_df = plot_df[plot_df["average_gaze_x"].notna() & plot_df["average_gaze_y"].notna()]
    plot_df = plot_df[plot_df[label_column].notna()]
    if plot_df.empty:
        return

    counts = plot_df[label_column].value_counts()
    labels = [label for label in counts.index.tolist() if label not in {"OTHER", "MISS"}][:top_n]
    plot_df = plot_df[plot_df[label_column].isin(labels)]
    if plot_df.empty:
        return

    players = sorted(plot_df["player_name"].fillna("Unknown").unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    player_markers = {player: markers[index % len(markers)] for index, player in enumerate(players)}
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))
    label_colors = {label: color for label, color in zip(labels, colors, strict=True)}

    figure, axis = plt.subplots(figsize=(11, 9))
    for _, row in plot_df.iterrows():
        player_name = row.get("player_name") or "Unknown"
        label = row[label_column]
        axis.scatter(
            row["average_gaze_x"],
            row["average_gaze_y"],
            color=label_colors[label],
            marker=player_markers[player_name],
            s=90,
            alpha=0.75,
            edgecolors=PALETTE["ink"],
            linewidths=0.6,
            zorder=3,
        )

    centroids = plot_df.groupby(label_column)[["average_gaze_x", "average_gaze_y"]].mean()
    for label, centroid in centroids.iterrows():
        axis.scatter(
            centroid["average_gaze_x"],
            centroid["average_gaze_y"],
            color=label_colors[label],
            s=1800,
            alpha=0.18,
            marker="o",
            edgecolors="none",
            zorder=1,
        )
        axis.text(
            centroid["average_gaze_x"] + 0.008,
            centroid["average_gaze_y"],
            str(label),
            fontsize=13,
            ha="right",
            va="center",
            color=PALETTE["ink"],
            fontweight="bold",
            zorder=5,
        )

    score_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[label], markeredgecolor=PALETTE["ink"], markersize=10, label=str(label))
        for label in labels
    ]
    player_handles = [
        Line2D([0], [0], marker=player_markers[player], color="w", markerfacecolor=PALETTE["steel"], markeredgecolor=PALETTE["ink"], markersize=10, label=player)
        for player in players
    ]
    axis.legend(handles=score_handles + player_handles, title="Area And Player", loc="best")
    axis.set_title(title)
    axis.set_xlabel("Horizontal Gaze (Normalized)")
    axis.set_ylabel("Vertical Gaze (Normalized)")
    axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.invert_yaxis()
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_player_centered_gaze_trends(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str,
    title: str,
    top_n: int = 6,
) -> None:
    _setup_axes()
    required_columns = {"average_gaze_x", "average_gaze_y", "player_name", label_column}
    if df.empty or not required_columns.issubset(df.columns):
        return

    plot_df = df.copy()
    plot_df = plot_df[plot_df["average_gaze_x"].notna() & plot_df["average_gaze_y"].notna()]
    plot_df = plot_df[plot_df["player_name"].notna() & plot_df[label_column].notna()]
    if plot_df.empty:
        return

    label_counts = plot_df[label_column].value_counts()
    labels = [label for label in label_counts.index.tolist() if label not in {"OTHER", "MISS"}][:top_n]
    plot_df = plot_df[plot_df[label_column].isin(labels)]
    if plot_df.empty:
        return

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))
    label_colors = {label: color for label, color in zip(labels, colors, strict=True)}
    players = sorted(plot_df["player_name"].unique().tolist())
    center_markers = ["P", "X", "*", "D", "h", "8"]
    player_markers = ["o", "s", "^", "v", "D", "H"]
    center_marker_map = {player: center_markers[index % len(center_markers)] for index, player in enumerate(players)}
    point_marker_map = {player: player_markers[index % len(player_markers)] for index, player in enumerate(players)}

    figure, axis = plt.subplots(figsize=(11, 9))
    for player_index, player_name in enumerate(players):
        player_df = plot_df[plot_df["player_name"] == player_name]
        if player_df.empty:
            continue
        center = player_df[["average_gaze_x", "average_gaze_y"]].mean()
        axis.scatter(
            center["average_gaze_x"],
            center["average_gaze_y"],
            color="black",
            marker=center_marker_map[player_name],
            s=180,
            zorder=6,
        )
        for label in labels:
            label_df = player_df[player_df[label_column] == label]
            if label_df.empty:
                continue
            centroid = label_df[["average_gaze_x", "average_gaze_y"]].mean()
            axis.plot(
                [center["average_gaze_x"], centroid["average_gaze_x"]],
                [center["average_gaze_y"], centroid["average_gaze_y"]],
                color=label_colors[label],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                zorder=2,
            )
            axis.scatter(
                centroid["average_gaze_x"],
                centroid["average_gaze_y"],
                color=label_colors[label],
                marker=point_marker_map[player_name],
                s=220,
                alpha=0.9,
                edgecolors=PALETTE["ink"],
                linewidths=0.8,
                zorder=4,
            )
            axis.text(
                centroid["average_gaze_x"] + 0.01,
                centroid["average_gaze_y"],
                str(label),
                fontsize=12,
                ha="right",
                va="center",
                color=PALETTE["ink"],
                fontweight="bold",
                zorder=5,
            )

    score_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[label], markeredgecolor=PALETTE["ink"], markersize=10, label=str(label))
        for label in labels
    ]
    player_handles = [
        Line2D([0], [0], marker=point_marker_map[player], color="w", markerfacecolor=PALETTE["steel"], markeredgecolor=PALETTE["ink"], markersize=10, label=player)
        for player in players
    ]
    axis.legend(handles=score_handles + player_handles, title="Area And Player", loc="best", ncol=2)
    axis.set_title(title)
    axis.set_xlabel("Horizontal Gaze (Normalized)")
    axis.set_ylabel("Vertical Gaze (Normalized)")
    axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.invert_yaxis()
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_binary_probability_distribution(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    output_path: str | Path,
    *,
    positive_label: str = "20",
    negative_label: str = "19",
) -> None:
    _setup_axes()
    y_true_series = pd.Series(y_true).astype(str)
    y_prob_series = pd.Series(y_prob).astype(float)
    if y_true_series.empty or y_prob_series.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    neg_probs = y_prob_series[y_true_series == negative_label]
    pos_probs = y_prob_series[y_true_series == positive_label]

    axes[0].hist(neg_probs, bins=16, alpha=0.7, color=PALETTE["steel"], label=f"True {negative_label}")
    axes[0].hist(pos_probs, bins=16, alpha=0.7, color=PALETTE["coral"], label=f"True {positive_label}")
    axes[0].set_xlabel(f"Predicted P({positive_label})")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{negative_label} vs {positive_label} Probability Separation")
    axes[0].legend(loc="best")

    box_data = [neg_probs.to_numpy(), pos_probs.to_numpy()]
    boxplot = axes[1].boxplot(box_data, tick_labels=[negative_label, positive_label], patch_artist=True)
    for patch, color in zip(boxplot["boxes"], [PALETTE["steel"], PALETTE["coral"]], strict=False):
        patch.set_facecolor(color)
    axes[1].set_ylabel(f"Predicted P({positive_label})")
    axes[1].set_title("Probability Distribution By True Label")

    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_binary_ranking_curves(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    output_path: str | Path,
    *,
    positive_label: str = "20",
) -> None:
    _setup_axes()
    y_true_series = (pd.Series(y_true).astype(str) == positive_label).astype(int)
    y_prob_series = pd.Series(y_prob).astype(float)
    if y_true_series.nunique() < 2:
        return

    fpr, tpr, _ = roc_curve(y_true_series, y_prob_series)
    precision, recall, _ = precision_recall_curve(y_true_series, y_prob_series)
    roc_auc = roc_auc_score(y_true_series, y_prob_series)
    ap = average_precision_score(y_true_series, y_prob_series)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color=PALETTE["blue"], linewidth=2, label=f"AUC {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["steel"], alpha=0.6)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color=PALETTE["teal"], linewidth=2, label=f"AP {ap:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_match_shaped_score_scatter(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str,
    title: str,
) -> None:
    _setup_axes()
    required_columns = {"average_gaze_x", "average_gaze_y", "sport_event_id", label_column}
    if df.empty or not required_columns.issubset(df.columns):
        return

    plot_df = df.copy()
    plot_df = plot_df[plot_df["average_gaze_x"].notna() & plot_df["average_gaze_y"].notna() & plot_df[label_column].notna()]
    if plot_df.empty:
        return

    labels = plot_df[label_column].astype(str).value_counts().index.tolist()
    def label_sort_key(value: str) -> tuple[int, str]:
        return (0, f"{int(value):02d}") if value.isdigit() else (1, value)
    labels = sorted(labels, key=label_sort_key)
    matches = sorted(plot_df["sport_event_id"].astype(str).unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    match_markers = {match_id: markers[index % len(markers)] for index, match_id in enumerate(matches)}
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(labels), 1)))
    label_colors = {label: color for label, color in zip(labels, colors, strict=True)}

    figure, axis = plt.subplots(figsize=(12, 10))
    for _, row in plot_df.iterrows():
        label = str(row[label_column])
        match_id = str(row["sport_event_id"])
        axis.scatter(
            row["average_gaze_x"],
            row["average_gaze_y"],
            color=label_colors[label],
            marker=match_markers[match_id],
            s=90,
            alpha=0.82,
            edgecolors=PALETTE["ink"],
            linewidths=0.7,
            zorder=3,
        )

    centroids = plot_df.groupby(label_column)[["average_gaze_x", "average_gaze_y"]].mean()
    for label, centroid in centroids.iterrows():
        label_str = str(label)
        axis.scatter(
            centroid["average_gaze_x"],
            centroid["average_gaze_y"],
            color=label_colors[label_str],
            s=2100,
            alpha=0.2,
            marker="o",
            edgecolors="none",
            zorder=1,
        )
        axis.text(
            centroid["average_gaze_x"] + 0.01,
            centroid["average_gaze_y"],
            label_str,
            fontsize=13,
            ha="right",
            va="center",
            color=PALETTE["ink"],
            fontweight="bold",
            zorder=5,
        )

    score_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[label], markeredgecolor=PALETTE["ink"], markersize=10, label=label)
        for label in labels
    ]
    match_handles = [
        Line2D([0], [0], marker=match_markers[match_id], color="w", markerfacecolor=PALETTE["steel"], markeredgecolor=PALETTE["ink"], markersize=10, label=match_id.split(":")[-1])
        for match_id in matches
    ]
    axis.legend(handles=score_handles + match_handles, title="Score And Match", loc="best", ncol=2)
    axis.set_title(title)
    axis.set_xlabel("Horizontal Gaze (Normalized)")
    axis.set_ylabel("Vertical Gaze (Normalized)")
    axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.4)
    axis.invert_yaxis()
    axis.grid(True)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_player_score_scatter_series(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    label_column: str,
    title_prefix: str,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "player_name" not in df.columns:
        return []

    output_paths: list[Path] = []
    for player_name in sorted(df["player_name"].dropna().astype(str).unique().tolist()):
        player_df = df[df["player_name"].astype(str) == player_name].copy()
        if player_df.empty:
            continue
        output_path = output_dir / f"{_slugify(player_name)}_score_scatter.pdf"
        save_match_shaped_score_scatter(
            player_df,
            output_path,
            label_column=label_column,
            title=f"{title_prefix}: {player_name}",
        )
        output_paths.append(output_path)
    return output_paths


def save_metric_ci_panels(
    *,
    metric_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    metric_order: list[str],
    metric_labels: dict[str, str],
    lower_is_better: set[str] | None = None,
) -> None:
    _setup_axes()
    if metric_df.empty:
        return

    lower_is_better = lower_is_better or set()
    available_metrics = [metric for metric in metric_order if metric in set(metric_df["metric"].astype(str))]
    if not available_metrics:
        return

    num_metrics = len(available_metrics)
    figure, axes = plt.subplots(1, num_metrics, figsize=(5.5 * num_metrics, 6), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for axis, metric_name in zip(axes[0], available_metrics, strict=True):
        subset = metric_df[metric_df["metric"] == metric_name].copy()
        subset = subset.sort_values("estimate", ascending=metric_name in lower_is_better).reset_index(drop=True)
        y_positions = np.arange(len(subset))
        colors = [cmap(index % 10) for index in range(len(subset))]
        lower_error = (subset["estimate"] - subset["ci_low"]).clip(lower=0).to_numpy()
        upper_error = (subset["ci_high"] - subset["estimate"]).clip(lower=0).to_numpy()
        axis.errorbar(
            subset["estimate"],
            y_positions,
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            color=PALETTE["ink"],
            ecolor=PALETTE["steel"],
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
        for y_pos, (_, row), color in zip(y_positions, subset.iterrows(), colors, strict=True):
            axis.scatter(row["estimate"], y_pos, s=90, color=color, edgecolors=PALETTE["ink"], linewidths=0.8, zorder=4)
        axis.set_yticks(y_positions, labels=subset["model"].tolist())
        axis.set_title(metric_labels.get(metric_name, metric_name))
        axis.grid(True, axis="x", alpha=0.4)
        if metric_name in {"accuracy", "balanced_accuracy", "macro_f1", "three_wedge_accuracy", "roc_auc", "average_precision"}:
            axis.set_xlim(0.0, 1.02)
        axis.invert_yaxis()

    figure.suptitle(title, fontsize=15, y=1.02)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_binary_model_curves(
    *,
    y_true: pd.Series | np.ndarray,
    probabilities_by_model: dict[str, pd.Series],
    output_path: str | Path,
    positive_label: str = "20",
) -> None:
    _setup_axes()
    if not probabilities_by_model:
        return

    y_true_series = pd.Series(y_true).astype(int)
    if y_true_series.nunique() < 2:
        return

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")

    for index, (model_name, probabilities) in enumerate(probabilities_by_model.items()):
        y_prob = pd.Series(probabilities).astype(float).clip(1e-6, 1 - 1e-6)
        color = cmap(index % 10)
        fpr, tpr, _ = roc_curve(y_true_series, y_prob)
        precision, recall, _ = precision_recall_curve(y_true_series, y_prob)
        roc_auc = roc_auc_score(y_true_series, y_prob)
        ap = average_precision_score(y_true_series, y_prob)
        axes[0].plot(fpr, tpr, linewidth=2, color=color, label=f"{model_name} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, linewidth=2, color=color, label=f"{model_name} ({ap:.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["steel"], alpha=0.6)
    axes[0].set_title(f"ROC Curves For P({positive_label})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_title(f"Precision-Recall For P({positive_label})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=9)

    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_binary_calibration_curves(
    *,
    y_true: pd.Series | np.ndarray,
    probabilities_by_model: dict[str, pd.Series],
    output_path: str | Path,
    num_bins: int = 8,
) -> None:
    _setup_axes()
    if not probabilities_by_model:
        return

    y_true_series = pd.Series(y_true).astype(int)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")

    for index, (model_name, probabilities) in enumerate(probabilities_by_model.items()):
        y_prob = pd.Series(probabilities).astype(float).clip(1e-6, 1 - 1e-6)
        color = cmap(index % 10)
        frac_pos, mean_pred = calibration_curve(y_true_series, y_prob, n_bins=num_bins, strategy="uniform")
        axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=2, color=color, label=model_name)
        axes[1].hist(y_prob, bins=num_bins, alpha=0.35, color=color, label=model_name)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["steel"], alpha=0.6)
    axes[0].set_title("Calibration Curve")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Observed Positive Rate")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_title("Probability Mass By Model")
    axes[1].set_xlabel("Predicted Positive Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="best", fontsize=9)

    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)
