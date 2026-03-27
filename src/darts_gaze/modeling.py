"""Baseline training and leave-one-match-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import PROCESSED_DIR, ensure_data_directories
from .plots import (
    save_binary_calibration_curves,
    save_binary_model_curves,
    save_binary_probability_distribution,
    save_binary_ranking_curves,
    save_confusion_matrix,
    save_dataset_distribution,
    save_gaze_trend_scatter,
    save_metric_ci_panels,
    save_match_shaped_score_scatter,
    save_player_centered_gaze_trends,
    save_player_score_scatter_series,
    save_regression_scatter,
)
from .targets import BOARD_WEDGE_ORDER, circular_wedge_distance, coerce_segment_number, is_three_wedge_hit

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

FEATURE_COLUMNS = [
    "valid_face",
    "detector_confidence",
    "left_gaze_x",
    "left_gaze_y",
    "left_gaze_z",
    "right_gaze_x",
    "right_gaze_y",
    "right_gaze_z",
    "average_gaze_x",
    "average_gaze_y",
    "average_gaze_z",
    "head_x_axis_x",
    "head_x_axis_y",
    "head_x_axis_z",
    "head_y_axis_x",
    "head_y_axis_y",
    "head_y_axis_z",
    "head_z_axis_x",
    "head_z_axis_y",
    "head_z_axis_z",
    "ipd",
    "eye_agreement",
    "face_bbox_x_norm",
    "face_bbox_y_norm",
    "face_bbox_width_norm",
    "face_bbox_height_norm",
]

RERANKER_CATEGORICAL_COLUMNS = [
    "player_name",
    "competitor_qualifier",
]

MODEL_COMPARISON_BOOTSTRAP_SAMPLES = 1000


if TORCH_AVAILABLE:
    class _TorchFeedForwardNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)
else:
    class _TorchFeedForwardNetwork:  # pragma: no cover - exercised only when torch is unavailable
        pass


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 400,
        patience: int = 40,
        validation_fraction: float = 0.15,
        random_state: int = 0,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    @staticmethod
    def _to_numpy(features: Any) -> np.ndarray:
        if hasattr(features, "toarray"):
            features = features.toarray()
        return np.asarray(features, dtype=np.float32)

    def _build_model(self, input_dim: int, output_dim: int) -> _TorchFeedForwardNetwork:
        return _TorchFeedForwardNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
        )

    def fit(self, X: Any, y: Any) -> "TorchTabularClassifier":
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is not available")

        features = self._to_numpy(X)
        raw_targets = np.asarray(y)
        self.classes_, encoded_targets = np.unique(raw_targets, return_inverse=True)
        encoded_targets = encoded_targets.astype(np.int64, copy=False)
        if features.ndim != 2:
            raise ValueError("Expected 2D input features")

        self.input_dim_ = int(features.shape[1])
        self.output_dim_ = int(len(self.classes_))
        if self.output_dim_ < 2:
            raise ValueError("TorchTabularClassifier requires at least two classes")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        stratify_targets = encoded_targets if self.validation_fraction > 0 and np.bincount(encoded_targets).min() >= 2 else None
        if self.validation_fraction > 0 and len(features) >= max(20, self.output_dim_ * 4):
            x_train, x_valid, y_train, y_valid = train_test_split(
                features,
                encoded_targets,
                test_size=self.validation_fraction,
                stratify=stratify_targets,
                random_state=self.random_state,
            )
        else:
            x_train, y_train = features, encoded_targets
            x_valid, y_valid = None, None

        class_counts = np.bincount(y_train, minlength=self.output_dim_).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (self.output_dim_ * class_counts)

        self.model_ = self._build_model(self.input_dim_, self.output_dim_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)

        valid_inputs = torch.tensor(x_valid, dtype=torch.float32) if x_valid is not None else None
        valid_targets = torch.tensor(y_valid, dtype=torch.long) if y_valid is not None else None

        best_state: dict[str, torch.Tensor] | None = None
        best_valid_loss = float("inf")
        epochs_without_improvement = 0

        for _ in range(self.max_epochs):
            self.model_.train()
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_inputs)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

            if valid_inputs is None or valid_targets is None:
                continue

            self.model_.eval()
            with torch.no_grad():
                valid_logits = self.model_(valid_inputs)
                valid_loss = float(criterion(valid_logits, valid_targets).item())
            if valid_loss + 1e-6 < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model has not been fitted")
        features = torch.tensor(self._to_numpy(X), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(features)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        labels = probabilities.argmax(axis=1)
        return self.classes_[labels]


def _regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )


def _segment_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _knn_classifier_pipeline(n_neighbors: int = 11) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=max(1, n_neighbors), weights="distance")),
        ]
    )


def _extra_trees_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=0,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _mixed_feature_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), FEATURE_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RERANKER_CATEGORICAL_COLUMNS),
        ]
    )


def _mixed_logistic_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def _mixed_knn_classifier_pipeline(n_neighbors: int = 11) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            ("model", KNeighborsClassifier(n_neighbors=max(1, n_neighbors), weights="distance")),
        ]
    )


def _mixed_extra_trees_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=0,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _mixed_random_forest_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=0,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _mixed_svc_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            ("model", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=0)),
        ]
    )


def _mixed_mlp_classifier_pipeline(train_size: int | None = None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=5e-3,
                    learning_rate_init=3e-4,
                    max_iter=5000,
                    early_stopping=False,
                    n_iter_no_change=60,
                    random_state=0,
                ),
            ),
        ]
    )


def _mixed_torch_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            (
                "model",
                TorchTabularClassifier(
                    hidden_dims=(96, 48, 24),
                    dropout=0.2,
                    learning_rate=8e-4,
                    weight_decay=2e-4,
                    batch_size=32,
                    max_epochs=500,
                    patience=50,
                    validation_fraction=0.15,
                    random_state=0,
                ),
            ),
        ]
    )


def _reranker_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )


def _reranker_extra_trees_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=0,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _reranker_mlp_pipeline(train_size: int | None = None) -> Pipeline:
    effective_train_size = train_size or 0
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-2,
                    learning_rate_init=5e-4,
                    max_iter=4000,
                    early_stopping=effective_train_size >= 20,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=0,
                ),
            ),
        ]
    )


def _reranker_svc_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            ("model", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=0)),
        ]
    )


def _reranker_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=0,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _reranker_torch_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _mixed_feature_preprocessor(scale_numeric=True)),
            (
                "model",
                TorchTabularClassifier(
                    hidden_dims=(96, 48, 24),
                    dropout=0.2,
                    learning_rate=8e-4,
                    weight_decay=2e-4,
                    batch_size=24,
                    max_epochs=500,
                    patience=50,
                    validation_fraction=0.15,
                    random_state=0,
                ),
            ),
        ]
    )


def _as_python(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def _majority_prediction(y_train: pd.Series, size: int) -> np.ndarray:
    mode = y_train.mode(dropna=False)
    label = mode.iloc[0] if not mode.empty else y_train.iloc[0]
    return np.repeat(label, size)


def _wedge_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    three_hits: list[bool] = []
    circular_distances: list[float] = []
    for actual_label, predicted_label in zip(y_true.astype(str), y_pred.astype(str), strict=True):
        if actual_label == "BULL" and predicted_label == "BULL":
            three_hits.append(True)
            circular_distances.append(0.0)
            continue
        actual_number = coerce_segment_number(actual_label)
        predicted_number = coerce_segment_number(predicted_label)
        three_hits.append(is_three_wedge_hit(predicted_number, actual_number))
        distance = circular_wedge_distance(predicted_number, actual_number)
        if distance is not None:
            circular_distances.append(float(distance))
    return {
        "three_wedge_accuracy": float(np.mean(three_hits)) if three_hits else None,
        "circular_wedge_mae": float(np.mean(circular_distances)) if circular_distances else None,
    }


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, *, wedge_task: bool = False) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if wedge_task:
        metrics.update(_wedge_metrics(y_true, y_pred))
    return metrics


def _binary_reranker_metrics(y_true: pd.Series, y_prob: pd.Series, *, threshold: float = 0.5) -> dict[str, Any]:
    y_true_int = pd.Series(y_true).astype(int)
    y_prob_float = pd.Series(y_prob).astype(float).clip(1e-6, 1 - 1e-6)
    y_pred_int = (y_prob_float >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_int, y_pred_int)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_int, y_pred_int)),
        "macro_f1": float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)),
        "average_precision": float(average_precision_score(y_true_int, y_prob_float)),
        "brier_score": float(brier_score_loss(y_true_int, y_prob_float)),
        "log_loss": float(log_loss(y_true_int, y_prob_float, labels=[0, 1])),
        "positive_rate": float(y_prob_float.mean()),
    }
    if y_true_int.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_int, y_prob_float))
    else:
        metrics["roc_auc"] = None
    return metrics


def _expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    y_true_int = pd.Series(y_true).astype(int).to_numpy()
    y_prob_float = pd.Series(y_prob).astype(float).to_numpy()
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if upper >= 1.0:
            mask = (y_prob_float >= lower) & (y_prob_float <= upper)
        else:
            mask = (y_prob_float >= lower) & (y_prob_float < upper)
        if not np.any(mask):
            continue
        observed_rate = float(np.mean(y_true_int[mask]))
        predicted_rate = float(np.mean(y_prob_float[mask]))
        ece += abs(observed_rate - predicted_rate) * (float(mask.sum()) / float(len(y_prob_float)))
    return float(ece)


def _bootstrap_indices(
    size: int,
    *,
    rng: np.random.Generator,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    if labels is None:
        return rng.integers(0, size, size=size)

    unique_labels = np.unique(labels)
    sampled_blocks: list[np.ndarray] = []
    for label in unique_labels:
        label_indices = np.flatnonzero(labels == label)
        sampled_blocks.append(rng.choice(label_indices, size=len(label_indices), replace=True))
    sampled = np.concatenate(sampled_blocks)
    rng.shuffle(sampled)
    return sampled


def _bootstrap_metric_interval(
    metric_fn: Any,
    *,
    size: int,
    seed: int,
    stratify_labels: np.ndarray | None = None,
    num_samples: int = MODEL_COMPARISON_BOOTSTRAP_SAMPLES,
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(num_samples):
        sample_indices = _bootstrap_indices(size=size, rng=rng, labels=stratify_labels)
        try:
            value = metric_fn(sample_indices)
        except ValueError:
            continue
        if value is None or np.isnan(value):
            continue
        estimates.append(float(value))
    if not estimates:
        return None, None
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _wedge_model_bootstrap_summary(
    *,
    y_true: pd.Series,
    predictions_by_model: dict[str, pd.Series],
) -> pd.DataFrame:
    y_true_series = y_true.astype(str)
    summary_rows: list[dict[str, Any]] = []
    metrics = {
        "accuracy": lambda actual, predicted: float(accuracy_score(actual, predicted)),
        "macro_f1": lambda actual, predicted: float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "three_wedge_accuracy": lambda actual, predicted: _wedge_metrics(actual, predicted)["three_wedge_accuracy"],
        "circular_wedge_mae": lambda actual, predicted: _wedge_metrics(actual, predicted)["circular_wedge_mae"],
    }
    for model_name, predictions in predictions_by_model.items():
        predicted_series = predictions.loc[y_true_series.index].astype(str)
        for metric_name, metric_fn in metrics.items():
            estimate = metric_fn(y_true_series, predicted_series)
            ci_low, ci_high = _bootstrap_metric_interval(
                lambda indices: metric_fn(y_true_series.iloc[indices], predicted_series.iloc[indices]),
                size=len(y_true_series),
                seed=abs(hash((model_name, metric_name, "wedge"))) % (2**32),
            )
            summary_rows.append(
                {
                    "task": "wedge_number",
                    "model": model_name,
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(summary_rows)


def _binary_model_bootstrap_summary(
    *,
    y_true: pd.Series,
    probabilities_by_model: dict[str, pd.Series],
    task_name: str,
) -> pd.DataFrame:
    y_true_series = y_true.astype(int)
    stratify_labels = y_true_series.to_numpy()
    summary_rows: list[dict[str, Any]] = []
    metric_fns = {
        "accuracy": lambda actual, prob: float(accuracy_score(actual, (prob >= 0.5).astype(int))),
        "balanced_accuracy": lambda actual, prob: float(balanced_accuracy_score(actual, (prob >= 0.5).astype(int))),
        "macro_f1": lambda actual, prob: float(f1_score(actual, (prob >= 0.5).astype(int), average="macro", zero_division=0)),
        "roc_auc": lambda actual, prob: float(roc_auc_score(actual, prob)),
        "average_precision": lambda actual, prob: float(average_precision_score(actual, prob)),
        "brier_score": lambda actual, prob: float(brier_score_loss(actual, prob)),
        "log_loss": lambda actual, prob: float(log_loss(actual, prob, labels=[0, 1])),
        "ece_10bin": lambda actual, prob: _expected_calibration_error(actual, prob, num_bins=10),
    }
    for model_name, probabilities in probabilities_by_model.items():
        probability_series = probabilities.loc[y_true_series.index].astype(float).clip(1e-6, 1 - 1e-6)
        for metric_name, metric_fn in metric_fns.items():
            estimate = metric_fn(y_true_series, probability_series)
            ci_low, ci_high = _bootstrap_metric_interval(
                lambda indices: metric_fn(y_true_series.iloc[indices], probability_series.iloc[indices]),
                size=len(y_true_series),
                seed=abs(hash((model_name, metric_name, task_name))) % (2**32),
                stratify_labels=stratify_labels,
            )
            summary_rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(summary_rows)


def _sorted_labels(labels: list[str], *, wedge_order: bool = False) -> list[str]:
    if not wedge_order:
        return sorted(labels)

    ordering = {str(number): index for index, number in enumerate(BOARD_WEDGE_ORDER)}
    ordering.update({"BULL": len(ordering), "MISS": len(ordering) + 1, "OTHER": len(ordering) + 2})
    return sorted(labels, key=lambda label: ordering.get(str(label), len(ordering) + 10))


def _evaluate_binary_reranker_models(
    *,
    feature_frame: pd.DataFrame,
    target: pd.Series,
    match_ids: pd.Series,
    task_name: str,
    model_builders: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.Series]]:
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    probabilities_by_model: dict[str, pd.Series] = {}

    for model_name, builder in model_builders.items():
        probabilities = pd.Series(index=target.index, dtype=float)
        for match_id in sorted(match_ids.unique()):
            test_mask = match_ids == match_id
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            x_train = feature_frame.loc[train_mask]
            x_test = feature_frame.loc[test_mask]
            y_train = target.loc[train_mask].astype(int)
            y_test = target.loc[test_mask].astype(int)

            if builder is None or y_train.nunique() < 2:
                positive_prob = float(y_train.mean())
                fold_probabilities = np.full(shape=int(test_mask.sum()), fill_value=positive_prob, dtype=float)
            else:
                try:
                    model = builder(int(train_mask.sum()))
                except TypeError:
                    model = builder()
                model.fit(x_train, y_train)
                fold_probabilities = model.predict_proba(x_test)[:, 1]

            fold_prob_series = pd.Series(fold_probabilities, index=y_test.index, dtype=float)
            probabilities.loc[test_mask] = fold_prob_series
            fold_row = {
                "task": task_name,
                "model": model_name,
                "sport_event_id": match_id,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
            fold_row.update(_binary_reranker_metrics(y_test, fold_prob_series))
            fold_rows.append(fold_row)

        valid_mask = probabilities.notna()
        if not valid_mask.any():
            continue
        y_true = target.loc[valid_mask].astype(int)
        y_prob = probabilities.loc[valid_mask].astype(float)
        overall_row = {"task": task_name, "model": model_name, "num_rows": int(valid_mask.sum())}
        overall_row.update(_binary_reranker_metrics(y_true, y_prob))
        overall_row["ece_10bin"] = _expected_calibration_error(y_true, y_prob, num_bins=10)
        model_rows.append(overall_row)
        probabilities_by_model[model_name] = probabilities.astype(float)

    return model_rows, fold_rows, probabilities_by_model


def _evaluate_classifier_models(
    *,
    feature_frame: pd.DataFrame,
    target: pd.Series,
    match_ids: pd.Series,
    task_name: str,
    model_builders: dict[str, Any],
    wedge_task: bool = False,
    label_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.Series]]:
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.Series] = {}

    for model_name, builder in model_builders.items():
        predictions = pd.Series(index=target.index, dtype=object)
        for match_id in sorted(match_ids.unique()):
            test_mask = match_ids == match_id
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            x_train = feature_frame.loc[train_mask]
            x_test = feature_frame.loc[test_mask]
            y_train = target.loc[train_mask]
            y_test = target.loc[test_mask]

            if builder is None:
                predicted = _majority_prediction(y_train, int(test_mask.sum()))
            else:
                try:
                    model = builder(int(train_mask.sum()))
                except TypeError:
                    model = builder()
                model.fit(x_train, y_train)
                predicted = model.predict(x_test)

            y_pred = pd.Series(predicted, index=y_test.index).astype(str)
            predictions.loc[test_mask] = y_pred
            fold_row = {
                "task": task_name,
                "model": model_name,
                "sport_event_id": match_id,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
            fold_row.update(_classification_metrics(y_test.astype(str), y_pred, wedge_task=wedge_task))
            fold_rows.append(fold_row)

        valid_mask = predictions.notna()
        if not valid_mask.any():
            continue
        y_true = target.loc[valid_mask].astype(str)
        y_pred = predictions.loc[valid_mask].astype(str)
        overall_row = {"task": task_name, "model": model_name}
        overall_row.update(_classification_metrics(y_true, y_pred, wedge_task=wedge_task))
        overall_row["num_rows"] = int(valid_mask.sum())
        model_rows.append(overall_row)
        predictions_by_model[model_name] = predictions.astype(str)

    if label_order is not None:
        for row in model_rows:
            row["labels"] = label_order
    return model_rows, fold_rows, predictions_by_model


def _save_best_classifier(
    *,
    feature_frame: pd.DataFrame,
    target: pd.Series,
    output_path: Path,
    builder: Any,
) -> None:
    try:
        model = builder(int(len(feature_frame)))
    except TypeError:
        model = builder()
    model.fit(feature_frame, target)
    joblib.dump(model, output_path)


def train_baselines(
    dataset_csv: str | Path,
    output_dir: str | Path = PROCESSED_DIR,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_data_directories()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_dir is None:
        tables_dir = output_dir
        figures_dir = output_dir
    else:
        report_root = Path(report_dir)
        tables_dir = report_root / "tables"
        figures_dir = report_root / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise RuntimeError("Training dataset is empty")

    train_df = df[df["review_status"].isin(["matched", "verified"])].copy()
    if train_df.empty:
        raise RuntimeError("No matched or verified rows are available for training")
    modeling_mask = _as_bool_series(train_df["entered_modeling"]) if "entered_modeling" in train_df.columns else _as_bool_series(train_df["valid_face"])
    modeling_df = train_df[modeling_mask].copy()
    if modeling_df.empty:
        raise RuntimeError("No valid-face rows are available for modeling")

    feature_frame = modeling_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mixed_feature_frame = modeling_df[FEATURE_COLUMNS + RERANKER_CATEGORICAL_COLUMNS].copy()
    target_score = modeling_df["resulting_score"].astype(float)
    target_segment = modeling_df["segment_label"].astype(str)
    target_wedge = modeling_df["wedge_number_label"].astype(str)
    target_coarse_area = modeling_df["coarse_wedge_area_label"].astype(str)
    match_ids = modeling_df["sport_event_id"].astype(str)

    fold_rows: list[dict[str, Any]] = []

    regression_predictions = pd.Series(index=modeling_df.index, dtype=float)
    naive_regression_predictions = pd.Series(index=modeling_df.index, dtype=float)
    exact_segment_predictions = pd.Series(index=modeling_df.index, dtype=object)

    for match_id in sorted(match_ids.unique()):
        test_mask = match_ids == match_id
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        x_train = feature_frame.loc[train_mask]
        x_test = feature_frame.loc[test_mask]
        y_train_score = target_score.loc[train_mask]
        y_test_score = target_score.loc[test_mask]
        y_train_segment = target_segment.loc[train_mask]
        y_test_segment = target_segment.loc[test_mask]

        naive_value = float(y_train_score.mean())
        naive_pred = np.full(shape=len(y_test_score), fill_value=naive_value, dtype=float)
        naive_regression_predictions.loc[test_mask] = naive_pred

        regression_model = _regression_pipeline()
        regression_model.fit(x_train, y_train_score)
        score_pred = regression_model.predict(x_test)
        regression_predictions.loc[test_mask] = score_pred

        segment_model = _segment_classifier_pipeline()
        segment_model.fit(x_train, y_train_segment)
        segment_pred = segment_model.predict(x_test)
        exact_segment_predictions.loc[test_mask] = segment_pred

        fold_row = {
            "task": "score_regression",
            "model": "ridge",
            "sport_event_id": match_id,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "naive_regression_mae": float(mean_absolute_error(y_test_score, naive_pred)),
            "naive_regression_rmse": float(np.sqrt(mean_squared_error(y_test_score, naive_pred))),
            "naive_regression_r2": float(r2_score(y_test_score, naive_pred)) if len(y_test_score) > 1 else None,
            "regression_mae": float(mean_absolute_error(y_test_score, score_pred)),
            "regression_rmse": float(np.sqrt(mean_squared_error(y_test_score, score_pred))),
            "regression_r2": float(r2_score(y_test_score, score_pred)) if len(y_test_score) > 1 else None,
        }
        fold_rows.append(fold_row)

        exact_segment_fold_row = {
            "task": "exact_segment",
            "model": "logistic",
            "sport_event_id": match_id,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        }
        exact_segment_fold_row.update(
            _classification_metrics(y_test_segment.astype(str), pd.Series(segment_pred, index=y_test_segment.index).astype(str))
        )
        fold_rows.append(exact_segment_fold_row)

    metrics: dict[str, Any] = {
        "total_rows": int(len(df)),
        "matched_rows": int(len(train_df)),
        "modeling_rows": int(len(modeling_df)),
        "valid_face_rows": int(_as_bool_series(train_df["valid_face"]).sum()),
        "invalid_face_rows": int((~_as_bool_series(train_df["valid_face"])).sum()),
        "num_matches": int(match_ids.nunique()),
    }

    valid_regression = regression_predictions.notna()
    metrics["overall_regression"] = {
        "mae": float(mean_absolute_error(target_score.loc[valid_regression], regression_predictions.loc[valid_regression])),
        "rmse": float(np.sqrt(mean_squared_error(target_score.loc[valid_regression], regression_predictions.loc[valid_regression]))),
        "r2": float(r2_score(target_score.loc[valid_regression], regression_predictions.loc[valid_regression])),
    }
    metrics["overall_naive_regression"] = {
        "mae": float(mean_absolute_error(target_score.loc[valid_regression], naive_regression_predictions.loc[valid_regression])),
        "rmse": float(np.sqrt(mean_squared_error(target_score.loc[valid_regression], naive_regression_predictions.loc[valid_regression]))),
        "r2": float(r2_score(target_score.loc[valid_regression], naive_regression_predictions.loc[valid_regression])),
    }
    save_regression_scatter(
        y_true=target_score.loc[valid_regression].to_numpy(),
        y_pred=regression_predictions.loc[valid_regression].to_numpy(),
        output_path=figures_dir / "score_regression_scatter.pdf",
    )

    valid_exact_segment = exact_segment_predictions.notna()
    exact_segment_labels = sorted(target_segment.unique().tolist())
    metrics["overall_classification"] = {
        "accuracy": float(accuracy_score(target_segment.loc[valid_exact_segment], exact_segment_predictions.loc[valid_exact_segment])),
        "macro_f1": float(
            f1_score(
                target_segment.loc[valid_exact_segment],
                exact_segment_predictions.loc[valid_exact_segment],
                average="macro",
                zero_division=0,
            )
        ),
        "labels": exact_segment_labels,
    }
    segment_cm = confusion_matrix(
        target_segment.loc[valid_exact_segment],
        exact_segment_predictions.loc[valid_exact_segment],
        labels=exact_segment_labels,
    )
    save_confusion_matrix(segment_cm, exact_segment_labels, figures_dir / "segment_confusion_matrix.pdf")

    wedge_models: dict[str, Any] = {
        "majority": None,
        "logistic": _mixed_logistic_classifier_pipeline,
        "knn": _mixed_knn_classifier_pipeline,
        "extra_trees": _mixed_extra_trees_classifier_pipeline,
        "random_forest": _mixed_random_forest_classifier_pipeline,
        "svc_rbf": _mixed_svc_classifier_pipeline,
        "mlp_deep": _mixed_mlp_classifier_pipeline,
    }
    if TORCH_AVAILABLE:
        wedge_models["torch_deep"] = _mixed_torch_classifier_pipeline
    wedge_labels = _sorted_labels(target_wedge.unique().tolist(), wedge_order=True)
    wedge_model_rows, wedge_fold_rows, wedge_predictions = _evaluate_classifier_models(
        feature_frame=mixed_feature_frame,
        target=target_wedge,
        match_ids=match_ids,
        task_name="wedge_number",
        model_builders=wedge_models,
        wedge_task=True,
        label_order=wedge_labels,
    )
    fold_rows.extend(wedge_fold_rows)
    wedge_selected_row = max(
        [row for row in wedge_model_rows if row["model"] != "majority"],
        key=lambda row: (
            row.get("three_wedge_accuracy") or float("-inf"),
            row.get("macro_f1") or float("-inf"),
            -(row.get("circular_wedge_mae") or float("inf")),
            row.get("accuracy") or float("-inf"),
        ),
    )
    wedge_selected_model = wedge_selected_row["model"]
    wedge_predictions_selected = wedge_predictions[wedge_selected_model]
    wedge_cm = confusion_matrix(target_wedge.astype(str), wedge_predictions_selected.astype(str), labels=wedge_labels)
    save_confusion_matrix(wedge_cm, wedge_labels, figures_dir / "wedge_number_confusion_matrix.pdf")
    _save_best_classifier(
        feature_frame=mixed_feature_frame,
        target=target_wedge.astype(str),
        output_path=output_dir / "wedge_number_classifier.joblib",
        builder=wedge_models[wedge_selected_model],
    )
    wedge_capture_ids = (
        modeling_df["capture_id"]
        if "capture_id" in modeling_df.columns
        else pd.Series(modeling_df.index, index=modeling_df.index, name="capture_id")
    )
    wedge_prediction_export = pd.DataFrame(
        {
            "capture_id": wedge_capture_ids,
            "sport_event_id": modeling_df["sport_event_id"],
            "player_name": modeling_df["player_name"],
            "true_wedge_number_label": target_wedge.astype(str),
        }
    )
    for model_name, predictions in wedge_predictions.items():
        wedge_prediction_export[f"{model_name}_predicted_label"] = predictions.astype(str)
    wedge_prediction_export.to_csv(tables_dir / "wedge_number_model_predictions.csv", index=False)

    coarse_models = {
        "majority": None,
        "knn": _knn_classifier_pipeline,
        "extra_trees": _extra_trees_classifier_pipeline,
    }
    coarse_labels = _sorted_labels(target_coarse_area.unique().tolist(), wedge_order=True)
    coarse_model_rows, coarse_fold_rows, coarse_predictions = _evaluate_classifier_models(
        feature_frame=feature_frame,
        target=target_coarse_area,
        match_ids=match_ids,
        task_name="coarse_wedge_area",
        model_builders=coarse_models,
        wedge_task=False,
        label_order=coarse_labels,
    )
    fold_rows.extend(coarse_fold_rows)
    coarse_selected_row = max(
        [row for row in coarse_model_rows if row["model"] != "majority"],
        key=lambda row: (row.get("macro_f1") or float("-inf"), row.get("accuracy") or float("-inf")),
    )
    coarse_selected_model = coarse_selected_row["model"]
    coarse_predictions_selected = coarse_predictions[coarse_selected_model]
    coarse_cm = confusion_matrix(target_coarse_area.astype(str), coarse_predictions_selected.astype(str), labels=coarse_labels)
    save_confusion_matrix(coarse_cm, coarse_labels, figures_dir / "coarse_wedge_area_confusion_matrix.pdf")
    _save_best_classifier(
        feature_frame=feature_frame,
        target=target_coarse_area.astype(str),
        output_path=output_dir / "coarse_wedge_area_classifier.joblib",
        builder=coarse_models[coarse_selected_model],
    )

    metrics["wedge_number_classification"] = {
        "selected_model": wedge_selected_model,
        "models": wedge_model_rows,
    }
    metrics["coarse_wedge_area_classification"] = {
        "selected_model": coarse_selected_model,
        "models": coarse_model_rows,
    }

    reranker_model_rows: list[dict[str, Any]] = []
    reranker_probabilities: dict[str, pd.Series] = {}
    reranker_df = modeling_df[modeling_df["wedge_number_label"].astype(str).isin(["19", "20"])].copy()
    if not reranker_df.empty and reranker_df["wedge_number_label"].nunique() >= 2:
        reranker_feature_frame = reranker_df[FEATURE_COLUMNS + RERANKER_CATEGORICAL_COLUMNS].copy()
        reranker_target = (reranker_df["wedge_number_label"].astype(str) == "20").astype(int)
        reranker_match_ids = reranker_df["sport_event_id"].astype(str)
        reranker_models: dict[str, Any] = {
            "majority": None,
            "logistic": _reranker_logistic_pipeline,
            "extra_trees": _reranker_extra_trees_pipeline,
            "random_forest": _reranker_random_forest_pipeline,
            "svc_rbf": _reranker_svc_pipeline,
            "mlp_deep": _reranker_mlp_pipeline,
        }
        if TORCH_AVAILABLE:
            reranker_models["torch_deep"] = _reranker_torch_pipeline
        reranker_model_rows, reranker_fold_rows, reranker_probabilities = _evaluate_binary_reranker_models(
            feature_frame=reranker_feature_frame,
            target=reranker_target,
            match_ids=reranker_match_ids,
            task_name="wedge_19_vs_20_reranker",
            model_builders=reranker_models,
        )
        fold_rows.extend(reranker_fold_rows)
        reranker_selected_row = max(
            [row for row in reranker_model_rows if row["model"] != "majority"],
            key=lambda row: (
                row.get("roc_auc") or float("-inf"),
                row.get("average_precision") or float("-inf"),
                row.get("balanced_accuracy") or float("-inf"),
                row.get("macro_f1") or float("-inf"),
                -(row.get("ece_10bin") or float("inf")),
                -(row.get("log_loss") or float("inf")),
            ),
        )
        reranker_selected_model = reranker_selected_row["model"]
        reranker_selected_probabilities = reranker_probabilities[reranker_selected_model].astype(float)
        reranker_predictions = (reranker_selected_probabilities >= 0.5).map({True: "20", False: "19"}).astype(str)
        reranker_cm = confusion_matrix(
            reranker_df["wedge_number_label"].astype(str),
            reranker_predictions,
            labels=["19", "20"],
        )
        save_confusion_matrix(reranker_cm, ["19", "20"], figures_dir / "wedge_19_vs_20_confusion_matrix.pdf")
        save_binary_probability_distribution(
            reranker_df["wedge_number_label"].astype(str),
            reranker_selected_probabilities,
            figures_dir / "wedge_19_vs_20_probability_distribution.pdf",
            positive_label="20",
            negative_label="19",
        )
        save_binary_ranking_curves(
            reranker_df["wedge_number_label"].astype(str),
            reranker_selected_probabilities,
            figures_dir / "wedge_19_vs_20_ranking_curves.pdf",
            positive_label="20",
        )
        save_binary_model_curves(
            y_true=reranker_target,
            probabilities_by_model=reranker_probabilities,
            output_path=figures_dir / "wedge_19_vs_20_model_curves.pdf",
            positive_label="20",
        )
        save_binary_calibration_curves(
            y_true=reranker_target,
            probabilities_by_model=reranker_probabilities,
            output_path=figures_dir / "wedge_19_vs_20_calibration_curves.pdf",
        )
        save_match_shaped_score_scatter(
            reranker_df,
            figures_dir / "wedge_19_vs_20_gaze_scatter.pdf",
            label_column="wedge_number_label",
            title="19 vs 20 Gaze Scatter",
        )
        save_player_centered_gaze_trends(
            reranker_df,
            figures_dir / "wedge_19_vs_20_player_centers.pdf",
            label_column="wedge_number_label",
            title="19 vs 20 Relative To Player Center",
            top_n=2,
        )
        _save_best_classifier(
            feature_frame=reranker_feature_frame,
            target=reranker_target,
            output_path=output_dir / "wedge_19_vs_20_reranker.joblib",
            builder=reranker_models[reranker_selected_model],
        )
        capture_ids = (
            reranker_df["capture_id"]
            if "capture_id" in reranker_df.columns
            else pd.Series(reranker_df.index, index=reranker_df.index, name="capture_id")
        )
        reranker_probability_export = pd.DataFrame(
            {
                "capture_id": capture_ids,
                "sport_event_id": reranker_df["sport_event_id"],
                "player_name": reranker_df["player_name"],
                "wedge_number_label": reranker_df["wedge_number_label"],
                "predicted_label": reranker_predictions,
                "predicted_p20": reranker_selected_probabilities,
            }
        )
        for model_name, probabilities in reranker_probabilities.items():
            reranker_probability_export[f"{model_name}_predicted_p20"] = probabilities.astype(float)
        reranker_probability_export.to_csv(tables_dir / "wedge_19_vs_20_predictions.csv", index=False)
        metrics["wedge_19_vs_20_reranker"] = {
            "selected_model": reranker_selected_model,
            "models": reranker_model_rows,
            "num_rows": int(len(reranker_df)),
        }

    metrics["fold_metrics"] = fold_rows

    save_dataset_distribution(train_df, figures_dir / "dataset_distribution.pdf")
    save_gaze_trend_scatter(
        modeling_df,
        figures_dir / "coarse_wedge_area_gaze_trend.pdf",
        label_column="coarse_wedge_area_label",
        title="Coarse Target-Area Gaze Trends",
    )
    save_player_centered_gaze_trends(
        modeling_df,
        figures_dir / "coarse_wedge_area_player_centers.pdf",
        label_column="coarse_wedge_area_label",
        title="Coarse Target Areas Relative To Player Center",
    )
    save_player_score_scatter_series(
        modeling_df,
        figures_dir / "player_score_scatters",
        label_column="wedge_number_label",
        title_prefix="All Score Gaze Scatter",
    )

    full_regression_model = _regression_pipeline()
    full_regression_model.fit(feature_frame, target_score)
    joblib.dump(full_regression_model, output_dir / "score_regression.joblib")

    full_segment_classifier = _segment_classifier_pipeline()
    full_segment_classifier.fit(feature_frame, target_segment)
    joblib.dump(full_segment_classifier, output_dir / "segment_classifier.joblib")

    (output_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (output_dir / "mixed_feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS + RERANKER_CATEGORICAL_COLUMNS, indent=2))

    model_comparison_rows = pd.DataFrame(
        [
            {
                "task": "score_regression",
                "model": "naive_mean_score",
                "mae": metrics["overall_naive_regression"]["mae"],
                "rmse": metrics["overall_naive_regression"]["rmse"],
                "r2": metrics["overall_naive_regression"]["r2"],
            },
            {
                "task": "score_regression",
                "model": "ridge",
                "mae": metrics["overall_regression"]["mae"],
                "rmse": metrics["overall_regression"]["rmse"],
                "r2": metrics["overall_regression"]["r2"],
            },
            {
                "task": "exact_segment",
                "model": "logistic",
                "accuracy": metrics["overall_classification"]["accuracy"],
                "macro_f1": metrics["overall_classification"]["macro_f1"],
            },
        ]
        + wedge_model_rows
        + coarse_model_rows
        + reranker_model_rows
    )
    baseline_comparison_rows = model_comparison_rows[model_comparison_rows["task"] == "score_regression"].copy()
    wedge_ci_rows = _wedge_model_bootstrap_summary(
        y_true=target_wedge,
        predictions_by_model={name: prediction.astype(str) for name, prediction in wedge_predictions.items()},
    )
    reranker_ci_rows = (
        _binary_model_bootstrap_summary(
            y_true=reranker_target,
            probabilities_by_model=reranker_probabilities,
            task_name="wedge_19_vs_20_reranker",
        )
        if reranker_probabilities
        else pd.DataFrame(columns=["task", "model", "metric", "estimate", "ci_low", "ci_high"])
    )
    wedge_ci_rows.to_csv(tables_dir / "wedge_number_model_bootstrap.csv", index=False)
    reranker_ci_rows.to_csv(tables_dir / "wedge_19_vs_20_model_bootstrap.csv", index=False)
    if not wedge_ci_rows.empty:
        save_metric_ci_panels(
            metric_df=wedge_ci_rows,
            output_path=figures_dir / "wedge_number_model_comparison.pdf",
            title="Wedge Model Comparison With 95% Bootstrap CI",
            metric_order=["three_wedge_accuracy", "macro_f1", "accuracy", "circular_wedge_mae"],
            metric_labels={
                "three_wedge_accuracy": "3-Wedge Accuracy",
                "macro_f1": "Macro-F1",
                "accuracy": "Exact Wedge Accuracy",
                "circular_wedge_mae": "Circular Wedge MAE",
            },
            lower_is_better={"circular_wedge_mae"},
        )
    if not reranker_ci_rows.empty:
        save_metric_ci_panels(
            metric_df=reranker_ci_rows,
            output_path=figures_dir / "wedge_19_vs_20_model_comparison.pdf",
            title="19 vs 20 Reranker Comparison With 95% Bootstrap CI",
            metric_order=["roc_auc", "average_precision", "balanced_accuracy", "ece_10bin"],
            metric_labels={
                "roc_auc": "ROC-AUC",
                "average_precision": "Average Precision",
                "balanced_accuracy": "Balanced Accuracy",
                "ece_10bin": "ECE (10 bins)",
            },
            lower_is_better={"ece_10bin"},
        )
    pd.DataFrame(fold_rows).to_csv(tables_dir / "fold_metrics.csv", index=False)
    model_comparison_rows.to_csv(tables_dir / "model_comparison.csv", index=False)
    baseline_comparison_rows.to_csv(tables_dir / "baseline_comparison.csv", index=False)
    metrics_json = json.dumps(metrics, indent=2, default=_as_python)
    (tables_dir / "metrics.json").write_text(metrics_json)
    return metrics
