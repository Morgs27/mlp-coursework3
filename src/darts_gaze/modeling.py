"""Baseline training and leave-one-match-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, ensure_data_directories
from .plots import save_confusion_matrix, save_dataset_distribution, save_regression_scatter

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


def _regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )


def _classification_pipeline() -> Pipeline:
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


def _as_python(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def train_baselines(dataset_csv: str | Path, output_dir: str | Path = PROCESSED_DIR) -> dict[str, Any]:
    ensure_data_directories()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise RuntimeError("Training dataset is empty")

    train_df = df[df["review_status"].isin(["matched", "verified"])].copy()
    if train_df.empty:
        raise RuntimeError("No matched or verified rows are available for training")

    feature_frame = train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    target_score = train_df["resulting_score"].astype(float)
    target_segment = train_df["segment_label"].astype(str)
    match_ids = train_df["sport_event_id"].astype(str)

    regression_predictions = pd.Series(index=train_df.index, dtype=float)
    classification_predictions = pd.Series(index=train_df.index, dtype=object)
    fold_rows: list[dict[str, Any]] = []

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

        regression_model = _regression_pipeline()
        regression_model.fit(x_train, y_train_score)
        score_pred = regression_model.predict(x_test)
        regression_predictions.loc[test_mask] = score_pred

        fold_result: dict[str, Any] = {
            "sport_event_id": match_id,
            "regression_mae": float(mean_absolute_error(y_test_score, score_pred)),
            "regression_rmse": float(np.sqrt(mean_squared_error(y_test_score, score_pred))),
            "regression_r2": float(r2_score(y_test_score, score_pred)) if len(y_test_score) > 1 else None,
        }

        if y_train_segment.nunique() >= 2:
            classifier = _classification_pipeline()
            classifier.fit(x_train, y_train_segment)
            segment_pred = classifier.predict(x_test)
            classification_predictions.loc[test_mask] = segment_pred
            fold_result.update(
                {
                    "classification_accuracy": float(accuracy_score(y_test_segment, segment_pred)),
                    "classification_macro_f1": float(f1_score(y_test_segment, segment_pred, average="macro", zero_division=0)),
                }
            )
        else:
            fold_result.update(
                {
                    "classification_accuracy": None,
                    "classification_macro_f1": None,
                }
            )
        fold_rows.append(fold_result)

    metrics: dict[str, Any] = {
        "num_rows": int(len(train_df)),
        "num_matches": int(match_ids.nunique()),
        "fold_metrics": fold_rows,
    }

    if regression_predictions.notna().any():
        valid_regression = regression_predictions.notna()
        metrics["overall_regression"] = {
            "mae": float(mean_absolute_error(target_score.loc[valid_regression], regression_predictions.loc[valid_regression])),
            "rmse": float(np.sqrt(mean_squared_error(target_score.loc[valid_regression], regression_predictions.loc[valid_regression]))),
            "r2": float(r2_score(target_score.loc[valid_regression], regression_predictions.loc[valid_regression])),
        }
        save_regression_scatter(
            y_true=target_score.loc[valid_regression].to_numpy(),
            y_pred=regression_predictions.loc[valid_regression].to_numpy(),
            output_path=output_dir / "score_regression_scatter.pdf",
        )

    if classification_predictions.notna().any():
        valid_classification = classification_predictions.notna()
        labels = sorted(target_segment.unique().tolist())
        cm = confusion_matrix(
            target_segment.loc[valid_classification],
            classification_predictions.loc[valid_classification],
            labels=labels,
        )
        metrics["overall_classification"] = {
            "accuracy": float(accuracy_score(target_segment.loc[valid_classification], classification_predictions.loc[valid_classification])),
            "macro_f1": float(f1_score(target_segment.loc[valid_classification], classification_predictions.loc[valid_classification], average="macro", zero_division=0)),
            "labels": labels,
        }
        save_confusion_matrix(cm, labels, output_dir / "segment_confusion_matrix.pdf")

    save_dataset_distribution(train_df, output_dir / "dataset_distribution.pdf")

    full_regression_model = _regression_pipeline()
    full_regression_model.fit(feature_frame, target_score)
    joblib.dump(full_regression_model, output_dir / "score_regression.joblib")

    if target_segment.nunique() >= 2:
        full_classifier = _classification_pipeline()
        full_classifier.fit(feature_frame, target_segment)
        joblib.dump(full_classifier, output_dir / "segment_classifier.joblib")

    (output_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (output_dir / "fold_metrics.csv").write_text(pd.DataFrame(fold_rows).to_csv(index=False))
    metrics_json = json.dumps(metrics, indent=2, default=_as_python)
    (output_dir / "metrics.json").write_text(metrics_json)
    return metrics
