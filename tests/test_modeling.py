from __future__ import annotations

from pathlib import Path

import pandas as pd

from darts_gaze.modeling import FEATURE_COLUMNS, train_baselines


def test_train_baselines_writes_models_metrics_and_plots(tmp_path: Path) -> None:
    rows = []
    match_ids = [
        "sr:sport_event:66098020",
        "sr:sport_event:66098024",
        "sr:sport_event:66098028",
        "sr:sport_event:66098032",
    ]
    for match_index, match_id in enumerate(match_ids):
        for sample_index in range(2):
            row = {column: 0.0 for column in FEATURE_COLUMNS}
            row.update(
                {
                    "review_status": "matched",
                    "sport_event_id": match_id,
                    "resulting_score": 60 if sample_index == 0 else 32,
                    "segment_label": "T20" if sample_index == 0 else "D16",
                }
            )
            row["valid_face"] = 1.0
            row["average_gaze_x"] = float(match_index + sample_index)
            row["average_gaze_y"] = float(sample_index) * 0.5
            row["head_z_axis_z"] = 1.0
            row["ipd"] = 0.12 + (0.01 * sample_index)
            rows.append(row)

    dataset_path = tmp_path / "training_samples.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    metrics = train_baselines(dataset_path, output_dir=tmp_path)

    assert metrics["num_matches"] == 4
    assert (tmp_path / "score_regression.joblib").exists()
    assert (tmp_path / "segment_classifier.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "dataset_distribution.pdf").exists()
    assert (tmp_path / "segment_confusion_matrix.pdf").exists()
