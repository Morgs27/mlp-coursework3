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
        for sample_index in range(3):
            row = {column: 0.0 for column in FEATURE_COLUMNS}
            score_value = 60 if sample_index == 0 else 19 if sample_index == 1 else 32
            segment_label = "T20" if sample_index == 0 else "S19" if sample_index == 1 else "D16"
            wedge_label = "20" if sample_index == 0 else "19" if sample_index == 1 else "16"
            row.update(
                {
                    "review_status": "matched",
                    "sport_event_id": match_id,
                    "resulting_score": score_value,
                    "segment_label": segment_label,
                    "wedge_number_label": wedge_label,
                    "coarse_wedge_area_label": wedge_label,
                    "player_name": "Fixture Player" if match_index % 2 == 0 else "Second Player",
                    "competitor_qualifier": "home" if match_index % 2 == 0 else "away",
                    "entered_modeling": True,
                }
            )
            row["valid_face"] = 1.0
            row["average_gaze_x"] = float(match_index + sample_index)
            row["average_gaze_y"] = float(sample_index) * 0.5
            row["head_z_axis_z"] = 1.0
            row["ipd"] = 0.12 + (0.01 * sample_index)
            rows.append(row)

    invalid_row = {column: 0.0 for column in FEATURE_COLUMNS}
    invalid_row.update(
        {
            "review_status": "matched",
            "sport_event_id": match_ids[0],
            "resulting_score": 10,
            "segment_label": "S10",
            "wedge_number_label": "10",
            "coarse_wedge_area_label": "OTHER",
            "player_name": "Fixture Player",
            "competitor_qualifier": "home",
            "valid_face": 0.0,
            "entered_modeling": False,
        }
    )
    rows.append(invalid_row)

    dataset_path = tmp_path / "training_samples.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    report_dir = tmp_path / "outputs"
    metrics = train_baselines(dataset_path, output_dir=tmp_path, report_dir=report_dir)

    assert metrics["num_matches"] == 4
    assert metrics["matched_rows"] == 13
    assert metrics["modeling_rows"] == 12
    assert metrics["invalid_face_rows"] == 1
    assert metrics["wedge_number_classification"]["selected_model"] in {"logistic", "knn", "extra_trees", "random_forest", "svc_rbf", "mlp_deep", "torch_deep"}
    assert metrics["coarse_wedge_area_classification"]["selected_model"] in {"knn", "extra_trees"}
    assert metrics["wedge_19_vs_20_reranker"]["selected_model"] in {"logistic", "extra_trees", "random_forest", "svc_rbf", "mlp_deep", "torch_deep"}
    assert (tmp_path / "score_regression.joblib").exists()
    assert (tmp_path / "segment_classifier.joblib").exists()
    assert (tmp_path / "wedge_number_classifier.joblib").exists()
    assert (tmp_path / "coarse_wedge_area_classifier.joblib").exists()
    assert (tmp_path / "wedge_19_vs_20_reranker.joblib").exists()
    assert (report_dir / "tables" / "metrics.json").exists()
    assert (report_dir / "tables" / "baseline_comparison.csv").exists()
    assert (report_dir / "tables" / "model_comparison.csv").exists()
    assert (report_dir / "tables" / "wedge_number_model_bootstrap.csv").exists()
    assert (report_dir / "tables" / "wedge_19_vs_20_model_bootstrap.csv").exists()
    assert (report_dir / "tables" / "wedge_19_vs_20_predictions.csv").exists()
    assert (report_dir / "figures" / "dataset_distribution.pdf").exists()
    assert (report_dir / "figures" / "dataset_distribution.png").exists()
    assert (report_dir / "figures" / "segment_confusion_matrix.pdf").exists()
    assert (report_dir / "figures" / "wedge_number_confusion_matrix.pdf").exists()
    assert (report_dir / "figures" / "coarse_wedge_area_confusion_matrix.pdf").exists()
    assert (report_dir / "figures" / "coarse_wedge_area_gaze_trend.pdf").exists()
    assert (report_dir / "figures" / "coarse_wedge_area_player_centers.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_confusion_matrix.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_probability_distribution.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_ranking_curves.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_model_curves.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_calibration_curves.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_gaze_scatter.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_player_centers.pdf").exists()
    assert (report_dir / "figures" / "wedge_number_model_comparison.pdf").exists()
    assert (report_dir / "figures" / "wedge_19_vs_20_model_comparison.pdf").exists()
    assert (report_dir / "figures" / "player_score_scatters" / "fixture_player_score_scatter.pdf").exists()
