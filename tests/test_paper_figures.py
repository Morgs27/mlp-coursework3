from __future__ import annotations

from pathlib import Path

import pandas as pd

from darts_gaze.paper_figures import export_paper_figures


def test_export_paper_figures_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "report"
    paper_dir = tmp_path / "paper"
    processed_dir.mkdir()
    (report_dir / "tables").mkdir(parents=True)

    training_rows = []
    matches = [
        "sr:sport_event:66098020",
        "sr:sport_event:66098024",
        "sr:sport_event:66098028",
        "sr:sport_event:66098032",
    ]
    for index, match_id in enumerate(matches):
        training_rows.extend(
            [
                {
                    "capture_id": f"cap-{index}-20",
                    "sport_event_id": match_id,
                    "player_name": "Fixture Player" if index % 2 == 0 else "Second Player",
                    "review_status": "matched",
                    "entered_modeling": True,
                    "valid_face": True,
                    "resulting_score": 60,
                    "wedge_number_label": "20",
                    "average_gaze_x": -0.35 + (index * 0.01),
                    "average_gaze_y": 0.10 + (index * 0.01),
                },
                {
                    "capture_id": f"cap-{index}-19",
                    "sport_event_id": match_id,
                    "player_name": "Fixture Player" if index % 2 == 0 else "Second Player",
                    "review_status": "matched",
                    "entered_modeling": True,
                    "valid_face": True,
                    "resulting_score": 19,
                    "wedge_number_label": "19",
                    "average_gaze_x": -0.42 + (index * 0.01),
                    "average_gaze_y": 0.06 + (index * 0.01),
                },
                {
                    "capture_id": f"cap-{index}-18",
                    "sport_event_id": match_id,
                    "player_name": "Fixture Player" if index % 2 == 0 else "Second Player",
                    "review_status": "matched",
                    "entered_modeling": True,
                    "valid_face": True,
                    "resulting_score": 18,
                    "wedge_number_label": "18",
                    "average_gaze_x": -0.48 + (index * 0.01),
                    "average_gaze_y": 0.14 + (index * 0.01),
                },
            ]
        )
    pd.DataFrame(training_rows).to_csv(processed_dir / "training_samples.csv", index=False)
    pd.DataFrame(training_rows).to_csv(processed_dir / "capture_quality.csv", index=False)

    model_comparison_rows = [
        {"task": "wedge_number", "model": "majority", "accuracy": 0.60, "macro_f1": 0.05, "three_wedge_accuracy": 0.67, "circular_wedge_mae": 2.2},
        {"task": "wedge_number", "model": "extra_trees", "accuracy": 0.57, "macro_f1": 0.16, "three_wedge_accuracy": 0.64, "circular_wedge_mae": 1.9},
        {"task": "wedge_number", "model": "mlp_deep", "accuracy": 0.55, "macro_f1": 0.18, "three_wedge_accuracy": 0.63, "circular_wedge_mae": 2.2},
        {"task": "wedge_19_vs_20_reranker", "model": "majority", "accuracy": 0.84, "balanced_accuracy": 0.50, "macro_f1": 0.46, "roc_auc": 0.42, "average_precision": 0.81, "brier_score": 0.14, "log_loss": 0.45, "ece_10bin": 0.01},
        {"task": "wedge_19_vs_20_reranker", "model": "logistic", "accuracy": 0.68, "balanced_accuracy": 0.61, "macro_f1": 0.56, "roc_auc": 0.71, "average_precision": 0.91, "brier_score": 0.21, "log_loss": 0.69, "ece_10bin": 0.24},
        {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "accuracy": 0.85, "balanced_accuracy": 0.57, "macro_f1": 0.58, "roc_auc": 0.69, "average_precision": 0.91, "brier_score": 0.13, "log_loss": 0.42, "ece_10bin": 0.07},
        {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "accuracy": 0.79, "balanced_accuracy": 0.65, "macro_f1": 0.63, "roc_auc": 0.70, "average_precision": 0.90, "brier_score": 0.17, "log_loss": 0.60, "ece_10bin": 0.17},
    ]
    pd.DataFrame(model_comparison_rows).to_csv(report_dir / "tables" / "model_comparison.csv", index=False)

    wedge_bootstrap_rows = [
        {"task": "wedge_number", "model": "majority", "metric": "three_wedge_accuracy", "estimate": 0.67, "ci_low": 0.62, "ci_high": 0.72},
        {"task": "wedge_number", "model": "majority", "metric": "macro_f1", "estimate": 0.05, "ci_low": 0.05, "ci_high": 0.06},
        {"task": "wedge_number", "model": "majority", "metric": "accuracy", "estimate": 0.60, "ci_low": 0.54, "ci_high": 0.65},
        {"task": "wedge_number", "model": "majority", "metric": "circular_wedge_mae", "estimate": 2.2, "ci_low": 1.8, "ci_high": 2.5},
        {"task": "wedge_number", "model": "extra_trees", "metric": "three_wedge_accuracy", "estimate": 0.64, "ci_low": 0.59, "ci_high": 0.70},
        {"task": "wedge_number", "model": "extra_trees", "metric": "macro_f1", "estimate": 0.16, "ci_low": 0.12, "ci_high": 0.20},
        {"task": "wedge_number", "model": "extra_trees", "metric": "accuracy", "estimate": 0.57, "ci_low": 0.51, "ci_high": 0.62},
        {"task": "wedge_number", "model": "extra_trees", "metric": "circular_wedge_mae", "estimate": 1.9, "ci_low": 1.6, "ci_high": 2.3},
        {"task": "wedge_number", "model": "mlp_deep", "metric": "three_wedge_accuracy", "estimate": 0.63, "ci_low": 0.58, "ci_high": 0.69},
        {"task": "wedge_number", "model": "mlp_deep", "metric": "macro_f1", "estimate": 0.18, "ci_low": 0.13, "ci_high": 0.24},
        {"task": "wedge_number", "model": "mlp_deep", "metric": "accuracy", "estimate": 0.55, "ci_low": 0.50, "ci_high": 0.60},
        {"task": "wedge_number", "model": "mlp_deep", "metric": "circular_wedge_mae", "estimate": 2.2, "ci_low": 1.9, "ci_high": 2.6},
    ]
    pd.DataFrame(wedge_bootstrap_rows).to_csv(report_dir / "tables" / "wedge_number_model_bootstrap.csv", index=False)

    reranker_bootstrap_rows = [
        {"task": "wedge_19_vs_20_reranker", "model": "logistic", "metric": "roc_auc", "estimate": 0.71, "ci_low": 0.60, "ci_high": 0.80},
        {"task": "wedge_19_vs_20_reranker", "model": "logistic", "metric": "average_precision", "estimate": 0.91, "ci_low": 0.87, "ci_high": 0.95},
        {"task": "wedge_19_vs_20_reranker", "model": "logistic", "metric": "balanced_accuracy", "estimate": 0.61, "ci_low": 0.52, "ci_high": 0.69},
        {"task": "wedge_19_vs_20_reranker", "model": "logistic", "metric": "ece_10bin", "estimate": 0.24, "ci_low": 0.19, "ci_high": 0.29},
        {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "metric": "roc_auc", "estimate": 0.69, "ci_low": 0.59, "ci_high": 0.79},
        {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "metric": "average_precision", "estimate": 0.91, "ci_low": 0.87, "ci_high": 0.95},
        {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "metric": "balanced_accuracy", "estimate": 0.57, "ci_low": 0.49, "ci_high": 0.65},
        {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "metric": "ece_10bin", "estimate": 0.07, "ci_low": 0.03, "ci_high": 0.12},
        {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "metric": "roc_auc", "estimate": 0.70, "ci_low": 0.60, "ci_high": 0.80},
        {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "metric": "average_precision", "estimate": 0.90, "ci_low": 0.85, "ci_high": 0.95},
        {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "metric": "balanced_accuracy", "estimate": 0.65, "ci_low": 0.56, "ci_high": 0.74},
        {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "metric": "ece_10bin", "estimate": 0.17, "ci_low": 0.11, "ci_high": 0.22},
    ]
    pd.DataFrame(reranker_bootstrap_rows).to_csv(report_dir / "tables" / "wedge_19_vs_20_model_bootstrap.csv", index=False)

    fold_rows = []
    for match_id in matches:
        fold_rows.extend(
            [
                {"task": "wedge_number", "model": "extra_trees", "sport_event_id": match_id, "three_wedge_accuracy": 0.60},
                {"task": "wedge_number", "model": "mlp_deep", "sport_event_id": match_id, "three_wedge_accuracy": 0.58},
                {"task": "wedge_19_vs_20_reranker", "model": "logistic", "sport_event_id": match_id, "roc_auc": 0.70},
                {"task": "wedge_19_vs_20_reranker", "model": "random_forest", "sport_event_id": match_id, "roc_auc": 0.68},
                {"task": "wedge_19_vs_20_reranker", "model": "torch_deep", "sport_event_id": match_id, "roc_auc": 0.69},
            ]
        )
    pd.DataFrame(fold_rows).to_csv(report_dir / "tables" / "fold_metrics.csv", index=False)

    reranker_prediction_rows = []
    for index, match_id in enumerate(matches):
        reranker_prediction_rows.extend(
            [
                {
                    "capture_id": f"cap-{index}-19",
                    "sport_event_id": match_id,
                    "player_name": "Fixture Player",
                    "wedge_number_label": "19",
                    "predicted_label": "19",
                    "predicted_p20": 0.22,
                    "logistic_predicted_p20": 0.22,
                    "random_forest_predicted_p20": 0.35,
                    "torch_deep_predicted_p20": 0.28,
                },
                {
                    "capture_id": f"cap-{index}-20",
                    "sport_event_id": match_id,
                    "player_name": "Fixture Player",
                    "wedge_number_label": "20",
                    "predicted_label": "20",
                    "predicted_p20": 0.81,
                    "logistic_predicted_p20": 0.81,
                    "random_forest_predicted_p20": 0.74,
                    "torch_deep_predicted_p20": 0.77,
                },
            ]
        )
    pd.DataFrame(reranker_prediction_rows).to_csv(report_dir / "tables" / "wedge_19_vs_20_predictions.csv", index=False)

    artifacts = export_paper_figures(processed_dir=processed_dir, report_dir=report_dir, output_dir=paper_dir)

    assert artifacts.notes_path.exists()
    assert (artifacts.figures_dir / "figure_01_dataset_funnel.pdf").exists()
    assert (artifacts.figures_dir / "figure_01_dataset_funnel.txt").exists()
    assert (artifacts.figures_dir / "figure_05_19_20_player_shift_vectors.pdf").exists()
    assert (artifacts.figures_dir / "figure_05_19_20_player_shift_vectors.txt").exists()
    assert (artifacts.figures_dir / "figure_09_reranker_model_ci.pdf").exists()
    assert (artifacts.figures_dir / "figure_09_reranker_model_ci.txt").exists()
    assert (artifacts.figures_dir / "figure_14_reranker_fold_stability.pdf").exists()
    assert (artifacts.figures_dir / "figure_14_reranker_fold_stability.txt").exists()
    assert (artifacts.figures_dir / "figure_15_player_average_map.pdf").exists()
    assert (artifacts.figures_dir / "figure_15_player_average_map.txt").exists()
    assert (artifacts.figures_dir / "figure_16_player_tendency_fixture_player.pdf").exists()
    assert (artifacts.figures_dir / "figure_16_player_tendency_fixture_player.txt").exists()
