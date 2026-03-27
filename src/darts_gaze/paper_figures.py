"""Paper-ready figure generation from processed datasets and evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.errors import EmptyDataError

from .config import FIRST_PASS_OUTPUTS_DIR, PROCESSED_DIR, ensure_data_directories
from .plots import (
    PALETTE,
    _save_figure,
    _slugify,
    _setup_axes,
    save_binary_calibration_curves,
    save_binary_model_curves,
    save_metric_ci_panels,
)


@dataclass(slots=True)
class PaperFigureArtifacts:
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    notes_path: Path


def export_paper_figures(
    *,
    processed_dir: str | Path = PROCESSED_DIR,
    report_dir: str | Path = FIRST_PASS_OUTPUTS_DIR,
    output_dir: str | Path = "outputs/paper",
) -> PaperFigureArtifacts:
    ensure_data_directories()
    processed_dir = Path(processed_dir)
    report_dir = Path(report_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    training_df = _read_csv_or_empty(processed_dir / "training_samples.csv")
    capture_quality_df = _read_csv_or_empty(processed_dir / "capture_quality.csv")
    model_comparison_df = _read_csv_or_empty(report_dir / "tables" / "model_comparison.csv")
    fold_metrics_df = _read_csv_or_empty(report_dir / "tables" / "fold_metrics.csv")
    wedge_bootstrap_df = _read_csv_or_empty(report_dir / "tables" / "wedge_number_model_bootstrap.csv")
    reranker_bootstrap_df = _read_csv_or_empty(report_dir / "tables" / "wedge_19_vs_20_model_bootstrap.csv")
    reranker_prediction_df = _read_csv_or_empty(report_dir / "tables" / "wedge_19_vs_20_predictions.csv")

    notes: list[tuple[str, str]] = []
    matched_rows = int(len(training_df))
    modeling_df = _modeling_subset(training_df) if not training_df.empty else pd.DataFrame()
    modeling_rows = int(len(modeling_df))
    reranker_subset_df = _reranker_subset(modeling_df) if not modeling_df.empty else pd.DataFrame()
    reranker_rows = int(len(reranker_subset_df))
    invalid_rows = int(matched_rows - modeling_rows)

    if not training_df.empty:
        if not modeling_df.empty:
            figure_path = figures_dir / "figure_01_dataset_funnel.pdf"
            _save_dataset_funnel(training_df, modeling_df, figure_path)
            _write_caption_file(
                figure_path,
                "Figure 1. Dataset funnel from all matched captures to the valid-face modeling set and the final 19-vs-20 reranker subset.",
                (
                    f"The current dataset contains {matched_rows} matched captures. Of these, {modeling_rows} rows "
                    f"({100.0 * modeling_rows / matched_rows:.1f}%) entered modeling after valid-face filtering, leaving "
                    f"{invalid_rows} rows ({100.0 * invalid_rows / matched_rows:.1f}%) excluded because gaze was not reliable enough. "
                    f"The 19-vs-20 reranker subset contains {reranker_rows} rows, which is {100.0 * reranker_rows / matched_rows:.1f}% "
                    f"of all matched captures and {100.0 * reranker_rows / modeling_rows:.1f}% of the valid-face modeling set."
                ),
            )
            notes.append(
                (
                    "figure_01_dataset_funnel.pdf",
                    "Summarizes the usable-data attrition from all matched captures to the valid-face modeling subset and the final 19-vs-20 reranker subset.",
                )
            )
            figure_path = figures_dir / "figure_02_match_player_heatmap.pdf"
            _save_match_player_heatmap(modeling_df, figure_path)
            player_counts = modeling_df["player_name"].value_counts()
            match_counts = modeling_df["sport_event_id"].value_counts()
            dominant_player = player_counts.index[0]
            dominant_player_count = int(player_counts.iloc[0])
            dominant_match = str(match_counts.index[0]).split(":")[-1]
            dominant_match_count = int(match_counts.iloc[0])
            dominant_cell = (
                modeling_df.groupby(["player_name", "sport_event_id"], dropna=False)
                .size()
                .sort_values(ascending=False)
                .iloc[0]
            )
            dominant_pair = (
                modeling_df.groupby(["player_name", "sport_event_id"], dropna=False)
                .size()
                .sort_values(ascending=False)
                .index[0]
            )
            _write_caption_file(
                figure_path,
                "Figure 2. Heatmap of valid-face modeling rows by player and match.",
                (
                    f"The modeling subset covers {modeling_df['player_name'].nunique()} players across "
                    f"{modeling_df['sport_event_id'].nunique()} matches. The largest player contribution is {dominant_player} "
                    f"with {dominant_player_count} of {modeling_rows} rows ({100.0 * dominant_player_count / modeling_rows:.1f}%), "
                    f"while match {dominant_match} contributes {dominant_match_count} rows ({100.0 * dominant_match_count / modeling_rows:.1f}%). "
                    f"The single densest player-match cell is {dominant_pair[0]} in {str(dominant_pair[1]).split(':')[-1]} with "
                    f"{int(dominant_cell)} rows, illustrating that the present dataset is still heavily concentrated in a few strata."
                ),
            )
            notes.append(
                (
                    "figure_02_match_player_heatmap.pdf",
                    "Shows which players and matches dominate the current modeling set, making the match/player imbalance explicit.",
                )
            )
            figure_path = figures_dir / "figure_03_wedge_long_tail.pdf"
            _save_wedge_long_tail(modeling_df, figure_path)
            wedge_counts = modeling_df["wedge_number_label"].astype(str).value_counts()
            top_two = int(wedge_counts.get("20", 0) + wedge_counts.get("19", 0))
            tail_count = int(wedge_counts.iloc[10:].sum()) if len(wedge_counts) > 10 else 0
            _write_caption_file(
                figure_path,
                "Figure 3. Long-tail distribution of wedge labels in the valid-face modeling subset.",
                (
                    f"Wedge 20 dominates the dataset with {int(wedge_counts.get('20', 0))} rows "
                    f"({100.0 * wedge_counts.get('20', 0) / modeling_rows:.1f}%), followed by wedge 19 with "
                    f"{int(wedge_counts.get('19', 0))} rows ({100.0 * wedge_counts.get('19', 0) / modeling_rows:.1f}%). "
                    f"Together, wedges 19 and 20 account for {top_two} of {modeling_rows} rows "
                    f"({100.0 * top_two / modeling_rows:.1f}%). The top ten wedge labels cover "
                    f"{int(wedge_counts.head(10).sum())} rows ({100.0 * wedge_counts.head(10).sum() / modeling_rows:.1f}%), "
                    f"leaving only {tail_count} rows ({100.0 * tail_count / modeling_rows:.1f}%) in the remaining tail."
                ),
            )
            notes.append(
                (
                    "figure_03_wedge_long_tail.pdf",
                    "Communicates the long-tail wedge distribution and highlights that 19 and 20 are only a small part of the full label space.",
                )
            )
            if not reranker_subset_df.empty:
                figure_path = figures_dir / "figure_04_19_20_match_balance.pdf"
                _save_19_20_breakdown(reranker_subset_df, figure_path)
                reranker_counts = reranker_subset_df["wedge_number_label"].astype(str).value_counts()
                reranker_breakdown = pd.crosstab(
                    reranker_subset_df["sport_event_id"].astype(str),
                    reranker_subset_df["wedge_number_label"].astype(str),
                ).sort_index()
                largest_match_id = reranker_breakdown.sum(axis=1).sort_values(ascending=False).index[0]
                largest_match_total = int(reranker_breakdown.loc[largest_match_id].sum())
                _write_caption_file(
                    figure_path,
                    "Figure 4. Match-level class balance for the 19-vs-20 reranker subset.",
                    (
                        f"The reranker subset contains {int(reranker_counts.get('20', 0))} examples of wedge 20 and "
                        f"{int(reranker_counts.get('19', 0))} examples of wedge 19, giving a class balance of "
                        f"{100.0 * reranker_counts.get('20', 0) / reranker_rows:.1f}% versus {100.0 * reranker_counts.get('19', 0) / reranker_rows:.1f}%. "
                        f"The largest held-out fold is match {largest_match_id.split(':')[-1]} with {largest_match_total} reranker rows "
                        f"({100.0 * largest_match_total / reranker_rows:.1f}% of the reranker set), which explains why raw accuracy alone is not an adequate metric."
                    ),
                )
                notes.append(
                    (
                        "figure_04_19_20_match_balance.pdf",
                        "Shows the class imbalance of the reranker subset by match, which explains why raw accuracy alone is misleading.",
                    )
                )
                figure_path = figures_dir / "figure_05_19_20_player_shift_vectors.pdf"
                _save_player_shift_vectors(reranker_subset_df, figure_path)
                shifts = _player_shift_summary(reranker_subset_df)
                largest_shift = max(shifts, key=lambda item: item["distance"]) if shifts else None
                smallest_shift = min(shifts, key=lambda item: item["distance"]) if shifts else None
                explanation = "The arrows join each player's centered 19 cluster to their centered 20 cluster."
                if largest_shift is not None and smallest_shift is not None:
                    explanation = (
                        f"{explanation} The largest separation is for {largest_shift['player']} "
                        f"with a centroid distance of {largest_shift['distance']:.3f} normalized gaze units "
                        f"(Δx={largest_shift['dx']:.3f}, Δy={largest_shift['dy']:.3f}), while the smallest separation is for "
                        f"{smallest_shift['player']} at {smallest_shift['distance']:.3f}. This figure gives the clearest qualitative evidence "
                        f"that the reranker task is driven by within-player relative shifts rather than a single global gaze location."
                    )
                _write_caption_file(
                    figure_path,
                    "Figure 5. Within-player centered gaze shift from wedge 19 to wedge 20.",
                    explanation,
                )
                notes.append(
                    (
                        "figure_05_19_20_player_shift_vectors.pdf",
                        "Visualizes the within-player gaze shift from 19 to 20 targets, which is the clearest qualitative reranker signal.",
                    )
                )
            figure_path = figures_dir / "figure_15_player_average_map.pdf"
            highlighted_labels = _select_highlight_labels(modeling_df, top_n=6)
            _save_player_average_map(
                modeling_df,
                figure_path,
                highlight_labels=highlighted_labels,
            )
            highlighted_rows = int(
                modeling_df["wedge_number_label"].astype(str).isin(highlighted_labels).sum()
            )
            _write_caption_file(
                figure_path,
                "Figure 15. Combined player-average gaze map for the dominant wedge areas in the modeling subset.",
                (
                    f"This figure overlays per-player wedge centroids for the highlighted areas {', '.join(highlighted_labels)}. "
                    f"These wedges account for {highlighted_rows} of {modeling_rows} valid-face rows "
                    f"({100.0 * highlighted_rows / modeling_rows:.1f}%). Black markers denote each player's overall gaze center, "
                    "and colored centroid markers show how each player's typical gaze location shifts relative to that center for the most common wedge targets."
                ),
            )
            notes.append(
                (
                    "figure_15_player_average_map.pdf",
                    "Summarizes the dominant player-specific gaze tendencies by connecting each player's overall center to their wedge-average centroids.",
                )
            )
            for figure_index, player_name in enumerate(
                sorted(modeling_df["player_name"].dropna().astype(str).unique().tolist()),
                start=16,
            ):
                player_df = modeling_df[modeling_df["player_name"].astype(str) == player_name].copy()
                figure_path = figures_dir / f"figure_{figure_index:02d}_player_tendency_{_slugify(player_name)}.pdf"
                selected_labels = _save_player_tendency_scatter(
                    player_df,
                    figure_path,
                    highlight_limit=6,
                )
                if not selected_labels:
                    continue
                highlighted_count = int(
                    player_df["wedge_number_label"].astype(str).isin(selected_labels).sum()
                )
                top_summary = _top_label_summary(player_df, selected_labels, limit=4)
                match_summary = _player_match_summary(player_df)
                _write_caption_file(
                    figure_path,
                    f"Figure {figure_index}. {player_name} side-by-side gaze tendency panels by match and in aggregate.",
                    (
                        f"This panel contains {len(player_df)} valid-face rows for {player_name}. "
                        f"The highlighted wedges {', '.join(selected_labels)} account for {highlighted_count} rows "
                        f"({100.0 * highlighted_count / max(len(player_df), 1):.1f}% of this player's modeling data). "
                        f"The match-specific panels are {match_summary}, followed by a combined panel that overlays all matches using distinct markers. "
                        f"Large translucent disks mark per-wedge average gaze position, and the strongest highlighted counts are {top_summary}."
                    ),
                )
                notes.append(
                    (
                        figure_path.name,
                        f"Player-specific multi-panel tendency map for {player_name}, split by match with a final combined panel.",
                    )
                )

    if not wedge_bootstrap_df.empty:
        figure_path = figures_dir / "figure_06_wedge_model_ci.pdf"
        save_metric_ci_panels(
            metric_df=wedge_bootstrap_df,
            output_path=figure_path,
            title="Model Comparison For The Wedge Task",
            metric_order=["three_wedge_accuracy", "macro_f1", "accuracy", "circular_wedge_mae"],
            metric_labels={
                "three_wedge_accuracy": "3-Wedge Accuracy",
                "macro_f1": "Macro-F1",
                "accuracy": "Exact Wedge Accuracy",
                "circular_wedge_mae": "Circular Wedge MAE",
            },
            lower_is_better={"circular_wedge_mae"},
        )
        best_three = _best_metric_row(wedge_bootstrap_df, "three_wedge_accuracy")
        best_macro = _best_metric_row(wedge_bootstrap_df, "macro_f1")
        best_mae = _best_metric_row(wedge_bootstrap_df, "circular_wedge_mae", lower_is_better=True)
        _write_caption_file(
            figure_path,
            "Figure 6. Wedge-model comparison with 95% bootstrap confidence intervals.",
            (
                f"The highest 3-wedge accuracy is achieved by {best_three['model']} at {best_three['estimate']:.3f} "
                f"(95% CI {best_three['ci_low']:.3f}-{best_three['ci_high']:.3f}), but that performance is largely driven by dataset skew. "
                f"Among non-degenerate learned models, `mlp_deep` gives the best macro-F1 at {best_macro['estimate']:.3f} "
                f"(95% CI {best_macro['ci_low']:.3f}-{best_macro['ci_high']:.3f}), while `extra_trees` yields the lowest circular wedge MAE at "
                f"{best_mae['estimate']:.3f} (95% CI {best_mae['ci_low']:.3f}-{best_mae['ci_high']:.3f})."
            ),
        )
        notes.append(
            (
                "figure_06_wedge_model_ci.pdf",
                "Provides the main wedge-task comparison with 95% bootstrap confidence intervals for exact, coarse, and circular metrics.",
            )
        )
        figure_path = figures_dir / "figure_07_wedge_model_tradeoff.pdf"
        _save_wedge_tradeoff(model_comparison_df, figure_path)
        wedge_subset = model_comparison_df[model_comparison_df["task"] == "wedge_number"].copy()
        best_tradeoff = wedge_subset.loc[wedge_subset["macro_f1"].idxmax()]
        best_three_overall = wedge_subset.loc[wedge_subset["three_wedge_accuracy"].idxmax()]
        _write_caption_file(
            figure_path,
            "Figure 7. Trade-off between coarse 3-wedge targeting and exact wedge discrimination.",
            (
                f"`{best_three_overall['model']}` sits at the top-right for 3-wedge accuracy with {best_three_overall['three_wedge_accuracy']:.3f}, "
                f"but its macro-F1 remains only {best_three_overall['macro_f1']:.3f}, indicating weak exact-wedge discrimination. "
                f"`{best_tradeoff['model']}` provides the best exact-label discrimination in this comparison with macro-F1 {best_tradeoff['macro_f1']:.3f} "
                f"at 3-wedge accuracy {best_tradeoff['three_wedge_accuracy']:.3f}, which makes it the most defensible learned trade-off model."
            ),
        )
        notes.append(
            (
                "figure_07_wedge_model_tradeoff.pdf",
                "Shows the trade-off between 3-wedge targeting and exact multiclass discrimination, separating useful from degenerate models.",
                )
        )

    if not fold_metrics_df.empty:
        figure_path = figures_dir / "figure_08_wedge_fold_stability.pdf"
        _save_fold_stability(
            fold_metrics_df,
            task="wedge_number",
            metric="three_wedge_accuracy",
            output_path=figure_path,
            title="3-Wedge Accuracy By Match",
        )
        fold_summary = _fold_metric_summary(fold_metrics_df, task="wedge_number", metric="three_wedge_accuracy")
        _write_caption_file(
            figure_path,
            "Figure 8. Match-wise stability of 3-wedge accuracy for the strongest wedge models.",
            fold_summary,
        )
        notes.append(
            (
                "figure_08_wedge_fold_stability.pdf",
                "Shows whether wedge performance is stable across held-out matches rather than being driven by a single fixture.",
            )
        )

    if not reranker_bootstrap_df.empty:
        figure_path = figures_dir / "figure_09_reranker_model_ci.pdf"
        save_metric_ci_panels(
            metric_df=reranker_bootstrap_df,
            output_path=figure_path,
            title="Model Comparison For The 19 vs 20 Reranker",
            metric_order=["roc_auc", "average_precision", "balanced_accuracy", "ece_10bin"],
            metric_labels={
                "roc_auc": "ROC-AUC",
                "average_precision": "Average Precision",
                "balanced_accuracy": "Balanced Accuracy",
                "ece_10bin": "ECE (10 bins)",
            },
            lower_is_better={"ece_10bin"},
        )
        best_auc = _best_metric_row(reranker_bootstrap_df, "roc_auc")
        best_ap = _best_metric_row(reranker_bootstrap_df, "average_precision")
        best_bal = _best_metric_row(reranker_bootstrap_df, "balanced_accuracy")
        learned_reranker_bootstrap_df = reranker_bootstrap_df[reranker_bootstrap_df["model"] != "majority"].copy()
        best_ece = _best_metric_row(learned_reranker_bootstrap_df, "ece_10bin", lower_is_better=True)
        _write_caption_file(
            figure_path,
            "Figure 9. Reranker comparison with 95% bootstrap confidence intervals for ranking, threshold, and calibration metrics.",
            (
                f"`{best_auc['model']}` has the strongest ranking performance with ROC-AUC {best_auc['estimate']:.3f} "
                f"(95% CI {best_auc['ci_low']:.3f}-{best_auc['ci_high']:.3f}) and average precision {best_ap['estimate']:.3f} "
                f"(95% CI {best_ap['ci_low']:.3f}-{best_ap['ci_high']:.3f}). The highest balanced accuracy is delivered by "
                f"`{best_bal['model']}` at {best_bal['estimate']:.3f}, while the lowest calibration error among the learned models is achieved by "
                f"`{best_ece['model']}` at ECE {best_ece['estimate']:.3f}. This split shows that ranking quality and calibration are not optimized by the same model."
            ),
        )
        notes.append(
            (
                "figure_09_reranker_model_ci.pdf",
                "Summarizes ranking quality, threshold performance, and confidence calibration for the 19-vs-20 reranker.",
            )
        )
        figure_path = figures_dir / "figure_10_reranker_tradeoff.pdf"
        _save_reranker_tradeoff(model_comparison_df, figure_path)
        reranker_subset = model_comparison_df[model_comparison_df["task"] == "wedge_19_vs_20_reranker"].copy()
        logistic_row = _model_row(reranker_subset, "logistic")
        forest_row = _model_row(reranker_subset, "random_forest")
        torch_row = _model_row(reranker_subset, "torch_deep")
        _write_caption_file(
            figure_path,
            "Figure 10. Trade-off between reranker ranking quality, threshold performance, and calibration.",
            (
                f"`logistic` yields the best ranking score with ROC-AUC {logistic_row['roc_auc']:.3f} but has the worst calibration among the learned models "
                f"(ECE {logistic_row['ece_10bin']:.3f}). `random_forest` sacrifices some ROC-AUC ({forest_row['roc_auc']:.3f}) but substantially improves calibration "
                f"(ECE {forest_row['ece_10bin']:.3f}) and log loss ({forest_row['log_loss']:.3f}). `torch_deep` achieves the strongest threshold performance "
                f"with balanced accuracy {torch_row['balanced_accuracy']:.3f}, making it a useful comparison point even though its calibration is weaker than the forest model."
            ),
        )
        notes.append(
            (
                "figure_10_reranker_tradeoff.pdf",
                "Highlights the ranking-calibration trade-off: logistic is best for ranking, while forest-style models are better calibrated.",
            )
        )

    selected_reranker_models = _select_reranker_models(model_comparison_df)
    if not reranker_prediction_df.empty and selected_reranker_models:
        probability_map = _probability_map(reranker_prediction_df, selected_reranker_models)
        y_true = (reranker_prediction_df["wedge_number_label"].astype(str) == "20").astype(int)
        reranker_rows_df = model_comparison_df[model_comparison_df["task"] == "wedge_19_vs_20_reranker"]
        logistic_row = _model_row(reranker_rows_df, "logistic")
        forest_row = _model_row(reranker_rows_df, "random_forest")
        torch_row = _model_row(reranker_rows_df, "torch_deep")
        figure_path = figures_dir / "figure_11_reranker_selected_curves.pdf"
        save_binary_model_curves(
            y_true=y_true,
            probabilities_by_model=probability_map,
            output_path=figure_path,
            positive_label="20",
        )
        selected_rows = [_model_row(model_comparison_df[model_comparison_df["task"] == "wedge_19_vs_20_reranker"], model_name) for model_name in selected_reranker_models]
        auc_summary = ", ".join(f"{row['model']}={row['roc_auc']:.3f}" for row in selected_rows)
        _write_caption_file(
            figure_path,
            "Figure 11. ROC and precision-recall curves for the selected reranker models.",
            (
                f"The selected comparison set contains {', '.join(selected_reranker_models)}. Their overall ROC-AUC values are "
                + auc_summary
                + ". Average precision remains high for all three models, ranging from "
                + f"{min(row['average_precision'] for row in selected_rows):.3f} to {max(row['average_precision'] for row in selected_rows):.3f}, "
                "showing that the reranker is useful as a ranking stage even though hard-threshold accuracy is damped by class imbalance."
            ),
        )
        notes.append(
            (
                "figure_11_reranker_selected_curves.pdf",
                "Restricts ROC and precision-recall curves to the most relevant reranker models so the ranking differences are readable.",
            )
        )
        figure_path = figures_dir / "figure_12_reranker_selected_calibration.pdf"
        save_binary_calibration_curves(
            y_true=y_true,
            probabilities_by_model=probability_map,
            output_path=figure_path,
            num_bins=8,
        )
        _write_caption_file(
            figure_path,
            "Figure 12. Calibration curves for the selected reranker models.",
            (
                f"Among the selected rerankers, calibration is strongest for `random_forest` with ECE {forest_row['ece_10bin']:.3f}, "
                f"followed by `torch_deep` at {torch_row['ece_10bin']:.3f}, while `logistic` is the least calibrated at {logistic_row['ece_10bin']:.3f}. "
                "The histogram panel shows that most models still concentrate probability mass toward the high-P(20) region because the reranker subset itself is 83.8% wedge 20."
            ),
        )
        notes.append(
            (
                "figure_12_reranker_selected_calibration.pdf",
                "Focuses on calibration behavior for the selected reranker models, which matters for downstream confidence-aware reranking.",
            )
        )
        figure_path = figures_dir / "figure_13_reranker_probability_boxplots.pdf"
        _save_probability_boxplots(
            reranker_prediction_df,
            selected_reranker_models,
            figure_path,
        )
        prob_summary = _probability_boxplot_summary(reranker_prediction_df, selected_reranker_models)
        _write_caption_file(
            figure_path,
            "Figure 13. Probability separation between true wedge 19 and true wedge 20 examples for the selected rerankers.",
            prob_summary,
        )
        notes.append(
            (
                "figure_13_reranker_probability_boxplots.pdf",
                "Shows how strongly each selected reranker separates true 19s from true 20s in probability space.",
            )
        )

    if not fold_metrics_df.empty:
        figure_path = figures_dir / "figure_14_reranker_fold_stability.pdf"
        _save_fold_stability(
            fold_metrics_df,
            task="wedge_19_vs_20_reranker",
            metric="roc_auc",
            output_path=figure_path,
            title="19 vs 20 ROC-AUC By Match",
        )
        _write_caption_file(
            figure_path,
            "Figure 14. Match-wise stability of ROC-AUC for the strongest reranker models.",
            _fold_metric_summary(fold_metrics_df, task="wedge_19_vs_20_reranker", metric="roc_auc"),
        )
        notes.append(
            (
                "figure_14_reranker_fold_stability.pdf",
                "Shows whether reranker ranking quality generalizes across held-out matches instead of depending on a single fold.",
            )
        )

    notes_path = tables_dir / "paper_figure_notes.md"
    notes_path.write_text(_render_notes(notes))

    return PaperFigureArtifacts(
        output_dir=output_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        notes_path=notes_path,
    )


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _write_caption_file(figure_path: Path, caption: str, explanation: str) -> Path:
    caption_path = figure_path.with_suffix(".txt")
    caption_path.write_text(f"Caption:\n{caption}\n\nExplanation:\n{explanation}\n")
    return caption_path


def _best_metric_row(metric_df: pd.DataFrame, metric_name: str, *, lower_is_better: bool = False) -> pd.Series:
    subset = metric_df[metric_df["metric"] == metric_name].copy()
    if subset.empty:
        raise ValueError(f"Metric {metric_name} not found")
    subset = subset.sort_values("estimate", ascending=lower_is_better)
    return subset.iloc[0]


def _model_row(model_df: pd.DataFrame, model_name: str) -> pd.Series:
    subset = model_df[model_df["model"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"Model {model_name} not found")
    return subset.iloc[0]


def _modeling_subset(training_df: pd.DataFrame) -> pd.DataFrame:
    if training_df.empty:
        return training_df
    if "entered_modeling" in training_df.columns:
        modeling_mask = training_df["entered_modeling"].fillna(False).astype(bool)
    else:
        modeling_mask = training_df["valid_face"].fillna(False).astype(bool)
    return training_df[modeling_mask].copy()


def _reranker_subset(modeling_df: pd.DataFrame) -> pd.DataFrame:
    subset = modeling_df[modeling_df["wedge_number_label"].astype(str).isin(["19", "20"])].copy()
    subset = subset[subset["average_gaze_x"].notna() & subset["average_gaze_y"].notna()]
    return subset


def _save_dataset_funnel(training_df: pd.DataFrame, modeling_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    counts = pd.Series(
        {
            "Matched Captures": len(training_df),
            "Valid-Face Rows": len(modeling_df),
            "19 vs 20 Subset": int(modeling_df["wedge_number_label"].astype(str).isin(["19", "20"]).sum()),
        }
    )
    axis.barh(counts.index, counts.values, color=[PALETTE["steel"], PALETTE["teal"], PALETTE["coral"]])
    axis.set_xlabel("Rows")
    axis.set_title("Data Funnel")
    max_value = float(max(counts.max(), 1))
    for idx, value in enumerate(counts.values):
        axis.text(value + (max_value * 0.01), idx, f"{int(value)}", va="center", color=PALETTE["ink"], fontweight="bold")
    axis.invert_yaxis()
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_match_player_heatmap(modeling_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    heatmap_df = modeling_df.pivot_table(
        index="player_name",
        columns="sport_event_id",
        values="resulting_score",
        aggfunc="size",
        fill_value=0,
    )
    if heatmap_df.empty:
        return
    figure, axis = plt.subplots(figsize=(8.5, max(4.5, 0.8 * len(heatmap_df))))
    image = axis.imshow(heatmap_df.to_numpy(), cmap="YlGnBu")
    axis.set_xticks(range(len(heatmap_df.columns)), labels=[column.split(":")[-1] for column in heatmap_df.columns], rotation=30, ha="right")
    axis.set_yticks(range(len(heatmap_df.index)), labels=heatmap_df.index.tolist())
    axis.set_xlabel("Match")
    axis.set_ylabel("Player")
    axis.set_title("Modeling Coverage By Player And Match")
    for row_index in range(heatmap_df.shape[0]):
        for column_index in range(heatmap_df.shape[1]):
            axis.text(column_index, row_index, int(heatmap_df.iat[row_index, column_index]), ha="center", va="center", color=PALETTE["ink"])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_wedge_long_tail(modeling_df: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    _setup_axes()
    counts = modeling_df["wedge_number_label"].astype(str).value_counts()
    top_counts = counts.head(top_n)
    other_count = int(counts.iloc[top_n:].sum())
    labels = top_counts.index.tolist()
    values = top_counts.values.tolist()
    if other_count:
        labels.append("OTHER")
        values.append(other_count)

    color_map = []
    for label in labels:
        if label == "20":
            color_map.append(PALETTE["coral"])
        elif label == "19":
            color_map.append(PALETTE["sand"])
        else:
            color_map.append(PALETTE["steel"])

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(labels, values, color=color_map)
    axis.set_ylabel("Samples")
    axis.set_title("Wedge Label Distribution")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_19_20_breakdown(reranker_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    breakdown = pd.crosstab(reranker_df["sport_event_id"].astype(str), reranker_df["wedge_number_label"].astype(str)).sort_index()
    for label in ["19", "20"]:
        if label not in breakdown.columns:
            breakdown[label] = 0
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    matches = [match_id.split(":")[-1] for match_id in breakdown.index.tolist()]
    axis.bar(matches, breakdown["19"], color=PALETTE["sand"], label="19")
    axis.bar(matches, breakdown["20"], bottom=breakdown["19"], color=PALETTE["coral"], label="20")
    axis.set_ylabel("Samples")
    axis.set_title("19 vs 20 Subset By Match")
    axis.legend(loc="upper right")
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_player_shift_vectors(reranker_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    figure, axis = plt.subplots(figsize=(8.5, 7))
    players = sorted(reranker_df["player_name"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    for index, player_name in enumerate(players):
        player_df = reranker_df[reranker_df["player_name"].astype(str) == player_name]
        if player_df["wedge_number_label"].astype(str).nunique() < 2:
            continue
        center = player_df[["average_gaze_x", "average_gaze_y"]].mean()
        by_label = player_df.groupby(player_df["wedge_number_label"].astype(str))[["average_gaze_x", "average_gaze_y"]].mean()
        if not {"19", "20"}.issubset(set(by_label.index)):
            continue
        point_19 = by_label.loc["19"] - center
        point_20 = by_label.loc["20"] - center
        color = cmap(index % 10)
        axis.scatter(point_19["average_gaze_x"], point_19["average_gaze_y"], marker="s", s=90, color=color, edgecolors=PALETTE["ink"], linewidths=0.8)
        axis.scatter(point_20["average_gaze_x"], point_20["average_gaze_y"], marker="o", s=90, color=color, edgecolors=PALETTE["ink"], linewidths=0.8)
        axis.annotate("", xy=(point_20["average_gaze_x"], point_20["average_gaze_y"]), xytext=(point_19["average_gaze_x"], point_19["average_gaze_y"]), arrowprops={"arrowstyle": "->", "linewidth": 2, "color": color})
        axis.text(point_20["average_gaze_x"] + 0.01, point_20["average_gaze_y"], player_name.split(",")[0], color=PALETTE["ink"], fontsize=10)

    axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.5)
    axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.5)
    axis.set_xlabel("Centered Horizontal Gaze")
    axis.set_ylabel("Centered Vertical Gaze")
    axis.set_title("Within-Player Gaze Shift From 19 To 20")
    axis.invert_yaxis()
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_wedge_tradeoff(model_comparison_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    subset = model_comparison_df[model_comparison_df["task"] == "wedge_number"].copy()
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    for _, row in subset.iterrows():
        model_name = str(row["model"])
        is_baseline = model_name == "majority"
        color = PALETTE["steel"] if is_baseline else PALETTE["teal"]
        axis.scatter(
            row["three_wedge_accuracy"],
            row["macro_f1"],
            s=max(120, 400 / max(float(row["circular_wedge_mae"]), 0.5)),
            color=color,
            alpha=0.85,
            edgecolors=PALETTE["ink"],
            linewidths=0.8,
        )
        axis.text(row["three_wedge_accuracy"] + 0.005, row["macro_f1"] + 0.004, model_name, fontsize=10, color=PALETTE["ink"])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, max(0.25, float(subset["macro_f1"].max()) + 0.04))
    axis.set_xlabel("3-Wedge Accuracy")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Wedge Model Trade-Off")
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_reranker_tradeoff(model_comparison_df: pd.DataFrame, output_path: Path) -> None:
    _setup_axes()
    subset = model_comparison_df[model_comparison_df["task"] == "wedge_19_vs_20_reranker"].copy()
    if subset.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for _, row in subset.iterrows():
        model_name = str(row["model"])
        color = PALETTE["steel"] if model_name == "majority" else PALETTE["coral"]
        axes[0].scatter(row["roc_auc"], row["balanced_accuracy"], s=180, color=color, alpha=0.85, edgecolors=PALETTE["ink"], linewidths=0.8)
        axes[0].text(row["roc_auc"] + 0.004, row["balanced_accuracy"] + 0.004, model_name, fontsize=9, color=PALETTE["ink"])
        axes[1].scatter(row["ece_10bin"], row["average_precision"], s=180, color=color, alpha=0.85, edgecolors=PALETTE["ink"], linewidths=0.8)
        axes[1].text(row["ece_10bin"] + 0.004, row["average_precision"] + 0.004, model_name, fontsize=9, color=PALETTE["ink"])
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title("Ranking vs Threshold Performance")
    axes[1].set_xlabel("ECE (10 bins)")
    axes[1].set_ylabel("Average Precision")
    axes[1].set_title("Calibration vs Ranking Quality")
    axes[0].set_xlim(0.35, 1.0)
    axes[0].set_ylim(0.4, 0.9)
    axes[1].set_xlim(-0.01, max(0.3, float(subset["ece_10bin"].max()) + 0.02))
    axes[1].set_ylim(0.75, 1.0)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_probability_boxplots(reranker_prediction_df: pd.DataFrame, selected_models: list[str], output_path: Path) -> None:
    _setup_axes()
    if not selected_models:
        return
    figure, axes = plt.subplots(1, len(selected_models), figsize=(4.5 * len(selected_models), 4.5), squeeze=False)
    true_labels = reranker_prediction_df["wedge_number_label"].astype(str)
    for axis, model_name in zip(axes[0], selected_models, strict=True):
        probability_column = f"{model_name}_predicted_p20"
        if probability_column not in reranker_prediction_df.columns:
            continue
        probabilities = reranker_prediction_df[probability_column].astype(float)
        box_data = [
            probabilities[true_labels == "19"].to_numpy(),
            probabilities[true_labels == "20"].to_numpy(),
        ]
        boxplot = axis.boxplot(box_data, tick_labels=["19", "20"], patch_artist=True)
        for patch, color in zip(boxplot["boxes"], [PALETTE["sand"], PALETTE["coral"]], strict=True):
            patch.set_facecolor(color)
        axis.set_ylim(0, 1)
        axis.set_title(model_name)
        axis.set_ylabel("Predicted P(20)")
    figure.suptitle("Probability Separation For Selected Rerankers", fontsize=14, y=1.02)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _save_fold_stability(
    fold_metrics_df: pd.DataFrame,
    *,
    task: str,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    _setup_axes()
    subset = fold_metrics_df[fold_metrics_df["task"] == task].copy()
    if subset.empty or metric not in subset.columns:
        return
    overall_metric = "roc_auc" if task == "wedge_19_vs_20_reranker" else "three_wedge_accuracy"
    model_order = (
        subset.groupby("model", dropna=False)[metric]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    model_order = [model for model in model_order if model != "majority"][:4]
    if not model_order:
        return
    figure, axis = plt.subplots(figsize=(8.5, 5))
    matches = sorted(subset["sport_event_id"].astype(str).unique().tolist())
    x_positions = np.arange(len(matches))
    cmap = plt.get_cmap("tab10")
    for index, model_name in enumerate(model_order):
        model_df = subset[subset["model"] == model_name].copy()
        model_df = model_df.set_index("sport_event_id").reindex(matches)
        axis.plot(
            x_positions,
            model_df[metric].to_numpy(),
            marker="o",
            linewidth=2,
            color=cmap(index % 10),
            label=model_name,
        )
    axis.set_xticks(x_positions, labels=[match.split(":")[-1] for match in matches], rotation=30)
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_xlabel("Held-Out Match")
    axis.set_title(title)
    axis.legend(loc="best")
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _select_reranker_models(model_comparison_df: pd.DataFrame) -> list[str]:
    subset = model_comparison_df[model_comparison_df["task"] == "wedge_19_vs_20_reranker"].copy()
    if subset.empty:
        return []
    subset = subset[subset["model"] != "majority"].copy()
    if subset.empty:
        return []
    preferred = ["logistic", "random_forest", "torch_deep", "extra_trees", "svc_rbf"]
    selected = [model for model in preferred if model in set(subset["model"].astype(str))]
    return selected[:3]


def _probability_map(reranker_prediction_df: pd.DataFrame, selected_models: list[str]) -> dict[str, pd.Series]:
    probability_map: dict[str, pd.Series] = {}
    for model_name in selected_models:
        column = f"{model_name}_predicted_p20"
        if column in reranker_prediction_df.columns:
            probability_map[model_name] = reranker_prediction_df[column].astype(float)
    return probability_map


def _player_shift_summary(reranker_df: pd.DataFrame) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for player_name, player_df in reranker_df.groupby("player_name", dropna=False):
        label_set = set(player_df["wedge_number_label"].astype(str))
        if not {"19", "20"}.issubset(label_set):
            continue
        center = player_df[["average_gaze_x", "average_gaze_y"]].mean()
        by_label = player_df.groupby(player_df["wedge_number_label"].astype(str))[["average_gaze_x", "average_gaze_y"]].mean()
        point_19 = by_label.loc["19"] - center
        point_20 = by_label.loc["20"] - center
        dx = float(point_20["average_gaze_x"] - point_19["average_gaze_x"])
        dy = float(point_20["average_gaze_y"] - point_19["average_gaze_y"])
        summaries.append(
            {
                "player": str(player_name).split(",")[0],
                "dx": dx,
                "dy": dy,
                "distance": float((dx**2 + dy**2) ** 0.5),
            }
        )
    return summaries


def _fold_metric_summary(fold_metrics_df: pd.DataFrame, *, task: str, metric: str, top_n: int = 4) -> str:
    subset = fold_metrics_df[fold_metrics_df["task"] == task].copy()
    if subset.empty or metric not in subset.columns:
        return "The corresponding fold-level data were not available when this caption file was generated."
    model_means = subset.groupby("model", dropna=False)[metric].mean().sort_values(ascending=False)
    selected_models = [model for model in model_means.index.tolist() if model != "majority"][:top_n]
    if not selected_models:
        return "No non-majority models were available for fold-level summary."
    summaries: list[str] = []
    for model_name in selected_models:
        model_values = subset[subset["model"] == model_name][metric].astype(float)
        summaries.append(
            f"{model_name}: mean {model_values.mean():.3f}, range {model_values.min():.3f}-{model_values.max():.3f}"
        )
    metric_label = metric.replace("_", " ")
    return (
        f"The figure plots the strongest non-majority models by held-out-match {metric_label}. "
        + " ; ".join(summaries)
        + ". The spread across folds shows that current performance is still sensitive to match composition and limited sample size."
    )


def _probability_boxplot_summary(reranker_prediction_df: pd.DataFrame, selected_models: list[str]) -> str:
    summaries: list[str] = []
    true_labels = reranker_prediction_df["wedge_number_label"].astype(str)
    for model_name in selected_models:
        column = f"{model_name}_predicted_p20"
        if column not in reranker_prediction_df.columns:
            continue
        probabilities = reranker_prediction_df[column].astype(float)
        probabilities_19 = probabilities[true_labels == "19"]
        probabilities_20 = probabilities[true_labels == "20"]
        summaries.append(
            f"{model_name}: median P(20) is {probabilities_19.median():.3f} for true 19 and {probabilities_20.median():.3f} for true 20"
        )
    if not summaries:
        return "Probability summaries were not available for the selected reranker models."
    return (
        "The boxplots compare predicted P(20) for true 19 and true 20 samples. "
        + " ; ".join(summaries)
        + ". Wider separation between the medians and interquartile ranges indicates a more useful reranking signal."
    )


def _select_highlight_labels(df: pd.DataFrame, *, top_n: int, min_count: int = 2) -> list[str]:
    if df.empty or "wedge_number_label" not in df.columns:
        return []
    counts = df["wedge_number_label"].astype(str).value_counts()
    counts = counts[counts >= min_count]
    labels = [label for label in counts.index.tolist() if label not in {"MISS", "OTHER"}]
    labels = sorted(labels[:top_n], key=_label_sort_key)
    return labels


def _label_sort_key(value: str) -> tuple[int, int | str]:
    upper_value = value.upper()
    if value.isdigit():
        return (0, int(value))
    if upper_value == "BULL":
        return (1, 25)
    return (2, upper_value)


def _build_label_color_map(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    if not labels:
        return {}
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(labels)))
    return {label: color for label, color in zip(labels, colors, strict=True)}


def _save_player_tendency_scatter(
    player_df: pd.DataFrame,
    output_path: Path,
    *,
    highlight_limit: int,
) -> list[str]:
    _setup_axes()
    required_columns = {"average_gaze_x", "average_gaze_y", "sport_event_id", "wedge_number_label"}
    if player_df.empty or not required_columns.issubset(player_df.columns):
        return []

    plot_df = player_df.copy()
    plot_df = plot_df[
        plot_df["average_gaze_x"].notna()
        & plot_df["average_gaze_y"].notna()
        & plot_df["wedge_number_label"].notna()
    ]
    if plot_df.empty:
        return []

    selected_labels = _select_highlight_labels(plot_df, top_n=highlight_limit, min_count=1)
    if not selected_labels:
        return []

    label_colors = _build_label_color_map(selected_labels)
    matches = sorted(plot_df["sport_event_id"].astype(str).unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    match_markers = {match_id: markers[index % len(markers)] for index, match_id in enumerate(matches)}
    panel_specs = [(match_id, plot_df[plot_df["sport_event_id"].astype(str) == match_id].copy()) for match_id in matches]
    panel_specs.append(("combined", plot_df.copy()))

    x_limits, y_limits = _compute_gaze_limits(plot_df)
    figure, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(5.0 * len(panel_specs), 5.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes[0]

    for axis, (panel_name, panel_df) in zip(flat_axes, panel_specs, strict=True):
        _draw_player_tendency_panel(
            axis,
            panel_df,
            panel_name=panel_name,
            selected_labels=selected_labels,
            label_colors=label_colors,
            match_markers=match_markers,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.35)
        axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.35)
        axis.invert_yaxis()
        axis.grid(True, alpha=0.28)

    player_name = str(plot_df["player_name"].iloc[0]) if "player_name" in plot_df.columns else "Player"
    flat_axes[0].set_ylabel("Vertical Gaze (Normalized)")
    for axis in flat_axes:
        axis.set_xlabel("Horizontal Gaze (Normalized)")

    label_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=label_colors[label],
            markeredgecolor=PALETTE["ink"],
            markersize=9,
            label=str(label),
        )
        for label in selected_labels
    ]
    match_handles = [
        Line2D(
            [0],
            [0],
            marker=match_markers[match_id],
            color="w",
            markerfacecolor=PALETTE["steel"],
            markeredgecolor=PALETTE["ink"],
            markersize=9,
            label=match_id.split(":")[-1],
        )
        for match_id in matches
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#C7CDD4",
            markeredgecolor="none",
            markersize=8,
            label="Other wedges",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor=PALETTE["ink"],
            markeredgecolor=PALETTE["ink"],
            markersize=9,
            label="Panel center",
        ),
    ]
    figure.legend(
        handles=label_handles + match_handles + context_handles,
        title="Highlighted Wedge And Match",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(label_handles + match_handles + context_handles)),
        frameon=True,
    )
    figure.suptitle(f"{player_name}: Gaze Tendencies By Match And In Aggregate", fontsize=15, y=0.98)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save_figure(figure, output_path)
    plt.close(figure)
    return selected_labels


def _save_player_average_map(
    modeling_df: pd.DataFrame,
    output_path: Path,
    *,
    highlight_labels: list[str],
) -> None:
    _setup_axes()
    required_columns = {"average_gaze_x", "average_gaze_y", "player_name", "wedge_number_label"}
    if modeling_df.empty or not required_columns.issubset(modeling_df.columns) or not highlight_labels:
        return

    plot_df = modeling_df.copy()
    plot_df = plot_df[
        plot_df["average_gaze_x"].notna()
        & plot_df["average_gaze_y"].notna()
        & plot_df["player_name"].notna()
        & plot_df["wedge_number_label"].notna()
    ]
    plot_df = plot_df[plot_df["wedge_number_label"].astype(str).isin(highlight_labels)]
    if plot_df.empty:
        return

    label_colors = _build_label_color_map(highlight_labels)
    players = sorted(plot_df["player_name"].astype(str).unique().tolist())
    player_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    center_markers = ["X", "P", "*", "D", "h", "8", "d", "H", "p"]
    marker_map = {player: player_markers[index % len(player_markers)] for index, player in enumerate(players)}
    center_map = {player: center_markers[index % len(center_markers)] for index, player in enumerate(players)}

    figure, axis = plt.subplots(figsize=(12, 10))
    for player_name in players:
        player_df = plot_df[plot_df["player_name"].astype(str) == player_name]
        center = player_df[["average_gaze_x", "average_gaze_y"]].mean()
        axis.scatter(
            center["average_gaze_x"],
            center["average_gaze_y"],
            color=PALETTE["ink"],
            marker=center_map[player_name],
            s=180,
            zorder=6,
        )
        by_label = player_df.groupby(player_df["wedge_number_label"].astype(str))[["average_gaze_x", "average_gaze_y"]].mean()
        for label in highlight_labels:
            if label not in by_label.index:
                continue
            centroid = by_label.loc[label]
            axis.plot(
                [center["average_gaze_x"], centroid["average_gaze_x"]],
                [center["average_gaze_y"], centroid["average_gaze_y"]],
                color=label_colors[label],
                linestyle="--",
                linewidth=2,
                alpha=0.85,
                zorder=2,
            )
            axis.scatter(
                centroid["average_gaze_x"],
                centroid["average_gaze_y"],
                color=label_colors[label],
                marker=marker_map[player_name],
                s=230,
                alpha=0.92,
                edgecolors=PALETTE["ink"],
                linewidths=0.8,
                zorder=4,
                )

    label_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=label_colors[label],
            markeredgecolor=PALETTE["ink"],
            markersize=10,
            label=str(label),
        )
        for label in highlight_labels
    ]
    player_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[player],
            color="w",
            markerfacecolor=PALETTE["steel"],
            markeredgecolor=PALETTE["ink"],
            markersize=10,
            label=player,
        )
        for player in players
    ]
    axis.legend(handles=label_handles + player_handles, title="Wedge And Player", loc="best", ncol=2)
    axis.set_title("Player-Average Gaze Tendencies For Dominant Wedges")
    axis.set_xlabel("Horizontal Gaze (Normalized)")
    axis.set_ylabel("Vertical Gaze (Normalized)")
    axis.axhline(0, color=PALETTE["steel"], linestyle="--", alpha=0.45)
    axis.axvline(0, color=PALETTE["steel"], linestyle="--", alpha=0.45)
    axis.invert_yaxis()
    axis.grid(True, alpha=0.35)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def _top_label_summary(player_df: pd.DataFrame, labels: list[str], *, limit: int) -> str:
    if player_df.empty or "wedge_number_label" not in player_df.columns:
        return "no highlighted wedges"
    counts = player_df[player_df["wedge_number_label"].astype(str).isin(labels)]["wedge_number_label"].astype(str).value_counts()
    if counts.empty:
        return "no highlighted wedges"
    summary_parts = [f"{label} (n={int(counts[label])})" for label in counts.index.tolist()[:limit]]
    return ", ".join(summary_parts)


def _draw_player_tendency_panel(
    axis: plt.Axes,
    panel_df: pd.DataFrame,
    *,
    panel_name: str,
    selected_labels: list[str],
    label_colors: dict[str, tuple[float, float, float, float]],
    match_markers: dict[str, str],
) -> None:
    background_df = panel_df[~panel_df["wedge_number_label"].astype(str).isin(selected_labels)].copy()
    highlight_df = panel_df[panel_df["wedge_number_label"].astype(str).isin(selected_labels)].copy()

    if not background_df.empty:
        axis.scatter(
            background_df["average_gaze_x"],
            background_df["average_gaze_y"],
            color="#C7CDD4",
            s=30,
            alpha=0.4,
            edgecolors="none",
            zorder=1,
        )

    for _, row in highlight_df.iterrows():
        label = str(row["wedge_number_label"])
        match_id = str(row["sport_event_id"])
        marker = match_markers.get(match_id, "o") if panel_name == "combined" else "o"
        axis.scatter(
            row["average_gaze_x"],
            row["average_gaze_y"],
            color=label_colors[label],
            marker=marker,
            s=68 if panel_name != "combined" else 78,
            alpha=0.82,
            edgecolors=PALETTE["ink"],
            linewidths=0.6,
            zorder=3,
        )

    panel_center = panel_df[["average_gaze_x", "average_gaze_y"]].mean()
    axis.scatter(
        panel_center["average_gaze_x"],
        panel_center["average_gaze_y"],
        marker="X",
        s=120,
        color=PALETTE["ink"],
        linewidths=0.8,
        zorder=5,
    )

    centroids = highlight_df.groupby(highlight_df["wedge_number_label"].astype(str))[["average_gaze_x", "average_gaze_y"]].mean()
    for label, centroid in centroids.iterrows():
        axis.scatter(
            centroid["average_gaze_x"],
            centroid["average_gaze_y"],
            color=label_colors[label],
            s=1700 if panel_name != "combined" else 2200,
            alpha=0.18,
            marker="o",
            edgecolors="none",
            zorder=2,
        )

    if panel_name == "combined":
        axis.set_title(f"Combined (n={len(panel_df)})")
    else:
        axis.set_title(f"Match {panel_name.split(':')[-1]} (n={len(panel_df)})")


def _compute_gaze_limits(plot_df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = plot_df["average_gaze_x"].astype(float).to_numpy()
    y_values = plot_df["average_gaze_y"].astype(float).to_numpy()
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    x_pad = max(0.015, (x_max - x_min) * 0.12 if x_max > x_min else 0.03)
    y_pad = max(0.015, (y_max - y_min) * 0.12 if y_max > y_min else 0.03)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _player_match_summary(player_df: pd.DataFrame) -> str:
    if player_df.empty or "sport_event_id" not in player_df.columns:
        return "no match panels"
    counts = (
        player_df.groupby("sport_event_id", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    parts = [f"{str(match_id).split(':')[-1]} (n={int(count)})" for match_id, count in counts.items()]
    if not parts:
        return "no match panels"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _render_notes(notes: list[tuple[str, str]]) -> str:
    lines = ["# Paper Figure Notes", ""]
    for filename, takeaway in notes:
        lines.append(f"- `{filename}`: {takeaway}")
    lines.append("")
    lines.append("Regenerate these figures with `just paper` after adding new annotated data.")
    return "\n".join(lines)
