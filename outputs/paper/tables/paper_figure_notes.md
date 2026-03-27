# Paper Figure Notes

- `figure_01_dataset_funnel.pdf`: Summarizes the usable-data attrition from all matched captures to the valid-face modeling subset and the final 19-vs-20 reranker subset.
- `figure_02_match_player_heatmap.pdf`: Shows which players and matches dominate the current modeling set, making the match/player imbalance explicit.
- `figure_03_wedge_long_tail.pdf`: Communicates the long-tail wedge distribution and highlights that 19 and 20 are only a small part of the full label space.
- `figure_04_19_20_match_balance.pdf`: Shows the class imbalance of the reranker subset by match, which explains why raw accuracy alone is misleading.
- `figure_05_19_20_player_shift_vectors.pdf`: Visualizes the within-player gaze shift from 19 to 20 targets, which is the clearest qualitative reranker signal.
- `figure_15_player_average_map.pdf`: Summarizes the dominant player-specific gaze tendencies by connecting each player's overall center to their wedge-average centroids.
- `figure_16_player_tendency_anderson_gary.pdf`: Player-specific multi-panel tendency map for Anderson, Gary, split by match with a final combined panel.
- `figure_17_player_tendency_littler_luke.pdf`: Player-specific multi-panel tendency map for Littler, Luke, split by match with a final combined panel.
- `figure_18_player_tendency_searle_ryan.pdf`: Player-specific multi-panel tendency map for Searle, Ryan, split by match with a final combined panel.
- `figure_19_player_tendency_van_veen_gian.pdf`: Player-specific multi-panel tendency map for van Veen, Gian, split by match with a final combined panel.
- `figure_06_wedge_model_ci.pdf`: Provides the main wedge-task comparison with 95% bootstrap confidence intervals for exact, coarse, and circular metrics.
- `figure_07_wedge_model_tradeoff.pdf`: Shows the trade-off between 3-wedge targeting and exact multiclass discrimination, separating useful from degenerate models.
- `figure_08_wedge_fold_stability.pdf`: Shows whether wedge performance is stable across held-out matches rather than being driven by a single fixture.
- `figure_09_reranker_model_ci.pdf`: Summarizes ranking quality, threshold performance, and confidence calibration for the 19-vs-20 reranker.
- `figure_10_reranker_tradeoff.pdf`: Highlights the ranking-calibration trade-off: logistic is best for ranking, while forest-style models are better calibrated.
- `figure_11_reranker_selected_curves.pdf`: Restricts ROC and precision-recall curves to the most relevant reranker models so the ranking differences are readable.
- `figure_12_reranker_selected_calibration.pdf`: Focuses on calibration behavior for the selected reranker models, which matters for downstream confidence-aware reranking.
- `figure_13_reranker_probability_boxplots.pdf`: Shows how strongly each selected reranker separates true 19s from true 20s in probability space.
- `figure_14_reranker_fold_stability.pdf`: Shows whether reranker ranking quality generalizes across held-out matches instead of depending on a single fold.

Regenerate these figures with `just paper` after adding new annotated data.