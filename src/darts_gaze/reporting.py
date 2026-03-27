"""First-pass QA report export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .config import FIRST_PASS_OUTPUTS_DIR, PROCESSED_DIR, ensure_data_directories
from .plots import (
    save_dataset_distribution,
    save_segment_imbalance,
    save_valid_face_rate_by_match,
)


@dataclass(slots=True)
class QualityReportArtifacts:
    tables_dir: Path
    figures_dir: Path
    capture_quality_csv: Path
    dataset_summary_csv: Path
    qa_summary_csv: Path


def export_quality_reports(
    processed_dir: str | Path = PROCESSED_DIR,
    report_dir: str | Path = FIRST_PASS_OUTPUTS_DIR,
) -> QualityReportArtifacts:
    ensure_data_directories()
    processed_dir = Path(processed_dir)
    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    capture_quality_df = _read_csv_or_empty(processed_dir / "capture_quality.csv")
    dataset_summary_df = _read_csv_or_empty(processed_dir / "dataset_summary.csv")
    qa_summary_df = _read_csv_or_empty(processed_dir / "qa_summary.csv")
    training_df = _read_csv_or_empty(processed_dir / "training_samples.csv")

    capture_quality_csv = tables_dir / "capture_quality.csv"
    dataset_summary_csv = tables_dir / "dataset_summary.csv"
    qa_summary_csv = tables_dir / "qa_summary.csv"
    capture_quality_df.to_csv(capture_quality_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    qa_summary_df.to_csv(qa_summary_csv, index=False)

    if not training_df.empty:
        save_dataset_distribution(training_df, figures_dir / "dataset_distribution.pdf")
        save_segment_imbalance(training_df, figures_dir / "segment_top10_distribution.pdf")
    if not capture_quality_df.empty:
        save_valid_face_rate_by_match(capture_quality_df, figures_dir / "valid_face_rate_by_match.pdf")

    return QualityReportArtifacts(
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        capture_quality_csv=capture_quality_csv,
        dataset_summary_csv=dataset_summary_csv,
        qa_summary_csv=qa_summary_csv,
    )


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
