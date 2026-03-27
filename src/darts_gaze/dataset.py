"""Dataset building from annotated captures and Sportradar timelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROCESSED_DIR, ensure_data_directories
from .gaze import estimate_gaze
from .sportradar import SportradarClient
from .storage import AnnotationStore
from .sync import resolve_throw_for_capture
from .targets import target_columns
from .types import EnrichedThrowSample, GazeResult, ThrowLabel


@dataclass(slots=True)
class DatasetArtifacts:
    enriched_csv: Path
    training_csv: Path
    unresolved_csv: Path
    capture_quality_csv: Path
    dataset_summary_csv: Path
    qa_summary_csv: Path
    total_captures: int
    enriched_rows: int
    unresolved_rows: int
    verified_captures: int
    valid_face_rows: int
    modeling_rows: int


class DatasetBuilder:
    """Build processed dataset tables from stored frame captures."""

    def __init__(
        self,
        store: AnnotationStore,
        client: SportradarClient,
        output_dir: Path | str = PROCESSED_DIR,
    ) -> None:
        ensure_data_directories()
        self.store = store
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, write_back: bool = True) -> DatasetArtifacts:
        captures = self.store.list_captures()
        anchors_by_key: dict[tuple[int, str], list] = {}
        throws_by_match: dict[str, list[ThrowLabel]] = {}
        event_times_by_match: dict[str, dict[int, str]] = {}

        enriched_rows: list[dict[str, Any]] = []
        unresolved_rows: list[dict[str, Any]] = []
        capture_quality_rows: list[dict[str, Any]] = []

        for capture in captures:
            gaze_result = self._estimate_capture_gaze(capture.frame_path, capture.face_bbox)
            quality_row = self._base_quality_row(capture, gaze_result)

            if not capture.sport_event_id:
                quality_row["resolution_reason"] = "missing_sport_event_id"
                capture_quality_rows.append(quality_row)
                unresolved_rows.append(
                    {
                        "capture_id": capture.id,
                        "video_id": capture.video_id,
                        "video_time_s": capture.video_time_s,
                        "reason": "missing_sport_event_id",
                    }
                )
                continue

            match_id = capture.sport_event_id
            key = (capture.video_id, match_id)
            if key not in anchors_by_key:
                anchors_by_key[key] = self.store.list_anchors(video_id=capture.video_id, sport_event_id=match_id)
            if match_id not in throws_by_match:
                timeline_payload = self.client.get_timeline(match_id)
                throws_by_match[match_id] = self.client.parse_throw_labels(timeline_payload)
                event_times_by_match[match_id] = self.client.timeline_event_times(timeline_payload)

            throws = throws_by_match[match_id]
            throw_index = {throw.throw_event_id: throw for throw in throws}
            resolution = resolve_throw_for_capture(
                video_time_s=capture.video_time_s,
                anchors=anchors_by_key[key],
                timeline_event_times=event_times_by_match[match_id],
                throw_labels=throws,
                selected_throw_event_id=capture.matched_throw_event_id,
            )

            use_manual_match = capture.matched_throw_event_id is not None
            matched_throw_id = capture.matched_throw_event_id
            match_resolution = capture.review_status
            if not use_manual_match:
                if resolution.ambiguous:
                    matched_throw_id = None
                    match_resolution = "needs_review"
                else:
                    matched_throw_id = resolution.matched_throw_event_id
                    match_resolution = resolution.resolution_status

            if matched_throw_id is None:
                quality_row.update(
                    {
                        "review_status": "needs_review",
                        "matched_throw_event_id": None,
                        "mapped_capture_time_utc": resolution.mapped_time_utc,
                        "candidate_throw_event_ids": ",".join(map(str, resolution.candidate_throw_event_ids)),
                        "resolution_reason": "ambiguous_match" if resolution.ambiguous else "no_match",
                    }
                )
                capture_quality_rows.append(quality_row)
                unresolved_rows.append(
                    {
                        "capture_id": capture.id,
                        "video_id": capture.video_id,
                        "sport_event_id": match_id,
                        "video_time_s": capture.video_time_s,
                        "mapped_capture_time_utc": resolution.mapped_time_utc,
                        "candidate_throw_event_ids": ",".join(map(str, resolution.candidate_throw_event_ids)),
                        "reason": "ambiguous_match" if resolution.ambiguous else "no_match",
                    }
                )
                if write_back and capture.id is not None:
                    self.store.update_capture(
                        capture.id,
                        review_status="needs_review",
                        resolved_timeline_time_utc=resolution.mapped_time_utc,
                    )
                continue

            throw_label = throw_index.get(matched_throw_id)
            if throw_label is None:
                quality_row.update(
                    {
                        "matched_throw_event_id": matched_throw_id,
                        "resolution_reason": f"missing_throw_label:{matched_throw_id}",
                    }
                )
                capture_quality_rows.append(quality_row)
                unresolved_rows.append(
                    {
                        "capture_id": capture.id,
                        "video_id": capture.video_id,
                        "sport_event_id": match_id,
                        "video_time_s": capture.video_time_s,
                        "reason": f"missing_throw_label:{matched_throw_id}",
                    }
                )
                continue

            updated_status = "verified" if use_manual_match else match_resolution
            if write_back and capture.id is not None:
                capture = self.store.update_capture(
                    capture.id,
                    review_status=updated_status,
                    matched_throw_event_id=matched_throw_id,
                    resolved_timeline_time_utc=resolution.mapped_time_utc or capture.resolved_timeline_time_utc,
                )

            sample = EnrichedThrowSample(
                capture=capture,
                throw_label=throw_label,
                gaze_result=gaze_result,
                mapped_capture_time_utc=resolution.mapped_time_utc or capture.resolved_timeline_time_utc,
                match_resolution=updated_status,
            )
            flat_sample = sample.to_flat_dict()
            flat_sample.update(
                target_columns(
                    flat_sample["segment_number"],
                    segment_label=flat_sample["segment_label"],
                )
            )
            flat_sample["roi_present"] = capture.face_bbox is not None
            flat_sample["entered_modeling"] = bool(
                flat_sample["review_status"] in {"matched", "verified"} and flat_sample["valid_face"]
            )
            enriched_rows.append(flat_sample)
            quality_row.update(
                {
                    "review_status": flat_sample["review_status"],
                    "matched_throw_event_id": flat_sample["matched_throw_event_id"],
                    "mapped_capture_time_utc": flat_sample["mapped_capture_time_utc"],
                    "player_name": flat_sample["player_name"],
                    "segment_label": flat_sample["segment_label"],
                    "wedge_number_label": flat_sample["wedge_number_label"],
                    "coarse_wedge_area_label": flat_sample["coarse_wedge_area_label"],
                    "resulting_score": flat_sample["resulting_score"],
                    "is_bust": flat_sample["is_bust"],
                    "entered_modeling": flat_sample["entered_modeling"],
                    "candidate_throw_event_ids": ",".join(map(str, resolution.candidate_throw_event_ids)),
                    "resolution_reason": None,
                }
            )
            capture_quality_rows.append(quality_row)

        enriched_df = pd.DataFrame(enriched_rows)
        training_df = (
            enriched_df[enriched_df["review_status"].isin(["matched", "verified"])].copy()
            if not enriched_df.empty
            else pd.DataFrame()
        )
        unresolved_df = pd.DataFrame(unresolved_rows)
        capture_quality_df = pd.DataFrame(capture_quality_rows)
        dataset_summary_df = self._dataset_summary(capture_quality_df)
        qa_summary_df = self._qa_summary(capture_quality_df)

        enriched_csv = self.output_dir / "enriched_samples.csv"
        training_csv = self.output_dir / "training_samples.csv"
        unresolved_csv = self.output_dir / "unresolved_captures.csv"
        capture_quality_csv = self.output_dir / "capture_quality.csv"
        dataset_summary_csv = self.output_dir / "dataset_summary.csv"
        qa_summary_csv = self.output_dir / "qa_summary.csv"
        enriched_df.to_csv(enriched_csv, index=False)
        training_df.to_csv(training_csv, index=False)
        unresolved_df.to_csv(unresolved_csv, index=False)
        capture_quality_df.to_csv(capture_quality_csv, index=False)
        dataset_summary_df.to_csv(dataset_summary_csv, index=False)
        qa_summary_df.to_csv(qa_summary_csv, index=False)
        verified_captures = (
            int(capture_quality_df["review_status"].eq("verified").sum())
            if not capture_quality_df.empty
            else 0
        )
        valid_face_rows = (
            int(capture_quality_df["valid_face"].fillna(False).astype(bool).sum())
            if not capture_quality_df.empty
            else 0
        )
        modeling_rows = (
            int(capture_quality_df["entered_modeling"].fillna(False).astype(bool).sum())
            if not capture_quality_df.empty
            else 0
        )
        return DatasetArtifacts(
            enriched_csv=enriched_csv,
            training_csv=training_csv,
            unresolved_csv=unresolved_csv,
            capture_quality_csv=capture_quality_csv,
            dataset_summary_csv=dataset_summary_csv,
            qa_summary_csv=qa_summary_csv,
            total_captures=len(captures),
            enriched_rows=len(enriched_df),
            unresolved_rows=len(unresolved_df),
            verified_captures=verified_captures,
            valid_face_rows=valid_face_rows,
            modeling_rows=modeling_rows,
        )

    @staticmethod
    def _estimate_capture_gaze(frame_path: str, face_bbox: Any) -> GazeResult:
        try:
            return estimate_gaze(frame_path, face_bbox=face_bbox)
        except Exception as exc:
            return GazeResult(
                valid_face=False,
                detector_confidence=0.0,
                left_gaze=None,
                right_gaze=None,
                average_gaze=None,
                head_x_axis=None,
                head_y_axis=None,
                head_z_axis=None,
                ipd=None,
                eye_agreement=None,
                face_bbox=face_bbox,
                image_width=0,
                image_height=0,
                metadata={"error": str(exc)},
            )

    @staticmethod
    def _base_quality_row(capture: Any, gaze_result: GazeResult) -> dict[str, Any]:
        return {
            "capture_id": capture.id,
            "video_id": capture.video_id,
            "sport_event_id": capture.sport_event_id,
            "video_time_s": capture.video_time_s,
            "frame_path": capture.frame_path,
            "review_status": capture.review_status,
            "matched_throw_event_id": capture.matched_throw_event_id,
            "mapped_capture_time_utc": capture.resolved_timeline_time_utc,
            "roi_present": capture.face_bbox is not None,
            "valid_face": gaze_result.valid_face,
            "detector_confidence": gaze_result.detector_confidence,
            "player_name": None,
            "segment_label": None,
            "wedge_number_label": None,
            "coarse_wedge_area_label": None,
            "resulting_score": None,
            "is_bust": None,
            "entered_modeling": False,
            "candidate_throw_event_ids": "",
            "resolution_reason": None,
            "gaze_error": gaze_result.metadata.get("error"),
        }

    @staticmethod
    def _dataset_summary(capture_quality_df: pd.DataFrame) -> pd.DataFrame:
        if capture_quality_df.empty:
            return pd.DataFrame(
                columns=["summary_type", "summary_value", "valid_face", "entered_modeling", "count"]
            )

        summary_frames: list[pd.DataFrame] = []
        summary_columns = [
            ("sport_event_id", "sport_event_id"),
            ("player_name", "player_name"),
            ("segment_label", "segment_label"),
            ("wedge_number_label", "wedge_number_label"),
            ("coarse_wedge_area_label", "coarse_wedge_area_label"),
            ("resulting_score", "resulting_score"),
        ]
        for source_column, summary_type in summary_columns:
            subset = capture_quality_df[capture_quality_df[source_column].notna()].copy()
            if subset.empty:
                continue
            grouped = (
                subset.groupby([source_column, "valid_face", "entered_modeling"], dropna=False)
                .size()
                .reset_index(name="count")
                .rename(columns={source_column: "summary_value"})
            )
            grouped.insert(0, "summary_type", summary_type)
            summary_frames.append(grouped)
        if not summary_frames:
            return pd.DataFrame(
                columns=["summary_type", "summary_value", "valid_face", "entered_modeling", "count"]
            )
        return pd.concat(summary_frames, ignore_index=True)

    @staticmethod
    def _qa_summary(capture_quality_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["section", "metric", "group_value", "value"]
        if capture_quality_df.empty:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        valid_face_mask = capture_quality_df["valid_face"].fillna(False).astype(bool)
        entered_modeling_mask = capture_quality_df["entered_modeling"].fillna(False).astype(bool)
        verified_mask = capture_quality_df["review_status"].eq("verified")

        rows.extend(
            [
                {"section": "overall", "metric": "total_captures", "group_value": None, "value": int(len(capture_quality_df))},
                {"section": "overall", "metric": "verified_captures", "group_value": None, "value": int(verified_mask.sum())},
                {"section": "overall", "metric": "valid_gaze_captures", "group_value": None, "value": int(valid_face_mask.sum())},
                {
                    "section": "overall",
                    "metric": "invalid_gaze_captures",
                    "group_value": None,
                    "value": int((~valid_face_mask).sum()),
                },
                {
                    "section": "overall",
                    "metric": "modeling_subset_captures",
                    "group_value": None,
                    "value": int(entered_modeling_mask.sum()),
                },
            ]
        )

        match_subset = capture_quality_df[capture_quality_df["sport_event_id"].notna()].copy()
        if not match_subset.empty:
            match_rates = (
                match_subset.groupby("sport_event_id", dropna=False)["valid_face"]
                .mean()
                .sort_index()
            )
            rows.extend(
                {
                    "section": "match_valid_face_rate",
                    "metric": "valid_face_rate",
                    "group_value": match_id,
                    "value": float(rate),
                }
                for match_id, rate in match_rates.items()
            )

        resolved_segments = capture_quality_df[capture_quality_df["segment_label"].notna()]
        if not resolved_segments.empty:
            segment_counts = resolved_segments["segment_label"].value_counts()
            top_segments = segment_counts.head(10)
            rows.extend(
                {
                    "section": "segment_count",
                    "metric": "count",
                    "group_value": segment,
                    "value": int(count),
                }
                for segment, count in top_segments.items()
            )
            tail_count = int(segment_counts.iloc[10:].sum())
            rows.append(
                {
                    "section": "segment_count",
                    "metric": "count",
                    "group_value": "OTHER",
                    "value": tail_count,
                }
            )

        return pd.DataFrame(rows, columns=columns)
