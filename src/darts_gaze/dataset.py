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
from .types import EnrichedThrowSample, ThrowLabel


@dataclass(slots=True)
class DatasetArtifacts:
    enriched_csv: Path
    training_csv: Path
    unresolved_csv: Path
    total_captures: int
    enriched_rows: int
    unresolved_rows: int


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

        for capture in captures:
            if not capture.sport_event_id:
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

            gaze_result = estimate_gaze(capture.frame_path, face_bbox=capture.face_bbox)
            sample = EnrichedThrowSample(
                capture=capture,
                throw_label=throw_label,
                gaze_result=gaze_result,
                mapped_capture_time_utc=resolution.mapped_time_utc or capture.resolved_timeline_time_utc,
                match_resolution=updated_status,
            )
            enriched_rows.append(sample.to_flat_dict())

        enriched_df = pd.DataFrame(enriched_rows)
        training_df = (
            enriched_df[enriched_df["review_status"].isin(["matched", "verified"])].copy()
            if not enriched_df.empty
            else pd.DataFrame()
        )
        unresolved_df = pd.DataFrame(unresolved_rows)

        enriched_csv = self.output_dir / "enriched_samples.csv"
        training_csv = self.output_dir / "training_samples.csv"
        unresolved_csv = self.output_dir / "unresolved_captures.csv"
        enriched_df.to_csv(enriched_csv, index=False)
        training_df.to_csv(training_csv, index=False)
        unresolved_df.to_csv(unresolved_csv, index=False)
        return DatasetArtifacts(
            enriched_csv=enriched_csv,
            training_csv=training_csv,
            unresolved_csv=unresolved_csv,
            total_captures=len(captures),
            enriched_rows=len(enriched_df),
            unresolved_rows=len(unresolved_df),
        )
