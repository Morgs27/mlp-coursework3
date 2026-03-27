from __future__ import annotations

from pathlib import Path

import pandas as pd

from darts_gaze.dataset import DatasetBuilder
from darts_gaze.sportradar import SportradarClient
from darts_gaze.storage import AnnotationStore
from darts_gaze.types import CaptureRecord, SyncAnchor


class FakeSportradarClient(SportradarClient):
    def __init__(self, payload: dict) -> None:
        super().__init__(api_key="test")
        self.payload = payload

    def get_timeline(self, sport_event_id: str, force_refresh: bool = False) -> dict:
        assert sport_event_id == self.payload["sport_event"]["id"]
        return self.payload


def test_dataset_builder_creates_enriched_and_training_rows(
    tmp_path: Path,
    sample_timeline_payload: dict,
    sample_debug_image_path: Path,
) -> None:
    db_path = tmp_path / "annotations.sqlite3"
    store = AnnotationStore(db_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    video = store.upsert_video(
        display_name="fixture-video",
        original_filename="fixture-video.mp4",
        stored_path=str(tmp_path / "fixture-video.mp4"),
        fps=30.0,
        duration_s=120.0,
        frame_width=854,
        frame_height=480,
    )

    store.create_anchor(
        SyncAnchor(
            video_id=video["id"],
            sport_event_id="sr:sport_event:test-1",
            video_time_s=5.0,
            timeline_event_id=10,
        )
    )
    store.create_anchor(
        SyncAnchor(
            video_id=video["id"],
            sport_event_id="sr:sport_event:test-1",
            video_time_s=18.0,
            timeline_event_id=22,
        )
    )

    store.create_capture(
        CaptureRecord(
            video_id=video["id"],
            sport_event_id="sr:sport_event:test-1",
            video_time_s=17.0,
            frame_path=str(sample_debug_image_path),
        )
    )

    builder = DatasetBuilder(
        store=store,
        client=FakeSportradarClient(sample_timeline_payload),
        output_dir=processed_dir,
    )
    artifacts = builder.build(write_back=True)

    assert artifacts.enriched_rows == 1
    assert artifacts.unresolved_rows == 0
    assert artifacts.valid_face_rows == 1
    assert artifacts.modeling_rows == 1
    enriched_df = pd.read_csv(artifacts.enriched_csv)
    assert list(enriched_df["segment_label"]) == ["T20"]
    assert list(enriched_df["resulting_score"]) == [0]
    assert list(enriched_df["raw_resulting_score"]) == [60]
    assert list(enriched_df["review_status"]) == ["matched"]
    assert list(enriched_df["entered_modeling"]) == [True]
    assert [str(value) for value in enriched_df["wedge_number_label"]] == ["20"]
    assert [str(value) for value in enriched_df["coarse_wedge_area_label"]] == ["20"]

    training_df = pd.read_csv(artifacts.training_csv)
    assert len(training_df) == 1

    capture_quality_df = pd.read_csv(artifacts.capture_quality_csv)
    assert len(capture_quality_df) == 1
    assert capture_quality_df.loc[0, "valid_face"] in (True, 1)
    assert capture_quality_df.loc[0, "entered_modeling"] in (True, 1)
    assert str(capture_quality_df.loc[0, "wedge_number_label"]) == "20"
    assert str(capture_quality_df.loc[0, "coarse_wedge_area_label"]) == "20"

    dataset_summary_df = pd.read_csv(artifacts.dataset_summary_csv)
    assert set(dataset_summary_df["summary_type"]) == {
        "sport_event_id",
        "player_name",
        "segment_label",
        "wedge_number_label",
        "coarse_wedge_area_label",
        "resulting_score",
    }

    qa_summary_df = pd.read_csv(artifacts.qa_summary_csv)
    assert "total_captures" in set(qa_summary_df["metric"])
