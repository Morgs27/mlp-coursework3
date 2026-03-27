from __future__ import annotations

from pathlib import Path

import pandas as pd

from darts_gaze.dataset import DatasetBuilder
from darts_gaze.reporting import export_quality_reports
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


def test_export_quality_reports_writes_tables_and_figures(
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
    builder.build(write_back=True)

    report_dir = tmp_path / "outputs" / "first_pass"
    artifacts = export_quality_reports(processed_dir=processed_dir, report_dir=report_dir)

    assert artifacts.capture_quality_csv.exists()
    assert artifacts.dataset_summary_csv.exists()
    assert artifacts.qa_summary_csv.exists()
    assert (report_dir / "figures" / "dataset_distribution.pdf").exists()
    assert (report_dir / "figures" / "dataset_distribution.png").exists()
    assert (report_dir / "figures" / "valid_face_rate_by_match.pdf").exists()
    assert (report_dir / "figures" / "segment_top10_distribution.pdf").exists()

    exported_quality_df = pd.read_csv(artifacts.capture_quality_csv)
    assert len(exported_quality_df) == 1
