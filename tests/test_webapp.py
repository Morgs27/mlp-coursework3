from __future__ import annotations

import base64
from pathlib import Path

from darts_gaze.storage import AnnotationStore
from darts_gaze.types import CaptureRecord
from darts_gaze.webapp import create_app


def test_webapp_routes_render_with_empty_database(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "annotations.sqlite3")
    client = app.test_client()

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert b"Darts Gaze Annotator" in index_response.data

    matches_response = client.get("/api/known-matches")
    assert matches_response.status_code == 200
    payload = matches_response.get_json()
    assert isinstance(payload, list)
    assert any(item["sport_event_id"] == "sr:sport_event:66098028" for item in payload)


def test_webapp_returns_annotated_capture_and_live_overlay(tmp_path: Path, sample_debug_image_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite3"
    store = AnnotationStore(db_path)
    video = store.upsert_video(
        display_name="fixture-video",
        original_filename="fixture-video.mp4",
        stored_path=str(tmp_path / "fixture-video.mp4"),
        fps=30.0,
        duration_s=60.0,
        frame_width=854,
        frame_height=480,
    )
    capture = store.create_capture(
        CaptureRecord(
            video_id=video["id"],
            sport_event_id=None,
            video_time_s=12.5,
            frame_path=str(sample_debug_image_path),
        )
    )

    app = create_app(db_path=db_path)
    client = app.test_client()

    annotated_response = client.get(f"/media/captures/{capture.id}/annotated")
    assert annotated_response.status_code == 200
    assert annotated_response.content_type == "image/png"

    image_payload = base64.b64encode(sample_debug_image_path.read_bytes()).decode("ascii")
    overlay_response = client.post("/api/gaze/annotate-frame", json={"image_data": image_payload})
    assert overlay_response.status_code == 200
    overlay_payload = overlay_response.get_json()
    assert overlay_payload["valid_face"] is True
    assert "overlay" in overlay_payload
