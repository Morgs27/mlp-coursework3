from __future__ import annotations

from pathlib import Path

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
