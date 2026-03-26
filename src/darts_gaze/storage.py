"""SQLite persistence for videos, sync anchors, and captures."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH, ensure_data_directories
from .types import CaptureRecord, FaceBoundingBox, SyncAnchor


class AnnotationStore:
    """Persist annotation state in SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        ensure_data_directories()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL UNIQUE,
                    fps REAL,
                    duration_s REAL,
                    frame_width INTEGER,
                    frame_height INTEGER,
                    source_url TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sync_anchors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    sport_event_id TEXT NOT NULL,
                    video_time_s REAL NOT NULL,
                    timeline_event_id INTEGER NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    sport_event_id TEXT,
                    video_time_s REAL NOT NULL,
                    frame_path TEXT NOT NULL,
                    face_x INTEGER,
                    face_y INTEGER,
                    face_width INTEGER,
                    face_height INTEGER,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    matched_throw_event_id INTEGER,
                    resolved_timeline_time_utc TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                );
                """
            )

    def upsert_video(
        self,
        *,
        display_name: str,
        original_filename: str,
        stored_path: str,
        fps: float | None = None,
        duration_s: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        source_url: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    display_name, original_filename, stored_path, fps, duration_s, frame_width, frame_height, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stored_path) DO UPDATE SET
                    display_name=excluded.display_name,
                    original_filename=excluded.original_filename,
                    fps=excluded.fps,
                    duration_s=excluded.duration_s,
                    frame_width=excluded.frame_width,
                    frame_height=excluded.frame_height,
                    source_url=excluded.source_url
                """,
                (display_name, original_filename, stored_path, fps, duration_s, frame_width, frame_height, source_url),
            )
            row = connection.execute("SELECT * FROM videos WHERE stored_path = ?", (stored_path,)).fetchone()
        return dict(row) if row else {}

    def list_videos(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC, id DESC").fetchall()
        return [dict(row) for row in rows]

    def get_video(self, video_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    def create_anchor(self, anchor: SyncAnchor) -> SyncAnchor:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sync_anchors (video_id, sport_event_id, video_time_s, timeline_event_id, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (anchor.video_id, anchor.sport_event_id, anchor.video_time_s, anchor.timeline_event_id, anchor.notes),
            )
            row = connection.execute("SELECT * FROM sync_anchors WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._anchor_from_row(row)

    def list_anchors(self, video_id: int | None = None, sport_event_id: str | None = None) -> list[SyncAnchor]:
        clauses = []
        parameters: list[Any] = []
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(video_id)
        if sport_event_id is not None:
            clauses.append("sport_event_id = ?")
            parameters.append(sport_event_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM sync_anchors {where_clause} ORDER BY video_time_s ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._anchor_from_row(row) for row in rows]

    def delete_anchor(self, anchor_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sync_anchors WHERE id = ?", (anchor_id,))

    def update_anchor(self, anchor_id: int, **changes: Any) -> SyncAnchor:
        allowed_columns = {"video_time_s", "timeline_event_id", "notes", "sport_event_id"}
        assignments = []
        parameters: list[Any] = []
        for key, value in changes.items():
            if key not in allowed_columns:
                continue
            assignments.append(f"{key} = ?")
            parameters.append(value)
        if not assignments:
            anchors = [anchor for anchor in self.list_anchors() if anchor.id == anchor_id]
            if not anchors:
                raise KeyError(f"Anchor {anchor_id} does not exist")
            return anchors[0]

        parameters.append(anchor_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE sync_anchors SET {', '.join(assignments)} WHERE id = ?", parameters)
            row = connection.execute("SELECT * FROM sync_anchors WHERE id = ?", (anchor_id,)).fetchone()
        if row is None:
            raise KeyError(f"Anchor {anchor_id} does not exist")
        return self._anchor_from_row(row)

    def create_capture(self, capture: CaptureRecord) -> CaptureRecord:
        bbox = capture.face_bbox
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO captures (
                    video_id, sport_event_id, video_time_s, frame_path, face_x, face_y, face_width, face_height,
                    review_status, matched_throw_event_id, resolved_timeline_time_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture.video_id,
                    capture.sport_event_id,
                    capture.video_time_s,
                    capture.frame_path,
                    bbox.x if bbox else None,
                    bbox.y if bbox else None,
                    bbox.width if bbox else None,
                    bbox.height if bbox else None,
                    capture.review_status,
                    capture.matched_throw_event_id,
                    capture.resolved_timeline_time_utc,
                    capture.notes,
                ),
            )
            row = connection.execute("SELECT * FROM captures WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._capture_from_row(row)

    def get_capture(self, capture_id: int) -> CaptureRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return self._capture_from_row(row) if row else None

    def delete_capture(self, capture_id: int) -> None:
        capture = self.get_capture(capture_id)
        if capture and capture.frame_path:
            frame_file = Path(capture.frame_path)
            if frame_file.exists():
                frame_file.unlink()
        with self._connect() as connection:
            connection.execute("DELETE FROM captures WHERE id = ?", (capture_id,))

    def list_captures(
        self,
        video_id: int | None = None,
        sport_event_id: str | None = None,
        review_status: str | None = None,
    ) -> list[CaptureRecord]:
        clauses = []
        parameters: list[Any] = []
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(video_id)
        if sport_event_id is not None:
            clauses.append("sport_event_id = ?")
            parameters.append(sport_event_id)
        if review_status is not None:
            clauses.append("review_status = ?")
            parameters.append(review_status)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM captures {where_clause} ORDER BY created_at DESC, id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._capture_from_row(row) for row in rows]

    def update_capture(self, capture_id: int, **changes: Any) -> CaptureRecord:
        allowed_columns = {
            "sport_event_id",
            "video_time_s",
            "frame_path",
            "face_x",
            "face_y",
            "face_width",
            "face_height",
            "review_status",
            "matched_throw_event_id",
            "resolved_timeline_time_utc",
            "notes",
        }
        assignments = []
        parameters: list[Any] = []

        face_bbox = changes.pop("face_bbox", None)
        if face_bbox is not None:
            if isinstance(face_bbox, FaceBoundingBox):
                changes["face_x"] = face_bbox.x
                changes["face_y"] = face_bbox.y
                changes["face_width"] = face_bbox.width
                changes["face_height"] = face_bbox.height
            else:
                raise TypeError("face_bbox must be a FaceBoundingBox")

        for key, value in changes.items():
            if key not in allowed_columns:
                continue
            assignments.append(f"{key} = ?")
            parameters.append(value)
        if not assignments:
            capture = self.get_capture(capture_id)
            if capture is None:
                raise KeyError(f"Capture {capture_id} does not exist")
            return capture

        parameters.append(capture_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE captures SET {', '.join(assignments)} WHERE id = ?", parameters)
            row = connection.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        if row is None:
            raise KeyError(f"Capture {capture_id} does not exist")
        return self._capture_from_row(row)

    @staticmethod
    def _anchor_from_row(row: sqlite3.Row) -> SyncAnchor:
        created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        return SyncAnchor(
            id=row["id"],
            video_id=row["video_id"],
            sport_event_id=row["sport_event_id"],
            video_time_s=row["video_time_s"],
            timeline_event_id=row["timeline_event_id"],
            notes=row["notes"],
            created_at=created_at,
        )

    @staticmethod
    def _capture_from_row(row: sqlite3.Row) -> CaptureRecord:
        face_bbox = None
        if row["face_x"] is not None and row["face_y"] is not None and row["face_width"] is not None and row["face_height"] is not None:
            face_bbox = FaceBoundingBox(
                x=row["face_x"],
                y=row["face_y"],
                width=row["face_width"],
                height=row["face_height"],
            )
        created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        return CaptureRecord(
            id=row["id"],
            video_id=row["video_id"],
            sport_event_id=row["sport_event_id"],
            video_time_s=row["video_time_s"],
            frame_path=row["frame_path"],
            face_bbox=face_bbox,
            review_status=row["review_status"],
            matched_throw_event_id=row["matched_throw_event_id"],
            resolved_timeline_time_utc=row["resolved_timeline_time_utc"],
            notes=row["notes"],
            created_at=created_at,
        )
