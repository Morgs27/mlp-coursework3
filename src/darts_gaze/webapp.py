"""Flask-based local annotation app."""

from __future__ import annotations

import base64
import binascii
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .config import CAPTURES_DIR, DEFAULT_DB_PATH, ROOT_DIR, VIDEOS_DIR, ensure_data_directories
from .matches import KNOWN_MATCHES
from .sportradar import SportradarClient
from .storage import AnnotationStore
from .sync import resolve_throw_for_capture
from .types import CaptureRecord, FaceBoundingBox, SyncAnchor, ThrowLabel
from .video import probe_video


def _json_error(message: str, status_code: int = 400):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def _capture_to_dict(capture: CaptureRecord) -> dict[str, Any]:
    return {
        "id": capture.id,
        "video_id": capture.video_id,
        "sport_event_id": capture.sport_event_id,
        "video_time_s": capture.video_time_s,
        "frame_path": capture.frame_path,
        "face_bbox": capture.face_bbox.to_dict() if capture.face_bbox else None,
        "review_status": capture.review_status,
        "matched_throw_event_id": capture.matched_throw_event_id,
        "resolved_timeline_time_utc": capture.resolved_timeline_time_utc,
        "notes": capture.notes,
        "created_at": capture.created_at.isoformat() if capture.created_at else None,
    }


def _anchor_to_dict(anchor: SyncAnchor) -> dict[str, Any]:
    return {
        "id": anchor.id,
        "video_id": anchor.video_id,
        "sport_event_id": anchor.sport_event_id,
        "video_time_s": anchor.video_time_s,
        "timeline_event_id": anchor.timeline_event_id,
        "notes": anchor.notes,
        "created_at": anchor.created_at.isoformat() if anchor.created_at else None,
    }


def _throw_to_dict(throw: ThrowLabel) -> dict[str, Any]:
    return {
        "throw_event_id": throw.throw_event_id,
        "throw_time_utc": throw.throw_time_utc,
        "player_id": throw.player_id,
        "player_name": throw.player_name,
        "competitor_qualifier": throw.competitor_qualifier,
        "resulting_score": throw.resulting_score,
        "raw_resulting_score": throw.raw_resulting_score,
        "segment_label": throw.segment_label,
        "segment_ring": throw.segment_ring,
        "segment_number": throw.segment_number,
        "is_bust": throw.is_bust,
        "dart_in_visit": throw.dart_in_visit,
        "period": throw.period,
    }


def _decode_image_data(image_data: str) -> bytes:
    if "," in image_data:
        _, encoded = image_data.split(",", 1)
    else:
        encoded = image_data
    try:
        return base64.b64decode(encoded)
    except binascii.Error as exc:
        raise ValueError("Invalid image payload") from exc


def create_app(
    db_path: str | Path = DEFAULT_DB_PATH,
    template_dir: str | Path | None = None,
    static_dir: str | Path | None = None,
) -> Flask:
    ensure_data_directories()
    template_root = Path(template_dir or ROOT_DIR / "apps" / "annotator" / "templates")
    static_root = Path(static_dir or ROOT_DIR / "apps" / "annotator" / "static")

    app = Flask(__name__, template_folder=str(template_root), static_folder=str(static_root))
    store = AnnotationStore(db_path)
    client = SportradarClient()
    youtube_jobs: dict[str, dict[str, Any]] = {}

    def get_throws_for_event(sport_event_id: str) -> list[ThrowLabel]:
        timeline = client.get_timeline(sport_event_id)
        return client.parse_throw_labels(timeline)

    def known_match_payload() -> list[dict[str, str]]:
        return [
            {
                "sport_event_id": match.sport_event_id,
                "match_date": match.match_date.isoformat(),
                "title": match.title,
                "youtube_url": match.youtube_url,
            }
            for match in KNOWN_MATCHES
        ]

    @app.get("/")
    def index():
        return render_template("index.html", known_matches=known_match_payload(), videos=store.list_videos())

    @app.after_request
    def _cors_for_media(response):
        if request.path.startswith("/media/"):
            response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    @app.get("/media/videos/<int:video_id>")
    def serve_video(video_id: int):
        video = store.get_video(video_id)
        if video is None:
            return _json_error("Unknown video", 404)
        return send_file(video["stored_path"], conditional=True)

    @app.get("/media/captures/<int:capture_id>")
    def serve_capture(capture_id: int):
        capture = store.get_capture(capture_id)
        if capture is None:
            return _json_error("Unknown capture", 404)
        return send_file(capture.frame_path, conditional=True)

    @app.get("/api/known-matches")
    def known_matches():
        return jsonify(known_match_payload())

    @app.get("/api/videos")
    def list_videos():
        return jsonify(store.list_videos())

    @app.post("/api/videos/upload")
    def upload_video():
        uploaded = request.files.get("video")
        if uploaded is None or not uploaded.filename:
            return _json_error("No video file provided")

        safe_name = secure_filename(uploaded.filename)
        target_path = VIDEOS_DIR / safe_name
        if target_path.exists():
            target_path = VIDEOS_DIR / f"{target_path.stem}-{uuid.uuid4().hex[:8]}{target_path.suffix}"
        uploaded.save(target_path)
        metadata = probe_video(target_path)
        record = store.upsert_video(
            display_name=target_path.stem,
            original_filename=uploaded.filename,
            stored_path=str(target_path),
            fps=float(metadata["fps"]),
            duration_s=float(metadata["duration_s"]),
            frame_width=int(metadata["frame_width"]),
            frame_height=int(metadata["frame_height"]),
        )
        return jsonify(record)

    @app.post("/api/videos/youtube")
    def download_youtube_video():
        payload = request.get_json(force=True)
        url = (payload.get("url") or "").strip()
        if not url:
            return _json_error("No YouTube URL provided")
        job_id = uuid.uuid4().hex[:12]
        youtube_jobs[job_id] = {"status": "downloading", "progress": 0, "error": None, "video": None}

        def _download() -> None:
            try:
                from yt_dlp import YoutubeDL

                def _hook(d: dict) -> None:
                    if d.get("status") == "downloading":
                        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                        downloaded = d.get("downloaded_bytes", 0)
                        youtube_jobs[job_id]["progress"] = int(downloaded / total * 100) if total else 0
                    elif d.get("status") == "finished":
                        youtube_jobs[job_id]["progress"] = 100

                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": str(VIDEOS_DIR / "%(title)s.%(ext)s"),
                    "progress_hooks": [_hook],
                    "merge_output_format": "mp4",
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    filepath = Path(filename)
                    if not filepath.suffix:
                        filepath = filepath.with_suffix(".mp4")
                    if not filepath.exists():
                        mp4_path = filepath.with_suffix(".mp4")
                        if mp4_path.exists():
                            filepath = mp4_path

                metadata = probe_video(filepath)
                record = store.upsert_video(
                    display_name=filepath.stem,
                    original_filename=filepath.name,
                    stored_path=str(filepath),
                    fps=float(metadata["fps"]),
                    duration_s=float(metadata["duration_s"]),
                    frame_width=int(metadata["frame_width"]),
                    frame_height=int(metadata["frame_height"]),
                    source_url=url,
                )
                youtube_jobs[job_id]["status"] = "done"
                youtube_jobs[job_id]["video"] = record
            except Exception as exc:
                youtube_jobs[job_id]["status"] = "error"
                youtube_jobs[job_id]["error"] = str(exc)

        thread = threading.Thread(target=_download, daemon=True)
        thread.start()
        return jsonify({"job_id": job_id})

    @app.get("/api/videos/youtube/status/<job_id>")
    def youtube_status(job_id: str):
        job = youtube_jobs.get(job_id)
        if job is None:
            return _json_error("Unknown job", 404)
        return jsonify(job)

    @app.get("/api/matches/search")
    def search_matches():
        schedule_date = request.args.get("date")
        query = request.args.get("query")
        try:
            matches = client.search_matches(schedule_date=schedule_date, query=query)
        except Exception as exc:
            return _json_error(str(exc), 500)
        return jsonify(
            [
                {
                    "sport_event_id": match.sport_event_id,
                    "start_time": match.start_time,
                    "title": match.title,
                    "home_name": match.home_name,
                    "away_name": match.away_name,
                }
                for match in matches
            ]
        )

    @app.get("/api/events/<path:sport_event_id>/darts")
    def list_event_darts(sport_event_id: str):
        try:
            throws = get_throws_for_event(sport_event_id)
        except Exception as exc:
            return _json_error(str(exc), 500)
        return jsonify([_throw_to_dict(throw) for throw in throws])

    @app.get("/api/anchors")
    def list_anchors():
        video_id = request.args.get("video_id", type=int)
        sport_event_id = request.args.get("sport_event_id")
        anchors = store.list_anchors(video_id=video_id, sport_event_id=sport_event_id)
        return jsonify([_anchor_to_dict(anchor) for anchor in anchors])

    @app.post("/api/anchors")
    def create_anchor():
        payload = request.get_json(force=True)
        try:
            anchor = store.create_anchor(
                SyncAnchor(
                    video_id=int(payload["video_id"]),
                    sport_event_id=str(payload["sport_event_id"]),
                    video_time_s=float(payload["video_time_s"]),
                    timeline_event_id=int(payload["timeline_event_id"]),
                    notes=payload.get("notes"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _json_error(f"Invalid anchor payload: {exc}")
        return jsonify(_anchor_to_dict(anchor))

    @app.patch("/api/anchors/<int:anchor_id>")
    def patch_anchor(anchor_id: int):
        payload = request.get_json(force=True)
        try:
            anchor = store.update_anchor(anchor_id, **payload)
        except Exception as exc:
            return _json_error(str(exc), 400)
        return jsonify(_anchor_to_dict(anchor))

    @app.delete("/api/anchors/<int:anchor_id>")
    def delete_anchor(anchor_id: int):
        store.delete_anchor(anchor_id)
        return jsonify({"deleted": True})

    @app.get("/api/captures")
    def list_captures():
        video_id = request.args.get("video_id", type=int)
        sport_event_id = request.args.get("sport_event_id")
        review_status = request.args.get("review_status")
        captures = store.list_captures(video_id=video_id, sport_event_id=sport_event_id, review_status=review_status)
        return jsonify([_capture_to_dict(capture) for capture in captures])

    @app.get("/api/match-resolution")
    def match_resolution():
        video_id = request.args.get("video_id", type=int)
        sport_event_id = request.args.get("sport_event_id")
        video_time_s = request.args.get("video_time_s", type=float)
        selected_throw_event_id = request.args.get("selected_throw_event_id", type=int)
        if video_id is None or not sport_event_id or video_time_s is None:
            return _json_error("video_id, sport_event_id, and video_time_s are required")

        try:
            timeline = client.get_timeline(sport_event_id)
            throws = client.parse_throw_labels(timeline)
            throw_index = {throw.throw_event_id: throw for throw in throws}
            resolution = resolve_throw_for_capture(
                video_time_s=video_time_s,
                anchors=store.list_anchors(video_id=video_id, sport_event_id=sport_event_id),
                timeline_event_times=client.timeline_event_times(timeline),
                throw_labels=throws,
                selected_throw_event_id=selected_throw_event_id,
            )
        except Exception as exc:
            return _json_error(str(exc), 500)

        candidate_details = [_throw_to_dict(throw_index[event_id]) for event_id in resolution.candidate_throw_event_ids if event_id in throw_index]
        return jsonify(
            {
                "mapped_time_utc": resolution.mapped_time_utc,
                "matched_throw_event_id": resolution.matched_throw_event_id,
                "candidate_throw_event_ids": resolution.candidate_throw_event_ids,
                "ambiguous": resolution.ambiguous,
                "resolution_status": resolution.resolution_status,
                "candidates": candidate_details,
            }
        )

    @app.post("/api/captures")
    def create_capture():
        payload = request.get_json(force=True)
        try:
            image_data = payload["image_data"]
            video_id = int(payload["video_id"])
            video_time_s = float(payload["video_time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            return _json_error(f"Invalid capture payload: {exc}")

        face_bbox_payload = payload.get("face_bbox")
        face_bbox = None
        if face_bbox_payload:
            face_bbox = FaceBoundingBox(
                x=int(face_bbox_payload["x"]),
                y=int(face_bbox_payload["y"]),
                width=int(face_bbox_payload["width"]),
                height=int(face_bbox_payload["height"]),
            )

        capture_filename = f"capture-{uuid.uuid4().hex}.png"
        capture_path = CAPTURES_DIR / capture_filename
        capture_path.write_bytes(_decode_image_data(image_data))

        initial_status = "verified" if payload.get("matched_throw_event_id") else "pending"
        capture = store.create_capture(
            CaptureRecord(
                video_id=video_id,
                sport_event_id=payload.get("sport_event_id"),
                video_time_s=video_time_s,
                frame_path=str(capture_path),
                face_bbox=face_bbox,
                review_status=initial_status,
                matched_throw_event_id=payload.get("matched_throw_event_id"),
                notes=payload.get("notes"),
            )
        )

        resolution_payload: dict[str, Any] = {}
        if capture.sport_event_id:
            try:
                timeline = client.get_timeline(capture.sport_event_id)
                throws = client.parse_throw_labels(timeline)
                throw_index = {throw.throw_event_id: throw for throw in throws}
                resolution = resolve_throw_for_capture(
                    video_time_s=capture.video_time_s,
                    anchors=store.list_anchors(video_id=capture.video_id, sport_event_id=capture.sport_event_id),
                    timeline_event_times=client.timeline_event_times(timeline),
                    throw_labels=throws,
                    selected_throw_event_id=capture.matched_throw_event_id,
                )
                candidate_details = [_throw_to_dict(throw_index[event_id]) for event_id in resolution.candidate_throw_event_ids if event_id in throw_index]
                resolution_payload = {
                    "mapped_time_utc": resolution.mapped_time_utc,
                    "matched_throw_event_id": resolution.matched_throw_event_id,
                    "candidate_throw_event_ids": resolution.candidate_throw_event_ids,
                    "ambiguous": resolution.ambiguous,
                    "resolution_status": resolution.resolution_status,
                    "candidates": candidate_details,
                }
                if capture.id is not None and capture.matched_throw_event_id is None:
                    if resolution.ambiguous:
                        capture = store.update_capture(
                            capture.id,
                            review_status="needs_review",
                            resolved_timeline_time_utc=resolution.mapped_time_utc,
                        )
                    elif resolution.matched_throw_event_id is not None:
                        capture = store.update_capture(
                            capture.id,
                            review_status=resolution.resolution_status,
                            matched_throw_event_id=resolution.matched_throw_event_id,
                            resolved_timeline_time_utc=resolution.mapped_time_utc,
                        )
            except Exception as exc:
                resolution_payload = {"error": str(exc)}

        return jsonify({"capture": _capture_to_dict(capture), "resolution": resolution_payload})

    @app.patch("/api/captures/<int:capture_id>")
    def patch_capture(capture_id: int):
        payload = request.get_json(force=True)
        if "face_bbox" in payload and payload["face_bbox"] is not None:
            bbox_payload = payload.pop("face_bbox")
            payload["face_bbox"] = FaceBoundingBox(
                x=int(bbox_payload["x"]),
                y=int(bbox_payload["y"]),
                width=int(bbox_payload["width"]),
                height=int(bbox_payload["height"]),
            )
        try:
            capture = store.update_capture(capture_id, **payload)
        except Exception as exc:
            return _json_error(str(exc), 400)
        return jsonify(_capture_to_dict(capture))

    @app.delete("/api/captures/<int:capture_id>")
    def delete_capture(capture_id: int):
        store.delete_capture(capture_id)
        return jsonify({"deleted": True})

    return app
