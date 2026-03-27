"""Shared dataclasses used across the collection and modeling pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


Vector3 = tuple[float, float, float]


@dataclass(slots=True)
class FaceBoundingBox:
    """Bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(slots=True)
class GazeResult:
    """Face and gaze outputs for a single image."""

    valid_face: bool
    detector_confidence: float | None
    left_gaze: Vector3 | None
    right_gaze: Vector3 | None
    average_gaze: Vector3 | None
    head_x_axis: Vector3 | None
    head_y_axis: Vector3 | None
    head_z_axis: Vector3 | None
    ipd: float | None
    eye_agreement: float | None
    face_bbox: FaceBoundingBox | None
    image_width: int
    image_height: int
    model_name: str = "mediapipe_face_landmarker"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "valid_face": self.valid_face,
            "detector_confidence": self.detector_confidence,
            "ipd": self.ipd,
            "eye_agreement": self.eye_agreement,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
        }

        vector_fields = {
            "left_gaze": self.left_gaze,
            "right_gaze": self.right_gaze,
            "average_gaze": self.average_gaze,
            "head_x_axis": self.head_x_axis,
            "head_y_axis": self.head_y_axis,
            "head_z_axis": self.head_z_axis,
        }
        for prefix, vector in vector_fields.items():
            for axis, value in zip(("x", "y", "z"), vector or (None, None, None), strict=True):
                output[f"{prefix}_{axis}"] = value

        if self.face_bbox:
            output.update(
                {
                    "face_bbox_x": self.face_bbox.x,
                    "face_bbox_y": self.face_bbox.y,
                    "face_bbox_width": self.face_bbox.width,
                    "face_bbox_height": self.face_bbox.height,
                    "face_bbox_x_norm": self.face_bbox.x / self.image_width if self.image_width else None,
                    "face_bbox_y_norm": self.face_bbox.y / self.image_height if self.image_height else None,
                    "face_bbox_width_norm": self.face_bbox.width / self.image_width if self.image_width else None,
                    "face_bbox_height_norm": self.face_bbox.height / self.image_height if self.image_height else None,
                }
            )
        else:
            for name in (
                "face_bbox_x",
                "face_bbox_y",
                "face_bbox_width",
                "face_bbox_height",
                "face_bbox_x_norm",
                "face_bbox_y_norm",
                "face_bbox_width_norm",
                "face_bbox_height_norm",
            ):
                output[name] = None

        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                output[f"gaze_meta_{key}"] = value
        return output


@dataclass(slots=True)
class SyncAnchor:
    """Manual mapping between a video timestamp and a timeline event."""

    video_id: int
    sport_event_id: str
    video_time_s: float
    timeline_event_id: int
    id: int | None = None
    notes: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class CaptureRecord:
    """Stored frame capture from the annotation tool."""

    video_id: int
    sport_event_id: str | None
    video_time_s: float
    frame_path: str
    id: int | None = None
    face_bbox: FaceBoundingBox | None = None
    review_status: str = "pending"
    matched_throw_event_id: int | None = None
    resolved_timeline_time_utc: str | None = None
    notes: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ThrowLabel:
    """Structured label derived from a Sportradar dart event."""

    match_id: str
    throw_event_id: int
    throw_time_utc: str
    player_id: str
    player_name: str
    competitor_qualifier: str
    resulting_score: int
    raw_resulting_score: int
    segment_label: str
    segment_ring: str
    segment_number: int | None
    is_bust: bool
    is_checkout_attempt: bool
    is_gameshot: bool
    period: int | None
    dart_in_visit: int
    score_remaining_before: int | None
    score_remaining_after: int | None
    opponent_score_remaining_before: int | None


@dataclass(slots=True)
class EnrichedThrowSample:
    """Final joined row used for dataset export and modeling."""

    capture: CaptureRecord
    throw_label: ThrowLabel
    gaze_result: GazeResult
    mapped_capture_time_utc: str | None
    match_resolution: str

    def to_flat_dict(self) -> dict[str, Any]:
        output = {
            "capture_id": self.capture.id,
            "video_id": self.capture.video_id,
            "sport_event_id": self.throw_label.match_id,
            "frame_path": self.capture.frame_path,
            "video_time_s": self.capture.video_time_s,
            "review_status": self.capture.review_status,
            "matched_throw_event_id": self.capture.matched_throw_event_id,
            "resolved_timeline_time_utc": self.capture.resolved_timeline_time_utc,
            "mapped_capture_time_utc": self.mapped_capture_time_utc,
            "match_resolution": self.match_resolution,
            "player_id": self.throw_label.player_id,
            "player_name": self.throw_label.player_name,
            "competitor_qualifier": self.throw_label.competitor_qualifier,
            "throw_event_id": self.throw_label.throw_event_id,
            "throw_time_utc": self.throw_label.throw_time_utc,
            "resulting_score": self.throw_label.resulting_score,
            "raw_resulting_score": self.throw_label.raw_resulting_score,
            "segment_label": self.throw_label.segment_label,
            "segment_ring": self.throw_label.segment_ring,
            "segment_number": self.throw_label.segment_number,
            "is_bust": self.throw_label.is_bust,
            "is_checkout_attempt": self.throw_label.is_checkout_attempt,
            "is_gameshot": self.throw_label.is_gameshot,
            "period": self.throw_label.period,
            "dart_in_visit": self.throw_label.dart_in_visit,
            "score_remaining_before": self.throw_label.score_remaining_before,
            "score_remaining_after": self.throw_label.score_remaining_after,
            "opponent_score_remaining_before": self.throw_label.opponent_score_remaining_before,
        }
        output.update(self.gaze_result.to_flat_dict())
        return output


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    """Convert nested dataclasses to a plain dictionary."""
    return asdict(instance)
