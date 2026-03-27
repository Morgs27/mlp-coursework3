"""Video-to-timeline alignment and throw candidate resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .types import SyncAnchor, ThrowLabel


def parse_utc_timestamp(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class MatchResolution:
    mapped_time_utc: str | None
    matched_throw_event_id: int | None
    candidate_throw_event_ids: list[int]
    ambiguous: bool
    resolution_status: str


class SyncMapper:
    """Map video timestamps to timeline UTC timestamps using manual anchors."""

    def __init__(self, anchors: list[SyncAnchor], timeline_event_times: dict[int, str]):
        self.anchors = sorted(anchors, key=lambda anchor: anchor.video_time_s)
        self._points: list[tuple[float, datetime, int]] = []
        for anchor in self.anchors:
            event_time = timeline_event_times.get(anchor.timeline_event_id)
            if event_time is None:
                continue
            self._points.append((anchor.video_time_s, parse_utc_timestamp(event_time), anchor.timeline_event_id))

    def has_mapping(self) -> bool:
        return bool(self._points)

    def map_video_time(self, video_time_s: float) -> str | None:
        if not self._points:
            return None
        if len(self._points) == 1:
            anchor_time_s, anchor_utc, _ = self._points[0]
            mapped = anchor_utc.timestamp() + (video_time_s - anchor_time_s)
            return format_utc_timestamp(datetime.fromtimestamp(mapped, tz=timezone.utc))

        if video_time_s <= self._points[0][0]:
            left, right = self._points[0], self._points[1]
        elif video_time_s >= self._points[-1][0]:
            left, right = self._points[-2], self._points[-1]
        else:
            left = right = self._points[0]
            for start, end in zip(self._points, self._points[1:], strict=False):
                if start[0] <= video_time_s <= end[0]:
                    left, right = start, end
                    break

        mapped = self._interpolate(video_time_s, left, right)
        return format_utc_timestamp(mapped)

    @staticmethod
    def _interpolate(video_time_s: float, left: tuple[float, datetime, int], right: tuple[float, datetime, int]) -> datetime:
        left_video_s, left_utc, _ = left
        right_video_s, right_utc, _ = right
        if right_video_s == left_video_s:
            return left_utc
        ratio = (video_time_s - left_video_s) / (right_video_s - left_video_s)
        left_ts = left_utc.timestamp()
        right_ts = right_utc.timestamp()
        mapped_ts = left_ts + ((right_ts - left_ts) * ratio)
        return datetime.fromtimestamp(mapped_ts, tz=timezone.utc)


def resolve_throw_for_capture(
    video_time_s: float,
    anchors: list[SyncAnchor],
    timeline_event_times: dict[int, str],
    throw_labels: list[ThrowLabel],
    selected_throw_event_id: int | None = None,
    window_s: float = 5.0,
    pre_window_s: float = 0.0,
) -> MatchResolution:
    """Resolve the nearest next dart event for a video capture."""

    if selected_throw_event_id is not None:
        throw_ids = {throw.throw_event_id for throw in throw_labels}
        status = "verified" if selected_throw_event_id in throw_ids else "needs_review"
        return MatchResolution(
            mapped_time_utc=None,
            matched_throw_event_id=selected_throw_event_id if selected_throw_event_id in throw_ids else None,
            candidate_throw_event_ids=[],
            ambiguous=False,
            resolution_status=status,
        )

    mapper = SyncMapper(anchors=anchors, timeline_event_times=timeline_event_times)
    mapped_time = mapper.map_video_time(video_time_s)
    if mapped_time is None:
        return MatchResolution(
            mapped_time_utc=None,
            matched_throw_event_id=None,
            candidate_throw_event_ids=[],
            ambiguous=False,
            resolution_status="needs_review",
        )

    mapped_dt = parse_utc_timestamp(mapped_time)
    candidate_labels = []
    for label in throw_labels:
        throw_dt = parse_utc_timestamp(label.throw_time_utc)
        delta_s = (throw_dt - mapped_dt).total_seconds()
        if -pre_window_s <= delta_s <= window_s and delta_s >= 0:
            candidate_labels.append((delta_s, label))

    if not candidate_labels:
        return MatchResolution(
            mapped_time_utc=mapped_time,
            matched_throw_event_id=None,
            candidate_throw_event_ids=[],
            ambiguous=False,
            resolution_status="needs_review",
        )

    candidate_labels.sort(key=lambda item: (abs(item[0]), item[1].throw_event_id))
    candidate_ids = [label.throw_event_id for _, label in candidate_labels]
    nearest = candidate_labels[0][1]
    ambiguous = len(candidate_labels) > 1
    status = "needs_review" if ambiguous else "matched"
    return MatchResolution(
        mapped_time_utc=mapped_time,
        matched_throw_event_id=nearest.throw_event_id,
        candidate_throw_event_ids=candidate_ids,
        ambiguous=ambiguous,
        resolution_status=status,
    )
