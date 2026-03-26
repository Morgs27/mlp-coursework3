from __future__ import annotations

from darts_gaze.sportradar import SportradarClient
from darts_gaze.sync import SyncMapper, parse_utc_timestamp, resolve_throw_for_capture
from darts_gaze.types import SyncAnchor


def test_sync_mapper_supports_interpolation_and_extrapolation(sample_timeline_payload: dict) -> None:
    event_times = SportradarClient.timeline_event_times(sample_timeline_payload)
    anchors = [
        SyncAnchor(video_id=1, sport_event_id="sr:sport_event:test-1", video_time_s=5.0, timeline_event_id=10),
        SyncAnchor(video_id=1, sport_event_id="sr:sport_event:test-1", video_time_s=9.0, timeline_event_id=12),
    ]

    mapper = SyncMapper(anchors=anchors, timeline_event_times=event_times)

    mapped_mid = mapper.map_video_time(7.0)
    mapped_before = mapper.map_video_time(3.0)

    assert mapped_mid == "2026-01-02T20:00:03Z"
    assert parse_utc_timestamp(mapped_before).isoformat().startswith("2026-01-02T19:59:59")


def test_resolve_throw_for_capture_marks_ambiguous_matches(sample_timeline_payload: dict) -> None:
    event_times = SportradarClient.timeline_event_times(sample_timeline_payload)
    throws = SportradarClient.parse_throw_labels(sample_timeline_payload)
    anchors = [SyncAnchor(video_id=1, sport_event_id="sr:sport_event:test-1", video_time_s=5.0, timeline_event_id=10)]

    resolution = resolve_throw_for_capture(
        video_time_s=6.2,
        anchors=anchors,
        timeline_event_times=event_times,
        throw_labels=throws,
    )

    assert resolution.ambiguous is True
    assert resolution.resolution_status == "needs_review"
    assert resolution.matched_throw_event_id == 11
    assert resolution.candidate_throw_event_ids[:2] == [11, 12]


def test_resolve_throw_for_capture_respects_manual_override(sample_timeline_payload: dict) -> None:
    event_times = SportradarClient.timeline_event_times(sample_timeline_payload)
    throws = SportradarClient.parse_throw_labels(sample_timeline_payload)
    anchors = [SyncAnchor(video_id=1, sport_event_id="sr:sport_event:test-1", video_time_s=5.0, timeline_event_id=10)]

    resolution = resolve_throw_for_capture(
        video_time_s=6.2,
        anchors=anchors,
        timeline_event_times=event_times,
        throw_labels=throws,
        selected_throw_event_id=12,
    )

    assert resolution.resolution_status == "verified"
    assert resolution.matched_throw_event_id == 12
