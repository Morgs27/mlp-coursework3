"""Sportradar client, caching, and timeline parsing helpers."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .config import CACHE_DIR, ensure_data_directories
from .matches import KNOWN_MATCHES
from .types import ThrowLabel


def _segment_parts(score: int | None, multiplier: int | None) -> tuple[str, str, int | None]:
    if score in (None, 0) or multiplier in (None, 0):
        return "MISS", "MISS", None
    if score == 25 and multiplier == 1:
        return "SB", "SB", 25
    if score == 25 and multiplier == 2:
        return "DB", "DB", 25
    ring = {1: "S", 2: "D", 3: "T"}.get(multiplier, "MISS")
    number = None if ring == "MISS" else score
    label = "MISS" if ring == "MISS" else f"{ring}{score}"
    return label, ring, number


@dataclass(slots=True)
class MatchSummary:
    sport_event_id: str
    start_time: str
    title: str
    home_name: str
    away_name: str


class SportradarClient:
    """Small Sportradar client with file-backed caching."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        access_level: str = "trial",
        language: str = "en",
        timeout: int = 30,
        max_retries: int = 8,
        base_sleep: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SPORTRADAR_API_KEY")
        self.cache_dir = cache_dir or CACHE_DIR
        self.access_level = access_level
        self.language = language
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        ensure_data_directories()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("SPORTRADAR_API_KEY is not set")
        return self.api_key

    def _request_json(self, url: str, cache_path: Path | None = None, force_refresh: bool = False) -> dict[str, Any]:
        if cache_path and cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text())

        api_key = self.require_api_key()
        for attempt in range(self.max_retries):
            response = requests.get(url, params={"api_key": api_key}, timeout=self.timeout)
            if response.status_code == 200:
                payload = response.json()
                if cache_path:
                    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                return payload

            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else self.base_sleep * (2**attempt)
                time.sleep(sleep_seconds + random.uniform(0, 0.5))
                continue

            raise RuntimeError(f"Sportradar request failed with {response.status_code}: {response.text[:200]}")

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} retries")

    def get_schedule(self, schedule_date: date | str, force_refresh: bool = False) -> dict[str, Any]:
        if isinstance(schedule_date, str):
            date_key = schedule_date
        else:
            date_key = schedule_date.isoformat()
        cache_path = self.cache_dir / f"schedule-{date_key}.json"
        url = f"https://api.sportradar.com/darts/{self.access_level}/v2/{self.language}/schedules/{date_key}/summaries.json"
        return self._request_json(url=url, cache_path=cache_path, force_refresh=force_refresh)

    def get_timeline(self, sport_event_id: str, force_refresh: bool = False) -> dict[str, Any]:
        safe_event_id = sport_event_id.replace(":", "_")
        cache_path = self.cache_dir / f"timeline-{safe_event_id}.json"
        url = f"https://api.sportradar.com/darts/{self.access_level}/v2/{self.language}/sport_events/{sport_event_id}/timeline.json"
        return self._request_json(url=url, cache_path=cache_path, force_refresh=force_refresh)

    def search_matches(self, schedule_date: date | str | None = None, query: str | None = None) -> list[MatchSummary]:
        query_lower = query.lower() if query else None
        summaries: list[MatchSummary] = []

        if schedule_date is None and query_lower:
            for known_match in KNOWN_MATCHES:
                if query_lower in known_match.title.lower():
                    summaries.append(
                        MatchSummary(
                            sport_event_id=known_match.sport_event_id,
                            start_time=known_match.match_date.isoformat(),
                            title=known_match.title,
                            home_name=known_match.title.split(" v ")[0],
                            away_name=known_match.title.split(" v ")[1].split(" 2026")[0],
                        )
                    )
            return summaries

        if schedule_date is None:
            return []

        schedule = self.get_schedule(schedule_date)
        items = schedule.get("summaries") or schedule.get("sport_events") or []
        for item in items:
            event = item.get("sport_event", item)
            competitors = event.get("competitors", [])
            home = next((competitor for competitor in competitors if competitor.get("qualifier") == "home"), {})
            away = next((competitor for competitor in competitors if competitor.get("qualifier") == "away"), {})
            title = f"{home.get('name', 'Unknown')} vs {away.get('name', 'Unknown')}"
            if query_lower and query_lower not in title.lower():
                continue
            summaries.append(
                MatchSummary(
                    sport_event_id=event["id"],
                    start_time=event.get("start_time", ""),
                    title=title,
                    home_name=home.get("name", ""),
                    away_name=away.get("name", ""),
                )
            )
        return summaries

    @staticmethod
    def timeline_event_times(timeline_payload: dict[str, Any]) -> dict[int, str]:
        event_times: dict[int, str] = {}
        for event in timeline_payload.get("timeline", []):
            event_id = event.get("id")
            event_time = event.get("time")
            if event_id is not None and event_time:
                event_times[int(event_id)] = event_time
        return event_times

    @staticmethod
    def parse_throw_labels(timeline_payload: dict[str, Any]) -> list[ThrowLabel]:
        timeline_events = sorted(
            timeline_payload.get("timeline", []),
            key=lambda event: (event.get("time", ""), int(event.get("id", 0))),
        )
        sport_event = timeline_payload.get("sport_event", {})
        competitors = {
            competitor.get("qualifier"): competitor
            for competitor in sport_event.get("competitors", [])
            if competitor.get("qualifier")
        }

        remaining = {"home": 501, "away": 501}
        visit_remaining = remaining.copy()
        current_thrower: str | None = None
        current_period = 0
        dart_in_visit = 0
        throws: list[ThrowLabel] = []

        for event in timeline_events:
            event_type = event.get("type")
            if event_type == "period_start":
                current_period += 1
                current_thrower = None
                dart_in_visit = 0
                remaining = {"home": 501, "away": 501}
                visit_remaining = remaining.copy()
                continue
                
            if event_type == "leg_score_change":
                current_period += 1
                continue

            if event_type == "score_change":
                if event.get("home_score") is not None:
                    remaining["home"] = int(event["home_score"])
                if event.get("away_score") is not None:
                    remaining["away"] = int(event["away_score"])
                continue

            if event_type != "dart":
                continue

            competitor_qualifier = event.get("competitor")
            if competitor_qualifier not in {"home", "away"}:
                continue

            if current_thrower != competitor_qualifier:
                current_thrower = competitor_qualifier
                dart_in_visit = 0
                visit_remaining[competitor_qualifier] = remaining[competitor_qualifier]

            dart_in_visit += 1
            raw_score = int(event.get("dart_score_total") or (event.get("dart_score") or 0) * (event.get("dart_score_multiplier") or 1))
            is_bust = bool(event.get("is_bust"))
            rem_before = visit_remaining[competitor_qualifier]
            rem_after = rem_before if is_bust else rem_before - raw_score
            visit_remaining[competitor_qualifier] = remaining[competitor_qualifier] if is_bust else rem_after

            segment_label, segment_ring, segment_number = _segment_parts(
                score=event.get("dart_score"),
                multiplier=event.get("dart_score_multiplier"),
            )
            competitor = competitors.get(competitor_qualifier, {})
            opponent_key = "away" if competitor_qualifier == "home" else "home"
            throws.append(
                ThrowLabel(
                    match_id=sport_event.get("id", ""),
                    throw_event_id=int(event["id"]),
                    throw_time_utc=event.get("time", ""),
                    player_id=competitor.get("id", ""),
                    player_name=competitor.get("name", ""),
                    competitor_qualifier=competitor_qualifier,
                    resulting_score=0 if is_bust else raw_score,
                    raw_resulting_score=raw_score,
                    segment_label=segment_label,
                    segment_ring=segment_ring,
                    segment_number=segment_number,
                    is_bust=is_bust,
                    is_checkout_attempt=bool(event.get("is_checkout_attempt")),
                    is_gameshot=bool(event.get("is_gameshot")),
                    period=current_period or None,
                    dart_in_visit=dart_in_visit,
                    score_remaining_before=rem_before,
                    score_remaining_after=remaining[competitor_qualifier] if is_bust else rem_after,
                    opponent_score_remaining_before=remaining[opponent_key],
                )
            )
        return throws
