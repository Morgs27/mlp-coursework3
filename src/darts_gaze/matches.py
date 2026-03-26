"""Known match metadata and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class KnownMatch:
    sport_event_id: str
    match_date: date
    title: str
    youtube_url: str


KNOWN_MATCHES: tuple[KnownMatch, ...] = (
    KnownMatch(
        sport_event_id="sr:sport_event:66098020",
        match_date=date(2026, 1, 1),
        title="Anderson v Hood QF 2026 World Darts Championship",
        youtube_url="https://www.youtube.com/watch?v=p-u9VVLM-yo",
    ),
    KnownMatch(
        sport_event_id="sr:sport_event:66098024",
        match_date=date(2026, 1, 2),
        title="Littler v Searle SF 2026 World Darts Championship",
        youtube_url="https://www.youtube.com/watch?v=aF7Gk1ScqbU",
    ),
    KnownMatch(
        sport_event_id="sr:sport_event:66098028",
        match_date=date(2026, 1, 2),
        title="Van Veen v Anderson SF 2026 World Darts Championship",
        youtube_url="https://www.youtube.com/watch?v=LL-GWqNUmZ0&t=1380s",
    ),
    KnownMatch(
        sport_event_id="sr:sport_event:66098032",
        match_date=date(2026, 1, 3),
        title="Littler v Van Veen 2026 FINAL World Darts Championship",
        youtube_url="https://www.youtube.com/watch?v=EP1aNxTksCc",
    ),
)


def get_known_match(sport_event_id: str) -> KnownMatch | None:
    for match in KNOWN_MATCHES:
        if match.sport_event_id == sport_event_id:
            return match
    return None
