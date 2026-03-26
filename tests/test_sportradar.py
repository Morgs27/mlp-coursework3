from __future__ import annotations

from darts_gaze.sportradar import SportradarClient


def test_parse_throw_labels_reconstructs_segments_and_busts(sample_timeline_payload: dict) -> None:
    throws = SportradarClient.parse_throw_labels(sample_timeline_payload)

    assert len(throws) == 6

    first = throws[0]
    assert first.segment_label == "T20"
    assert first.resulting_score == 60
    assert first.score_remaining_before == 501
    assert first.score_remaining_after == 441

    bull = throws[3]
    assert bull.segment_label == "DB"
    assert bull.segment_ring == "DB"
    assert bull.segment_number == 25
    assert bull.player_name == "Away Player"

    miss = throws[4]
    assert miss.segment_label == "MISS"
    assert miss.resulting_score == 0
    assert miss.score_remaining_after == 451

    bust = throws[5]
    assert bust.segment_label == "T20"
    assert bust.raw_resulting_score == 60
    assert bust.resulting_score == 0
    assert bust.is_bust is True
    assert bust.score_remaining_before == 451
    assert bust.score_remaining_after == 501
