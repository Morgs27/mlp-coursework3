from __future__ import annotations

from darts_gaze.targets import (
    circular_wedge_distance,
    coarse_wedge_area_label,
    is_three_wedge_hit,
    target_columns,
    wedge_neighbors,
    wedge_number_label,
)


def test_target_helpers_map_wedges_and_coarse_areas() -> None:
    assert wedge_neighbors(20) == (5, 20, 1)
    assert wedge_number_label(20, "T20") == "20"
    assert wedge_number_label(25, "SB") == "BULL"
    assert coarse_wedge_area_label(1, "S1") == "20"
    assert coarse_wedge_area_label(7, "S7") == "19"
    assert coarse_wedge_area_label(9, "S9") == "OTHER"

    columns = target_columns(20, "S20")
    assert columns["wedge_number_label"] == "20"
    assert columns["coarse_wedge_area_label"] == "20"
    assert columns["coarse_wedge_area_members"] == "5|20|1"


def test_wedge_metrics_helpers_support_three_wedge_matching() -> None:
    assert circular_wedge_distance(20, 1) == 1
    assert circular_wedge_distance(20, 18) == 2
    assert is_three_wedge_hit(20, 5) is True
    assert is_three_wedge_hit(20, 1) is True
    assert is_three_wedge_hit(20, 19) is False
