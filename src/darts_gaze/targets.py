"""Helpers for ringless wedge targets and coarse target-area labels."""

from __future__ import annotations

from math import isnan
from typing import Any


BOARD_WEDGE_ORDER: tuple[int, ...] = (
    20,
    1,
    18,
    4,
    13,
    6,
    10,
    15,
    2,
    17,
    3,
    19,
    7,
    16,
    8,
    11,
    14,
    9,
    12,
    5,
)
BOARD_WEDGE_INDEX = {number: index for index, number in enumerate(BOARD_WEDGE_ORDER)}
DOMINANT_TARGET_CENTERS: tuple[int, ...] = (20, 19, 18, 17, 16)


def coerce_segment_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def wedge_neighbors(center_number: int) -> tuple[int, int, int]:
    index = BOARD_WEDGE_INDEX[center_number]
    return (
        BOARD_WEDGE_ORDER[(index - 1) % len(BOARD_WEDGE_ORDER)],
        center_number,
        BOARD_WEDGE_ORDER[(index + 1) % len(BOARD_WEDGE_ORDER)],
    )


def wedge_number_label(segment_number: Any, segment_label: str | None = None) -> str:
    number = coerce_segment_number(segment_number)
    if segment_label == "MISS" or number is None:
        return "MISS"
    if number in {25, 50} or segment_label in {"SB", "DB"}:
        return "BULL"
    return str(number)


def coarse_wedge_area_label(segment_number: Any, segment_label: str | None = None) -> str:
    wedge_label = wedge_number_label(segment_number, segment_label=segment_label)
    if wedge_label in {"MISS", "BULL"}:
        return wedge_label

    number = int(wedge_label)
    best_center = min(
        DOMINANT_TARGET_CENTERS,
        key=lambda center: (circular_wedge_distance(center, number), DOMINANT_TARGET_CENTERS.index(center)),
    )
    if circular_wedge_distance(best_center, number) <= 1:
        return str(best_center)
    return "OTHER"


def coarse_wedge_area_members(label: str) -> str:
    if label.isdigit():
        left, center, right = wedge_neighbors(int(label))
        return f"{left}|{center}|{right}"
    return label


def circular_wedge_distance(predicted_number: Any, actual_number: Any) -> int | None:
    predicted = coerce_segment_number(predicted_number)
    actual = coerce_segment_number(actual_number)
    if predicted is None or actual is None:
        return None
    if predicted in {25, 50} or actual in {25, 50}:
        return 0 if predicted == actual else None
    if predicted not in BOARD_WEDGE_INDEX or actual not in BOARD_WEDGE_INDEX:
        return None

    predicted_index = BOARD_WEDGE_INDEX[predicted]
    actual_index = BOARD_WEDGE_INDEX[actual]
    direct_distance = abs(predicted_index - actual_index)
    return min(direct_distance, len(BOARD_WEDGE_ORDER) - direct_distance)


def is_three_wedge_hit(predicted_number: Any, actual_number: Any) -> bool:
    if predicted_number == actual_number:
        return True
    predicted = coerce_segment_number(predicted_number)
    actual = coerce_segment_number(actual_number)
    if predicted is None or actual is None:
        return False
    if predicted in {25, 50} or actual in {25, 50}:
        return False
    if predicted not in BOARD_WEDGE_INDEX:
        return False
    return actual in set(wedge_neighbors(predicted))


def target_columns(segment_number: Any, segment_label: str | None = None) -> dict[str, Any]:
    wedge_label = wedge_number_label(segment_number, segment_label=segment_label)
    coarse_label = coarse_wedge_area_label(segment_number, segment_label=segment_label)
    return {
        "wedge_number_label": wedge_label,
        "coarse_wedge_area_label": coarse_label,
        "coarse_wedge_area_members": coarse_wedge_area_members(coarse_label),
    }
