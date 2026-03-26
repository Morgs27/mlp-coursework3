from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_timeline_payload() -> dict:
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_timeline.json"
    return json.loads(fixture_path.read_text())


@pytest.fixture
def sample_debug_image_path() -> Path:
    return PROJECT_ROOT / "debug-images" / "sv_a_20-1.png"
