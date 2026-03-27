"""Project-level configuration and filesystem paths."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
CAPTURES_DIR = DATA_DIR / "captures"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIRST_PASS_OUTPUTS_DIR = OUTPUTS_DIR / "first_pass"
FIRST_PASS_TABLES_DIR = FIRST_PASS_OUTPUTS_DIR / "tables"
FIRST_PASS_FIGURES_DIR = FIRST_PASS_OUTPUTS_DIR / "figures"
DEFAULT_DB_PATH = DATA_DIR / "annotations.sqlite3"
MODEL_ASSET_PATH = ROOT_DIR / "face_landmarker.task"


def ensure_data_directories() -> None:
    """Create the project data directories if they do not exist."""
    for directory in (
        DATA_DIR,
        CACHE_DIR,
        PROCESSED_DIR,
        CAPTURES_DIR,
        VIDEOS_DIR,
        OUTPUTS_DIR,
        FIRST_PASS_OUTPUTS_DIR,
        FIRST_PASS_TABLES_DIR,
        FIRST_PASS_FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
