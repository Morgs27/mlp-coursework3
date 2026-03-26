"""Public package surface for the darts gaze pipeline."""

from .config import DEFAULT_DB_PATH, MODEL_ASSET_PATH, ROOT_DIR
from .gaze import estimate_gaze
from .types import CaptureRecord, EnrichedThrowSample, GazeResult, SyncAnchor, ThrowLabel

__all__ = [
    "CaptureRecord",
    "DEFAULT_DB_PATH",
    "EnrichedThrowSample",
    "GazeResult",
    "MODEL_ASSET_PATH",
    "ROOT_DIR",
    "SyncAnchor",
    "ThrowLabel",
    "estimate_gaze",
]
