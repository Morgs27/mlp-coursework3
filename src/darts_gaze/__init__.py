"""Public package surface for the darts gaze pipeline."""

from .config import DEFAULT_DB_PATH, MODEL_ASSET_PATH, ROOT_DIR
from .gaze import annotate_frame, estimate_gaze, overlay_payload
from .types import CaptureRecord, EnrichedThrowSample, GazeResult, SyncAnchor, ThrowLabel

__all__ = [
    "annotate_frame",
    "CaptureRecord",
    "DEFAULT_DB_PATH",
    "EnrichedThrowSample",
    "GazeResult",
    "MODEL_ASSET_PATH",
    "overlay_payload",
    "ROOT_DIR",
    "SyncAnchor",
    "ThrowLabel",
    "estimate_gaze",
]
