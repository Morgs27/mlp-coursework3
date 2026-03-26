"""Video metadata helpers for the annotation app."""

from __future__ import annotations

from pathlib import Path

import cv2


def probe_video(path: str | Path) -> dict[str, float | int]:
    video_path = str(path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration_s = (frame_count / fps) if fps else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "duration_s": duration_s,
    }
