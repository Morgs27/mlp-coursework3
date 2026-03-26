from __future__ import annotations

from darts_gaze.gaze import estimate_gaze
from darts_gaze.types import FaceBoundingBox


def test_estimate_gaze_supports_full_frame_and_roi(sample_debug_image_path) -> None:
    full_frame_result = estimate_gaze(sample_debug_image_path)

    assert full_frame_result.valid_face is True
    assert full_frame_result.left_gaze is not None
    assert full_frame_result.right_gaze is not None
    assert full_frame_result.average_gaze is not None
    assert full_frame_result.face_bbox is not None

    face_bbox = full_frame_result.face_bbox
    padded_roi = FaceBoundingBox(
        x=max(0, face_bbox.x - 20),
        y=max(0, face_bbox.y - 20),
        width=face_bbox.width + 40,
        height=face_bbox.height + 40,
    )
    roi_result = estimate_gaze(sample_debug_image_path, face_bbox=padded_roi)

    assert roi_result.valid_face is True
    assert roi_result.face_bbox is not None
    assert roi_result.image_width == full_frame_result.image_width
    assert roi_result.image_height == full_frame_result.image_height
