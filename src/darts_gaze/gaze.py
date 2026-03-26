"""Gaze estimation using MediaPipe face landmarks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import MODEL_ASSET_PATH
from .types import FaceBoundingBox, GazeResult, Vector3

LEFT_EYE_INDICES = (33, 133)
RIGHT_EYE_INDICES = (362, 263)
LEFT_IRIS_INDICES = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES = (473, 474, 475, 476, 477)


def _coerce_image(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.copy()
    image_path = Path(image)
    loaded = cv2.imread(str(image_path))
    if loaded is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")
    return loaded


def _coerce_bbox(face_bbox: FaceBoundingBox | tuple[int, int, int, int] | None, shape: tuple[int, int, int]) -> FaceBoundingBox | None:
    if face_bbox is None:
        return None
    if isinstance(face_bbox, FaceBoundingBox):
        bbox = face_bbox
    else:
        bbox = FaceBoundingBox(*face_bbox)
    image_height, image_width = shape[:2]
    x = max(0, min(bbox.x, image_width - 1))
    y = max(0, min(bbox.y, image_height - 1))
    width = max(1, min(bbox.width, image_width - x))
    height = max(1, min(bbox.height, image_height - y))
    return FaceBoundingBox(x=x, y=y, width=width, height=height)


@lru_cache(maxsize=1)
def _get_detector() -> vision.FaceLandmarker:
    model_path = MODEL_ASSET_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Face landmarker model not found at {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=3,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _mean_point(face_landmarks: list, indices: Iterable[int]) -> np.ndarray:
    coords = np.array([[face_landmarks[index].x, face_landmarks[index].y, face_landmarks[index].z] for index in indices], dtype=np.float64)
    return coords.mean(axis=0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.zeros(3, dtype=np.float64)
    return vector / magnitude


def _compute_head_axes(face_landmarks: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.array([face_landmarks[454].x, face_landmarks[454].y, face_landmarks[454].z], dtype=np.float64)
    right = np.array([face_landmarks[234].x, face_landmarks[234].y, face_landmarks[234].z], dtype=np.float64)
    top = np.array([face_landmarks[10].x, face_landmarks[10].y, face_landmarks[10].z], dtype=np.float64)
    bottom = np.array([face_landmarks[152].x, face_landmarks[152].y, face_landmarks[152].z], dtype=np.float64)

    head_x = _normalize(right - left)
    head_y_approx = _normalize(top - bottom)
    head_z = _normalize(np.cross(head_x, head_y_approx))
    head_y = _normalize(np.cross(head_z, head_x))
    return head_x, head_y, head_z


def _compute_gaze_vector(face_landmarks: list, eye_indices: Iterable[int], iris_indices: Iterable[int], head_z_axis: np.ndarray, ipd: float) -> tuple[np.ndarray, np.ndarray]:
    eye_surface_center = _mean_point(face_landmarks, eye_indices)
    iris_center = _mean_point(face_landmarks, iris_indices)
    eye_center = eye_surface_center + (head_z_axis * (ipd * 0.3))
    gaze_vector = _normalize(iris_center - eye_center)
    return gaze_vector, iris_center


def _landmark_bbox(face_landmarks: list, image_width: int, image_height: int, offset_x: int = 0, offset_y: int = 0) -> FaceBoundingBox:
    xs = [landmark.x for landmark in face_landmarks]
    ys = [landmark.y for landmark in face_landmarks]
    min_x = max(0, int(min(xs) * image_width) + offset_x)
    min_y = max(0, int(min(ys) * image_height) + offset_y)
    max_x = int(max(xs) * image_width) + offset_x
    max_y = int(max(ys) * image_height) + offset_y
    return FaceBoundingBox(x=min_x, y=min_y, width=max(1, max_x - min_x), height=max(1, max_y - min_y))


def _select_prominent_face(face_landmarks_list: list[list], image_width: int, image_height: int) -> tuple[int, FaceBoundingBox]:
    boxes = [_landmark_bbox(face_landmarks, image_width, image_height) for face_landmarks in face_landmarks_list]
    index = max(range(len(boxes)), key=lambda idx: boxes[idx].width * boxes[idx].height)
    return index, boxes[index]


def _to_vector3(vector: np.ndarray | None) -> Vector3 | None:
    if vector is None:
        return None
    return tuple(float(component) for component in vector.tolist())


def estimate_gaze(image: np.ndarray | str | Path, face_bbox: FaceBoundingBox | tuple[int, int, int, int] | None = None) -> GazeResult:
    """Estimate left/right/average gaze vectors and head pose from a frame.

    If ``face_bbox`` is provided, detection is restricted to that region. The
    returned bounding box is always expressed in full-frame pixel coordinates.
    """

    frame = _coerce_image(image)
    image_height, image_width = frame.shape[:2]
    roi = _coerce_bbox(face_bbox, frame.shape)
    if roi is None:
        cropped = frame
        offset_x = 0
        offset_y = 0
    else:
        cropped = frame[roi.y : roi.y2, roi.x : roi.x2]
        offset_x = roi.x
        offset_y = roi.y
        if cropped.size == 0:
            raise ValueError("Provided face_bbox produced an empty crop")

    detector = _get_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return GazeResult(
            valid_face=False,
            detector_confidence=0.0,
            left_gaze=None,
            right_gaze=None,
            average_gaze=None,
            head_x_axis=None,
            head_y_axis=None,
            head_z_axis=None,
            ipd=None,
            eye_agreement=None,
            face_bbox=roi,
            image_width=image_width,
            image_height=image_height,
            metadata={"roi_applied": roi is not None, "selected_face_index": None},
        )

    crop_height, crop_width = cropped.shape[:2]
    face_index, _ = _select_prominent_face(result.face_landmarks, crop_width, crop_height)
    face_landmarks = result.face_landmarks[face_index]
    resolved_face_bbox = _landmark_bbox(face_landmarks, crop_width, crop_height, offset_x=offset_x, offset_y=offset_y)

    left_iris_center = _mean_point(face_landmarks, LEFT_IRIS_INDICES)
    right_iris_center = _mean_point(face_landmarks, RIGHT_IRIS_INDICES)
    ipd = float(np.linalg.norm(left_iris_center - right_iris_center))
    head_x_axis, head_y_axis, head_z_axis = _compute_head_axes(face_landmarks)
    left_gaze, _ = _compute_gaze_vector(face_landmarks, LEFT_EYE_INDICES, LEFT_IRIS_INDICES, head_z_axis, ipd)
    right_gaze, _ = _compute_gaze_vector(face_landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES, head_z_axis, ipd)
    average_gaze = _normalize(left_gaze + right_gaze)
    eye_agreement = float(np.clip(np.dot(left_gaze, right_gaze), -1.0, 1.0))

    # MediaPipe FaceLandmarker does not expose a per-face score here, so this is
    # a binary confidence proxy rather than a calibrated detector probability.
    detector_confidence = 1.0
    return GazeResult(
        valid_face=True,
        detector_confidence=detector_confidence,
        left_gaze=_to_vector3(left_gaze),
        right_gaze=_to_vector3(right_gaze),
        average_gaze=_to_vector3(average_gaze),
        head_x_axis=_to_vector3(head_x_axis),
        head_y_axis=_to_vector3(head_y_axis),
        head_z_axis=_to_vector3(head_z_axis),
        ipd=ipd,
        eye_agreement=eye_agreement,
        face_bbox=resolved_face_bbox,
        image_width=image_width,
        image_height=image_height,
        metadata={"roi_applied": roi is not None, "selected_face_index": face_index},
    )
