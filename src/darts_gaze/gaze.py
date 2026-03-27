"""Gaze estimation using MediaPipe face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
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


@dataclass(slots=True)
class _SearchCandidate:
    name: str
    crop_bbox: FaceBoundingBox
    image: np.ndarray


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


def _expand_bbox(
    bbox: FaceBoundingBox,
    shape: tuple[int, int, int],
    *,
    x_ratio: float,
    y_ratio: float,
) -> FaceBoundingBox:
    image_height, image_width = shape[:2]
    expand_x = int(round(bbox.width * x_ratio))
    expand_y = int(round(bbox.height * y_ratio))
    expanded = FaceBoundingBox(
        x=max(0, bbox.x - expand_x),
        y=max(0, bbox.y - expand_y),
        width=min(image_width, bbox.x2 + expand_x) - max(0, bbox.x - expand_x),
        height=min(image_height, bbox.y2 + expand_y) - max(0, bbox.y - expand_y),
    )
    return _coerce_bbox(expanded, shape) or bbox


def _scaled_crop(crop: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _add_candidate(
    candidates: list[_SearchCandidate],
    seen: set[tuple[str, int, int, int, int, int, int]],
    frame: np.ndarray,
    bbox: FaceBoundingBox | None,
    *,
    name: str,
    scale: float = 1.0,
) -> None:
    if bbox is None or bbox.width <= 1 or bbox.height <= 1:
        return
    key = (
        name,
        bbox.x,
        bbox.y,
        bbox.width,
        bbox.height,
        int(round(scale * 100)),
        frame.shape[1],
    )
    if key in seen:
        return
    crop = frame[bbox.y : bbox.y2, bbox.x : bbox.x2]
    if crop.size == 0:
        return
    seen.add(key)
    candidates.append(_SearchCandidate(name=name, crop_bbox=bbox, image=_scaled_crop(crop, scale)))


def _build_search_candidates(
    frame: np.ndarray,
    roi: FaceBoundingBox | None,
) -> list[_SearchCandidate]:
    image_height, image_width = frame.shape[:2]
    full_bbox = FaceBoundingBox(x=0, y=0, width=image_width, height=image_height)
    candidates: list[_SearchCandidate] = []
    seen: set[tuple[str, int, int, int, int, int, int]] = set()

    if roi is not None:
        _add_candidate(candidates, seen, frame, roi, name="roi_exact")
        _add_candidate(
            candidates,
            seen,
            frame,
            _expand_bbox(roi, frame.shape, x_ratio=0.25, y_ratio=0.35),
            name="roi_expanded_medium",
        )
        _add_candidate(
            candidates,
            seen,
            frame,
            _expand_bbox(roi, frame.shape, x_ratio=0.5, y_ratio=0.7),
            name="roi_expanded_large",
        )

    _add_candidate(candidates, seen, frame, full_bbox, name="full_frame")

    aspect_ratio = image_width / max(image_height, 1)
    if aspect_ratio >= 1.2:
        right_half = FaceBoundingBox(
            x=image_width // 2,
            y=0,
            width=image_width - (image_width // 2),
            height=image_height,
        )
        right_60 = FaceBoundingBox(
            x=int(image_width * 0.4),
            y=0,
            width=image_width - int(image_width * 0.4),
            height=image_height,
        )
        upper_right = FaceBoundingBox(
            x=int(image_width * 0.52),
            y=0,
            width=image_width - int(image_width * 0.52),
            height=int(image_height * 0.92),
        )
        left_half = FaceBoundingBox(
            x=0,
            y=0,
            width=max(1, image_width // 2),
            height=image_height,
        )
        _add_candidate(candidates, seen, frame, right_half, name="right_half")
        _add_candidate(candidates, seen, frame, right_60, name="right_sixty")
        _add_candidate(candidates, seen, frame, upper_right, name="upper_right")
        _add_candidate(candidates, seen, frame, left_half, name="left_half")
        _add_candidate(candidates, seen, frame, right_half, name="right_half_upscaled", scale=2.0)
        _add_candidate(candidates, seen, frame, right_60, name="right_sixty_upscaled", scale=2.0)

    _add_candidate(candidates, seen, frame, full_bbox, name="full_frame_upscaled", scale=2.0)
    return candidates


def _detect_on_candidate(
    detector: vision.FaceLandmarker,
    candidate: _SearchCandidate,
) -> tuple[list | None, int | None]:
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(candidate.image, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    face_index, _ = _select_prominent_face(
        result.face_landmarks,
        candidate.crop_bbox.width,
        candidate.crop_bbox.height,
    )
    return result.face_landmarks[face_index], face_index


def _to_vector3(vector: np.ndarray | None) -> Vector3 | None:
    if vector is None:
        return None
    return tuple(float(component) for component in vector.tolist())


def _point_to_pixels(point: np.ndarray, image_width: int, image_height: int, offset_x: int = 0, offset_y: int = 0) -> tuple[int, int]:
    return (
        int(point[0] * image_width) + offset_x,
        int(point[1] * image_height) + offset_y,
    )


def _overlay_metadata(
    *,
    crop_width: int,
    crop_height: int,
    offset_x: int,
    offset_y: int,
    face_bbox: FaceBoundingBox,
    head_x_axis: np.ndarray,
    head_y_axis: np.ndarray,
    head_z_axis: np.ndarray,
    left_gaze: np.ndarray,
    right_gaze: np.ndarray,
    average_gaze: np.ndarray,
    left_iris_center: np.ndarray,
    right_iris_center: np.ndarray,
    nose_tip: np.ndarray,
) -> dict[str, Any]:
    face_scale = max(face_bbox.width, face_bbox.height)
    gaze_scale = max(36, int(face_scale * 0.75))
    axis_scale = max(28, int(face_scale * 0.45))

    def _arrow(start_px: tuple[int, int], vector: np.ndarray, scale: int, label: str) -> dict[str, Any]:
        return {
            "label": label,
            "start": {"x": int(start_px[0]), "y": int(start_px[1])},
            "end": {"x": int(start_px[0] + (vector[0] * scale)), "y": int(start_px[1] + (vector[1] * scale))},
        }

    left_start = _point_to_pixels(left_iris_center, crop_width, crop_height, offset_x, offset_y)
    right_start = _point_to_pixels(right_iris_center, crop_width, crop_height, offset_x, offset_y)
    avg_start = (
        int((left_start[0] + right_start[0]) / 2),
        int((left_start[1] + right_start[1]) / 2),
    )
    nose_px = _point_to_pixels(nose_tip, crop_width, crop_height, offset_x, offset_y)
    return {
        "face_bbox": face_bbox.to_dict(),
        "gaze_arrows": [
            _arrow(left_start, left_gaze, gaze_scale, "left"),
            _arrow(right_start, right_gaze, gaze_scale, "right"),
            _arrow(avg_start, average_gaze, gaze_scale, "average"),
        ],
        "head_axes": [
            _arrow(nose_px, head_x_axis, axis_scale, "x"),
            _arrow(nose_px, head_y_axis, axis_scale, "y"),
            _arrow(nose_px, -head_z_axis, axis_scale, "z"),
        ],
        "nose_point": {"x": int(nose_px[0]), "y": int(nose_px[1])},
    }


def overlay_payload(gaze_result: GazeResult) -> dict[str, Any] | None:
    overlay = gaze_result.metadata.get("overlay")
    return overlay if isinstance(overlay, dict) else None


def annotate_frame(
    image: np.ndarray | str | Path,
    gaze_result: GazeResult | None = None,
    *,
    face_bbox: FaceBoundingBox | tuple[int, int, int, int] | None = None,
    crop_to_roi: bool = False,
) -> np.ndarray:
    """Draw gaze vectors and head axes on an image."""

    frame = _coerce_image(image)
    result = gaze_result or estimate_gaze(frame, face_bbox=face_bbox)
    annotated = frame.copy()

    overlay = overlay_payload(result)
    if overlay:
        face_box = overlay.get("face_bbox")
        if face_box:
            cv2.rectangle(
                annotated,
                (int(face_box["x"]), int(face_box["y"])),
                (int(face_box["x"] + face_box["width"]), int(face_box["y"] + face_box["height"])),
                (42, 157, 143),
                2,
                cv2.LINE_AA,
            )
        color_map = {
            "left": (44, 160, 255),
            "right": (231, 111, 81),
            "average": (233, 196, 106),
            "x": (0, 0, 255),
            "y": (0, 255, 0),
            "z": (255, 0, 0),
        }
        for arrow in overlay.get("gaze_arrows", []):
            cv2.arrowedLine(
                annotated,
                (int(arrow["start"]["x"]), int(arrow["start"]["y"])),
                (int(arrow["end"]["x"]), int(arrow["end"]["y"])),
                color_map.get(arrow["label"], (255, 255, 255)),
                2,
                cv2.LINE_AA,
                tipLength=0.2,
            )
        for axis in overlay.get("head_axes", []):
            cv2.arrowedLine(
                annotated,
                (int(axis["start"]["x"]), int(axis["start"]["y"])),
                (int(axis["end"]["x"]), int(axis["end"]["y"])),
                color_map.get(axis["label"], (255, 255, 255)),
                2,
                cv2.LINE_AA,
                tipLength=0.16,
            )

    status_color = (63, 185, 80) if result.valid_face else (244, 112, 103)
    cv2.putText(
        annotated,
        f"Face: {'yes' if result.valid_face else 'no'}  IPD: {result.ipd:.4f}" if result.ipd is not None else f"Face: {'yes' if result.valid_face else 'no'}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if result.average_gaze is not None:
        avg_x, avg_y, avg_z = result.average_gaze
        cv2.putText(
            annotated,
            f"Avg gaze: ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f})",
            (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (230, 237, 243),
            1,
            cv2.LINE_AA,
        )

    if crop_to_roi and result.face_bbox is not None:
        x1, y1 = result.face_bbox.x, result.face_bbox.y
        x2, y2 = result.face_bbox.x2, result.face_bbox.y2
        return annotated[y1:y2, x1:x2].copy()
    return annotated


def estimate_gaze(image: np.ndarray | str | Path, face_bbox: FaceBoundingBox | tuple[int, int, int, int] | None = None) -> GazeResult:
    """Estimate left/right/average gaze vectors and head pose from a frame.

    If ``face_bbox`` is provided, detection is restricted to that region. The
    returned bounding box is always expressed in full-frame pixel coordinates.
    """

    frame = _coerce_image(image)
    image_height, image_width = frame.shape[:2]
    roi = _coerce_bbox(face_bbox, frame.shape)
    detector = _get_detector()
    attempted_candidates: list[str] = []
    selected_candidate: _SearchCandidate | None = None
    selected_face_landmarks: list | None = None
    face_index: int | None = None

    for candidate in _build_search_candidates(frame, roi):
        attempted_candidates.append(candidate.name)
        face_landmarks, detected_face_index = _detect_on_candidate(detector, candidate)
        if face_landmarks is None:
            continue
        selected_candidate = candidate
        selected_face_landmarks = face_landmarks
        face_index = detected_face_index
        break

    if selected_candidate is None or selected_face_landmarks is None:
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
            metadata={
                "roi_applied": roi is not None,
                "selected_face_index": None,
                "search_candidate": None,
                "search_attempts": ",".join(attempted_candidates),
            },
        )

    crop_height = selected_candidate.crop_bbox.height
    crop_width = selected_candidate.crop_bbox.width
    offset_x = selected_candidate.crop_bbox.x
    offset_y = selected_candidate.crop_bbox.y
    face_landmarks = selected_face_landmarks
    resolved_face_bbox = _landmark_bbox(face_landmarks, crop_width, crop_height, offset_x=offset_x, offset_y=offset_y)

    left_iris_center = _mean_point(face_landmarks, LEFT_IRIS_INDICES)
    right_iris_center = _mean_point(face_landmarks, RIGHT_IRIS_INDICES)
    nose_tip = np.array([face_landmarks[4].x, face_landmarks[4].y, face_landmarks[4].z], dtype=np.float64)
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
        metadata={
            "roi_applied": roi is not None,
            "selected_face_index": face_index,
            "search_candidate": selected_candidate.name,
            "search_attempts": ",".join(attempted_candidates),
            "overlay": _overlay_metadata(
                crop_width=crop_width,
                crop_height=crop_height,
                offset_x=offset_x,
                offset_y=offset_y,
                face_bbox=resolved_face_bbox,
                head_x_axis=head_x_axis,
                head_y_axis=head_y_axis,
                head_z_axis=head_z_axis,
                left_gaze=left_gaze,
                right_gaze=right_gaze,
                average_gaze=average_gaze,
                left_iris_center=left_iris_center,
                right_iris_center=right_iris_center,
                nose_tip=nose_tip,
            ),
        },
    )
