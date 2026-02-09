
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import glob

# Constants
INPUT_DIR = '/Users/morgandaniel/Documents/mlp/mlp-coursework3/debug-images'
OUTPUT_DIR = '/Users/morgandaniel/Documents/mlp/mlp-coursework3/annotated-images'
MODEL_PATH = '/Users/morgandaniel/Documents/mlp/mlp-coursework3/face_landmarker.task'

# Initialize Face Landmarker (lazy loaded or global)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector

def get_pixel_coords(landmark, image_shape):
    return int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])

def get_gaze_vector(face_landmarks, eye_indices, iris_indices, image_shape):
    p1 = face_landmarks[eye_indices[0]] # Corner 1
    p2 = face_landmarks[eye_indices[1]] # Corner 2
    
    eye_center_x = (p1.x + p2.x) / 2
    eye_center_y = (p1.y + p2.y) / 2
    
    iris_center = face_landmarks[iris_indices[0]]
    
    # Gaze vector relative to eye width/height to be somewhat invariant to distance?
    # For now, just raw relative coordinates
    dx = iris_center.x - eye_center_x
    dy = iris_center.y - eye_center_y
    
    return (dx, dy)

def process_image(image_path, filename=None, save_output=True):
    if filename is None:
        filename = os.path.basename(image_path)
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return None

    detector = get_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print(f"No face detected in {filename}")
        return None

    face_landmarks = detection_result.face_landmarks[0]
    h, w, _ = image.shape

    # Eye indices
    LEFT_EYE_INDICES = [33, 133]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_EYE_INDICES = [362, 263]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    left_gaze = get_gaze_vector(face_landmarks, LEFT_EYE_INDICES, LEFT_IRIS, image.shape)
    right_gaze = get_gaze_vector(face_landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS, image.shape)
    
    if save_output:
        # Draw visualization
        # ... (drawing logic similar to before, can be kept or simplified)
        # Re-implement drawing for saving output
        for idx in LEFT_IRIS + RIGHT_IRIS:
            point = face_landmarks[idx]
            cv2.circle(image, get_pixel_coords(point, image.shape), 2, (0, 255, 0), -1, cv2.LINE_AA)

        def draw_gaze(gaze, eye_indices, iris_indices):
            iris_center = face_landmarks[iris_indices[0]]
            start_point = get_pixel_coords(iris_center, image.shape)
            # Scale vector for visualization
            end_point = (int(start_point[0] + gaze[0] * w * 10), int(start_point[1] + gaze[1] * h * 10))
            cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA)

        draw_gaze(left_gaze, LEFT_EYE_INDICES, LEFT_IRIS)
        draw_gaze(right_gaze, RIGHT_EYE_INDICES, RIGHT_IRIS)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, image)
        
        # Zoom crop logic...
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for point in face_landmarks:
            x, y = get_pixel_coords(point, image.shape)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
        
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        cropped_face = image[y_min:y_max, x_min:x_max]
        crop_output_path = os.path.join(OUTPUT_DIR, f"crop_{filename}")
        if cropped_face.size > 0:
            cv2.imwrite(crop_output_path, cropped_face)

    return {
        'left_gaze': left_gaze,
        'right_gaze': right_gaze,
        'filename': filename
    }

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        exit(1)
        
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    print(f"Found {len(image_files)} images.")
    for img_path in image_files:
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
             process_image(img_path)
