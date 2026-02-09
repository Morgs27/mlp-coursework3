
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

def get_3d_gaze_vector(face_landmarks, eye_indices, iris_indices, transform_matrix=None):
    # 1. Calculate Face Orientation Frame
    # Define landmarks for Head Frame
    # Lateral: 454 (Left Ear tragus) - 234 (Right Ear tragus)
    # Vertical: 10 (Top Hairline/Forehead) - 152 (Chin)
    
    p_left = np.array([face_landmarks[454].x, face_landmarks[454].y, face_landmarks[454].z])
    p_right = np.array([face_landmarks[234].x, face_landmarks[234].y, face_landmarks[234].z])
    p_top = np.array([face_landmarks[10].x, face_landmarks[10].y, face_landmarks[10].z])
    p_bottom = np.array([face_landmarks[152].x, face_landmarks[152].y, face_landmarks[152].z])
    
    # Head X Axis (Right)
    head_x = p_right - p_left
    head_x /= np.linalg.norm(head_x)
    
    # Head Y Axis (Up - relative to face) roughly
    # Actually, let's use the cross product to be sure.
    # Vector top->bottom is roughly local Down (-Y) or Up?
    # Landmarks: 10 is top, 152 is bottom. Vector 10->152 is DOWN.
    # Let's define Head_Y as UP. So 152->10.
    head_y_approx = p_top - p_bottom
    head_y_approx /= np.linalg.norm(head_y_approx)
    
    # Head Forward (Z) = Cross(X, Y)
    # Right x Up = Forward (Standard RHS?)
    # Right (x) cross Up (y) = Out of Face (z) ?
    # Let's check: x=(1,0,0), y=(0,1,0) -> z=(0,0,1)
    head_forward = np.cross(head_x, head_y_approx)
    head_forward /= np.linalg.norm(head_forward)
    
    # Re-orthogonalize Y
    head_y = np.cross(head_forward, head_x)
    head_y /= np.linalg.norm(head_y)
    
    # 2. Estimate Scale (IPD)
    # Left Eye Center: 468 (Left Iris)
    # Right Eye Center: 473 (Right Iris)
    # Or use corners mean
    
    # Calculate Mean Eye Centers (Corners)
    def get_mean_point(indices):
        xs = [face_landmarks[i].x for i in indices]
        ys = [face_landmarks[i].y for i in indices]
        zs = [face_landmarks[i].z for i in indices]
        return np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
    
    left_enc_surf = get_mean_point(eye_indices) # Surface center
    
    # Calculate IPD using Iris centers for scale
    p_left_iris = np.array([face_landmarks[468].x, face_landmarks[468].y, face_landmarks[468].z])
    p_right_iris = np.array([face_landmarks[473].x, face_landmarks[473].y, face_landmarks[473].z])
    ipd = np.linalg.norm(p_left_iris - p_right_iris)
    
    # ... (inside get_3d_gaze_vector)
    # 3. Estimate True Eyeball Center
    # Shift backwards by K * IPD
    # Standard anatomical ratio: Eye center depth / IPD approx 13mm / 64mm ~= 0.2
    # But since landmarks are surface, we might need slightly more?
    # Let's try 0.25 to start
    K_depth = 0.3 # Moving eye center back
    center_offset = head_forward * (ipd * K_depth) # Use positive head_forward to go deeper (IN)
    
    # We apply this offset to the surface eye center
    true_eye_center = left_enc_surf + center_offset
        
    # 4. Gaze Vector
    # Current Iris position
    # Iris landmarks are on the surface of the cornea (bulging out)
    # So Iris - True Center is the optical axis
    
    # Iris Center
    iris_center = get_mean_point(iris_indices)
    
    gaze_vector = iris_center - true_eye_center
    
    gaze_vector /= np.linalg.norm(gaze_vector)
    
    return gaze_vector, true_eye_center, iris_center

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

    # Eye indices (Corners)
    LEFT_EYE_INDICES = [33, 133]
    RIGHT_EYE_INDICES = [362, 263]
    
    # Iris indices
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    left_gaze, left_eye_center, left_iris_center = get_3d_gaze_vector(face_landmarks, LEFT_EYE_INDICES, LEFT_IRIS)
    right_gaze, right_eye_center, right_iris_center = get_3d_gaze_vector(face_landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS)
    
    if save_output:
        # Draw visualization
        for idx in LEFT_IRIS + RIGHT_IRIS:
            point = face_landmarks[idx]
            cv2.circle(image, get_pixel_coords(point, image.shape), 1, (0, 255, 0), -1, cv2.LINE_AA)

        def draw_gaze_3d(gaze_vec, iris_center_3d):
             # We project the start and end points of the gaze vector back to 2D
             # Start point is the iris center
             start_point_3d = iris_center_3d
             # End point is scaled along the gaze vector
             scale = 0.5 # Arbitrary scale for visualization length
             end_point_3d = start_point_3d + gaze_vec * scale
             
             # Simple orthographic projection (ignoring depth perspective for drawing on 2D image)
             # Since landmarks are normalized [0, 1], we map to pixel coords
             
             start_2d = (int(start_point_3d[0] * w), int(start_point_3d[1] * h))
             
             # For the end point, we need to be careful. The Z coordinate in MediaPipe is relative to the image plane? 
             # MediaPipe Z is "depth", where the origin is at the center of the head approx?
             # For visualization "lazers", we mainly care about x and y direction in the image plane
             # But the 3D vector allows us to see "into" the image.
             # Let's simple project the 3D end point to 2D x,y
             
             end_2d = (int(end_point_3d[0] * w), int(end_point_3d[1] * h))
             
             cv2.arrowedLine(image, start_2d, end_2d, (0, 0, 255), 2, cv2.LINE_AA)

        draw_gaze_3d(left_gaze, left_iris_center)
        draw_gaze_3d(right_gaze, right_iris_center)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, image)
        
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
