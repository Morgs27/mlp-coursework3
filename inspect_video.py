import cv2
import easyocr
import glob
import os
import sys

def inspect_video(video_path):
    print(f"Inspecting {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

def inspect_video(video_path):
    print(f"Inspecting {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Throw at 476.0s (from CSV)
    # Check 466 to 476 (10s window) to find the CUT
    target_time = 476.0
    start_time = target_time - 10.0
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    prev_hist = None
    
    print(f"\nScanning for cuts between {start_time}s and {target_time}s...")
    
    for i in range(int(fps * 10)):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = start_time + (i / fps)
        
        # Calculate Histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        if prev_hist is not None:
             # Compare
             val = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
             # Correlation < 0.5 usually implies scene change? Or < 0.8
             if val < 0.7:
                 print(f"[{current_time:.2f}s] Potential CUT detected (Corr: {val:.3f})")
                 # Check if face present?
                 pass
        
        prev_hist = hist
    
    cap.release()

if __name__ == "__main__":
    # Find the video file
    video_files = glob.glob("video/*.mp4")
    if not video_files:
        print("No video files found in video/ directory.")
        sys.exit(1)
    
    video_path = video_files[0]
    inspect_video(video_path)
