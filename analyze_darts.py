import cv2
import easyocr
import glob
import os
import pandas as pd
import numpy as np
import re

def get_video_path(video_dir="video"):
    files = glob.glob(f"{video_dir}/*.mp4")
    if files:
        return files[0]
    return None

def parse_score(text):
    # Keep only digits
    text = re.sub(r'\D', '', text)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None

def analyze_video(video_path):
    print(f"Analyzing {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    reader = easyocr.Reader(['en'], gpu=True) # Use GPU if available (MPS on mac? EasyOCR handles it via torch)

    # ROIs based on finding:
    # P1: x=1651, y=925, w=76, h=40
    # P2: x=1649, y=973, w=76, h=40
    # We add some padding
    p1_roi = (1640, 920, 100, 50)
    p2_roi = (1640, 970, 100, 50)

    data = []
    
    p1_score_prev = None
    p2_score_prev = None
    
    frame_step = int(fps * 0.5) # Check every 0.5 seconds
    
    frame_count = 0
    
    # Start loop from 5 minutes (300s) to skip intros
    start_frame = int(fps * 300) # Start 5 mins in
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    # State
    current_leg = 1
    p1_current_score = 501
    p2_current_score = 501
    
    # Store last N valid readings to smooth
    # But simple logic: if score drops by X <= 180 and holds for a few frames?
    # Actually, just taking the first valid change is risky.
    # We will log raw changes and clean in post-processing or add simple heuristic:
    # Score must be < previous score.
    # If score > previous, maybe leg reset?
    
    last_p1_val = None
    last_p2_val = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_step != 0:
            frame_count += 1
            continue

        # OCR with simple retry or just accept
        # P1
        x, y, w, h = p1_roi
        p1_img = frame[y:y+h, x:x+w]
        res_p1 = reader.readtext(p1_img, detail=0)
        val_p1 = parse_score(res_p1[0]) if res_p1 else None
        
        # P2
        x, y, w, h = p2_roi
        p2_img = frame[y:y+h, x:x+w]
        res_p2 = reader.readtext(p2_img, detail=0)
        val_p2 = parse_score(res_p2[0]) if res_p2 else None
        
        current_time = frame_count / fps

        # Process P1
        if val_p1 is not None and val_p1 <= 501:
            if last_p1_val is not None:
                # Detect Leg Reset
                if val_p1 == 501 and last_p1_val < 100:
                    print(f"[{current_time:.2f}s] P1 Leg Reset (Likely new leg)")
                    # Maybe increment leg if both reset?
                    # We'll rely on post-processing for Leg number, or simple heuristic here.
                    if p1_current_score < 100: # Only if we were finishing
                         current_leg += 1
                    p1_current_score = 501
                
                # Detect Score Drop
                elif val_p1 < p1_current_score:
                    diff = p1_current_score - val_p1
                    # Valid single dart is <= 60. Valid 3-dart is <= 180.
                    # If we see < 60 update, it's a throw.
                    # If we see 180 update time-to-time, it's a visit.
                    if 0 < diff <= 180:
                         print(f"[{current_time:.2f}s] L{current_leg} P1 Throw: {diff} (rem: {val_p1})")
                         data.append({
                             'Leg': current_leg,
                             'Time': current_time,
                             'Player': 'Player 1',
                             'ThrowScore': diff,
                             'RemainingScore': val_p1
                         })
                         p1_current_score = val_p1
            
            last_p1_val = val_p1
            # Correction: if OCR reads 501, 100, 501 (noise), we shouldn't reset.
            # But we rely on p1_current_score logic.
            # If val_p1 > p1_current_score and val_p1 != 501, it's likely noise or correction.
            # We ignore increases unless it's 501.

        # Process P2
        if val_p2 is not None and val_p2 <= 501:
            if last_p2_val is not None:
                if val_p2 == 501 and last_p2_val < 100:
                    if p2_current_score < 100:
                        # Only increment leg if we haven't already for this leg boundary
                        # This is tricky without unified state.
                        pass 
                    p2_current_score = 501
                elif val_p2 < p2_current_score:
                    diff = p2_current_score - val_p2
                    if 0 < diff <= 180:
                         print(f"[{current_time:.2f}s] L{current_leg} P2 Throw: {diff} (rem: {val_p2})")
                         data.append({
                             'Leg': current_leg,
                             'Time': current_time,
                             'Player': 'Player 2',
                             'ThrowScore': diff,
                             'RemainingScore': val_p2
                         })
                         p2_current_score = val_p2
            last_p2_val = val_p2

        if frame_count % (int(fps) * 60) == 0:
             print(f"Processed {current_time:.0f}s...")
        
        # Fast forward
        frames_to_skip = frame_step - 1
        if frames_to_skip > 0:
            for _ in range(frames_to_skip):
                cap.grab()
            frame_count += frames_to_skip

    cap.release()
    
    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv("darts_scores.csv", index=False)
    print("Saved darts_scores.csv")

if __name__ == "__main__":
    video_path = get_video_path()
    if video_path:
        analyze_video(video_path)
    else:
        print("No video found.")
