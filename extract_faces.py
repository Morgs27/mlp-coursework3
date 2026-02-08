import pandas as pd
import cv2
import os
import numpy as np

def extract_face_clips(csv_path, video_path, output_dir="clips"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Group by Leg, Player to identify Visits
    # A visit is a sequence of throws by one player.
    # Throws are rows in the CSV.
    # Group consecutive rows for same player within same leg?
    # Simple logic: Iterate rows. If Player changes or Time gap > 15s (unlikely in throw), New Visit.
    
    visits = []
    current_visit = []
    
    for index, row in df.iterrows():
        if not current_visit:
            current_visit.append(row)
        else:
            last = current_visit[-1]
            if row['Leg'] == last['Leg'] and row['Player'] == last['Player'] and (row['Time'] - last['Time'] < 20):
                current_visit.append(row)
            else:
                visits.append(current_visit)
                current_visit = [row]
    if current_visit:
        visits.append(current_visit)

    print(f"Identified {len(visits)} visits.")
    
    # Process each visit
    for i, visit in enumerate(visits):
        player = visit[0]['Player']
        leg = visit[0]['Leg']
        
        # Determine time window
        # End time: Time of the LAST score update (or first? Usually face is shown *before* throws start)
        # Actually, face is shown while aiming -> Throw 1 -> Score 1 -> ...
        # So we want the time leading up to the FIRST score update of the visit.
        
        first_score_time = visit[0]['Time']
        
        # Scan window: [T-12s, T-2s]
        # Why 12? Give enough buffer.
        start_scan = max(0, first_score_time - 12.0)
        end_scan = max(0, first_score_time - 2.0)
        
        # Finding the face
        cap.set(cv2.CAP_PROP_POS_MSEC, start_scan * 1000)
        
        frames_to_check = int((end_scan - start_scan) * fps)
        
        face_frames = [] # (frame_idx, frame, score)
        
        # Optimize: Check every 5th frame for face?
        step = 5
        
        best_face_score = 0
        best_face_time = -1
        
        # We want a continuous clip.
        # Strategy: Find the "best" face frame (largest face), then take +/- 2 seconds around it?
        # Or just take the whole window if face is present?
        
        # Let's read the window frames into a buffer first (memory intensive? 10s * 25fps = 250 frames. Fine.)
        window_frames = []
        
        start_frame_idx = int(start_scan * fps)
        
        # Read frames
        for f in range(frames_to_check):
            ret, frame = cap.read()
            if not ret:
                break
            window_frames.append(frame)
        
        if not window_frames:
            continue

        # Detect faces in sparse frames to find "center of attention"
        # We prioritize LARGE faces (taking up significant frame height).
        # Normal crowd faces are small.
        # Player face is usually > 15-20% of height?
        
        scores = []
        
        for idx in range(0, len(window_frames), step):
            frame = window_frames[idx]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            max_size = 0
            for (x, y, w, h) in faces:
                if h > max_size:
                    max_size = h
            
            # Simple score: max face height
            # Filter minimal size to ignore crowd
            if max_size > (height * 0.15): # Face must be > 15% of screen height
                scores.append((idx, max_size))
        
        if scores:
            # Find the time with the best/largest face
            scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = scores[0][0]
            
            # Clip Window: Center around best face, say 3 seconds total?
            # Or from best face - 1.5s to best face + 1.5s
            
            clip_center_idx = best_idx
            clip_len_frames = int(fps * 3.0) # 3 second clip
            
            clip_start_local = max(0, clip_center_idx - clip_len_frames // 2)
            clip_end_local = min(len(window_frames), clip_start_local + clip_len_frames)
            
            final_clip_frames = window_frames[clip_start_local:clip_end_local]
            
            if len(final_clip_frames) > 0:
                clip_filename = f"{output_dir}/L{leg}_{player.replace(' ', '')}_Visit{i}.mp4"
                print(f"Saving clip {clip_filename} (Face size: {scores[0][1]})")
                
                # Write video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
                for frame in final_clip_frames:
                    out.write(frame)
                out.release()
        
        # Limit for testing? No, run full but maybe print progress
        if i % 10 == 0:
            print(f"Processed {i}/{len(visits)} visits...")

    cap.release()
    print("Done.")

if __name__ == "__main__":
    extract_face_clips("darts_scores.csv", "video/Finales van de NK Darts 2025 @ Driebergen.mp4")
