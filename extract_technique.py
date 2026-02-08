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
        
        # Time of the first score update (dart has already hit)
        first_score_time = visit[0]['Time']
        
        # We look for the cut to the board, which happens just before the dart hits.
        # Scan window: [T-10s, T-0s]
        start_scan = max(0, first_score_time - 10.0)
        end_scan = first_score_time
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_scan * 1000)
        
        frames_to_check = int((end_scan - start_scan) * fps)
        
        prev_hist = None
        cut_frame_idx = -1
        
        # Buffer frames to save clip later
        window_frames = []
        
        # Scan for cuts
        for j in range(frames_to_check):
            ret, frame = cap.read()
            if not ret:
                break
            
            window_frames.append(frame)
            
            # Check for cut every frame or skip? Every frame for precision.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                 val = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                 # Lower threshold for distinct cuts
                 if val < 0.6: 
                     # Found a cut at frame j
                     # We want the LAST cut before the score update (Player -> Board)
                     cut_frame_idx = j
            
            prev_hist = hist

        if cut_frame_idx != -1:
            # We found the cut. The throw happens BEFORE this cut.
            # Clip end = cut_frame_idx
            # Clip start = cut_frame_idx - 6s
            
            clip_len_frames = int(fps * 6.0)
            
            start_idx = max(0, cut_frame_idx - clip_len_frames)
            end_idx = cut_frame_idx # Include the cut? Maybe 1-2 frames after to show transition
            
            final_clip_frames = window_frames[start_idx:end_idx]
            
            if len(final_clip_frames) > fps * 2: # Min duration 2s
                clip_filename = f"{output_dir}/L{leg}_{player.replace(' ', '')}_Technique_Visit{i}.mp4"
                print(f"Saving clip {clip_filename} (Cut at -{(len(window_frames)-cut_frame_idx)/fps:.2f}s before score)")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
                for frame in final_clip_frames:
                    out.write(frame)
                out.release()
            else:
                print(f"Skipping Visit {i}: Clip too short")
        else:
             print(f"Skipping Visit {i}: No cut detected")
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(visits)} visits...")

    cap.release()
    print("Done.")

if __name__ == "__main__":
    extract_face_clips("darts_scores.csv", "video/Finales van de NK Darts 2025 @ Driebergen.mp4")
