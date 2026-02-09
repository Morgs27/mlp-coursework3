
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from gaze_tracking import process_image, INPUT_DIR

def parse_score(filename):
    # Expected format: sv_a_{score}-{index}.png
    try:
        parts = filename.split('_')
        score_part = parts[2] # 19-1.png
        score = int(score_part.split('-')[0])
        return score
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return None

def main():
    image_files = glob.glob(os.path.join(INPUT_DIR, "sv_a_*.png"))
    
    data_19 = []
    data_20 = []
    
    print(f"Found {len(image_files)} images for analysis.")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        score = parse_score(filename)
        
        if score not in [19, 20]:
            continue
            
        print(f"Processing {filename} (Score: {score})...")
        result = process_image(img_path, save_output=False)
        
        if result is None:
            continue
            
        left_gaze = result['left_gaze']
        right_gaze = result['right_gaze']
        
        # Average gaze
        avg_gaze_x = (left_gaze[0] + right_gaze[0]) / 2
        avg_gaze_y = (left_gaze[1] + right_gaze[1]) / 2
        
        if score == 19:
            data_19.append((avg_gaze_x, avg_gaze_y))
        elif score == 20:
            data_20.append((avg_gaze_x, avg_gaze_y))

    # Plotting
    plt.figure(figsize=(10, 8))
    
    if data_19:
        x19, y19 = zip(*data_19)
        plt.scatter(x19, y19, c='blue', label='Score 19', s=100, alpha=0.7)
        
    if data_20:
        x20, y20 = zip(*data_20)
        plt.scatter(x20, y20, c='red', label='Score 20', s=100, alpha=0.7)
        
    plt.title("Gaze Vectors: Score 19 vs 20")
    plt.xlabel("Gaze X (relative)")
    plt.ylabel("Gaze Y (relative)")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(INPUT_DIR), "gaze_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
