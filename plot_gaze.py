
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
        
        # Average gaze vector (3D)
        avg_gaze = (left_gaze + right_gaze) / 2
        
        # In MediaPipe:
        # +y is down (image coordinates)
        # +x is right
        # +z is into the screen (depth)
        
        # If looking UP (at 20), y component should be smaller (more negative or less positive) 
        # relative to looking DOWN (at 19).
        # Actually, since it's a direction vector:
        # Looking UP -> Y component is negative (pointing to top of image)
        # Looking DOWN -> Y component is positive (pointing to bottom of image)
        
        # We expect Score 20 (Top) -> Lower Y value (more negative)
        # We expect Score 19 (Bottom) -> Higher Y value (more positive)
        
        vertical_gaze = avg_gaze[1] 
        horizontal_gaze = avg_gaze[0]
        
        if score == 19:
            data_19.append((horizontal_gaze, vertical_gaze))
        elif score == 20:
            data_20.append((horizontal_gaze, vertical_gaze))

    # Plotting
    plt.figure(figsize=(10, 8))
    
    if data_19:
        x19, y19 = zip(*data_19)
        plt.scatter(x19, y19, c='blue', label='Score 19 (Bottom)', s=100, alpha=0.7)
        
    if data_20:
        x20, y20 = zip(*data_20)
        plt.scatter(x20, y20, c='red', label='Score 20 (Top)', s=100, alpha=0.7)
        
    plt.title("3D Gaze Vectors (Projected): Score 19 vs 20")
    plt.xlabel("Horizontal Gaze (Normalized)")
    plt.ylabel("Vertical Gaze (Normalized)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(INPUT_DIR), "gaze_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
