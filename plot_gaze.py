
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import numpy as np
from gaze_tracking import process_image, INPUT_DIR

def parse_filename(filename):
    # Formats: 
    # sv_a_20-1.png
    # mv_a_10-1.png
    # mb_a_bull-1.png
    try:
        parts = filename.split('_')
        view = parts[0].lower() # sv, mv, mb
        # player = parts[1] # 'a'
        score_part = parts[2]
        score_label = score_part.split('-')[0] # 20, 19, bull
        return view, score_label
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return None, None

def main():
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    
    data = [] # List of (view, score, gaze_x, gaze_y)
    
    print(f"Found {len(image_files)} images for analysis.")
    
    for img_path in image_files:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        filename = os.path.basename(img_path)
        view, score = parse_filename(filename)
        
        if view is None:
            continue
            
        print(f"Processing {filename} (View: {view}, Score: {score})...")
        result = process_image(img_path, save_output=False)
        
        if result is None:
            continue
            
        left_gaze = result['left_gaze']
        right_gaze = result['right_gaze']
        
        # Average gaze vector (3D)
        avg_gaze = (left_gaze + right_gaze) / 2
        
        vertical_gaze = avg_gaze[1] 
        horizontal_gaze = avg_gaze[0]
        
        data.append({
            'view': view,
            'score': score,
            'x': horizontal_gaze,
            'y': vertical_gaze
        })

    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Unique scores for coloring
    scores = sorted(list(set(d['score'] for d in data)))
    
    # Custom sort to put numbers in order and bull at end?
    def sort_key(s):
        if s.isdigit(): return int(s)
        return 100 # Put Bull/others at end
    scores.sort(key=sort_key)
    
    # Colormap
    colors = cm.rainbow(np.linspace(0, 1, len(scores)))
    score_color_map = {score: color for score, color in zip(scores, colors)}
    
    # Markers for views
    view_markers = {
        'sv': 'o', # Circle
        'mv': '^', # Triangle Up
        'mb': 's', # Square
        'default': 'x'
    }
    
    # Plot points
    for d in data:
        view = d['view']
        score = d['score']
        x = d['x']
        y = d['y']
        
        marker = view_markers.get(view, 'x')
        color = score_color_map[score]
        
        plt.scatter(x, y, color=color, marker=marker, s=100, alpha=0.8, edgecolor='k', zorder=3)
        plt.text(x, y, score, fontsize=9, ha='right', va='bottom', zorder=4)

    # Plot Average Blobs
    for score in scores:
        score_data = [d for d in data if d['score'] == score]
        if not score_data:
            continue
            
        avg_x = np.mean([d['x'] for d in score_data])
        avg_y = np.mean([d['y'] for d in score_data])
        color = score_color_map[score]
        
        # Plot "blob"
        plt.scatter(avg_x, avg_y, color=color, s=2000, alpha=0.2, marker='o', edgecolor='none', zorder=1)

    # Create Legend manually
    # Score Legend
    from matplotlib.lines import Line2D
    score_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=s) for s, c in score_color_map.items()]
    view_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='k', markersize=10, label=v.upper()) for v, m in view_markers.items() if v != 'default']
    
    # Combined legend
    plt.legend(handles=score_handles + view_handles, title="Score & View", loc='best')

    plt.title("3D Gaze Vectors: All Scores & Views")
    plt.xlabel("Horizontal Gaze (Normalized)")
    plt.ylabel("Vertical Gaze (Normalized)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis() # Invert Y axis so Negative (Up) is Top
    plt.gca().invert_xaxis() # Invert X axis so Positive (Image Right/Player Left) becomes Left on plot?
    # Logic: Player looks Left -> Eyes move Image Right (+X) -> We want this to be Left on Plot (-X direction).
    # So we need to Invert X axis? 
    # Normal Plot: -X is Left, +X is Right.
    # Data: +X (Look Left).
    # We want +X data to be on Left.
    # So we invert X axis: Positive is Left, Negative is Right.
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(INPUT_DIR), "gaze_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
