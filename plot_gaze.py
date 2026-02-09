
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import numpy as np
from gaze_tracking import process_image, INPUT_DIR

def parse_filename(filename):
    # Formats: 
    # sv_a_20-1.png (Anderson)
    # mv_v_13-1.png (Van Veen)
    try:
        parts = filename.split('_')
        view = parts[0].lower() # sv, mv, mb
        player = parts[1] # 'a', 'v'
        score_part = parts[2]
        score_label = score_part.split('-')[0] # 20, 19, bull
        return view, player, score_label
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return None, None, None

def plot_for_player(player_code, data, output_suffix):
    player_data = [d for d in data if d['player'] == player_code]
    if not player_data:
        print(f"No data for player {player_code}")
        return

    plt.figure(figsize=(12, 10))
    
    # Unique scores
    scores = sorted(list(set(d['score'] for d in player_data)))
    
    def sort_key(s):
        if s.isdigit(): return int(s)
        return 100 
    scores.sort(key=sort_key)
    
    # Colormap
    colors = cm.rainbow(np.linspace(0, 1, len(scores)))
    score_color_map = {score: color for score, color in zip(scores, colors)}
    
    view_markers = {
        'sv': 'o', 
        'mv': '^', 
        'mb': 's', 
        'default': 'x'
    }
    
    # Plot points
    for d in player_data:
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
        score_data = [d for d in player_data if d['score'] == score]
        if not score_data:
            continue
            
        avg_x = np.mean([d['x'] for d in score_data])
        avg_y = np.mean([d['y'] for d in score_data])
        color = score_color_map[score]
        
        plt.scatter(avg_x, avg_y, color=color, s=2000, alpha=0.2, marker='o', edgecolor='none', zorder=1)

    # Legend
    from matplotlib.lines import Line2D
    score_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=s) for s, c in score_color_map.items()]
    view_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='k', markersize=10, label=v.upper()) for v, m in view_markers.items() if v != 'default']
    
    plt.legend(handles=score_handles + view_handles, title="Score & View", loc='best')

    player_name = "Anderson" if player_code == 'a' else "Van Veen" if player_code == 'v' else player_code
    plt.title(f"3D Gaze Vectors: {player_name}")
    plt.xlabel("Horizontal Gaze (Normalized)")
    plt.ylabel("Vertical Gaze (Normalized)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis() 
    plt.gca().invert_xaxis() 
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(INPUT_DIR), f"gaze_plot_{output_suffix}.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    image_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    
    data = [] 
    
    print(f"Found {len(image_files)} images for analysis.")
    
    for img_path in image_files:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        filename = os.path.basename(img_path)
        view, player, score = parse_filename(filename)
        
        if view is None:
            continue
            
        # print(f"Processing {filename}...") 
        result = process_image(img_path, save_output=False)
        
        if result is None:
            continue
            
        left_gaze = result['left_gaze']
        right_gaze = result['right_gaze']
        
        avg_gaze = (left_gaze + right_gaze) / 2
        
        vertical_gaze = avg_gaze[1] 
        horizontal_gaze = avg_gaze[0]
        
        data.append({
            'view': view,
            'player': player,
            'score': score,
            'x': horizontal_gaze,
            'y': vertical_gaze
        })

    # Generate plots for known players
    plot_for_player('a', data, 'anderson')
    plot_for_player('v', data, 'van_veen')

if __name__ == "__main__":
    main()
