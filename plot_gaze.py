
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
        # plt.text(x, y, score, fontsize=9, ha='right', va='bottom', zorder=4) # Removed labels for individual points

    # Plot Average Blobs
    for score in scores:
        score_data = [d for d in player_data if d['score'] == score]
        if not score_data:
            continue
            
        avg_x = np.mean([d['x'] for d in score_data])
        avg_y = np.mean([d['y'] for d in score_data])
        color = score_color_map[score]
        
        plt.scatter(avg_x, avg_y, color=color, s=2000, alpha=0.2, marker='o', edgecolor='none', zorder=1)
        # Label the average blob - Move to LEFT
        plt.text(avg_x + 0.015, avg_y, f"{score}", fontsize=14, ha='right', va='center', color='black', fontweight='bold', zorder=5)

    # Legend
    from matplotlib.lines import Line2D
    score_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=s) for s, c in score_color_map.items()]
    view_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='k', markersize=12, label=v.upper()) for v, m in view_markers.items() if v != 'default' and v != 'mb']
    
    plt.legend(handles=score_handles + view_handles, title="Score & View", loc='best', fontsize='x-large', title_fontsize='x-large')

    player_name = "Anderson" if player_code == 'a' else "Van Veen" if player_code == 'v' else player_code
    plt.title(f"3D Gaze Vectors: {player_name}", fontsize=20)
    plt.xlabel("Horizontal Gaze (Normalized)", fontsize=16)
    plt.ylabel("Vertical Gaze (Normalized)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis() 
    
    # Set X axis limits to zoom in (User request: start from -0.15)
    plt.xlim(-0.15, -0.45)
    
    # plt.gca().invert_xaxis() # Already handled by xlim if we set it right? 
    # Actually if we set xlim(-0.1, -0.45), matplotlib puts -0.45 on left?
    # No, usually min is left.
    # If we want -0.1 on Left, we should set xlim(-0.1, -0.45) AND invert? 
    # Wait, previous code had invert_xaxis().
    # If I set xlim(-0.1, -0.45), that implies -0.1 is Min and -0.45 is Max? No.
    # Let's just keep invert_xaxis and set limits compatible with it.
    # Or just set limits in the order we want.
    # If I want -0.1 on Left and -0.45 on Right:
    # plt.xlim(-0.1, -0.45) should do it.
    
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
    plot_combined_averages(data)

def plot_combined_averages(data):
    plt.figure(figsize=(12, 10))
    
    # Normalize 'bool' to 'bull' in data
    for d in data:
        if d['score'] == 'bool':
            d['score'] = 'bull'
            
    # Unique scores across all data
    scores = sorted(list(set(d['score'] for d in data)))
    def sort_key(s):
        if s.isdigit(): return int(s)
        return 100 
    scores.sort(key=sort_key)
    
    # Colormap
    colors = cm.rainbow(np.linspace(0, 1, len(scores)))
    score_color_map = {score: color for score, color in zip(scores, colors)}
    
    # Player markers
    player_markers = {
        'a': 'o', # Anderson = Circle
        'v': 's', # Van Veen = Square
        'default': 'x'
    }
    
    players = sorted(list(set(d['player'] for d in data if d['player'])))
    
    for player in players:
        player_data = [d for d in data if d['player'] == player]
        if not player_data: 
            continue
            
        # Calculate Player Center (Mean of all throws)
        player_center_x = np.mean([d['x'] for d in player_data])
        player_center_y = np.mean([d['y'] for d in player_data])
        
        # Plot Center
        center_marker = 'P' if player == 'a' else 'X' # Plus for Anderson, X for Van Veen? Or just a common 'Star'?
        # Let's use a common distinctive marker for center, maybe just different colors or shapes again.
        # User said "make the 'center' have a distinct point"
        # Let's use a Black 'X' for both, or maybe match the player shape but black and hollow?
        # Let's try a bold Black Plus '+' for Anderson and Black Cross 'x' for Van Veen to distinguish centers if they are different.
        # Or simpler: A large Black Star for both? But their centers are different (relative to origin, though this plot is relative to center? No, it's relative to center logic but plotted in absolute coordinates).
        # Actually the plot title says "Relative to Center" but we are plotting absolute coordinates and drawing lines FROM center.
        
        center_shape = 'P' if player == 'a' else 'X'
        plt.scatter(player_center_x, player_center_y, color='black', marker=center_shape, s=200, label=f"Center ({player})", zorder=6)
        
        # For this player, find average for each score
        player_scores = sorted(list(set(d['score'] for d in player_data)))
        
        for score in player_scores:
            score_data = [d for d in player_data if d['score'] == score]
            if not score_data:
                continue
                
            avg_x = np.mean([d['x'] for d in score_data])
            avg_y = np.mean([d['y'] for d in score_data])
            
            color = score_color_map.get(score, 'gray')
            marker = player_markers.get(player, 'x')
            
            # Draw dotted line from center to this score average
            # User request: make the dashed lines more visible (wider)
            plt.plot([player_center_x, avg_x], [player_center_y, avg_y], color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=2)
            
            plt.scatter(avg_x, avg_y, color=color, marker=marker, s=300, alpha=0.9, edgecolor='k', zorder=3)
            
            # Text label calculation (offset slightly)
            # Direction from center
            dx = avg_x - player_center_x
            dy = avg_y - player_center_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Distance to move text away from marker center
            # Marker size 300 is roughly radius 10 points?
            # In data coordinates, this depends on the scale.
            # Let's try a fixed offset based on direction.
            offset_scale = 0.005 # Increased offset
            
            if dist > 0:
                off_x = ((dx / dist) * offset_scale) + 0.02
                off_y = ((dy / dist) * offset_scale) + 0.0
            else:
                # If exactly at center (unlikely), offsets 0
                off_x, off_y = 0.01, 0.01
                
            # Better label placement: Put text inside marker if it fits, or just next to it
            # Text color contrast?
            # Let's just put it slightly offset
            # User request: Move average labels to the left of the indicators
            plt.text(avg_x + 0.01, avg_y, f"{score}", fontsize=14, ha='right', va='center', color='black', fontweight='bold', zorder=5)


    # Legend construction
    from matplotlib.lines import Line2D
    # Score Legend (Colors)
    score_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=s) for s, c in score_color_map.items()]
    
    # Player Legend (Markers)
    player_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='Anderson'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, label='Van Veen'),
        Line2D([0], [0], marker='P', color='black', markerfacecolor='black', markersize=12, linestyle='None', label='Center (Anderson)'),
        Line2D([0], [0], marker='X', color='black', markerfacecolor='black', markersize=12, linestyle='None', label='Center (Van Veen)')
    ]
    
    plt.legend(handles=score_handles + player_handles, title="Score & Player", loc='best', fontsize='x-large', title_fontsize='x-large', ncol=2)

    plt.title("Combined Average Gaze Vectors: Anderson vs Van Veen (Relative to Center)", fontsize=20)
    plt.xlabel("Horizontal Gaze (Normalized)", fontsize=16)
    plt.ylabel("Vertical Gaze (Normalized)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Invert Y axis so Negative (Up) is Top
    plt.gca().invert_yaxis() 
    
    # Set X axis limits to zoom in (User request: start from -0.15)
    plt.xlim(-0.15, -0.45)
    
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(INPUT_DIR), "gaze_plot_combined.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
