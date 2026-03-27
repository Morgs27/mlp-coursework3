import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

# Set British English and scientific plotting style
plt.style.use('seaborn-v0_8-paper')
plt_sns.set_theme(style="whitegrid", rc={"axes.labelsize": 12, "axes.titlesize": 14})

INPUT_FILE = "../data/processed/enriched_samples.csv"
OUTPUT_DIR = "."

def plot_1_all_shots_and_games(df):
    """
    Plot 1: Scatter graph of horizontal vs vertical gaze normalized for each shot, 
    colour coded by score. Include area average point. Flipped horizontal axis.
    """
    # Create output subdirectories
    os.makedirs(os.path.join(OUTPUT_DIR, "plot1_overall"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plot1_games"), exist_ok=True)
    
    # Get distinct players
    players = df['player_name'].unique()
    
    for player in players:
        player_df = df[df['player_name'] == player]
        if len(player_df) == 0:
            continue
            
        # Get top 5 most frequent scores for this player
        top_scores = player_df['segment_number'].value_counts().nlargest(5).index
        plot_df = player_df[player_df['segment_number'].isin(top_scores)]
        
        # Get colour palette for these specific scores
        unique_scores = sorted(plot_df['segment_number'].unique())
        palette = plt_sns.color_palette("husl", max(len(unique_scores), 1))
        score_colours = dict(zip(unique_scores, palette))
        
        # 1A. Overall plot for player
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Plot individual shots
        plt_sns.scatterplot(
            data=plot_df, 
            x='average_gaze_x', 
            y='average_gaze_y', 
            hue='segment_number', 
            palette=score_colours, 
            s=20, # points aren't too big
            alpha=0.6,
            ax=ax,
            legend='full'
        )
        
        # Add area average point for each score
        score_means = plot_df.groupby('segment_number')[['average_gaze_x', 'average_gaze_y']].mean().reset_index()
        
        # Draw center point and lines
        center_x = player_df['average_gaze_x'].mean()
        center_y = player_df['average_gaze_y'].mean()
        ax.scatter(center_x, center_y, marker='P', s=200, color='black', edgecolor='white', linewidth=1.5, label='_nolegend_', zorder=5)
        
        for _, row in score_means.iterrows():
            ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax.scatter(row['average_gaze_x'], row['average_gaze_y'], 
                       marker='X', s=150, edgecolor='black', linewidth=1.5,
                       color=score_colours[row['segment_number']], 
                       label='_nolegend_', zorder=6) # Avoid duplicate legends
        
        ax.invert_xaxis() # Flip horizontal axis
        ax.invert_yaxis() # Flip vertical axis
        ax.set_title(f"Horizontal vs Vertical Gaze (All Shots) - {player}")
        ax.set_xlabel("Horizontal Gaze (Inverted)")
        ax.set_ylabel("Vertical Gaze")
        # Fix legend title to British English
        # ax.legend(title="Score (Colour Code)", bbox_to_anchor=(1.05, 1), loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles, labels, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"plot1_overall/gaze_all_shots_{player.replace(' ', '_')}.png"), dpi=300)
        plt.close()
        
        # 1B. Individual games plot for player
        game_ids = player_df['sport_event_id'].unique()
        for game_id in game_ids:
            game_df = player_df[player_df['sport_event_id'] == game_id]
            if len(game_df) == 0:
                continue
                
            # Get top 5 most frequent scores for this game
            game_top_scores = game_df['segment_number'].value_counts().nlargest(5).index
            game_plot_df = game_df[game_df['segment_number'].isin(game_top_scores)]
            
            # Get colour palette for this game's top scores
            game_unique_scores = sorted(game_plot_df['segment_number'].unique())
            game_palette = plt_sns.color_palette("husl", max(len(game_unique_scores), 1))
            game_score_colours = dict(zip(game_unique_scores, game_palette))
                
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            
            plt_sns.scatterplot(
                data=game_plot_df, 
                x='average_gaze_x', 
                y='average_gaze_y', 
                hue='segment_number', 
                palette=game_score_colours, 
                s=20, 
                alpha=0.6,
                ax=ax,
                legend='full'
            )
            
            # Area average points for game
            game_score_means = game_plot_df.groupby('segment_number')[['average_gaze_x', 'average_gaze_y']].mean().reset_index()
            
            center_x = game_df['average_gaze_x'].mean()
            center_y = game_df['average_gaze_y'].mean()
            ax.scatter(center_x, center_y, marker='P', s=200, color='black', edgecolor='white', linewidth=1.5, label='_nolegend_', zorder=5)
            
            for _, row in game_score_means.iterrows():
                if pd.notna(row['average_gaze_x']) and pd.notna(row['average_gaze_y']):
                    ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle='--', alpha=0.5, zorder=1)
                    ax.scatter(row['average_gaze_x'], row['average_gaze_y'], 
                               marker='X', s=150, edgecolor='black', linewidth=1.5,
                               color=game_score_colours[row['segment_number']],
                               label='_nolegend_', zorder=6)
            
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_title(f"Horizontal vs Vertical Gaze (Game {game_id}) - {player}")
            ax.set_xlabel("Horizontal Gaze (Inverted)")
            ax.set_ylabel("Vertical Gaze")
            
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(handles, labels, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"plot1_games/gaze_game_{game_id}_{player.replace(' ', '_')}.png"), dpi=300)
            plt.close()

def plot_2_average_per_score(df):
    """
    Plot 2: Same number of plots as Plot 1, but only plot the average for each score.
    """
    os.makedirs(os.path.join(OUTPUT_DIR, "plot2_overall"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plot2_games"), exist_ok=True)
    
    players = df['player_name'].unique()
    for player in players:
        player_df = df[df['player_name'] == player]
        if len(player_df) == 0:
            continue
            
        # Get top 5 most frequent scores for this player
        top_scores = player_df['segment_number'].value_counts().nlargest(5).index
        plot_df = player_df[player_df['segment_number'].isin(top_scores)]
        
        # Get colour palette for these specific scores
        unique_scores = sorted(plot_df['segment_number'].unique())
        palette = plt_sns.color_palette("husl", max(len(unique_scores), 1))
        score_colours = dict(zip(unique_scores, palette))
            
        # 2A. Overall average plot
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        score_means = plot_df.groupby('segment_number', as_index=False)[['average_gaze_x', 'average_gaze_y']].mean()
        # Drop nans
        score_means = score_means.dropna(subset=['average_gaze_x', 'average_gaze_y'])
        
        center_x = player_df['average_gaze_x'].mean()
        center_y = player_df['average_gaze_y'].mean()
        ax.scatter(center_x, center_y, marker='P', s=200, color='black', edgecolor='white', linewidth=1.5, label='_nolegend_', zorder=5)
        
        for _, row in score_means.iterrows():
            ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        plt_sns.scatterplot(
            data=score_means, 
            x='average_gaze_x', 
            y='average_gaze_y', 
            hue='segment_number', 
            palette=score_colours, 
            s=200, # Large points for averages
            marker='o',
            edgecolor='black',
            linewidth=1.5,
            ax=ax,
            legend='full',
            zorder=6
        )
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title(f"Average Horizontal vs Vertical Gaze per Score (All Shots) - {player}")
        ax.set_xlabel("Horizontal Gaze (Inverted)")
        ax.set_ylabel("Vertical Gaze")
        
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles, labels, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"plot2_overall/avg_gaze_all_shots_{player.replace(' ', '_')}.png"), dpi=300)
        plt.close()
        
        # 2B. Individual games average plot
        game_ids = player_df['sport_event_id'].unique()
        for game_id in game_ids:
            game_df = player_df[player_df['sport_event_id'] == game_id]
            if len(game_df) == 0:
                continue
                
            # Get top 5 most frequent scores for this game
            game_top_scores = game_df['segment_number'].value_counts().nlargest(5).index
            game_plot_df = game_df[game_df['segment_number'].isin(game_top_scores)]
            
            # Get colour palette for this game's top scores
            game_unique_scores = sorted(game_plot_df['segment_number'].unique())
            game_palette = plt_sns.color_palette("husl", max(len(game_unique_scores), 1))
            game_score_colours = dict(zip(game_unique_scores, game_palette))
                
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            
            game_score_means = game_plot_df.groupby('segment_number', as_index=False)[['average_gaze_x', 'average_gaze_y']].mean()
            game_score_means = game_score_means.dropna(subset=['average_gaze_x', 'average_gaze_y'])
            
            center_x = game_df['average_gaze_x'].mean()
            center_y = game_df['average_gaze_y'].mean()
            ax.scatter(center_x, center_y, marker='P', s=200, color='black', edgecolor='white', linewidth=1.5, label='_nolegend_', zorder=5)
            
            for _, row in game_score_means.iterrows():
                ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            plt_sns.scatterplot(
                data=game_score_means, 
                x='average_gaze_x', 
                y='average_gaze_y', 
                hue='segment_number', 
                palette=game_score_colours, 
                s=200, 
                marker='o',
                edgecolor='black',
                linewidth=1.5,
                ax=ax,
                legend='full',
                zorder=6
            )
            
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_title(f"Average Horizontal vs Vertical Gaze per Score (Game {game_id}) - {player}")
            ax.set_xlabel("Horizontal Gaze (Inverted)")
            ax.set_ylabel("Vertical Gaze")
            
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(handles, labels, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
                
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"plot2_games/avg_gaze_game_{game_id}_{player.replace(' ', '_')}.png"), dpi=300)
            plt.close()

def plot_3_specific_scores(df):
    """
    Plot 3: One plot with the averages for 19, 20 & 17 for each player over all games.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_scores = [17, 19, 20]
    filtered_df = df[df['segment_number'].isin(target_scores)]
    
    avg_df = filtered_df.groupby(['player_name', 'segment_number'], as_index=False)[['average_gaze_x', 'average_gaze_y']].mean()
    avg_df = avg_df.dropna(subset=['average_gaze_x', 'average_gaze_y'])
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    players = avg_df['player_name'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    player_markers = dict(zip(players, markers[:len(players)]))
    
    palette = plt_sns.color_palette("Set1", len(target_scores))
    score_colours = dict(zip(target_scores, palette))
    
    # Plot each player/score combination
    for _, row in avg_df.iterrows():
        ax.scatter(
            row['average_gaze_x'], 
            row['average_gaze_y'],
            color=score_colours[row['segment_number']],
            marker=player_markers[row['player_name']],
            s=250,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            zorder=6
        )
        
    # Draw center lines for each player's average points
    for player in players:
        player_df = df[df['player_name'] == player]
        center_x = player_df['average_gaze_x'].mean()
        center_y = player_df['average_gaze_y'].mean()
        
        ax.scatter(center_x, center_y, marker='P', s=200, color='black', edgecolor='white', linewidth=1.5, zorder=5)
        
        player_avg_df = avg_df[avg_df['player_name'] == player]
        for _, row in player_avg_df.iterrows():
            ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle='--', alpha=0.5, zorder=1)
        
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title("Average Gaze for Scores 17, 19, & 20 by Player")
    ax.set_xlabel("Horizontal Gaze (Inverted)")
    ax.set_ylabel("Vertical Gaze")
    
    # Custom legends for scores (colours) and players (markers)
    import matplotlib.lines as mlines
    
    score_handles = [mlines.Line2D([], [], color=score_colours[s], marker='o', linestyle='None',
                                  markersize=10, label=str(s)) for s in target_scores]
                                  
    player_handles = [mlines.Line2D([], [], color='gray', marker=player_markers[p], linestyle='None',
                                    markersize=10, label=p) for p in players]
                                    
    legend1 = ax.legend(handles=score_handles, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(legend1)
    ax.legend(handles=player_handles, title="Player", bbox_to_anchor=(1.05, 0.6), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot3_specific_scores_averages.png"), dpi=300)
    plt.close()

def plot_4_top_5_scores_per_player(df):
    """
    Plot 4: Tidied up for an academic paper.
    - No title
    - 'Horizontal Gaze' without '(Inverted)'
    - Legends aligned properly
    - Centre marker defined in legend
    - Colorblind friendly palette
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Increase font scales for academic readability
    plt_sns.set_context("paper", font_scale=1.5)
    plt_sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    players = df['player_name'].unique()
    # Academic-friendly distinct markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    player_markers = dict(zip(players, markers[:len(players)]))
    
    all_top_scores = set()
    player_top_scores = {}
    
    for player in players:
        player_df = df[df['player_name'] == player]
        if len(player_df) == 0:
            continue
        top_scores = player_df['segment_number'].value_counts().nlargest(5).index.tolist()
        player_top_scores[player] = top_scores
        all_top_scores.update(top_scores)
        
    all_top_scores = sorted(list(all_top_scores))
    
    # Use colorblind friendly palette or tab20 if too many
    if len(all_top_scores) <= 10:
        palette = plt_sns.color_palette("colorblind", len(all_top_scores))
    else:
        palette = plt_sns.color_palette("tab20", len(all_top_scores))
        
    score_colours = dict(zip(all_top_scores, palette))
    
    for player in players:
        if player not in player_top_scores: 
            continue
            
        player_df = df[df['player_name'] == player]
        top_scores = player_top_scores[player]
        
        center_x = player_df['average_gaze_x'].mean()
        center_y = player_df['average_gaze_y'].mean()
        ax.scatter(center_x, center_y, marker='P', s=150, color='black', edgecolor='white', linewidth=1.5, zorder=5)
        
        plot_df = player_df[player_df['segment_number'].isin(top_scores)]
        score_means = plot_df.groupby('segment_number', as_index=False)[['average_gaze_x', 'average_gaze_y']].mean()
        score_means = score_means.dropna(subset=['average_gaze_x', 'average_gaze_y'])
        
        for _, row in score_means.iterrows():
            # Thinner, dotted lines connection to reduce spaghetti effect
            ax.plot([center_x, row['average_gaze_x']], [center_y, row['average_gaze_y']], color='gray', linestyle=':', alpha=0.7, zorder=1)
            ax.scatter(
                row['average_gaze_x'], 
                row['average_gaze_y'],
                color=score_colours[row['segment_number']],
                marker=player_markers[player],
                s=200,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.85,
                zorder=6
            )
            
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Academic axes formatting
    ax.set_xlabel("Horizontal Gaze")
    ax.set_ylabel("Vertical Gaze")
    plt_sns.despine(ax=ax)
    
    # Custom legends configuration
    import matplotlib.lines as mlines
    from matplotlib.legend_handler import HandlerTuple
    
    score_handles = [mlines.Line2D([], [], color=score_colours[s], marker='o', linestyle='None',
                                  markersize=8, label=str(s)) for s in all_top_scores]
                                  
    player_handles = [mlines.Line2D([], [], color='white', markerfacecolor='gray', markeredgecolor='black', marker=player_markers[p], linestyle='None',
                                    markersize=8, label=p) for p in players if p in player_top_scores]
                                    
    # Add the player center point to the player legend
    center_handle = mlines.Line2D([], [], color='white', markerfacecolor='black', markeredgecolor='white', marker='P', linestyle='None', markersize=10, label='Player Centre')
    player_handles.append(center_handle)
                                    
    # Position legends neatly stacked in the top-right margin
    legend1 = ax.legend(handles=score_handles, title="Target Score", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=player_handles, title="Player & Features", bbox_to_anchor=(1.02, 0.5), loc='upper left', frameon=False)
    
    # Save precisely with bbox_inches to capture right-hand legends without ridiculous whitespace
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot4_top5_scores_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Loading data...")
    # Read CSV
    df = pd.read_csv(INPUT_FILE)
    
    # Clean data (keep necessary columns, drop NaNs in key columns)
    cols = ['player_name', 'sport_event_id', 'average_gaze_x', 'average_gaze_y', 'segment_number']
    
    if 'segment_number' not in df.columns and 'resulting_score' in df.columns:
        # fallback if exact name differs
        df['segment_number'] = df['resulting_score']
        
    df = df[cols].dropna(subset=['average_gaze_x', 'average_gaze_y', 'segment_number', 'player_name', 'sport_event_id'])
    
    # Convert segment_number to int if applicable, to make legend ordering nice
    try:
        df['segment_number'] = df['segment_number'].astype(float).astype(int)
    except:
        pass
        
    print("Generating Plot 1: Scatter per player (overall and distinct games)...")
    plot_1_all_shots_and_games(df)
    
    print("Generating Plot 2: Average scatter per score (overall and distinct games)...")
    plot_2_average_per_score(df)
    
    print("Generating Plot 3: Specific score averages (17, 19, 20)...")
    plot_3_specific_scores(df)
    
    print("Generating Plot 4: Top 5 scores per player combined...")
    plot_4_top_5_scores_per_player(df)
    
    print("Plots generated successfully in the 'analysis' directory.")

if __name__ == "__main__":
    main()
