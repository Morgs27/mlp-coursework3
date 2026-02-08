import pandas as pd

def validate_scores(csv_path):
    print(f"Validating {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort just in case, though it should be time-ordered
    df = df.sort_values(by=['Time'])
    
    issues = []
    
    for leg in df['Leg'].unique():
        leg_df = df[df['Leg'] == leg]
        
        for player in ['Player 1', 'Player 2']:
            play_df = leg_df[leg_df['Player'] == player]
            
            current_score = 501
            
            for index, row in play_df.iterrows():
                throw = row['ThrowScore']
                rem = row['RemainingScore']
                time = row['Time']
                
                # Check arithmetic of this row (implied)
                # The CSV 'RemainingScore' is what the OCR read *after* the throw.
                # The 'ThrowScore' was calculated as (Prev - Current).
                # So by definition in the script, Prev - Throw = Current.
                # However, we want to check if the 'Prev' of this row was actually the 'Current' of the last row.
                
                expected_rem = current_score - throw
                
                if expected_rem != rem:
                    # This means we likely missed a score update in between, 
                    # OR the OCR misread the previous or current value, causing a jump.
                    
                    # Example: 501 -> 441 (Throw 60). True.
                    # Next read: 300. Script calc Throw = 441 - 300 = 141. 
                    # Row says: Throw 141, Rem 300.
                    # Then Expected (441 - 141) == 300. Matches.
                    # So the script logic enforces row consistency self-contained.
                    
                    # BUT, current_score tracks the *running* validation from 501.
                    # If we missed a row, current_score will be higher than the row's starting point.
                    
                    diff = current_score - (rem + throw) # The gap before this throw
                    
                    issues.append({
                        'Leg': leg,
                        'Player': player,
                        'Time': time,
                        'Issue': 'Continuity Error',
                        'Details': f"Expected previous score {current_score}, but imply starting at {rem+throw}. Gap of {diff}."
                    })
                    
                    # Resync to continue validation
                    current_score = rem 
                else:
                    current_score = rem

                # Validity checks
                if throw > 180:
                     issues.append({
                        'Leg': leg,
                        'Player': player,
                         'Time': time,
                        'Issue': 'Invalid Throw',
                        'Details': f"Throw of {throw} > 180"
                    })
                if throw < 0:
                     issues.append({
                        'Leg': leg,
                        'Player': player,
                         'Time': time,
                        'Issue': 'Negative Throw',
                        'Details': f"Throw of {throw}"
                    })

    if not issues:
        print("No validation issues found!")
    else:
        print(f"Found {len(issues)} issues:")
        issue_df = pd.DataFrame(issues)
        print(issue_df.head(20))
        issue_df.to_csv("validation_issues.csv", index=False)
        print("Saved validation_issues.csv")

if __name__ == "__main__":
    validate_scores("darts_scores.csv")
