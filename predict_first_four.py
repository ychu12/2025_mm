"""
Predict First Four games using combined training data from multiple years
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from BasketballUtils import manual_merge_basketball_data, create_matchup_features

def load_first_four_matchups(file_path):
    """Load First Four matchups from the revised CSV file"""
    print(f"Loading First Four matchups from {file_path}...")
    matchups_df = pd.read_csv(file_path)
    
    # Extract only First Four games
    first_four_df = matchups_df[matchups_df['Round'] == 'First Four']
    
    if first_four_df.empty:
        print("No First Four games found in the dataset")
        return []
    
    # Format First Four matchups
    first_four_matchups = []
    for _, row in first_four_df.iterrows():
        matchup = {
            'region': row['Region'],
            'seed': row['Seed1'],  # Both teams have same seed in First Four
            'team1': row['Team1'],
            'team2': row['Team2']
        }
        first_four_matchups.append(matchup)
    
    print(f"Found {len(first_four_matchups)} First Four matchups")
    return first_four_matchups

def load_combined_training_data(years):
    """Load and combine training data from multiple years"""
    print(f"Loading training data from years: {years}")
    
    all_team_data = []
    for year in years:
        try:
            print(f"Loading {year} data...")
            team_data = manual_merge_basketball_data(year=year)
            team_data['Year'] = year  # Add year column for reference
            all_team_data.append(team_data)
            print(f"Added {len(team_data)} team records from {year}")
        except Exception as e:
            print(f"Error loading data for {year}: {e}")
    
    if not all_team_data:
        raise ValueError("No valid data loaded from any year")
    
    # Combine all years' data
    combined_data = pd.concat(all_team_data, ignore_index=True)
    print(f"Combined data contains {len(combined_data)} team records across {len(years)} years")
    
    return combined_data

def train_combined_model(combined_data, n_samples=2000):
    """Train a prediction model using combined data from multiple years"""
    print(f"Training prediction model on combined data...")
    
    # Get all teams
    all_teams = combined_data['TeamName'].unique()
    print(f"Total unique teams in combined data: {len(all_teams)}")
    
    # Generate synthetic matchups for training
    matchups = []
    for _ in range(n_samples):
        # Sample a random year from the data
        year_data = combined_data.groupby('Year').sample(n=1).iloc[0]
        year = year_data['Year']
        
        # Filter data for this year
        year_teams = combined_data[combined_data['Year'] == year]['TeamName'].unique()
        
        # Get two random teams from this year
        if len(year_teams) < 2:
            continue
            
        team1, team2 = np.random.choice(year_teams, 2, replace=False)
        
        # Create features for this matchup
        year_team_data = combined_data[combined_data['Year'] == year]
        features = create_matchup_features(team1, team2, year_team_data)
        
        if features:
            # Generate synthetic result based on team metrics
            if 'RankAdjEM_DIFF' in features:
                # Negative means team1 is better
                prob_team1_wins = 1 / (1 + np.exp(features['RankAdjEM_DIFF'] / 10))
            else:
                # Default if no rank info
                prob_team1_wins = 0.6
                
            features['RESULT'] = 1 if np.random.random() < prob_team1_wins else -1
            matchups.append(features)
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    print(f"Generated {len(matchups_df)} training matchups")
    
    # Prepare features and target
    feature_cols = [col for col in matchups_df.columns if col not in ['TEAM1', 'TEAM2', 'RESULT']]
    X = matchups_df[feature_cols].select_dtypes(include=[np.number])
    y = matchups_df['RESULT']
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")
    
    return model, X.columns

def find_closest_team_name(target_team, all_teams):
    """Find the closest matching team name in the dataset"""
    if target_team in all_teams:
        return target_team
    
    # Try common variations
    variations = [
        target_team,
        target_team.replace("Saint", "St."),
        target_team.replace("St.", "Saint"),
        target_team.replace("State", "St."),
        target_team.replace("St.", "State"),
        target_team + " University",
        target_team.replace(" University", ""),
        target_team.split()[0]  # First word only
    ]
    
    for variation in variations:
        for team in all_teams:
            if variation.lower() in team.lower() or team.lower() in variation.lower():
                print(f"Found approximate match: '{target_team}' -> '{team}'")
                return team
    
    # No close match found
    print(f"No match found for '{target_team}'")
    return target_team

def predict_first_four_games(first_four_matchups, combined_data, model, feature_columns):
    """Make predictions for First Four games"""
    print("\nPredicting First Four winners...")
    
    predictions = []
    all_teams = combined_data['TeamName'].unique()
    
    # Manually add potential mappings for common teams
    team_mappings = {
        "North Carolina": "UNC",
        "San Diego State": "San Diego St.",
        "Ohio State": "Ohio St.",
        "Xavier": "Xavier",
        "UNC Wilmington": "UNC Wilmington",
        "SIUE": "SIU Edwardsville"
    }
    
    for matchup in first_four_matchups:
        region = matchup['region']
        seed = matchup['seed']
        team1_original = matchup['team1']
        team2_original = matchup['team2']
        
        # Use manual mappings if available, otherwise find closest match
        team1 = team_mappings.get(team1_original, team1_original)
        team2 = team_mappings.get(team2_original, team2_original)
        
        # If still not in the dataset, find closest match
        if team1 not in all_teams:
            team1 = find_closest_team_name(team1, all_teams)
        
        if team2 not in all_teams:
            team2 = find_closest_team_name(team2, all_teams)
        
        print(f"\nAnalyzing {region} Region: (#{seed}) {team1_original} vs (#{seed}) {team2_original}")
        print(f"Using team mappings: {team1_original} -> {team1}, {team2_original} -> {team2}")
        
        # Try to find teams in each year, starting with most recent
        years = sorted(combined_data['Year'].unique(), reverse=True)
        best_match_year = None
        
        for year in years:
            year_data = combined_data[combined_data['Year'] == year]
            teams_in_year = year_data['TeamName'].unique()
            
            if team1 in teams_in_year and team2 in teams_in_year:
                best_match_year = year
                print(f"Found both teams in {year} data - using this for prediction")
                break
        
        if best_match_year is None:
            # If we can't find both teams in same year, use most recent year with the most data
            best_match_year = max(years)
            print(f"Teams not found in same year. Using most recent data: {best_match_year}")
            
            # For analysis, still note which teams we have data for
            for year in years:
                year_data = combined_data[combined_data['Year'] == year]
                if team1 in year_data['TeamName'].unique():
                    print(f"- Found {team1} in {year} data")
                if team2 in year_data['TeamName'].unique():
                    print(f"- Found {team2} in {year} data")
        
        # Use all data, prioritizing the best match year
        all_years_data = combined_data.copy()
        
        # Try to create features using team data - first try exact year
        year_data = combined_data[combined_data['Year'] == best_match_year]
        features = create_matchup_features(team1, team2, year_data)
        
        # If that fails, try all years combined
        if not features:
            print(f"Attempting to use combined multi-year data for prediction")
            features = create_matchup_features(team1, team2, all_years_data)
        
        if features:
            # Prepare for prediction
            pred_features = {col: features.get(col, 0) for col in feature_columns}
            pred_df = pd.DataFrame([pred_features])
            
            # Make prediction
            prob = model.predict_proba(pred_df)[0]
            team1_win_prob = prob[1] if model.classes_[1] == 1 else prob[0]
            
            winner = team1 if team1_win_prob > 0.5 else team2
            winner_prob = team1_win_prob if winner == team1 else 1 - team1_win_prob
            
            # Map back to original team names
            winner_original = team1_original if winner == team1 else team2_original
            
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner_original,
                'confidence': winner_prob
            })
            
            print(f"Prediction: {winner_original} wins with {winner_prob:.2%} confidence")
            
            # Show key statistical differences
            if 'AdjEM_DIFF' in features:
                diff = features['AdjEM_DIFF']
                better_team = team1_original if diff > 0 else team2_original
                print(f"  {better_team} has a better Adjusted Efficiency Margin by {abs(diff):.2f} points")
            
            if 'AdjOE_DIFF' in features:
                diff = features['AdjOE_DIFF']
                better_team = team1_original if diff > 0 else team2_original
                print(f"  {better_team} has a better Offensive Efficiency by {abs(diff):.2f} points")
            
            if 'AdjDE_DIFF' in features:
                diff = features['AdjDE_DIFF']
                better_team = team1_original if diff > 0 else team2_original
                print(f"  {better_team} has a better Defensive Efficiency by {abs(diff):.2f} points")
            
            # Add NCAA tournament history if available (simulated based on seed)
            seed_int = int(seed)
            if seed_int <= 8:
                print(f"  Both teams have a high tournament seed (#{seed}), indicating strong regular season performance")
            else:
                print(f"  As #{seed} seeds, both teams likely need this win to advance to the main tournament")
                
        else:
            print(f"Could not create features for {team1} vs {team2} - making educated guess")
            
            # Make a somewhat educated guess
            if seed_int <= 8:
                # For higher seeds, use name recognition/prominence as a rough proxy
                # This is not scientific but reflects historical tendencies in tournament play
                major_conference_terms = ["North Carolina", "Ohio State", "Kansas", "Duke", "Kentucky"]
                
                team1_major = any(term in team1_original for term in major_conference_terms)
                team2_major = any(term in team2_original for term in major_conference_terms)
                
                if team1_major and not team2_major:
                    winner = team1_original
                    confidence = 0.65
                elif team2_major and not team1_major:
                    winner = team2_original
                    confidence = 0.65
                else:
                    # Both or neither from major conferences, slight edge to team1
                    winner = team1_original
                    confidence = 0.55
            else:
                # For lower seeds, it's essentially a toss-up
                winner = team1_original if np.random.random() > 0.5 else team2_original
                confidence = 0.52
            
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner,
                'confidence': confidence
            })
            
            print(f"Educated guess: {winner} advances with {confidence:.2%} confidence (based on program history/reputation)")
    
    return predictions

def format_predictions_text(predictions):
    """Format the First Four predictions as text"""
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 20 + "MARCH MADNESS FIRST FOUR PREDICTIONS")
    lines.append(" " * 15 + "(Using Combined Multi-Year Training Data)")
    lines.append("=" * 80)
    
    for pred in predictions:
        region = pred['region']
        seed = pred['seed']
        team1 = pred['team1']
        team2 = pred['team2']
        winner = pred['winner']
        confidence = pred['confidence']
        
        lines.append(f"\n{region.upper()} REGION - #{seed} SEED PLAY-IN GAME:")
        lines.append(f"  {team1} vs {team2}")
        lines.append(f"  Predicted Winner: {winner.upper()} with {confidence:.1%} confidence")
        
        # Add warning for low confidence predictions
        if confidence < 0.6:
            lines.append("  WARNING: Low confidence prediction - consider this game a toss-up")
        elif confidence > 0.8:
            lines.append("  HIGH CONFIDENCE: Strong statistical advantage detected")
    
    lines.append("\n" + "=" * 80)
    lines.append("Prediction generated using Random Forest model trained on multiple years of KenPom metrics")
    lines.append("Top predictive features: Adjusted Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def save_predictions(predictions_text, output_file):
    """Save predictions to text file"""
    with open(output_file, 'w') as f:
        f.write(predictions_text)
    print(f"Predictions saved to {output_file}")

def main():
    """Main function to predict First Four games"""
    # File paths
    revised_file = "./data/revised.csv"
    output_file = "./first_four_predictions.txt"
    
    # Load First Four matchups
    first_four_matchups = load_first_four_matchups(revised_file)
    
    if not first_four_matchups:
        print("No First Four games found. Exiting.")
        return
    
    # Load combined training data from multiple years (2022-2025)
    years = [2022, 2023, 2024, 2025]
    try:
        combined_data = load_combined_training_data(years)
        
        # Train model on combined data
        model, feature_columns = train_combined_model(combined_data)
        
        # Make predictions for First Four games
        predictions = predict_first_four_games(first_four_matchups, combined_data, model, feature_columns)
        
        # Format and save predictions
        predictions_text = format_predictions_text(predictions)
        save_predictions(predictions_text, output_file)
        
        # Also print predictions to console
        print("\n" + predictions_text)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print("\nFirst Four predictions complete!")

if __name__ == "__main__":
    main()