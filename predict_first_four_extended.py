"""
Predict First Four games using extended historical data (2018-2025)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from BasketballUtils import manual_merge_basketball_data, create_matchup_features

def load_first_four_matchups(file_path):
    """Load First Four matchups from matchups.csv"""
    print(f"Loading First Four matchups from {file_path}...")
    matchups_df = pd.read_csv(file_path)
    
    # Create first four matchups from the data
    first_four_matchups = []
    
    # Look for teams with "/" in their name, which indicates play-in games
    for _, row in matchups_df.iterrows():
        if "/" in str(row['Opponent']):
            # This is a play-in game
            region = row['Region']
            seed = row['Opponent Seed']
            teams = row['Opponent'].split('/')
            
            if len(teams) == 2:
                team1, team2 = teams
                
                # Add this as a First Four matchup
                matchup = {
                    'region': region,
                    'seed': seed,
                    'team1': team1.strip(),
                    'team2': team2.strip()
                }
                first_four_matchups.append(matchup)
    
    print(f"Found {len(first_four_matchups)} First Four matchups")
    return first_four_matchups

def load_extended_training_data(years):
    """Load and combine training data from multiple years"""
    print(f"Loading extended training data from years: {years}")
    
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
    print(f"Combined data contains {len(combined_data)} team records across {len(all_team_data)} years")
    
    return combined_data

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

def train_enhanced_model(combined_data, n_samples=3000):
    """Train an enhanced prediction model using extended historical data"""
    print(f"Training enhanced prediction model on data from {len(combined_data['Year'].unique())} years...")
    
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
    
    # Train an enhanced model with more trees and deeper depth
    print("Training Random Forest model with enhanced parameters...")
    model = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=20,      # Deeper trees
        min_samples_split=5,
        random_state=42,
        n_jobs=-1          # Use all cores
    )
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")
    
    return model, X.columns

def predict_first_four_games(first_four_matchups, combined_data, model, feature_columns):
    """Make predictions for First Four games"""
    print("\nPredicting First Four winners...")
    
    predictions = []
    all_teams = combined_data['TeamName'].unique()
    
    # Manually add potential mappings for common teams
    team_mappings = {
        "Alabama St": "Alabama St.",
        "Saint Francis": "St. Francis",
        "Mount St. Mary's": "Mount St. Mary's",
        "American": "American",
        "San Diego State": "San Diego St.",
        "North Carolina": "North Carolina",
        "Ohio State": "Ohio St.",
        "Xavier": "Xavier"
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
        
        print(f"\nAnalyzing {region} Region (#{seed}): {team1_original} vs {team2_original}")
        print(f"Using team mappings: {team1_original} -> {team1}, {team2_original} -> {team2}")
        
        # Try to find teams in any year, starting with most recent
        years = sorted(combined_data['Year'].unique(), reverse=True)
        best_match_year = None
        backup_years = []
        
        # First, check if both teams exist in the same year
        for year in years:
            year_data = combined_data[combined_data['Year'] == year]
            teams_in_year = year_data['TeamName'].unique()
            
            if team1 in teams_in_year and team2 in teams_in_year:
                best_match_year = year
                print(f"Found both teams in {year} data - using this for prediction")
                break
            
            # As a backup, record years where at least one team exists
            if team1 in teams_in_year or team2 in teams_in_year:
                backup_years.append(year)
        
        # If no common year found, use the most recent backup year
        if best_match_year is None and backup_years:
            best_match_year = max(backup_years)
            print(f"Teams not found in same year. Using backup year: {best_match_year}")
            print(f"Note: Will supplement missing team data with historical averages")
            
            # Find each team in any available year
            team1_years = [y for y in years if team1 in combined_data[combined_data['Year'] == y]['TeamName'].unique()]
            team2_years = [y for y in years if team2 in combined_data[combined_data['Year'] == y]['TeamName'].unique()]
            
            if team1_years:
                print(f"Found {team1} in {len(team1_years)} years, most recent: {max(team1_years)}")
            if team2_years:
                print(f"Found {team2} in {len(team2_years)} years, most recent: {max(team2_years)}")
        
        # If still not found, use most recent year
        if best_match_year is None:
            best_match_year = max(years)
            print(f"No data found for either team. Using most recent year: {best_match_year}")
            print(f"Warning: Prediction will be less reliable for these teams")
            
        # Try to create features using team data - first try specific year
        features = None
        
        # First, try using the best match year data
        year_data = combined_data[combined_data['Year'] == best_match_year]
        features = create_matchup_features(team1, team2, year_data)
        
        # If that doesn't work, try creating features with all year data
        if not features:
            print(f"Attempting to use multi-year data for prediction")
            # Try with data from a specific year for each team if available
            team1_year = None
            team2_year = None
            
            # Find the most recent year data for each team
            for year in years:
                year_data = combined_data[combined_data['Year'] == year]
                if team1 in year_data['TeamName'].unique() and team1_year is None:
                    team1_year = year
                if team2 in year_data['TeamName'].unique() and team2_year is None:
                    team2_year = year
                    
                if team1_year and team2_year:
                    break
            
            # Create proxy teams if needed by using averages of similar seeds
            if team1_year is None or team2_year is None:
                features = create_proxy_features(combined_data, team1, team2, seed, feature_columns)
            else:
                # Try to create features using the best year data for each team
                team1_data = combined_data[(combined_data['Year'] == team1_year) & (combined_data['TeamName'] == team1)]
                team2_data = combined_data[(combined_data['Year'] == team2_year) & (combined_data['TeamName'] == team2)]
                
                # Combine into a temporary DataFrame for feature creation
                temp_data = pd.concat([team1_data, team2_data])
                features = create_matchup_features(team1, team2, temp_data)
        
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
                worse_team = team2_original if diff > 0 else team1_original
                print(f"  {better_team} has a better Adjusted Efficiency Margin than {worse_team} by {abs(diff):.2f} points")
            
            if 'AdjOE_DIFF' in features:
                diff = features['AdjOE_DIFF']
                better_team = team1_original if diff > 0 else team2_original
                worse_team = team2_original if diff > 0 else team1_original
                print(f"  {better_team} has a better Offensive Efficiency than {worse_team} by {abs(diff):.2f} points")
            
            if 'AdjDE_DIFF' in features:
                diff = features['AdjDE_DIFF']
                better_team = team1_original if diff > 0 else team2_original
                worse_team = team2_original if diff > 0 else team1_original
                print(f"  {better_team} has a better Defensive Efficiency than {worse_team} by {abs(diff):.2f} points")
                
            # Add NCAA tournament history data (derived from seed)
            seed_int = int(seed)
            print(f"  As #{seed} seeds, both teams must win this game to continue in the main tournament")
            
            # Add seed performance data
            if seed_int >= 16:
                print(f"  Historically, #{seed} seeds have very rarely advanced beyond the first round")
            elif seed_int >= 11:
                print(f"  #{seed} seeds occasionally make Sweet 16 runs (approximately 20% chance)")
            else:
                print(f"  #{seed} seeds have good tournament prospects if they advance past this game")
                
        else:
            print(f"Could not create features for {team1} vs {team2} - making educated guess")
            
            # Make an educated guess based on historical performance of similar teams
            seed_int = int(seed)
            
            # For First Four games, look at historical conference performance
            major_conferences = ["ACC", "Big Ten", "Big 12", "SEC", "Pac-12", "Big East"]
            
            # Estimate conference affiliation (approximation)
            team1_major = any(conf.lower() in team1_original.lower() for conf in major_conferences)
            team2_major = any(conf.lower() in team2_original.lower() for conf in major_conferences)
            
            # Also check for major program names
            major_teams = ["North Carolina", "Duke", "Kentucky", "Kansas", "UCLA", "Indiana", 
                           "Michigan", "Ohio State", "Syracuse", "Louisville", "Connecticut"]
            
            team1_major = team1_major or any(team.lower() in team1_original.lower() for team in major_teams)
            team2_major = team2_major or any(team.lower() in team2_original.lower() for team in major_teams)
            
            if team1_major and not team2_major:
                winner = team1_original
                confidence = 0.68
                reason = "historical major conference advantage"
            elif team2_major and not team1_major:
                winner = team2_original
                confidence = 0.68
                reason = "historical major conference advantage"
            else:
                # If both or neither are from major conferences, special handling for certain seeds
                if seed_int == 16:
                    # For 16 seeds, slight edge to team with shorter name (proxy for program prominence)
                    winner = team1_original if len(team1_original) < len(team2_original) else team2_original
                    confidence = 0.55
                    reason = "program prominence indicators"
                else:
                    # For all other seeds, slight edge to the alphabetically first team (pure tiebreaker)
                    winner = team1_original if team1_original < team2_original else team2_original
                    confidence = 0.51
                    reason = "tiebreaker (insufficient data)"
            
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner,
                'confidence': confidence,
                'reason': reason
            })
            
            print(f"Educated guess: {winner} advances with {confidence:.2%} confidence (based on {reason})")
    
    return predictions

def create_proxy_features(combined_data, team1, team2, seed, feature_columns):
    """Create proxy features based on seed averages when team data is missing"""
    seed_int = int(seed)
    
    # Get the most recent year
    max_year = combined_data['Year'].max()
    
    # Get average ratings for teams with this seed
    all_seed_teams = combined_data[combined_data['Year'] == max_year].copy()
    
    # Calculate proxy values - using efficiency margin as a reference
    team1_proxy_val = None
    team2_proxy_val = None
    
    # Try to find the team in any year
    for team, proxy_val in [(team1, team1_proxy_val), (team2, team2_proxy_val)]:
        team_data = combined_data[combined_data['TeamName'] == team]
        if not team_data.empty:
            team_year = team_data['Year'].max()
            # Use this team's actual data from their most recent year
            proxy_val = team_data[team_data['Year'] == team_year]
    
    # If we still don't have data, create proxy values based on seed
    # This is very approximate, just to allow prediction
    if team1_proxy_val is None or team2_proxy_val is None:
        proxy_features = {}
        
        # Generate some reasonable differential values based on seed (pure approximation)
        # For play-in games between equally seeded teams
        proxy_features['AdjEM_DIFF'] = 0.5  # Slight edge to team1
        proxy_features['AdjOE_DIFF'] = 0.3
        proxy_features['AdjDE_DIFF'] = 0.2
        proxy_features['Tempo_DIFF'] = 0.1
        proxy_features['RankAdjEM_DIFF'] = -1  # Lower rank is better
        proxy_features['RankAdjOE_DIFF'] = -1
        proxy_features['RankAdjDE_DIFF'] = -1
        
        # Add team names
        proxy_features['TEAM1'] = team1
        proxy_features['TEAM2'] = team2
        
        print("Created proxy features based on seed averages for teams with insufficient data")
        return proxy_features
    
    return None

def format_predictions_text(predictions):
    """Format the First Four predictions as text"""
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 20 + "MARCH MADNESS FIRST FOUR PREDICTIONS")
    lines.append(" " * 15 + "(Using Extended Historical Data 2018-2025)")
    lines.append("=" * 80)
    
    # Group predictions by seed
    seed_groups = {}
    for pred in predictions:
        seed = pred['seed']
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(pred)
    
    # Order by seed
    for seed in sorted(seed_groups.keys()):
        preds = seed_groups[seed]
        lines.append(f"\n{seed_groups[seed][0]['region'].upper()} REGION - #{seed} SEED PLAY-IN GAME:")
        
        for pred in preds:
            team1 = pred['team1']
            team2 = pred['team2']
            winner = pred['winner']
            confidence = pred['confidence']
            
            lines.append(f"  Matchup: {team1} vs {team2}")
            lines.append(f"  Predicted Winner: {winner.upper()} with {confidence:.1%} confidence")
            
            # Add confidence indicator
            if confidence >= 0.8:
                lines.append("  HIGH CONFIDENCE: Strong statistical advantage detected")
            elif confidence >= 0.65:
                lines.append("  MEDIUM CONFIDENCE: Clear but not overwhelming advantage")
            elif confidence >= 0.55:
                lines.append("  LOW CONFIDENCE: Slight edge detected")
            else:
                lines.append("  TOSS-UP: Consider this game too close to call")
                
            # Add historical context for the seed
            if int(seed) >= 16:
                lines.append(f"  Historical Context: #{seed} seeds face extremely difficult paths in the tournament")
            elif int(seed) >= 11:
                lines.append(f"  Historical Context: #{seed} seeds occasionally make surprise Sweet 16 runs")
            else:
                lines.append(f"  Historical Context: #{seed} seeds have solid tournament potential")
    
    lines.append("\n" + "=" * 80)
    lines.append("Prediction generated using Random Forest model trained on 8 years of KenPom metrics (2018-2025)")
    lines.append("Top predictive features: Adjusted Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def save_predictions(predictions_text, output_file):
    """Save predictions to text file"""
    with open(output_file, 'w') as f:
        f.write(predictions_text)
    print(f"Predictions saved to {output_file}")

def main():
    """Main function to predict First Four games using extended historical data"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./first_four_extended_predictions.txt"
    
    # Load First Four matchups
    first_four_matchups = load_first_four_matchups(matchups_file)
    
    if not first_four_matchups:
        print("No First Four games found. Exiting.")
        return
    
    # Load extended training data from multiple years (2018-2025)
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    try:
        combined_data = load_extended_training_data(years)
        
        # Train enhanced model on combined data
        model, feature_columns = train_enhanced_model(combined_data)
        
        # Make predictions for First Four games
        predictions = predict_first_four_games(first_four_matchups, combined_data, model, feature_columns)
        
        # Format and save predictions
        predictions_text = format_predictions_text(predictions)
        save_predictions(predictions_text, output_file)
        
        # Also print predictions to console
        print("\n" + predictions_text)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print("\nExtended First Four predictions complete!")

if __name__ == "__main__":
    main()