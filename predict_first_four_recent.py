"""
Predict First Four games using recent data (2023-2025)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def load_recent_data(years):
    """Load and combine recent years data"""
    print(f"Loading data from {len(years)} recent seasons: {min(years)}-{max(years)}")
    
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
    print(f"Combined data contains {len(combined_data)} team records across {len(years)} seasons")
    
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

def train_recent_model(combined_data, n_samples=2500):
    """Train a model using recent data"""
    print(f"Training model using data from {len(combined_data['Year'].unique())} recent seasons...")
    
    # Get all teams
    all_teams = combined_data['TeamName'].unique()
    print(f"Total unique teams in recent data: {len(all_teams)}")
    
    # Generate synthetic matchups for training, with weight to most recent year
    years = sorted(combined_data['Year'].unique())
    year_weights = {year: 1 + 0.5 * (idx / len(years)) for idx, year in enumerate(years)}
    print(f"Year weights for sampling: {year_weights}")
    
    matchups = []
    for _ in range(n_samples):
        # Sample a year with bias toward more recent years
        year_probs = np.array(list(year_weights.values()))
        year_probs = year_probs / year_probs.sum()  # Normalize to sum to 1
        year = np.random.choice(years, p=year_probs)
        
        # Get teams from this year
        year_data = combined_data[combined_data['Year'] == year]
        year_teams = year_data['TeamName'].unique()
        
        if len(year_teams) < 2:
            continue
            
        team1, team2 = np.random.choice(year_teams, 2, replace=False)
        features = create_matchup_features(team1, team2, year_data)
        
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
    model = RandomForestClassifier(
        n_estimators=250, 
        max_depth=15,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")
    
    return model, X.columns

def get_team_trend(team, combined_data):
    """Analyze a team's trend over recent seasons"""
    team_data = combined_data[combined_data['TeamName'] == team]
    
    if len(team_data) <= 1:
        return "Insufficient data for trend analysis"
    
    # Get metrics by year
    metrics_by_year = {}
    for year, year_data in team_data.groupby('Year'):
        metrics_by_year[year] = {
            'AdjEM': year_data['AdjEM'].values[0],
            'AdjOE': year_data['AdjOE'].values[0],
            'AdjDE': year_data['AdjDE'].values[0],
            'Rank': year_data['RankAdjEM'].values[0]
        }
    
    # Calculate trend direction for key metrics
    years = sorted(metrics_by_year.keys())
    if len(years) >= 2:
        first_year = years[0]
        last_year = years[-1]
        
        em_change = metrics_by_year[last_year]['AdjEM'] - metrics_by_year[first_year]['AdjEM']
        rank_change = metrics_by_year[first_year]['Rank'] - metrics_by_year[last_year]['Rank']  # Reverse so positive is improvement
        
        if em_change > 2 and rank_change > 10:
            trend = "Strong upward trend"
        elif em_change > 0.5 and rank_change > 0:
            trend = "Slight improvement"
        elif em_change < -2 and rank_change < -10:
            trend = "Notable decline"
        elif em_change < -0.5 and rank_change < 0:
            trend = "Slight decline"
        else:
            trend = "Relatively stable"
        
        return trend
    else:
        return "Insufficient data for trend analysis"

def predict_first_four_games(first_four_matchups, combined_data, model, feature_columns):
    """Make predictions using recent data"""
    print("\nPredicting First Four games using recent data (2023-2025)...")
    
    predictions = []
    all_teams = combined_data['TeamName'].unique()
    
    # Manually add potential mappings for common teams
    team_mappings = {
        "Alabama State": "Alabama St.",
        "Saint Francis": "St. Francis",
        "Mount St. Mary's": "Mount St. Mary's",
        "American": "American",
        "San Diego State": "San Diego St.",
        "North Carolina": "North Carolina",
        "Ohio State": "Ohio St.",
        "Xavier": "Xavier",
        "Texas": "Texas"
    }
    
    for matchup in first_four_matchups:
        region = matchup['region']
        seed = matchup['seed']
        team1_original = matchup['team1']
        team2_original = matchup['team2']
        
        # Fix potential prefix issues
        if "First Four:" in team1_original:
            team1_original = team1_original.replace("First Four:", "").strip()
        if "First Four:" in team2_original:
            team2_original = team2_original.replace("First Four:", "").strip()
        
        print(f"\nAnalyzing {region} Region (#{seed}): {team1_original} vs {team2_original}")
        
        # Use manual mappings if available, otherwise find closest match
        team1 = team_mappings.get(team1_original, team1_original)
        team2 = team_mappings.get(team2_original, team2_original)
        
        # If still not in the dataset, find closest match
        if team1 not in all_teams:
            team1 = find_closest_team_name(team1, all_teams)
        
        if team2 not in all_teams:
            team2 = find_closest_team_name(team2, all_teams)
        
        print(f"Using team mappings: {team1_original} -> {team1}, {team2_original} -> {team2}")
        
        # Get trend information
        team1_trend = get_team_trend(team1, combined_data)
        team2_trend = get_team_trend(team2, combined_data)
        
        print(f"{team1_original} trend: {team1_trend}")
        print(f"{team2_original} trend: {team2_trend}")
        
        # Try to find teams in the same year, preferring the most recent
        years = sorted(combined_data['Year'].unique(), reverse=True)
        best_match_year = None
        
        for year in years:
            year_data = combined_data[combined_data['Year'] == year]
            if team1 in year_data['TeamName'].unique() and team2 in year_data['TeamName'].unique():
                best_match_year = year
                print(f"Found both teams in {year} data - using this for prediction")
                break
        
        # If not found together, use most recent year with aggregated data
        if best_match_year is None:
            best_match_year = max(years)
            print(f"Teams not found in same year. Using most recent ({best_match_year}) with aggregated data")
        
        # Get data for prediction
        year_data = combined_data[combined_data['Year'] == best_match_year]
        features = create_matchup_features(team1, team2, year_data)
        
        if features:
            # Prepare features for prediction
            pred_features = {col: features.get(col, 0) for col in feature_columns}
            pred_df = pd.DataFrame([pred_features])
            
            # Make prediction
            prob = model.predict_proba(pred_df)[0]
            team1_win_prob = prob[1] if model.classes_[1] == 1 else prob[0]
            
            winner = team1 if team1_win_prob > 0.5 else team2
            winner_prob = team1_win_prob if winner == team1 else 1 - team1_win_prob
            
            # Map back to original team names
            winner_original = team1_original if winner == team1 else team2_original
            
            # Save prediction
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner_original,
                'confidence': winner_prob,
                'features': features,
                'team1_trend': team1_trend,
                'team2_trend': team2_trend
            })
            
            print(f"Prediction: {winner_original} wins with {winner_prob:.2%} confidence")
            
            # Show key statistical differences
            print("\nKey statistical differences:")
            for metric, name in [
                ('AdjEM_DIFF', 'Adjusted Efficiency Margin'),
                ('AdjOE_DIFF', 'Offensive Efficiency'),
                ('AdjDE_DIFF', 'Defensive Efficiency'),
                ('Tempo_DIFF', 'Pace of Play')
            ]:
                if metric in features:
                    diff = features[metric]
                    better_team = team1_original if diff > 0 else team2_original
                    worse_team = team2_original if diff > 0 else team1_original
                    print(f"• {better_team} has better {name} than {worse_team} by {abs(diff):.2f} points")
        else:
            print(f"Could not create features for {team1} vs {team2}")
            confidence = 0.51  # Default for toss-up
            
            # If we have trend data, use that to bias our prediction
            if team1_trend in ["Strong upward trend", "Slight improvement"] and team2_trend not in ["Strong upward trend"]:
                winner = team1_original
                confidence = 0.58
                reason = "recent improvement trend"
            elif team2_trend in ["Strong upward trend", "Slight improvement"] and team1_trend not in ["Strong upward trend"]:
                winner = team2_original
                confidence = 0.58
                reason = "recent improvement trend"
            else:
                # Default to alphabetical order as tiebreaker
                winner = team1_original if team1 < team2 else team2_original 
                reason = "insufficient data"
            
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner,
                'confidence': confidence,
                'reason': reason,
                'team1_trend': team1_trend,
                'team2_trend': team2_trend
            })
            
            print(f"Prediction based on {reason}: {winner} advances with {confidence:.2%} confidence")
    
    return predictions

def format_predictions_text(predictions):
    """Format First Four predictions as text"""
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 20 + "MARCH MADNESS FIRST FOUR PREDICTIONS")
    lines.append(" " * 22 + "(Using Recent Data: 2023-2025)")
    lines.append("=" * 80)
    
    # Group by seed
    seed_groups = {}
    for pred in predictions:
        seed = pred['seed']
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(pred)
    
    # Order by seed
    for seed in sorted(seed_groups.keys()):
        lines.append(f"\n{seed_groups[seed][0]['region'].upper()} REGION - #{seed} SEED PLAY-IN GAMES:")
        lines.append("-" * 80)
        
        for pred in seed_groups[seed]:
            team1 = pred['team1']
            team2 = pred['team2']
            winner = pred['winner']
            confidence = pred['confidence']
            
            lines.append(f"Matchup: {team1} vs {team2}")
            lines.append(f"Predicted Winner: {winner.upper()} with {confidence:.1%} confidence")
            
            # Add confidence indicator
            if confidence >= 0.8:
                lines.append("Confidence: VERY HIGH - Strong statistical advantage detected")
            elif confidence >= 0.7:
                lines.append("Confidence: HIGH - Clear statistical advantage")
            elif confidence >= 0.6:
                lines.append("Confidence: MEDIUM - Moderate advantage detected")
            elif confidence >= 0.55:
                lines.append("Confidence: LOW - Slight edge detected")
            else:
                lines.append("Confidence: TOSS-UP - Teams are very evenly matched")
            
            # Add team trends
            if 'team1_trend' in pred and 'team2_trend' in pred:
                lines.append(f"\nRecent Trends:")
                lines.append(f"• {team1}: {pred['team1_trend']}")
                lines.append(f"• {team2}: {pred['team2_trend']}")
            
            # Add key metrics if available
            if 'features' in pred:
                lines.append("\nStatistical Analysis (Recent Data):")
                
                features = pred['features']
                for metric, name in [
                    ('AdjEM_DIFF', 'Adjusted Efficiency Margin'),
                    ('AdjOE_DIFF', 'Offensive Efficiency'),
                    ('AdjDE_DIFF', 'Defensive Efficiency'),
                    ('Tempo_DIFF', 'Pace of Play')
                ]:
                    if metric in features and abs(features[metric]) > 0.5:
                        diff = features[metric]
                        better_team = team1 if diff > 0 else team2
                        worse_team = team2 if diff > 0 else team1
                        lines.append(f"• {better_team} has better {name} than {worse_team} by {abs(diff):.2f} points")
            
            # Add bracket impact
            lines.append(f"\nFirst Round Impact:")
            if seed == "16":
                lines.append(f"Winner will face a #1 seed with <5% historical chance of an upset")
            elif seed == "11":
                lines.append(f"Winner will face a #6 seed with ~30% historical chance of an upset")
            
            # Add separator between matchups
            if pred != seed_groups[seed][-1]:
                lines.append("\n" + "-" * 40)
    
    lines.append("\n" + "=" * 80)
    lines.append("MODEL INFORMATION:")
    lines.append("• Random Forest model trained on 2023-2025 KenPom metrics")
    lines.append("• Prediction weighs recent seasons more heavily (2025 > 2024 > 2023)")
    lines.append("• Analysis includes team performance trends across recent seasons")
    lines.append("• Top predictive features: Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def save_predictions(predictions_text, output_file):
    """Save predictions to text file"""
    with open(output_file, 'w') as f:
        f.write(predictions_text)
    print(f"Predictions saved to {output_file}")

def main():
    """Main function to predict First Four games using recent data"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./first_four_recent_predictions.txt"
    
    # Load First Four matchups
    first_four_matchups = load_first_four_matchups(matchups_file)
    
    if not first_four_matchups:
        print("No First Four games found. Exiting.")
        return
    
    try:
        # Load data from recent years
        years = [2023, 2024, 2025]
        combined_data = load_recent_data(years)
        
        # Train model on recent data
        model, feature_columns = train_recent_model(combined_data)
        
        # Make predictions
        predictions = predict_first_four_games(first_four_matchups, combined_data, model, feature_columns)
        
        # Format and save predictions
        predictions_text = format_predictions_text(predictions)
        save_predictions(predictions_text, output_file)
        
        print("\nFirst Four predictions using recent data complete!")
        
    except Exception as e:
        print(f"Error during prediction process: {str(e)}")

if __name__ == "__main__":
    main()