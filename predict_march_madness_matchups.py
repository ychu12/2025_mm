"""
Predict 3-20-2025 March Madness matchups using both historical (2022-2025) 
and current season (2025) data.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from BasketballUtils import manual_merge_basketball_data, create_matchup_features

def load_matchups(file_path, target_date="2025-03-20"):
    """Load matchups for a specific date from matchups.csv"""
    print(f"Loading matchups from {file_path} for date {target_date}...")
    matchups_df = pd.read_csv(file_path)
    
    # Filter for target date
    date_matchups = matchups_df[matchups_df['Date'] == target_date]
    print(f"Found {len(date_matchups)} matchups for {target_date}")
    
    # Create structured matchups
    day_matchups = []
    
    for _, row in date_matchups.iterrows():
        region = row['Region']
        team_seed = row['Seed']
        team = row['Team']
        opponent_seed = row['Opponent Seed']
        opponent = row['Opponent']
        
        # Check if this is a play-in team reference (contains '/')
        if '/' in opponent:
            # Skip these as they're uncertain until First Four games are played
            continue
        
        # Otherwise, add as a normal matchup
        matchup = {
            'region': region,
            'team1': team,
            'team1_seed': team_seed,
            'team2': opponent,
            'team2_seed': opponent_seed
        }
        day_matchups.append(matchup)
    
    print(f"Processed {len(day_matchups)} matchups for prediction")
    return day_matchups

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
    print(f"WARNING: No match found for '{target_team}'")
    return target_team

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

def train_model(team_data, recent=False, n_samples=2500):
    """Train a model using provided team data"""
    if recent:
        print(f"Training model using data from {len(team_data['Year'].unique())} seasons...")
        model_type = "recent"
    else:
        print(f"Training model using only 2025 data...")
        model_type = "current"
    
    # Get all teams
    all_teams = team_data['TeamName'].unique()
    print(f"Total unique teams in {model_type} data: {len(all_teams)}")
    
    if recent:
        # Generate synthetic matchups for training, with weight to most recent year
        years = sorted(team_data['Year'].unique())
        year_weights = {year: 1 + 0.5 * (idx / len(years)) for idx, year in enumerate(years)}
        print(f"Year weights for sampling: {year_weights}")
        
        matchups = []
        for _ in range(n_samples):
            # Sample a year with bias toward more recent years
            year_probs = np.array(list(year_weights.values()))
            year_probs = year_probs / year_probs.sum()  # Normalize to sum to 1
            year = np.random.choice(years, p=year_probs)
            
            # Get teams from this year
            year_data = team_data[team_data['Year'] == year]
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
    else:
        # Generate synthetic matchups for the current season only
        matchups = []
        for _ in range(n_samples):
            # Get two random teams
            team1, team2 = np.random.choice(all_teams, 2, replace=False)
            
            # Create features for this matchup
            features = create_matchup_features(team1, team2, team_data)
            
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
    print(f"Training Random Forest model for {model_type} data...")
    model = RandomForestClassifier(
        n_estimators=250, 
        max_depth=15,
        min_samples_split=5,
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

def predict_matchups(matchups, current_data, recent_data, current_model, recent_model,
                    current_features, recent_features):
    """
    Make predictions for matchups using both current and historical data models
    """
    print("\nPredicting March Madness matchups for 3-20-2025...")
    
    predictions = []
    current_teams = current_data['TeamName'].unique()
    recent_teams = recent_data['TeamName'].unique()
    
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
        "Texas": "Texas",
        "Saint Mary's": "St. Mary's"
    }
    
    for matchup in matchups:
        region = matchup['region']
        team1_original = matchup['team1']
        team1_seed = matchup['team1_seed']
        team2_original = matchup['team2']
        team2_seed = matchup['team2_seed']
        
        print(f"\nAnalyzing {region} Region: #{team1_seed} {team1_original} vs #{team2_seed} {team2_original}")
        
        # Use manual mappings or find closest match for current year data
        team1_current = team_mappings.get(team1_original, team1_original)
        team2_current = team_mappings.get(team2_original, team2_original)
        
        if team1_current not in current_teams:
            team1_current = find_closest_team_name(team1_current, current_teams)
        
        if team2_current not in current_teams:
            team2_current = find_closest_team_name(team2_current, current_teams)
            
        print(f"Using current season team mappings: {team1_original} -> {team1_current}, {team2_original} -> {team2_current}")
        
        # Use manual mappings or find closest match for historical data
        team1_recent = team_mappings.get(team1_original, team1_original)
        team2_recent = team_mappings.get(team2_original, team2_original)
        
        if team1_recent not in recent_teams:
            team1_recent = find_closest_team_name(team1_recent, recent_teams)
        
        if team2_recent not in recent_teams:
            team2_recent = find_closest_team_name(team2_recent, recent_teams)
            
        print(f"Using historical data team mappings: {team1_original} -> {team1_recent}, {team2_original} -> {team2_recent}")
        
        # Get trend information
        team1_trend = get_team_trend(team1_recent, recent_data)
        team2_trend = get_team_trend(team2_recent, recent_data)
        
        print(f"{team1_original} trend: {team1_trend}")
        print(f"{team2_original} trend: {team2_trend}")
        
        # Current season prediction
        current_features_dict = create_matchup_features(team1_current, team2_current, current_data)
        recent_features_dict = None
        
        # Try to find teams in the same recent year, preferring the most recent
        if team1_recent in recent_teams and team2_recent in recent_teams:
            years = sorted(recent_data['Year'].unique(), reverse=True)
            best_match_year = None
            
            for year in years:
                year_data = recent_data[recent_data['Year'] == year]
                if team1_recent in year_data['TeamName'].unique() and team2_recent in year_data['TeamName'].unique():
                    best_match_year = year
                    print(f"Found both teams in {year} data - using this for historical prediction")
                    break
            
            # If not found together, use most recent year with aggregated data
            if best_match_year is None:
                best_match_year = max(years)
                print(f"Teams not found in same year. Using most recent ({best_match_year}) with aggregated data")
            
            # Get data for prediction
            year_data = recent_data[recent_data['Year'] == best_match_year]
            recent_features_dict = create_matchup_features(team1_recent, team2_recent, year_data)
        
        # Make predictions with both models
        current_prediction = None
        recent_prediction = None
        current_confidence = 0
        recent_confidence = 0
        
        # Current season model prediction
        if current_features_dict:
            # Prepare features for prediction
            pred_features = {col: current_features_dict.get(col, 0) for col in current_features}
            pred_df = pd.DataFrame([pred_features])
            
            # Make prediction
            prob = current_model.predict_proba(pred_df)[0]
            team1_win_prob = prob[1] if current_model.classes_[1] == 1 else prob[0]
            
            current_winner = team1_current if team1_win_prob > 0.5 else team2_current
            current_confidence = team1_win_prob if current_winner == team1_current else 1 - team1_win_prob
            
            # Map back to original team names
            current_prediction = team1_original if current_winner == team1_current else team2_original
            
            print(f"Current season model: {current_prediction} wins with {current_confidence:.2%} confidence")
        
        # Recent seasons model prediction
        if recent_features_dict:
            # Prepare features for prediction
            pred_features = {col: recent_features_dict.get(col, 0) for col in recent_features}
            pred_df = pd.DataFrame([pred_features])
            
            # Make prediction
            prob = recent_model.predict_proba(pred_df)[0]
            team1_win_prob = prob[1] if recent_model.classes_[1] == 1 else prob[0]
            
            recent_winner = team1_recent if team1_win_prob > 0.5 else team2_recent
            recent_confidence = team1_win_prob if recent_winner == team1_recent else 1 - team1_win_prob
            
            # Map back to original team names
            recent_prediction = team1_original if recent_winner == team1_recent else team2_original
            
            print(f"Historical model: {recent_prediction} wins with {recent_confidence:.2%} confidence")
        
        # Combine predictions
        if current_prediction and recent_prediction:
            # If models agree, higher confidence
            if current_prediction == recent_prediction:
                winner = current_prediction
                confidence = (current_confidence * 0.6) + (recent_confidence * 0.4)
                model_agreement = "Models agree"
            else:
                # Models disagree - weight current season more heavily
                if current_confidence >= recent_confidence:
                    winner = current_prediction
                    confidence = current_confidence * 0.8
                    model_agreement = "Current season model preferred (higher confidence)"
                else:
                    winner = recent_prediction
                    confidence = recent_confidence * 0.7
                    model_agreement = "Historical model preferred (higher confidence)"
        elif current_prediction:
            winner = current_prediction
            confidence = current_confidence
            model_agreement = "Only current season model available"
        elif recent_prediction:
            winner = recent_prediction
            confidence = recent_confidence
            model_agreement = "Only historical model available"
        else:
            # No model predictions - use seeding
            winner = team1_original if team1_seed < team2_seed else team2_original
            confidence = 0.51 + (abs(team1_seed - team2_seed) * 0.02)  # Higher seed difference = higher confidence
            model_agreement = "No model predictions - using seeding"
            
        # Save prediction
        prediction = {
            'region': region,
            'team1': team1_original,
            'team1_seed': team1_seed,
            'team2': team2_original,
            'team2_seed': team2_seed,
            'winner': winner,
            'confidence': confidence,
            'model_agreement': model_agreement,
            'current_prediction': current_prediction,
            'current_confidence': current_confidence if current_prediction else None,
            'recent_prediction': recent_prediction,
            'recent_confidence': recent_confidence if recent_prediction else None,
            'team1_trend': team1_trend,
            'team2_trend': team2_trend,
            'current_features': current_features_dict,
            'recent_features': recent_features_dict
        }
        
        predictions.append(prediction)
        print(f"FINAL PREDICTION: {winner} wins with {confidence:.2%} confidence ({model_agreement})")
        
        # Show key statistical differences from current season
        if current_features_dict:
            print("\nKey statistical differences (Current Season):")
            for metric, name in [
                ('AdjEM_DIFF', 'Adjusted Efficiency Margin'),
                ('AdjOE_DIFF', 'Offensive Efficiency'),
                ('AdjDE_DIFF', 'Defensive Efficiency'),
                ('Tempo_DIFF', 'Pace of Play')
            ]:
                if metric in current_features_dict:
                    diff = current_features_dict[metric]
                    better_team = team1_original if diff > 0 else team2_original
                    worse_team = team2_original if diff > 0 else team1_original
                    print(f"• {better_team} has better {name} than {worse_team} by {abs(diff):.2f} points")
    
    return predictions

def format_predictions_text(predictions):
    """Format predictions as text"""
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 15 + "MARCH MADNESS PREDICTIONS FOR MARCH 20, 2025")
    lines.append("=" * 80)
    
    # Group by region
    region_groups = {}
    for pred in predictions:
        region = pred['region']
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append(pred)
    
    # Process each region
    for region in sorted(region_groups.keys()):
        lines.append(f"\n{region.upper()} REGION:")
        lines.append("-" * 80)
        
        # Sort by seed
        sorted_preds = sorted(region_groups[region], key=lambda x: x['team1_seed'])
        
        for pred in sorted_preds:
            team1 = pred['team1']
            team2 = pred['team2']
            team1_seed = pred['team1_seed']
            team2_seed = pred['team2_seed']
            winner = pred['winner']
            confidence = pred['confidence']
            agreement = pred['model_agreement']
            
            lines.append(f"Matchup: #{team1_seed} {team1} vs #{team2_seed} {team2}")
            lines.append(f"Predicted Winner: {winner.upper()} with {confidence:.1%} confidence")
            lines.append(f"Model Agreement: {agreement}")
            
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
            if 'current_features' in pred and pred['current_features']:
                lines.append("\nStatistical Analysis (Current Season):")
                
                features = pred['current_features']
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
            
            # Add separator between matchups
            lines.append("\n" + "-" * 40 + "\n")
    
    lines.append("=" * 80)
    lines.append("MODEL INFORMATION:")
    lines.append("• Predictions use both current season (2025) and historical (2022-2025) data")
    lines.append("• Current season model weighted more heavily (60% vs 40%)")
    lines.append("• Historical model includes team performance trends across seasons")
    lines.append("• Random Forest classifier with top features: Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("• For teams with conflicting model predictions, higher confidence model is preferred")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def save_predictions(predictions_text, output_file):
    """Save predictions to text file"""
    with open(output_file, 'w') as f:
        f.write(predictions_text)
    print(f"Predictions saved to {output_file}")

def main():
    """Main function to predict March Madness matchups"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./march_madness_predictions_3_20_2025.txt"
    target_date = "2025-03-20"
    
    # Load matchups for target date
    march_20_matchups = load_matchups(matchups_file, target_date)
    
    if not march_20_matchups:
        print("No matchups found for 3-20-2025. Exiting.")
        return
    
    try:
        # Load 2025 data only for current season model
        current_data = manual_merge_basketball_data(year=2025)
        
        # Load data from recent years for historical model
        recent_years = [2022, 2023, 2024, 2025]
        historical_data = load_recent_data(recent_years)
        
        # Train models
        print("\n--- Training Current Season Model ---")
        current_model, current_features = train_model(current_data, recent=False)
        
        print("\n--- Training Historical Model ---")
        historical_model, historical_features = train_model(historical_data, recent=True)
        
        # Make predictions using both models
        predictions = predict_matchups(
            march_20_matchups, 
            current_data, 
            historical_data,
            current_model, 
            historical_model,
            current_features, 
            historical_features
        )
        
        # Format and save predictions
        predictions_text = format_predictions_text(predictions)
        save_predictions(predictions_text, output_file)
        
        print(f"\nMarch Madness predictions for {target_date} complete and saved to {output_file}!")
        
    except Exception as e:
        print(f"Error during prediction process: {str(e)}")

if __name__ == "__main__":
    main()