"""
Predict First Four games using an ensemble approach combining multiple timeframe models
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

def load_data_for_timeframe(timeframe):
    """Load data for a specific timeframe"""
    print(f"Loading data for timeframe: {timeframe}")
    
    if timeframe == "current_only":
        # Load only 2025 data
        years = [2025]
    elif timeframe == "recent":
        # Load 2023-2025 data
        years = [2023, 2024, 2025]
    elif timeframe == "extended":
        # Load 2018-2025 data
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    elif timeframe == "comprehensive":
        # Load all available data (2003-2025)
        years = list(range(2003, 2026))
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    print(f"Years in timeframe: {years}")
    
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
    
    return combined_data, years

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

def train_model_for_timeframe(combined_data, timeframe, n_samples=2500):
    """Train a model for a specific timeframe"""
    print(f"Training model for timeframe: {timeframe}")
    
    # Get all teams
    all_teams = combined_data['TeamName'].unique()
    print(f"Total unique teams in {timeframe} data: {len(all_teams)}")
    
    # Configure year weights based on timeframe
    years = sorted(combined_data['Year'].unique())
    
    if timeframe == "current_only":
        # Current year only, no weighting needed
        year_weights = {year: 1.0 for year in years}
    elif timeframe == "recent":
        # Recent years: 2023-2025, with increasing weights
        year_weights = {year: 1 + 0.5 * (idx / len(years)) for idx, year in enumerate(years)}
    elif timeframe == "extended":
        # Extended history: 2018-2025, with slightly increasing weights
        year_weights = {year: 1 + 0.2 * (idx / len(years)) for idx, year in enumerate(years)}
    elif timeframe == "comprehensive":
        # Comprehensive: 2003-2025, with minimal weighting
        year_weights = {year: 1 + 0.1 * (idx / len(years)) for idx, year in enumerate(years)}
    
    print(f"Year weights for sampling: {year_weights}")
    
    # Generate synthetic matchups for training
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
    
    # Train model with hyperparameters based on timeframe
    print(f"Training Random Forest model for {timeframe}...")
    
    if timeframe == "current_only":
        # Current year model - needs to be more responsive to limited data
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )
    elif timeframe == "recent":
        # Recent years model - balanced approach
        model = RandomForestClassifier(
            n_estimators=250, 
            max_depth=15,
            min_samples_split=4,
            random_state=42
        )
    elif timeframe == "extended":
        # Extended history model - more trees for more data
        model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
    elif timeframe == "comprehensive":
        # Comprehensive model - deeper trees for complex patterns
        model = RandomForestClassifier(
            n_estimators=400, 
            max_depth=25,
            min_samples_split=6,
            random_state=42
        )
    
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(10)
    
    print(f"\nTop 10 important features for {timeframe} model:")
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

def predict_with_model(matchup, combined_data, model, feature_columns, timeframe, all_teams):
    """Make a prediction using a specific model"""
    region = matchup['region']
    seed = matchup['seed']
    team1_original = matchup['team1']
    team2_original = matchup['team2']
    
    # Fix potential prefix issues
    if "First Four:" in team1_original:
        team1_original = team1_original.replace("First Four:", "").strip()
    if "First Four:" in team2_original:
        team2_original = team2_original.replace("First Four:", "").strip()
    
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
    
    # Use manual mappings if available, otherwise find closest match
    team1 = team_mappings.get(team1_original, team1_original)
    team2 = team_mappings.get(team2_original, team2_original)
    
    # If still not in the dataset, find closest match
    if team1 not in all_teams:
        team1 = find_closest_team_name(team1, all_teams)
    
    if team2 not in all_teams:
        team2 = find_closest_team_name(team2, all_teams)
    
    # Try to find teams in the same year, preferring the most recent
    years = sorted(combined_data['Year'].unique(), reverse=True)
    best_match_year = None
    
    for year in years:
        year_data = combined_data[combined_data['Year'] == year]
        if team1 in year_data['TeamName'].unique() and team2 in year_data['TeamName'].unique():
            best_match_year = year
            break
    
    # If not found together, use most recent year with aggregated data
    if best_match_year is None:
        best_match_year = max(years)
    
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
        
        return {
            'winner': winner_original,
            'confidence': winner_prob,
            'features': features,
            'year_used': best_match_year
        }
    else:
        confidence = 0.51  # Default for toss-up
        
        # If recent timeframe and we have trend data, use that to bias prediction
        if timeframe == "recent":
            team1_trend = get_team_trend(team1, combined_data)
            team2_trend = get_team_trend(team2, combined_data)
            
            if team1_trend in ["Strong upward trend", "Slight improvement"] and team2_trend not in ["Strong upward trend"]:
                winner = team1_original
                confidence = 0.58
            elif team2_trend in ["Strong upward trend", "Slight improvement"] and team1_trend not in ["Strong upward trend"]:
                winner = team2_original
                confidence = 0.58
            else:
                # Default to alphabetical order as tiebreaker
                winner = team1_original if team1 < team2 else team2_original
        else:
            # For other timeframes, just use alphabetical
            winner = team1_original if team1 < team2 else team2_original
        
        return {
            'winner': winner,
            'confidence': confidence,
            'features': None,
            'year_used': None
        }

def ensemble_predict(first_four_matchups, models_data):
    """Predict using an ensemble of models"""
    print("\nPredicting First Four games using ensemble approach...")
    
    predictions = []
    
    # Define model weights - how much we trust each model
    model_weights = {
        "current_only": 0.25,
        "recent": 0.45,  # Highest weight to recent model with trend analysis
        "extended": 0.20,
        "comprehensive": 0.10
    }
    
    for matchup in first_four_matchups:
        region = matchup['region']
        seed = matchup['seed']
        team1 = matchup['team1']
        team2 = matchup['team2']
        
        # Strip "First Four:" prefix if present
        if "First Four:" in team1:
            team1 = team1.replace("First Four:", "").strip()
        if "First Four:" in team2:
            team2 = team2.replace("First Four:", "").strip()
            
        matchup['team1'] = team1
        matchup['team2'] = team2
        
        print(f"\nAnalyzing {region} Region (#{seed}): {team1} vs {team2}")
        
        # Collect predictions from each model
        model_predictions = {}
        for timeframe, (model, features, data, all_teams) in models_data.items():
            prediction = predict_with_model(matchup, data, model, features, timeframe, all_teams)
            model_predictions[timeframe] = prediction
            print(f"{timeframe.title()} model predicts: {prediction['winner']} with {prediction['confidence']:.2%} confidence")
        
        # Calculate weighted votes
        team_votes = {team1: 0, team2: 0}
        
        for timeframe, prediction in model_predictions.items():
            winner = prediction['winner']
            confidence = prediction['confidence']
            weight = model_weights[timeframe]
            
            # Check which team the model predicted as winner
            if winner == team1:
                team_votes[team1] += weight * confidence
                team_votes[team2] += weight * (1 - confidence)  # Add the complement for the other team
            else:
                team_votes[team2] += weight * confidence
                team_votes[team1] += weight * (1 - confidence)  # Add the complement for the other team
        
        # Normalize vote totals
        total_votes = sum(model_weights.values())  # Should sum to 1.0
        team1_ensemble_prob = team_votes[team1] / total_votes
        team2_ensemble_prob = team_votes[team2] / total_votes
        
        # Determine ensemble winner
        ensemble_winner = team1 if team1_ensemble_prob > team2_ensemble_prob else team2
        ensemble_confidence = max(team1_ensemble_prob, team2_ensemble_prob)
        
        # Get trend data from the recent model data
        recent_data = models_data["recent"][2]
        team1_mapped = find_closest_team_name(team1, recent_data['TeamName'].unique())
        team2_mapped = find_closest_team_name(team2, recent_data['TeamName'].unique())
        team1_trend = get_team_trend(team1_mapped, recent_data)
        team2_trend = get_team_trend(team2_mapped, recent_data)
        
        # Get key features from most recent data
        for timeframe in ["current_only", "recent"]:
            if timeframe in model_predictions and model_predictions[timeframe].get('features'):
                features = model_predictions[timeframe]['features']
                break
        else:
            # Fallback to any model with features
            for prediction in model_predictions.values():
                if prediction.get('features'):
                    features = prediction['features']
                    break
            else:
                features = None
        
        # Determine model consensus
        consensus_pct = sum(1 for p in model_predictions.values() if p['winner'] == ensemble_winner) / len(model_predictions) * 100
        
        # Store prediction with all information
        predictions.append({
            'region': region,
            'seed': seed,
            'team1': team1,
            'team2': team2,
            'winner': ensemble_winner,
            'confidence': ensemble_confidence,
            'model_consensus': consensus_pct,
            'model_predictions': model_predictions,
            'features': features,
            'team1_trend': team1_trend,
            'team2_trend': team2_trend
        })
        
        print(f"ENSEMBLE PREDICTION: {ensemble_winner} wins with {ensemble_confidence:.2%} confidence ({consensus_pct:.0f}% model consensus)")
        print(f"Team trends: {team1}: {team1_trend}, {team2}: {team2_trend}")
    
    return predictions

def format_ensemble_predictions(predictions):
    """Format ensemble predictions as text"""
    lines = []
    lines.append("=" * 90)
    lines.append(" " * 25 + "MARCH MADNESS FIRST FOUR ENSEMBLE PREDICTIONS")
    lines.append(" " * 30 + "(Weighted Model Ensemble)")
    lines.append("=" * 90)
    
    # Group by region, then seed
    region_groups = {}
    for pred in predictions:
        region = pred['region']
        if region not in region_groups:
            region_groups[region] = {}
        
        seed = pred['seed']
        if seed not in region_groups[region]:
            region_groups[region][seed] = []
        
        region_groups[region][seed].append(pred)
    
    # Order by region, then seed
    for region in sorted(region_groups.keys()):
        for seed in sorted(region_groups[region].keys()):
            lines.append(f"\n{region.upper()} REGION - #{seed} SEED PLAY-IN GAMES:")
            lines.append("-" * 90)
            
            for pred in region_groups[region][seed]:
                team1 = pred['team1']
                team2 = pred['team2']
                winner = pred['winner']
                confidence = pred['confidence']
                consensus = pred['model_consensus']
                
                lines.append(f"Matchup: {team1} vs {team2}")
                lines.append(f"Predicted Winner: {winner.upper()} with {confidence:.1%} confidence ({consensus:.0f}% model consensus)")
                
                # Add confidence indicator
                if confidence >= 0.7:
                    lines.append("Confidence: HIGH - Clear statistical advantage")
                elif confidence >= 0.6:
                    lines.append("Confidence: MEDIUM - Moderate advantage detected")
                elif confidence >= 0.55:
                    lines.append("Confidence: LOW - Slight edge detected")
                else:
                    lines.append("Confidence: TOSS-UP - Teams are very evenly matched")
                    
                # Add individual model predictions
                lines.append("\nIndividual Model Predictions:")
                model_results = []
                for model_name, prediction in pred['model_predictions'].items():
                    model_winner = prediction['winner']
                    model_conf = prediction['confidence']
                    agrees = "✓" if model_winner == winner else "✗"
                    model_results.append(f"• {model_name.title()}: {model_winner} ({model_conf:.1%}) {agrees}")
                
                lines.extend(sorted(model_results))
                
                # Add team trends
                if 'team1_trend' in pred and 'team2_trend' in pred:
                    lines.append(f"\nRecent Trends:")
                    lines.append(f"• {team1}: {pred['team1_trend']}")
                    lines.append(f"• {team2}: {pred['team2_trend']}")
                
                # Add key metrics if available
                if 'features' in pred and pred['features']:
                    lines.append("\nStatistical Analysis:")
                    
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
                
                # Add separator between matchups in same region/seed
                if pred != region_groups[region][seed][-1]:
                    lines.append("\n" + "-" * 40)
    
    lines.append("\n" + "=" * 90)
    lines.append("ENSEMBLE MODEL INFORMATION:")
    lines.append("• Weighted ensemble of four timeframe-based models:")
    lines.append("  - Current (2025): 25% weight - Most recent team performance")
    lines.append("  - Recent (2023-2025): 45% weight - Recent trends with performance trajectory analysis")
    lines.append("  - Extended (2018-2025): 20% weight - Medium-term historical performance") 
    lines.append("  - Comprehensive (2003-2025): 10% weight - Long-term historical patterns")
    lines.append("• Each model contributes a weighted vote proportional to its confidence")
    lines.append("• Model consensus indicates agreement across the four different timeframe models")
    lines.append("• Top predictive features: Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("=" * 90)
    
    return "\n".join(lines)

def main():
    """Main function to predict First Four games using ensemble approach"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./first_four_ensemble_predictions.txt"
    
    # Load First Four matchups
    first_four_matchups = load_first_four_matchups(matchups_file)
    
    if not first_four_matchups:
        print("No First Four games found. Exiting.")
        return
    
    try:
        # Define timeframes
        timeframes = ["current_only", "recent", "extended", "comprehensive"]
        
        # Load data and train models for each timeframe
        models_data = {}
        
        for timeframe in timeframes:
            # Load data for timeframe
            combined_data, years = load_data_for_timeframe(timeframe)
            
            # Train model
            model, feature_columns = train_model_for_timeframe(combined_data, timeframe)
            
            # Store model and data
            all_teams = combined_data['TeamName'].unique()
            models_data[timeframe] = (model, feature_columns, combined_data, all_teams)
        
        # Make ensemble predictions
        predictions = ensemble_predict(first_four_matchups, models_data)
        
        # Format and save predictions
        predictions_text = format_ensemble_predictions(predictions)
        
        with open(output_file, 'w') as f:
            f.write(predictions_text)
        
        print(f"\nEnsemble predictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error during prediction process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()