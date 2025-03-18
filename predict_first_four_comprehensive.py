"""
Predict First Four games using comprehensive historical data (2003-2025)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import traceback
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

def load_comprehensive_training_data(years):
    """Load and combine training data from multiple years"""
    print(f"Loading comprehensive training data from {len(years)} years: {min(years)}-{max(years)}")
    
    all_team_data = []
    valid_years = []
    
    # Create more efficient loading with batches for older years
    for year in years:
        try:
            # More verbose output for recent years
            if year >= 2018:
                print(f"Loading {year} data...")
            else:
                print(f"Loading {year}...", end='', flush=True)
                
            team_data = manual_merge_basketball_data(year=year)
            team_data['Year'] = year  # Add year column for reference
            all_team_data.append(team_data)
            valid_years.append(year)
            
            if year >= 2018:
                print(f"Added {len(team_data)} team records from {year}")
            else:
                print(f" Added {len(team_data)} teams", flush=True)
                
        except Exception as e:
            print(f"Error loading data for {year}: {e}")
    
    if not all_team_data:
        raise ValueError("No valid data loaded from any year")
    
    # Combine all years' data
    combined_data = pd.concat(all_team_data, ignore_index=True)
    print(f"\nSuccessfully loaded data from {len(valid_years)} years: {min(valid_years)}-{max(valid_years)}")
    print(f"Combined data contains {len(combined_data)} team records")
    print(f"Number of years with data: {len(combined_data['Year'].unique())}")
    
    return combined_data

def find_closest_team_name(target_team, all_teams):
    """Find the closest matching team name in the dataset"""
    if target_team in all_teams:
        return target_team
    
    # Common team name variations
    name_variations = {
        "St.": "Saint",
        "Saint": "St.",
        "NC": "North Carolina",
        "UNC": "North Carolina",
        "USC": "Southern California",
        "LSU": "Louisiana State",
        "SMU": "Southern Methodist",
        "UCF": "Central Florida",
        "UTEP": "Texas El Paso",
        "UNLV": "Nevada Las Vegas",
        "BYU": "Brigham Young",
        "VCU": "Virginia Commonwealth",
        "UAB": "Alabama Birmingham",
        "ETSU": "East Tennessee St.",
    }
    
    # Try common variations
    variations = [target_team]
    
    # Add variations with substitutions
    for old, new in name_variations.items():
        if old in target_team:
            variations.append(target_team.replace(old, new))
    
    # Add more general variations
    variations.extend([
        target_team.replace("State", "St."),
        target_team.replace("St.", "State"),
        target_team + " University",
        target_team.replace(" University", ""),
        target_team.split()[0]  # First word only
    ])
    
    # Try exact matches first
    for team in all_teams:
        if team.lower() == target_team.lower():
            return team
    
    # Try partial matches
    for variation in variations:
        for team in all_teams:
            if variation.lower() in team.lower() or team.lower() in variation.lower():
                print(f"Found approximate match: '{target_team}' -> '{team}'")
                return team
    
    # No close match found
    print(f"No match found for '{target_team}'")
    return target_team

def train_comprehensive_model(combined_data, n_samples=4000):
    """Train an advanced prediction model using comprehensive historical data"""
    print(f"Training comprehensive prediction model on data spanning {len(combined_data['Year'].unique())} years...")
    
    # Get all teams
    all_teams = combined_data['TeamName'].unique()
    print(f"Total unique teams in combined data: {len(all_teams)}")
    
    # Generate synthetic matchups for training, stratified by year
    all_years = sorted(combined_data['Year'].unique())
    recent_years = [y for y in all_years if y >= 2015]  # More weight to recent years
    older_years = [y for y in all_years if y < 2015]
    
    # Allocate samples: 70% from recent years, 30% from older years
    recent_samples = int(n_samples * 0.7)
    older_samples = n_samples - recent_samples
    
    print(f"Generating {recent_samples} samples from recent years (2015+)")
    print(f"Generating {older_samples} samples from older years (pre-2015)")
    
    matchups = []
    
    # Generate matchups from recent years (2015+)
    for _ in range(recent_samples):
        # Sample a random recent year
        year = np.random.choice(recent_years)
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
    
    # Generate matchups from older years
    for _ in range(older_samples):
        # Sample a random older year
        year = np.random.choice(older_years)
        year_data = combined_data[combined_data['Year'] == year]
        year_teams = year_data['TeamName'].unique()
        
        if len(year_teams) < 2:
            continue
            
        team1, team2 = np.random.choice(year_teams, 2, replace=False)
        features = create_matchup_features(team1, team2, year_data)
        
        if features:
            # Generate synthetic result based on team metrics
            if 'RankAdjEM_DIFF' in features:
                prob_team1_wins = 1 / (1 + np.exp(features['RankAdjEM_DIFF'] / 10))
            else:
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
    
    # Train an advanced model
    print("Training comprehensive Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=500,    # More trees for better accuracy
        max_depth=25,        # Deeper trees to capture complex patterns
        min_samples_split=4,
        class_weight='balanced',  # Address potential class imbalance
        random_state=42,
        n_jobs=-1            # Use all available cores
    )
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(15)
    
    print("\nTop 15 important features:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")
    
    return model, X.columns

def get_team_historical_stats(team_name, combined_data):
    """Get historical statistics for a team"""
    team_data = combined_data[combined_data['TeamName'] == team_name]
    
    if team_data.empty:
        return None
    
    # Get data from most recent available year
    most_recent_year = team_data['Year'].max()
    recent_data = team_data[team_data['Year'] == most_recent_year]
    
    # Calculate average stats across all available years
    historical_avg = {
        'Years': sorted(team_data['Year'].unique()),
        'Total_Years': len(team_data['Year'].unique()),
        'Most_Recent': most_recent_year,
        'Avg_AdjEM': team_data['AdjEM'].mean(),
        'Recent_AdjEM': recent_data['AdjEM'].values[0] if not recent_data.empty else None,
        'Avg_AdjOE': team_data['AdjOE'].mean(),
        'Avg_AdjDE': team_data['AdjDE'].mean(),
        'Best_Rank': team_data['RankAdjEM'].min(),
        'Avg_Rank': team_data['RankAdjEM'].mean()
    }
    
    return historical_avg

def predict_first_four_games(first_four_matchups, combined_data, model, feature_columns):
    """Make predictions for First Four games with historical context"""
    print("\nPredicting First Four winners with historical context...")
    
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
    
    # Loop through all First Four matchups
    for matchup in first_four_matchups:
        region = matchup['region']
        seed = matchup['seed']
        team1_original = matchup['team1']
        team2_original = matchup['team2']
        
        # Fix names that include "First Four:" prefix
        if "First Four:" in team1_original:
            team1_original = team1_original.replace("First Four:", "").strip()
        if "First Four:" in team2_original:
            team2_original = team2_original.replace("First Four:", "").strip()
        
        print(f"\n=== Analyzing {region} Region (#{seed}): {team1_original} vs {team2_original} ===")
        
        # Use manual mappings if available, otherwise find closest match
        team1 = team_mappings.get(team1_original, team1_original)
        team2 = team_mappings.get(team2_original, team2_original)
        
        # If still not in the dataset, find closest match
        if team1 not in all_teams:
            team1 = find_closest_team_name(team1, all_teams)
        
        if team2 not in all_teams:
            team2 = find_closest_team_name(team2, all_teams)
        
        print(f"Using team mappings: {team1_original} -> {team1}, {team2_original} -> {team2}")
        
        # Get historical statistics for both teams
        team1_history = get_team_historical_stats(team1, combined_data)
        team2_history = get_team_historical_stats(team2, combined_data)
        
        if team1_history:
            print(f"\n{team1_original} historical stats:")
            print(f"  Years in dataset: {team1_history['Total_Years']} (most recent: {team1_history['Most_Recent']})")
            print(f"  Average efficiency margin: {team1_history['Avg_AdjEM']:.2f}")
            print(f"  Best historical rank: {int(team1_history['Best_Rank'])}")
        
        if team2_history:
            print(f"\n{team2_original} historical stats:")
            print(f"  Years in dataset: {team2_history['Total_Years']} (most recent: {team2_history['Most_Recent']})")
            print(f"  Average efficiency margin: {team2_history['Avg_AdjEM']:.2f}")
            print(f"  Best historical rank: {int(team2_history['Best_Rank'])}")
        
        # Determine the best year to use for prediction
        best_match_year = None
        recent_years = sorted(combined_data['Year'].unique(), reverse=True)
        
        # First try to find a year with both teams
        for year in recent_years:
            year_data = combined_data[combined_data['Year'] == year]
            if team1 in year_data['TeamName'].unique() and team2 in year_data['TeamName'].unique():
                best_match_year = year
                print(f"\nFound both teams in {year} data - using this for prediction")
                break
        
        # If not found together, use most recent year with most data
        if best_match_year is None:
            best_match_year = max(recent_years)
            print(f"\nTeams not found in same year. Using most recent data: {best_match_year}")
            
            # Note which teams we have recent data for
            team1_years = combined_data[combined_data['TeamName'] == team1]['Year'].unique()
            team2_years = combined_data[combined_data['TeamName'] == team2]['Year'].unique()
            
            if len(team1_years) > 0:
                print(f"  Found {team1} in {len(team1_years)} years, most recent: {max(team1_years)}")
            if len(team2_years) > 0:
                print(f"  Found {team2} in {len(team2_years)} years, most recent: {max(team2_years)}")
        
        # Get matchup features using the best available data
        features = None
        
        # Try different approaches to get features
        approaches = [
            # Approach 1: Use data from the best match year
            (f"Using {best_match_year} data", 
             combined_data[combined_data['Year'] == best_match_year]),
            
            # Approach 2: Use data from most recent 3 years
            (f"Using data from most recent 3 years", 
             combined_data[combined_data['Year'] >= max(recent_years) - 2]),
            
            # Approach 3: Use all data
            (f"Using all historical data", 
             combined_data)
        ]
        
        for approach_desc, approach_data in approaches:
            if features is not None:
                break
                
            try:
                print(f"Attempt: {approach_desc}")
                features = create_matchup_features(team1, team2, approach_data)
                if features:
                    print(f"Success! Created matchup features using: {approach_desc}")
                    break
            except Exception as e:
                print(f"Error with {approach_desc}: {str(e)}")
        
        # If we have features, make a prediction
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
            
            # Save prediction
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner_original,
                'confidence': winner_prob,
                'features': features
            })
            
            print(f"\nPrediction: {winner_original} wins with {winner_prob:.2%} confidence")
            
            # Show key statistical differences
            key_metrics = [
                ('AdjEM_DIFF', 'Adjusted Efficiency Margin'),
                ('AdjOE_DIFF', 'Offensive Efficiency'),
                ('AdjDE_DIFF', 'Defensive Efficiency'),
                ('Tempo_DIFF', 'Tempo'),
                ('Experience_DIFF', 'Team Experience'),
                ('AvgHeight_DIFF', 'Average Height')
            ]
            
            print("\nKey statistical differences:")
            for metric_code, metric_name in key_metrics:
                if metric_code in features:
                    diff = features[metric_code]
                    better_team = team1_original if diff > 0 else team2_original
                    worse_team = team2_original if diff > 0 else team1_original
                    print(f"  {better_team} has better {metric_name} than {worse_team} by {abs(diff):.2f} points")
                
            # Add seed performance data
            seed_int = int(seed)
            if seed_int >= 14:
                print(f"\nHistorical context: #{seed} seeds rarely advance beyond the first round (< 5% chance)")
            elif seed_int >= 11:
                print(f"\nHistorical context: #{seed} seeds have pulled off upsets and occasionally make Sweet 16 runs")
            else:
                print(f"\nHistorical context: #{seed} seeds typically perform well in the tournament")
                
        else:
            print("\nCould not create features - making educated guess based on historical data")
            
            # Make an educated guess based on historical data if available
            historical_edge = None
            confidence = 0.51  # Default for true toss-up
            
            if team1_history and team2_history:
                # Compare historical performance
                team1_rating = team1_history['Avg_AdjEM']
                team2_rating = team2_history['Avg_AdjEM']
                
                if abs(team1_rating - team2_rating) > 5:
                    # Clear historical advantage
                    winner = team1_original if team1_rating > team2_rating else team2_original
                    confidence = 0.65
                    historical_edge = "clear historical advantage in efficiency metrics"
                elif abs(team1_rating - team2_rating) > 2:
                    # Slight historical advantage
                    winner = team1_original if team1_rating > team2_rating else team2_original
                    confidence = 0.58
                    historical_edge = "slight historical advantage in efficiency metrics"
                else:
                    # Very close historically
                    better_rank_team = team1_original if team1_history['Best_Rank'] < team2_history['Best_Rank'] else team2_original
                    winner = better_rank_team
                    confidence = 0.53
                    historical_edge = "marginally better historical peak rank"
            else:
                # Limited historical data - use seed and team reputation
                major_teams = ["North Carolina", "Duke", "Kentucky", "Kansas", "UCLA", "Indiana", 
                               "Michigan State", "Ohio State", "Connecticut", "Louisville", "Syracuse"]
                
                # Check if either team is a major program
                team1_major = any(team in team1_original for team in major_teams)
                team2_major = any(team in team2_original for team in major_teams)
                
                if team1_major and not team2_major:
                    winner = team1_original
                    confidence = 0.60
                    historical_edge = "program prestige and past tournament success"
                elif team2_major and not team1_major:
                    winner = team2_original
                    confidence = 0.60
                    historical_edge = "program prestige and past tournament success"
                else:
                    # True toss-up
                    winner = team1_original  # Default to team1 in true toss-ups
                    confidence = 0.51
                    historical_edge = "minimal data, statistical tie"
            
            # Save the prediction
            predictions.append({
                'region': region,
                'seed': seed,
                'team1': team1_original,
                'team2': team2_original,
                'winner': winner,
                'confidence': confidence,
                'reason': historical_edge
            })
            
            print(f"Educated guess: {winner} advances with {confidence:.2%} confidence based on {historical_edge}")
    
    return predictions

def format_comprehensive_predictions(predictions):
    """Format the First Four predictions with comprehensive analysis"""
    lines = []
    lines.append("=" * 90)
    lines.append(" " * 20 + "MARCH MADNESS FIRST FOUR COMPREHENSIVE PREDICTIONS")
    lines.append(" " * 25 + "(Using Historical Data from 2003-2025)")
    lines.append("=" * 90)
    
    # Group predictions by seed
    seed_groups = {}
    for pred in predictions:
        seed = pred['seed']
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(pred)
    
    # Order by seed
    for seed in sorted(seed_groups.keys()):
        seed_preds = seed_groups[seed]
        
        # Get the region for this section
        region = seed_preds[0]['region']
        lines.append(f"\n{region.upper()} REGION - #{seed} SEED PLAY-IN GAMES:")
        lines.append("-" * 90)
        
        for i, pred in enumerate(seed_preds):
            team1 = pred['team1']
            team2 = pred['team2']
            winner = pred['winner']
            confidence = pred['confidence']
            
            lines.append(f"Matchup: {team1} vs {team2}")
            lines.append(f"Predicted Winner: {winner.upper()} with {confidence:.1%} confidence")
            
            # Add confidence indicator
            if confidence >= 0.85:
                lines.append("Confidence: VERY HIGH - Strong statistical advantage detected")
            elif confidence >= 0.70:
                lines.append("Confidence: HIGH - Clear statistical advantage")
            elif confidence >= 0.60:
                lines.append("Confidence: MEDIUM - Moderate advantage detected")
            elif confidence >= 0.55:
                lines.append("Confidence: LOW - Slight edge detected")
            else:
                lines.append("Confidence: TOSS-UP - Teams are very evenly matched")
            
            # Add key metrics if available
            if 'features' in pred:
                features = pred['features']
                lines.append("\nKey statistical advantages:")
                
                metrics = [
                    ('AdjEM_DIFF', 'Adjusted Efficiency Margin'),
                    ('AdjOE_DIFF', 'Offensive Efficiency'),
                    ('AdjDE_DIFF', 'Defensive Efficiency'),
                    ('Tempo_DIFF', 'Pace of Play')
                ]
                
                for code, name in metrics:
                    if code in features and abs(features[code]) > 0.5:
                        diff = features[code]
                        better_team = team1 if diff > 0 else team2
                        worse_team = team2 if diff > 0 else team1
                        lines.append(f"• {better_team} has better {name} than {worse_team} by {abs(diff):.2f} points")
            elif 'reason' in pred:
                lines.append(f"Basis for prediction: {pred['reason']}")
            
            # Add historical context based on seed
            seed_int = int(seed)
            if seed_int >= 16:
                lines.append("\nHistorical Context: Only one #16 seed has ever defeated a #1 seed in tournament history")
                lines.append("                   (UMBC over Virginia in 2018)")
            elif seed_int >= 11:
                lines.append("\nHistorical Context: #11 seeds have made surprising tournament runs in recent years")
                lines.append("                   (UCLA Final Four 2021, Loyola Chicago Final Four 2018)")
            
            # Add separator between matchups in the same seed group
            if i < len(seed_preds) - 1:
                lines.append("\n" + "-" * 50)
    
    lines.append("\n" + "=" * 90)
    lines.append("ANALYSIS METHODOLOGY:")
    lines.append("• Predictions based on Random Forest model trained on 20+ years of basketball statistics")
    lines.append("• Model incorporates efficiency metrics, tempo, experience, and other team statistics")
    lines.append("• Historical team performance patterns weighted in prediction algorithm")
    lines.append("• Top predictive metrics: Adjusted Efficiency Margin, Team Rankings, Offensive/Defensive Efficiency")
    lines.append("=" * 90)
    
    return "\n".join(lines)

def save_predictions(predictions_text, output_file):
    """Save predictions to text file"""
    with open(output_file, 'w') as f:
        f.write(predictions_text)
    print(f"Comprehensive predictions saved to {output_file}")

def main():
    """Main function to predict First Four games using comprehensive historical data"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./first_four_comprehensive_predictions.txt"
    
    # Load First Four matchups
    first_four_matchups = load_first_four_matchups(matchups_file)
    
    if not first_four_matchups:
        print("No First Four games found. Exiting.")
        return
    
    try:
        # Load comprehensive training data from 2003-2025
        years = list(range(2003, 2026))  # 2003 through 2025
        combined_data = load_comprehensive_training_data(years)
        
        # Train comprehensive model on combined data
        model, feature_columns = train_comprehensive_model(combined_data)
        
        # Make predictions for First Four games
        predictions = predict_first_four_games(first_four_matchups, combined_data, model, feature_columns)
        
        # Format and save comprehensive predictions
        predictions_text = format_comprehensive_predictions(predictions)
        save_predictions(predictions_text, output_file)
        
        print("\nComprehensive First Four predictions complete!")
        
    except Exception as e:
        print(f"Error during prediction process: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()