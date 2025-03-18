"""
Generate complete March Madness predictions based on data from 2021-2025.
"""
import pandas as pd
import numpy as np
from BasketballUtils import manual_merge_basketball_data, create_matchup_features
from sklearn.ensemble import RandomForestClassifier

def load_matchups(file_path):
    """Load matchups from matchups.csv"""
    matchups_df = pd.read_csv(file_path)
    
    day_matchups = []
    
    for _, row in matchups_df.iterrows():
        region = row['Region']
        team_seed = row['Seed']
        team = row['Team']
        opponent_seed = row['Opponent Seed']
        opponent = row['Opponent']
        
        matchup = {
            'region': region,
            'team1': team,
            'team1_seed': team_seed,
            'team2': opponent,
            'team2_seed': opponent_seed
        }
        day_matchups.append(matchup)
    
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
                return team
    
    # No close match found
    return target_team

def load_recent_data(years):
    """Load and combine recent years data"""
    all_team_data = []
    for year in years:
        try:
            team_data = manual_merge_basketball_data(year=year)
            team_data['Year'] = year  # Add year column for reference
            all_team_data.append(team_data)
        except Exception as e:
            print(f"Error loading data for {year}: {e}")
    
    if not all_team_data:
        raise ValueError("No valid data loaded from any year")
    
    # Combine all years' data
    combined_data = pd.concat(all_team_data, ignore_index=True)
    
    return combined_data

def train_model(team_data, n_samples=2500):
    """Train a model using provided team data"""
    # Get all teams
    all_teams = team_data['TeamName'].unique()
    
    # Generate synthetic matchups for training, with weight to most recent year
    years = sorted(team_data['Year'].unique())
    year_weights = {year: 1 + 0.5 * (idx / len(years)) for idx, year in enumerate(years)}
    
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
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    
    # Prepare features and target
    feature_cols = [col for col in matchups_df.columns if col not in ['TEAM1', 'TEAM2', 'RESULT']]
    X = matchups_df[feature_cols].select_dtypes(include=[np.number])
    y = matchups_df['RESULT']
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=250, 
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X, y)
    
    return model, X.columns

def predict_matchup(team1_info, team2_info, team_data, model, model_features, teams_map):
    """Predict the winner of a single matchup"""
    team1_original, team1_seed = team1_info
    team2_original, team2_seed = team2_info
    
    # Get mapped team names
    team1 = teams_map.get(team1_original, team1_original)
    team2 = teams_map.get(team2_original, team2_original)
    
    # Get the latest (2025) data for these teams if available
    team1_data = team_data[(team_data['TeamName'] == team1) & (team_data['Year'] == 2025)]
    team2_data = team_data[(team_data['TeamName'] == team2) & (team_data['Year'] == 2025)]
    
    # If not found in 2025, try the most recent available year
    if team1_data.empty or team2_data.empty:
        years = sorted(team_data['Year'].unique(), reverse=True)
        for year in years:
            if team1_data.empty:
                team1_year_data = team_data[(team_data['TeamName'] == team1) & (team_data['Year'] == year)]
                if not team1_year_data.empty:
                    team1_data = team1_year_data
            
            if team2_data.empty:
                team2_year_data = team_data[(team_data['TeamName'] == team2) & (team_data['Year'] == year)]
                if not team2_year_data.empty:
                    team2_data = team2_year_data
            
            if not team1_data.empty and not team2_data.empty:
                break
    
    # Create features for prediction
    features_dict = None
    if not team1_data.empty and not team2_data.empty:
        team1_row = team1_data.iloc[0]
        team2_row = team2_data.iloc[0]
        
        # Create a small dataframe with just these two teams
        matchup_df = pd.DataFrame([team1_row, team2_row])
        features_dict = create_matchup_features(team1, team2, matchup_df)
    
    if features_dict:
        # Prepare features for prediction
        pred_features = {col: features_dict.get(col, 0) for col in model_features}
        pred_df = pd.DataFrame([pred_features])
        
        # Make prediction
        prob = model.predict_proba(pred_df)[0]
        team1_win_prob = prob[1] if model.classes_[1] == 1 else prob[0]
        
        winner = team1_original if team1_win_prob > 0.5 else team2_original
        winner_seed = team1_seed if team1_win_prob > 0.5 else team2_seed
        confidence = team1_win_prob if winner == team1_original else 1 - team1_win_prob
    else:
        # Default to higher seed if we can't make a prediction
        winner = team1_original if int(team1_seed) < int(team2_seed) else team2_original
        winner_seed = team1_seed if int(team1_seed) < int(team2_seed) else team2_seed
        confidence = 0.5 + (abs(int(team1_seed) - int(team2_seed)) * 0.02)
    
    return winner, winner_seed, confidence

def simulate_tournament(first_round_matchups, team_data, model, model_features):
    """Simulate the entire tournament bracket"""
    all_predictions = []
    teams_in_dataset = team_data['TeamName'].unique()
    
    # Manual mappings for team names
    team_mappings = {
        "Alabama State": "Alabama St.",
        "St. Francis (PA)": "St. Francis PA",
        "Saint Francis": "St. Francis",
        "Mount St. Mary's": "Mount St. Mary's",
        "American": "American",
        "San Diego State": "San Diego St.",
        "North Carolina": "North Carolina",
        "Xavier": "Xavier",
        "Texas": "Texas",
        "Saint Mary's": "St. Mary's",
        "UNC Wilmington": "UNC-Wilmington",
        "UC San Diego": "UC-San Diego"
    }
    
    # Find closest matches for teams not in the dataset
    for team in set([m['team1'] for m in first_round_matchups] + [m['team2'] for m in first_round_matchups]):
        if team_mappings.get(team, team) not in teams_in_dataset:
            closest_match = find_closest_team_name(team, teams_in_dataset)
            if closest_match != team:
                team_mappings[team] = closest_match
    
    # Process play-in games first (First Four)
    first_four_games = [m for m in first_round_matchups if m['region'] == 'First Four']
    first_four_results = {}
    
    for matchup in first_four_games:
        team1, team1_seed = matchup['team1'], matchup['team1_seed']
        team2, team2_seed = matchup['team2'], matchup['team2_seed']
        
        winner, winner_seed, confidence = predict_matchup(
            (team1, team1_seed), 
            (team2, team2_seed), 
            team_data, model, model_features, team_mappings
        )
        
        # Store the winner by seed combination
        key = f"{team1_seed} {team1}/{team2_seed} {team2}"
        first_four_results[key] = (winner, winner_seed)
        
        # Add to all predictions
        all_predictions.append({
            'round': 'First Four',
            'region': 'First Four',
            'team1': team1,
            'team1_seed': team1_seed,
            'team2': team2,
            'team2_seed': team2_seed,
            'winner': winner,
            'confidence': confidence
        })
    
    # Process main bracket by region
    regions = sorted(set([m['region'] for m in first_round_matchups if m['region'] != 'First Four']))
    region_winners = {}
    
    for region in regions:
        region_games = [m for m in first_round_matchups if m['region'] == region]
        
        # First round games in the region (Round of 64)
        first_round_results = []
        for matchup in region_games:
            team1, team1_seed = matchup['team1'], matchup['team1_seed']
            team2, team2_seed = matchup['team2'], matchup['team2_seed']
            
            # Check if this is a play-in reference
            if '/' in team2:
                # Find the First Four result
                key = team2
                if key in first_four_results:
                    team2, team2_seed = first_four_results[key]
            
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            first_round_results.append((winner, winner_seed))
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Round of 64',
                'region': region,
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
        
        # Second round games (Round of 32)
        second_round_matchups = []
        for i in range(0, len(first_round_results), 2):
            if i + 1 < len(first_round_results):
                second_round_matchups.append((first_round_results[i], first_round_results[i+1]))
        
        second_round_results = []
        for (team1, team1_seed), (team2, team2_seed) in second_round_matchups:
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            second_round_results.append((winner, winner_seed))
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Round of 32',
                'region': region,
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
        
        # Sweet 16 games
        sweet_16_matchups = []
        for i in range(0, len(second_round_results), 2):
            if i + 1 < len(second_round_results):
                sweet_16_matchups.append((second_round_results[i], second_round_results[i+1]))
        
        sweet_16_results = []
        for (team1, team1_seed), (team2, team2_seed) in sweet_16_matchups:
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            sweet_16_results.append((winner, winner_seed))
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Sweet 16',
                'region': region,
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
        
        # Elite 8 game (Regional Final)
        if len(sweet_16_results) >= 2:
            (team1, team1_seed), (team2, team2_seed) = sweet_16_results[0], sweet_16_results[1]
            
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            # Store region winner
            region_winners[region] = (winner, winner_seed)
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Elite 8',
                'region': region,
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
    
    # Final Four - semifinals
    if len(region_winners) >= 4:
        # Match regions for semifinals
        regions_list = list(region_winners.keys())
        
        # First semifinal: South vs East
        if 'South' in region_winners and 'East' in region_winners:
            (team1, team1_seed), (team2, team2_seed) = region_winners['South'], region_winners['East']
            
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            semifinal1_winner = (winner, winner_seed)
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Final Four',
                'region': 'Semifinal 1',
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
        
        # Second semifinal: Midwest vs West
        if 'Midwest' in region_winners and 'West' in region_winners:
            (team1, team1_seed), (team2, team2_seed) = region_winners['Midwest'], region_winners['West']
            
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            semifinal2_winner = (winner, winner_seed)
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Final Four',
                'region': 'Semifinal 2',
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
        
        # Championship game
        if 'semifinal1_winner' in locals() and 'semifinal2_winner' in locals():
            (team1, team1_seed), (team2, team2_seed) = semifinal1_winner, semifinal2_winner
            
            winner, winner_seed, confidence = predict_matchup(
                (team1, team1_seed), 
                (team2, team2_seed), 
                team_data, model, model_features, team_mappings
            )
            
            # Add to all predictions
            all_predictions.append({
                'round': 'Championship',
                'region': 'Final',
                'team1': team1,
                'team1_seed': team1_seed,
                'team2': team2,
                'team2_seed': team2_seed,
                'winner': winner,
                'confidence': confidence
            })
    
    return all_predictions

def format_predictions_text(predictions):
    """Format predictions as simple text with only matchups and winners"""
    lines = []
    
    # Split predictions by round
    rounds = {
        'First Four': [],
        'Round of 64': [],
        'Round of 32': [],
        'Sweet 16': [],
        'Elite 8': [],
        'Final Four': [],
        'Championship': []
    }
    
    for pred in predictions:
        rounds[pred['round']].append(pred)
    
    # First Four
    if rounds['First Four']:
        lines.append("First Four:")
        for pred in rounds['First Four']:
            team1 = pred['team1']
            team2 = pred['team2']
            team1_seed = pred['team1_seed']
            team2_seed = pred['team2_seed']
            winner = pred['winner']
            lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
        lines.append("")
    
    # Round of 64 (First Round)
    lines.append("First Round:")
    for region in sorted(set([pred['region'] for pred in rounds['Round of 64']])):
        region_preds = [p for p in rounds['Round of 64'] if p['region'] == region]
        for pred in sorted(region_preds, key=lambda x: x['team1_seed']):
            team1 = pred['team1']
            team2 = pred['team2']
            team1_seed = pred['team1_seed']
            team2_seed = pred['team2_seed']
            winner = pred['winner']
            lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
    lines.append("")
    
    # Round of 32 (Second Round)
    lines.append("Second Round:")
    for pred in rounds['Round of 32']:
        team1 = pred['team1']
        team2 = pred['team2']
        team1_seed = pred['team1_seed']
        team2_seed = pred['team2_seed']
        winner = pred['winner']
        lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
    lines.append("")
    
    # Sweet 16
    lines.append("Sweet 16:")
    for pred in rounds['Sweet 16']:
        team1 = pred['team1']
        team2 = pred['team2']
        team1_seed = pred['team1_seed']
        team2_seed = pred['team2_seed']
        winner = pred['winner']
        lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
    lines.append("")
    
    # Elite 8
    lines.append("Elite 8:")
    for pred in rounds['Elite 8']:
        team1 = pred['team1']
        team2 = pred['team2']
        team1_seed = pred['team1_seed']
        team2_seed = pred['team2_seed']
        winner = pred['winner']
        lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
    lines.append("")
    
    # Final Four
    if rounds['Final Four']:
        lines.append("Final Four:")
        for pred in rounds['Final Four']:
            team1 = pred['team1']
            team2 = pred['team2']
            team1_seed = pred['team1_seed']
            team2_seed = pred['team2_seed']
            winner = pred['winner']
            lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
        lines.append("")
    
    # Championship
    if rounds['Championship']:
        lines.append("Championship:")
        for pred in rounds['Championship']:
            team1 = pred['team1']
            team2 = pred['team2']
            team1_seed = pred['team1_seed']
            team2_seed = pred['team2_seed']
            winner = pred['winner']
            lines.append(f"#{team1_seed} {team1} vs #{team2_seed} {team2}: {winner}")
    
    return "\n".join(lines)

def main():
    """Main function to predict March Madness matchups"""
    # File paths
    matchups_file = "./data/matchups.csv"
    output_file = "./bracket_predictions.txt"
    
    # Load matchups
    matchups = load_matchups(matchups_file)
    
    # Load data from 2021-2025
    recent_years = [2021, 2022, 2023, 2024, 2025]
    team_data = load_recent_data(recent_years)
    
    # Train model
    model, model_features = train_model(team_data)
    
    # Simulate tournament and get all round predictions
    all_predictions = simulate_tournament(matchups, team_data, model, model_features)
    
    # Format and save predictions
    predictions_text = format_predictions_text(all_predictions)
    with open(output_file, 'w') as f:
        f.write(predictions_text)

if __name__ == "__main__":
    main()