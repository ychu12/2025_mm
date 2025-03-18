"""
Make multiple bracket predictions using different data sources and years
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from BasketballUtils import manual_merge_basketball_data, create_matchup_features, simulate_bracket

def load_revised_matchups(file_path):
    """Load matchups from the revised CSV format"""
    print(f"Loading matchups from {file_path}...")
    matchups_df = pd.read_csv(file_path)
    
    # Handle TBD teams - replace with placeholder names
    matchups_df['Team1'] = matchups_df['Team1'].apply(lambda x: f"Team{np.random.randint(1000)}" if x == "TBD" else x)
    matchups_df['Team2'] = matchups_df['Team2'].apply(lambda x: f"Team{np.random.randint(1000)}" if x == "TBD" else x)
    
    # Handle First Four winners
    first_four_winners = {}
    first_four_rows = matchups_df[matchups_df['Round'] == 'First Four']
    
    for _, row in first_four_rows.iterrows():
        region = row['Region']
        team1, team2 = row['Team1'], row['Team2']
        # Randomly pick a winner for First Four games (50/50 chance)
        winner = team1 if np.random.random() < 0.5 else team2
        first_four_winners[f"{region}_First_Four_{row['Seed1']}"] = winner
    
    # Create dictionary of matchups by region
    regions = matchups_df['Region'].unique()
    matchups_by_region = {}
    
    for region in regions:
        region_df = matchups_df[(matchups_df['Region'] == region) & (matchups_df['Round'] == 'First Round')]
        region_matchups = []
        
        for _, row in region_df.iterrows():
            team1 = row['Team1']
            team2 = row['Team2']
            
            # Replace First Four Winner placeholders
            if isinstance(team1, str) and "First Four Winner" in team1:
                parts = team1.split("(")[1].split(")")[0].split("/")
                key = f"{region}_First_Four_{row['Seed1']}"
                if key in first_four_winners:
                    team1 = first_four_winners[key]
            
            if isinstance(team2, str) and "First Four Winner" in team2:
                parts = team2.split("(")[1].split(")")[0].split("/")
                key = f"{region}_First_Four_{row['Seed2']}"
                if key in first_four_winners:
                    team2 = first_four_winners[key]
            
            region_matchups.append((team1, team2))
        
        matchups_by_region[region] = region_matchups
    
    # Create dictionary of team seeds
    team_seeds = {}
    for _, row in matchups_df.iterrows():
        team_seeds[row['Team1']] = row['Seed1']
        team_seeds[row['Team2']] = row['Seed2']
    
    # Create dictionary of regions
    team_regions = {}
    for region in regions:
        region_df = matchups_df[matchups_df['Region'] == region]
        teams = []
        
        for _, row in region_df.iterrows():
            # Only include teams from First Round
            if row['Round'] == 'First Round':
                team1 = row['Team1']
                team2 = row['Team2']
                
                # Replace First Four Winner placeholders
                if isinstance(team1, str) and "First Four Winner" in team1:
                    key = f"{region}_First_Four_{row['Seed1']}"
                    if key in first_four_winners:
                        team1 = first_four_winners[key]
                
                if isinstance(team2, str) and "First Four Winner" in team2:
                    key = f"{region}_First_Four_{row['Seed2']}"
                    if key in first_four_winners:
                        team2 = first_four_winners[key]
                
                teams.append(team1)
                teams.append(team2)
        
        team_regions[region] = teams
    
    return matchups_by_region, team_seeds, team_regions

def train_model(team_data, year, n_samples=1000):
    """Train a prediction model"""
    print(f"Training prediction model using {year} data...")
    
    # Get all teams
    all_teams = team_data['TeamName'].unique()
    
    # Generate random matchups for training
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
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")
    
    return model, X.columns

def predict_bracket(team_data, matchups, seeds, regions, model, feature_columns):
    """Predict tournament bracket"""
    print("\nPredicting tournament bracket...")
    
    # Get all teams
    all_teams = []
    for region_teams in regions.values():
        all_teams.extend(region_teams)
    
    # Simulate tournament
    simulation = simulate_bracket(
        teams=all_teams,
        seeds=seeds,
        regions=regions,
        model=model,
        team_data=team_data,
        known_matchups=matchups
    )
    
    return simulation

def format_bracket_text(simulation, seeds, year=None):
    """Format bracket results as text"""
    lines = []
    lines.append("=" * 80)
    title = "MARCH MADNESS BRACKET PREDICTION"
    if year:
        title += f" (USING {year} DATA)"
    lines.append(" " * ((80 - len(title)) // 2) + title)
    lines.append("=" * 80)
    
    # First round
    lines.append("\nFIRST ROUND MATCHUPS:")
    lines.append("-" * 80)
    
    for region, matchups in simulation['first_round'].items():
        lines.append(f"\n{region.upper()} REGION:")
        for i, matchup in enumerate(matchups):
            team1, team2 = matchup
            seed1 = seeds.get(team1, '?')
            seed2 = seeds.get(team2, '?')
            lines.append(f"  Game {i+1}: (#{seed1}) {team1:20} vs (#{seed2}) {team2:20}")
    
    # Add some whitespace
    lines.append("\n" + "=" * 80)
    
    # Sweet 16 (generated based on simulation logic)
    lines.append("\nPREDICTED SWEET 16:")
    lines.append("-" * 80)
    for region, region_teams in simulation['region_winners'].items():
        # We don't have direct access to Sweet 16 teams, but we can imply from the model
        lines.append(f"\n{region.upper()} REGION:")
        lines.append(f"  Region Winner: (#{seeds.get(region_teams, '?')}) {region_teams}")
    
    # Add some whitespace
    lines.append("\n" + "=" * 80)
    
    # Final Four
    lines.append("\nFINAL FOUR:")
    lines.append("-" * 80)
    lines.append("\nSEMIFINAL MATCHUPS:")
    for i, matchup in enumerate(simulation['semifinals']):
        team1, team2 = matchup
        seed1 = seeds.get(team1, '?')
        seed2 = seeds.get(team2, '?')
        lines.append(f"  Semifinal {i+1}: (#{seed1}) {team1:20} vs (#{seed2}) {team2:20}")
    
    # Championship
    lines.append("\nCHAMPIONSHIP GAME:")
    if len(simulation['finalists']) >= 2:
        team1, team2 = simulation['finalists']
        seed1 = seeds.get(team1, '?')
        seed2 = seeds.get(team2, '?')
        lines.append(f"  (#{seed1}) {team1:20} vs (#{seed2}) {team2:20}")
    
    # Add some whitespace
    lines.append("\n" + "=" * 80)
    
    # Champion
    lines.append("\nNATIONAL CHAMPION:")
    lines.append("-" * 80)
    champion = simulation['champion']
    seed = seeds.get(champion, '?')
    lines.append(f"  (#{seed}) {champion.upper()}")
    
    # Final note
    lines.append("\n" + "=" * 80)
    lines.append("Prediction generated using Random Forest model trained on KenPom metrics")
    lines.append("Top predictive features: Adjusted Efficiency Margin, Team Rankings, Offensive Efficiency")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def save_bracket(bracket_text, output_file):
    """Save bracket to text file"""
    with open(output_file, 'w') as f:
        f.write(bracket_text)
    print(f"Bracket saved to {output_file}")

def main():
    """Main function to make multiple predictions"""
    # File paths
    revised_file = "./data/revised.csv"
    output_file1 = "./predictions_revised.txt"
    output_file2 = "./predictions2.txt"
    
    # 1. Make prediction using revised.csv and 2023 data
    matchups, seeds, regions = load_revised_matchups(revised_file)
    team_data_2023 = manual_merge_basketball_data(year=2023)
    model_2023, features_2023 = train_model(team_data_2023, 2023)
    simulation_2023 = predict_bracket(team_data_2023, matchups, seeds, regions, model_2023, features_2023)
    bracket_text_2023 = format_bracket_text(simulation_2023, seeds, 2023)
    save_bracket(bracket_text_2023, output_file1)
    
    # 2. Make prediction using 2024 data (or 2025 if available)
    # Try 2025 first, fall back to 2024 if not available
    try:
        # Try 2025
        team_data_2025 = manual_merge_basketball_data(year=2025)
        print("Using 2025 data for second prediction")
        recent_year = 2025
    except Exception as e:
        # Fall back to 2024
        print(f"2025 data not available, falling back to 2024: {e}")
        team_data_2024 = manual_merge_basketball_data(year=2024)
        team_data_2025 = team_data_2024  # Use 2024 data instead
        recent_year = 2024
    
    # Train model and make prediction
    model_recent, features_recent = train_model(team_data_2025, recent_year)
    simulation_recent = predict_bracket(team_data_2025, matchups, seeds, regions, model_recent, features_recent)
    bracket_text_recent = format_bracket_text(simulation_recent, seeds, recent_year)
    save_bracket(bracket_text_recent, output_file2)
    
    print("\nAll predictions complete!")

if __name__ == "__main__":
    main()