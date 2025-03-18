"""
Test script for March Madness bracket simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from BasketballUtils import create_matchup_features, simulate_bracket

# Test data paths
SUMMARY_PATH = "./data/archive/INT _ KenPom _ Summary.csv"
EFFICIENCY_PATH = "./data/archive/INT _ KenPom _ Efficiency.csv" 
DEFENSE_PATH = "./data/archive/INT _ KenPom _ Defense.csv"
OFFENSE_PATH = "./data/archive/INT _ KenPom _ Offense.csv"
HEIGHT_PATH = "./data/archive/INT _ KenPom _ Height.csv"
TOURNAMENT_TEAMS_PATH = "./data/archive/REF _ Post-Season Tournament Teams.csv"

def manual_merge_basketball_data(year=2023):
    """Perform manual data merging due to column name inconsistencies"""
    print(f"Loading and merging basketball data for {year}...")
    
    # Load datasets
    kenpom_summary = pd.read_csv(SUMMARY_PATH)
    kenpom_efficiency = pd.read_csv(EFFICIENCY_PATH)
    kenpom_defense = pd.read_csv(DEFENSE_PATH)
    kenpom_offense = pd.read_csv(OFFENSE_PATH)
    kenpom_height = pd.read_csv(HEIGHT_PATH)
    
    # Rename columns to ensure consistency
    kenpom_efficiency = kenpom_efficiency.rename(columns={'Team': 'TeamName'})
    
    # Check if other files need column renaming
    if 'Team' in kenpom_defense.columns and 'TeamName' not in kenpom_defense.columns:
        kenpom_defense = kenpom_defense.rename(columns={'Team': 'TeamName'})
    
    if 'Team' in kenpom_offense.columns and 'TeamName' not in kenpom_offense.columns:
        kenpom_offense = kenpom_offense.rename(columns={'Team': 'TeamName'})
    
    if 'Team' in kenpom_height.columns and 'TeamName' not in kenpom_height.columns:
        kenpom_height = kenpom_height.rename(columns={'Team': 'TeamName'})
    
    # Filter for year
    summary = kenpom_summary[kenpom_summary['Season'] == year].copy()
    efficiency = kenpom_efficiency[kenpom_efficiency['Season'] == year].copy()
    defense = kenpom_defense[kenpom_defense['Season'] == year].copy() if 'Season' in kenpom_defense.columns else kenpom_defense.copy()
    offense = kenpom_offense[kenpom_offense['Season'] == year].copy() if 'Season' in kenpom_offense.columns else kenpom_offense.copy()
    height = kenpom_height[kenpom_height['Season'] == year].copy() if 'Season' in kenpom_height.columns else kenpom_height.copy()
    
    # Manually merge what we can
    merged = summary.merge(efficiency, on=['Season', 'TeamName'], how='inner')
    print(f"Merged summary and efficiency: {merged.shape}")
    
    # Continue merging if other datasets have TeamName
    if 'TeamName' in defense.columns:
        merged = merged.merge(defense, on=['Season', 'TeamName'], how='inner')
        print(f"Merged with defense: {merged.shape}")
    
    if 'TeamName' in offense.columns:
        merged = merged.merge(offense, on=['Season', 'TeamName'], how='inner')
        print(f"Merged with offense: {merged.shape}")
    
    if 'TeamName' in height.columns:
        merged = merged.merge(height, on=['Season', 'TeamName'], how='inner')
        print(f"Merged with height: {merged.shape}")
    
    print(f"Number of teams: {merged['TeamName'].nunique()}")
    
    return merged

def load_tournament_teams(year=2023):
    """Load tournament teams and their seedings"""
    print(f"Loading tournament teams for {year}...")
    
    try:
        tournament_data = pd.read_csv(TOURNAMENT_TEAMS_PATH)
        tournament_data = tournament_data[tournament_data['Season'] == year].copy()
        print(f"Found {len(tournament_data)} tournament teams for {year}")
        
        # Create a dictionary of team seeds
        team_seeds = {}
        teams_by_region = {}
        
        # Check if the required columns exist
        if 'Seed' in tournament_data.columns and 'Mapped ESPN Team Name' in tournament_data.columns and 'Region' in tournament_data.columns:
            # Group teams by region
            for _, row in tournament_data.iterrows():
                team = row['Mapped ESPN Team Name']
                seed = row['Seed']
                region = row['Region']
                
                team_seeds[team] = seed
                
                if region not in teams_by_region:
                    teams_by_region[region] = []
                teams_by_region[region].append(team)
            
            return team_seeds, teams_by_region
        else:
            # Use mock data if columns are missing
            print("Tournament data columns not found, using mock data...")
            return create_mock_tournament()
    except Exception as e:
        print(f"Error loading tournament data: {e}")
        print("Using mock tournament data instead...")
        return create_mock_tournament()

def create_mock_tournament():
    """Create a mock tournament structure for testing"""
    print("Creating mock tournament data...")
    
    # Sample teams by seed
    seeds = {
        "Duke": 1, "Gonzaga": 1, "Kansas": 1, "UConn": 1,
        "Tennessee": 2, "Arizona": 2, "Marquette": 2, "UCLA": 2,
        "Baylor": 3, "Creighton": 3, "Xavier": 3, "Houston": 3,
        "Indiana": 4, "Virginia": 4, "Kentucky": 4, "Iowa St.": 4,
        "Miami FL": 5, "San Diego St.": 5, "St. Mary's": 5, "Purdue": 5,
        "TCU": 6, "Iowa": 6, "Michigan St.": 6, "West Virginia": 6,
        "Northwestern": 7, "Missouri": 7, "Texas A&M": 7, "USC": 7,
        "Arkansas": 8, "Memphis": 8, "Utah St.": 8, "Maryland": 8,
        "Auburn": 9, "Florida Atlantic": 9, "Illinois": 9, "Providence": 9,
        "Boise St.": 10, "Penn St.": 10, "Utah": 10, "VCU": 10,
        "Arizona St.": 11, "Pittsburgh": 11, "NC State": 11, "Nevada": 11,
        "Drake": 12, "Charleston": 12, "Oral Roberts": 12, "VCU": 12,
        "Kent St.": 13, "Furman": 13, "Louisiana": 13, "Iona": 13,
        "UC Santa Barbara": 14, "Montana St.": 14, "Grand Canyon": 14, "Kennesaw St.": 14,
        "Princeton": 15, "UNC Asheville": 15, "Colgate": 15, "Vermont": 15,
        "Northern Kentucky": 16, "Howard": 16, "Texas A&M CC": 16, "FDU": 16
    }
    
    # Create region groupings
    regions = {
        "East": [
            "Duke", "Tennessee", "Baylor", "Indiana", "Miami FL", "TCU", "Northwestern", "Arkansas",
            "Auburn", "Boise St.", "Arizona St.", "Drake", "Kent St.", "UC Santa Barbara", "Princeton", "Northern Kentucky"
        ],
        "West": [
            "Gonzaga", "Arizona", "Creighton", "Virginia", "San Diego St.", "Iowa", "Missouri", "Memphis",
            "Florida Atlantic", "Penn St.", "Pittsburgh", "Charleston", "Furman", "Montana St.", "UNC Asheville", "Howard"
        ],
        "South": [
            "Kansas", "Marquette", "Xavier", "Kentucky", "St. Mary's", "Michigan St.", "Texas A&M", "Utah St.",
            "Illinois", "Utah", "NC State", "Oral Roberts", "Louisiana", "Grand Canyon", "Colgate", "Texas A&M CC"
        ],
        "Midwest": [
            "UConn", "UCLA", "Houston", "Iowa St.", "Purdue", "West Virginia", "USC", "Maryland",
            "Providence", "VCU", "Nevada", "VCU", "Iona", "Kennesaw St.", "Vermont", "FDU"
        ]
    }
    
    return seeds, regions

def generate_training_data(team_data, n_samples=500):
    """Generate synthetic training data for model training"""
    print(f"Generating {n_samples} synthetic training samples...")
    
    # Get all teams
    all_teams = team_data['TeamName'].unique()
    
    # Generate random matchups
    matchups = []
    for _ in range(n_samples):
        # Get two random teams
        team1, team2 = np.random.choice(all_teams, 2, replace=False)
        
        # Create features for this matchup
        features = create_matchup_features(team1, team2, team_data)
        
        if features:
            # Generate synthetic result
            # Higher ranked teams (lower RankAdjEM) tend to win more often
            if 'RankAdjEM_DIFF' in features:
                # Negative means team1 has better rank
                prob_team1_wins = 1 / (1 + np.exp(features['RankAdjEM_DIFF'] / 10))
            else:
                # 60% chance for team1 to win
                prob_team1_wins = 0.6
                
            features['RESULT'] = 1 if np.random.random() < prob_team1_wins else -1
            matchups.append(features)
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    print(f"Generated {len(matchups_df)} valid matchups")
    
    return matchups_df

def train_predictor(training_data):
    """Train a Random Forest model on matchup data"""
    print("Training predictor model...")
    
    # Separate features and target
    feature_cols = [col for col in training_data.columns 
                   if col not in ['TEAM1', 'TEAM2', 'RESULT']]
    
    # Select only numeric features
    X = training_data[feature_cols].select_dtypes(include=[np.number])
    y = training_data['RESULT']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Show feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features:")
    for feature, importance_value in top_features.items():
        print(f"{feature}: {importance_value:.4f}")
    
    return model, X.columns

def simulate_march_madness(team_data, seeds, regions, model, feature_columns):
    """Simulate the March Madness tournament"""
    print("\nSimulating March Madness tournament...")
    
    # Get all teams in the tournament
    all_tournament_teams = []
    for region_teams in regions.values():
        all_tournament_teams.extend(region_teams)
    
    # Simulate the tournament
    simulation_result = simulate_bracket(all_tournament_teams, seeds, regions, model, team_data)
    
    print("\n=== Tournament Results ===")
    print("\nRegion Winners:")
    for region, team in simulation_result['region_winners'].items():
        print(f"{region}: {team} (Seed #{seeds.get(team, 'Unknown')})")
    
    print("\nFinal Four:")
    for matchup in simulation_result['semifinals']:
        print(f"{matchup[0]} vs {matchup[1]}")
    
    print("\nChampionship Game:")
    if len(simulation_result['finalists']) >= 2:
        print(f"{simulation_result['finalists'][0]} vs {simulation_result['finalists'][1]}")
    
    print(f"\nChampion: {simulation_result['champion']} (Seed #{seeds.get(simulation_result['champion'], 'Unknown')})")
    
    return simulation_result

def analyze_upset_probabilities(team_data, seeds, model, feature_columns):
    """Analyze potential upsets in the tournament"""
    print("\nAnalyzing potential upsets...")
    
    # Define criteria for upsets based on seed difference
    upset_threshold = 4  # Seed difference to consider an upset
    
    # Get all teams
    all_teams = list(seeds.keys())
    
    # Generate all potential first round matchups
    potential_upsets = []
    for i, higher_seed_team in enumerate(all_teams):
        higher_seed = seeds.get(higher_seed_team)
        
        for lower_seed_team in all_teams[i+1:]:
            lower_seed = seeds.get(lower_seed_team)
            
            # Consider only if seed difference is significant
            if higher_seed and lower_seed and lower_seed - higher_seed >= upset_threshold:
                # Create features
                features = create_matchup_features(higher_seed_team, lower_seed_team, team_data)
                
                if features:
                    # Prepare features for prediction
                    pred_features = pd.DataFrame([features])[feature_columns]
                    
                    # Make prediction
                    prob = model.predict_proba(pred_features)[0]
                    prob_upset = 1 - prob[1] if model.classes_[1] == 1 else prob[1]
                    
                    potential_upsets.append({
                        'higher_seed_team': higher_seed_team,
                        'higher_seed': higher_seed,
                        'lower_seed_team': lower_seed_team,
                        'lower_seed': lower_seed, 
                        'upset_probability': prob_upset
                    })
    
    # Convert to DataFrame and sort by upset probability
    upset_df = pd.DataFrame(potential_upsets)
    upset_df = upset_df.sort_values('upset_probability', ascending=False)
    
    # Display top potential upsets
    print("\nTop 10 potential upsets:")
    for _, row in upset_df.head(10).iterrows():
        print(f"#{row['lower_seed']} {row['lower_seed_team']} over #{row['higher_seed']} {row['higher_seed_team']}: "
              f"{row['upset_probability']:.2f} probability")
    
    return upset_df

def main():
    """Main function to run the tournament simulation"""
    print("=== March Madness Tournament Simulator ===")
    
    # Load and merge team data
    team_data = manual_merge_basketball_data(year=2023)
    
    # Load tournament teams
    seeds, regions = load_tournament_teams(year=2023)
    
    # Generate training data
    training_data = generate_training_data(team_data)
    
    # Train model
    model, feature_columns = train_predictor(training_data)
    
    # Simulate tournament
    simulation = simulate_march_madness(team_data, seeds, regions, model, feature_columns)
    
    # Analyze upsets
    upset_analysis = analyze_upset_probabilities(team_data, seeds, model, feature_columns)
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()