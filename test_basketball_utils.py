"""
Test script for BasketballUtils functionality
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from BasketballUtils import create_matchup_features, simulate_bracket

# Test data paths
SUMMARY_PATH = "./data/archive/INT _ KenPom _ Summary.csv"
EFFICIENCY_PATH = "./data/archive/INT _ KenPom _ Efficiency.csv" 
DEFENSE_PATH = "./data/archive/INT _ KenPom _ Defense.csv"
OFFENSE_PATH = "./data/archive/INT _ KenPom _ Offense.csv"
HEIGHT_PATH = "./data/archive/INT _ KenPom _ Height.csv"

def manual_merge_basketball_data():
    """Perform manual data merging due to column name inconsistencies"""
    print("Testing data merging...")
    
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
    
    # Test merging for a specific year
    year = 2023
    # Filter for year
    summary = kenpom_summary[kenpom_summary['Season'] == year].copy()
    efficiency = kenpom_efficiency[kenpom_efficiency['Season'] == year].copy()
    defense = kenpom_defense[kenpom_defense['Season'] == year].copy() if 'Season' in kenpom_defense.columns else kenpom_defense.copy()
    offense = kenpom_offense[kenpom_offense['Season'] == year].copy() if 'Season' in kenpom_offense.columns else kenpom_offense.copy()
    height = kenpom_height[kenpom_height['Season'] == year].copy() if 'Season' in kenpom_height.columns else kenpom_height.copy()
    
    # Print column names for debugging
    print(f"Summary columns: {summary.columns.tolist()}")
    print(f"Efficiency columns: {efficiency.columns.tolist()}")
    print(f"Defense columns: {defense.columns.tolist() if 'TeamName' in defense.columns else 'No TeamName column'}")
    print(f"Offense columns: {offense.columns.tolist() if 'TeamName' in offense.columns else 'No TeamName column'}")
    print(f"Height columns: {height.columns.tolist() if 'TeamName' in height.columns else 'No TeamName column'}")
    
    # Manually merge what we can
    if 'TeamName' in summary.columns and 'TeamName' in efficiency.columns:
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
        print(f"Sample teams: {merged['TeamName'].sample(min(5, len(merged))).tolist()}")
        
        return merged
    else:
        print("Cannot merge due to missing TeamName column")
        return summary  # return just the summary data for testing

def test_create_matchup_features(team_data):
    """Test creation of matchup features"""
    print("\nTesting matchup feature creation...")
    
    # Get 2 random teams
    teams = team_data['TeamName'].sample(min(2, len(team_data))).tolist()
    if len(teams) < 2:
        # If we don't have enough teams, create a dummy team
        teams = [teams[0], "Dummy Team"]
    
    team1, team2 = teams
    
    print(f"Creating matchup features for {team1} vs {team2}")
    features = create_matchup_features(team1, team2, team_data)
    
    if features:
        print(f"Number of differential features: {len(features) - 2}")  # Subtract TEAM1 and TEAM2
        print(f"Sample features: {list(features.keys())[:5]}")
    else:
        print("Failed to create features, this is expected if one team doesn't exist in the data")
    
    return features

def test_simple_prediction(team_data):
    """Test a simple prediction without full bracket simulation"""
    print("\nTesting simple prediction...")
    
    # Select teams that we know are in the data
    teams = team_data['TeamName'].sample(min(10, len(team_data))).tolist()
    
    if len(teams) < 2:
        print("Not enough teams for testing")
        return None
    
    # Generate random matchups for training
    matchups = []
    for i in range(min(100, len(teams)*len(teams)//2)):
        idx1, idx2 = np.random.choice(range(len(teams)), 2, replace=False)
        team1, team2 = teams[idx1], teams[idx2]
        features = create_matchup_features(team1, team2, team_data)
        if features:
            # Add a random result for training
            features['RESULT'] = 1 if np.random.random() > 0.5 else -1
            matchups.append(features)
    
    if not matchups:
        print("Failed to create any valid matchups")
        return None
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    
    # Train Random Forest on a subset of features
    feature_cols = [col for col in matchups_df.columns if col not in ['TEAM1', 'TEAM2', 'RESULT']]
    if not feature_cols:
        print("No valid feature columns found")
        return None
    
    # Select only numeric columns
    X = matchups_df[feature_cols].select_dtypes(include=[np.number])
    if X.empty:
        print("No numeric features available")
        return None
        
    y = matchups_df['RESULT']
    
    # Train a simple model
    rf_model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    rf_model.fit(X, y)
    
    # Make a prediction for a test matchup
    test_team1, test_team2 = teams[0], teams[1]
    test_features = create_matchup_features(test_team1, test_team2, team_data)
    
    if test_features:
        # Extract features
        test_X = pd.DataFrame([test_features])[X.columns]
        
        # Make prediction
        prediction = rf_model.predict(test_X)[0]
        prob = rf_model.predict_proba(test_X)[0][1]
        
        winner = test_team1 if prediction == 1 else test_team2
        print(f"Predicted winner of {test_team1} vs {test_team2}: {winner} with {prob:.2f} probability")
        return winner, prob
    else:
        print(f"Failed to create features for {test_team1} vs {test_team2}")
        return None

def main():
    """Main test function"""
    print("Testing BasketballUtils module...")
    
    # Test data merging
    team_data = manual_merge_basketball_data()
    
    if team_data is not None and len(team_data) > 0:
        # Test matchup feature creation
        features = test_create_matchup_features(team_data)
        
        # Test simple prediction
        prediction = test_simple_prediction(team_data)
    else:
        print("Skipping tests due to data issues")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()