"""
Example script for using the March Madness prediction system with known first-round matchups
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from BasketballUtils import manual_merge_basketball_data, create_matchup_features, simulate_bracket

def load_data(year=2023):
    """Load and prepare data"""
    print(f"Loading data for {year}...")
    team_data = manual_merge_basketball_data(year=year)
    return team_data

def train_model(team_data, n_samples=500):
    """Train a prediction model"""
    print("Training prediction model...")
    
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
            # Generate synthetic result
            # Rank difference determines win probability
            if 'RankAdjEM_DIFF' in features:
                prob_team1_wins = 1 / (1 + np.exp(features['RankAdjEM_DIFF'] / 10))
            else:
                prob_team1_wins = 0.6
                
            features['RESULT'] = 1 if np.random.random() < prob_team1_wins else -1
            matchups.append(features)
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    
    # Prepare features and target
    X = matchups_df.drop(['TEAM1', 'TEAM2', 'RESULT'], axis=1).select_dtypes(include=[np.number])
    y = matchups_df['RESULT']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

def create_known_matchups():
    """Define known first-round matchups for each region"""
    # Example: 2023 NCAA Tournament first-round matchups
    # Replace with your actual first-round matchups
    known_matchups = {
        'East': [
            ('Purdue', 'Fairleigh Dickinson'),
            ('Memphis', 'FAU'),
            ('Duke', 'Oral Roberts'),
            ('Tennessee', 'Louisiana'),
            ('Kentucky', 'Providence'),
            ('Kansas State', 'Montana State'),
            ('Michigan State', 'USC'),
            ('Marquette', 'Vermont')
        ],
        'West': [
            ('Kansas', 'Howard'),
            ('Arkansas', 'Illinois'),
            ('Saint Mary\'s', 'VCU'),
            ('UConn', 'Iona'),
            ('TCU', 'Arizona State'),
            ('Gonzaga', 'Grand Canyon'),
            ('Northwestern', 'Boise State'),
            ('UCLA', 'UNC Asheville')
        ],
        'South': [
            ('Alabama', 'Texas A&M CC'),
            ('Maryland', 'West Virginia'),
            ('San Diego State', 'Charleston'),
            ('Virginia', 'Furman'),
            ('Creighton', 'NC State'),
            ('Baylor', 'UC Santa Barbara'),
            ('Missouri', 'Utah State'),
            ('Arizona', 'Princeton')
        ],
        'Midwest': [
            ('Houston', 'Northern Kentucky'),
            ('Iowa', 'Auburn'),
            ('Miami FL', 'Drake'),
            ('Indiana', 'Kent State'),
            ('Iowa State', 'Pittsburgh'),
            ('Xavier', 'Kennesaw State'),
            ('Texas A&M', 'Penn State'),
            ('Texas', 'Colgate')
        ]
    }
    
    return known_matchups

def create_team_seeds():
    """Create a dictionary mapping teams to their seeds"""
    # Example: 2023 NCAA Tournament seeds
    # Replace with your actual team seeds
    seeds = {
        # East Region
        'Purdue': 1, 'Fairleigh Dickinson': 16,
        'Memphis': 8, 'FAU': 9,
        'Duke': 5, 'Oral Roberts': 12,
        'Tennessee': 4, 'Louisiana': 13,
        'Kentucky': 6, 'Providence': 11,
        'Kansas State': 3, 'Montana State': 14,
        'Michigan State': 7, 'USC': 10,
        'Marquette': 2, 'Vermont': 15,
        
        # West Region
        'Kansas': 1, 'Howard': 16,
        'Arkansas': 8, 'Illinois': 9,
        'Saint Mary\'s': 5, 'VCU': 12,
        'UConn': 4, 'Iona': 13,
        'TCU': 6, 'Arizona State': 11,
        'Gonzaga': 3, 'Grand Canyon': 14,
        'Northwestern': 7, 'Boise State': 10,
        'UCLA': 2, 'UNC Asheville': 15,
        
        # South Region
        'Alabama': 1, 'Texas A&M CC': 16,
        'Maryland': 8, 'West Virginia': 9,
        'San Diego State': 5, 'Charleston': 12,
        'Virginia': 4, 'Furman': 13,
        'Creighton': 6, 'NC State': 11,
        'Baylor': 3, 'UC Santa Barbara': 14,
        'Missouri': 7, 'Utah State': 10,
        'Arizona': 2, 'Princeton': 15,
        
        # Midwest Region
        'Houston': 1, 'Northern Kentucky': 16,
        'Iowa': 8, 'Auburn': 9,
        'Miami FL': 5, 'Drake': 12,
        'Indiana': 4, 'Kent State': 13,
        'Iowa State': 6, 'Pittsburgh': 11,
        'Xavier': 3, 'Kennesaw State': 14,
        'Texas A&M': 7, 'Penn State': 10,
        'Texas': 2, 'Colgate': 15
    }
    
    return seeds

def create_team_regions():
    """Create a dictionary mapping teams to their regions"""
    # Example: 2023 NCAA Tournament regions
    regions = {
        'East': [
            'Purdue', 'Fairleigh Dickinson',
            'Memphis', 'FAU',
            'Duke', 'Oral Roberts',
            'Tennessee', 'Louisiana',
            'Kentucky', 'Providence',
            'Kansas State', 'Montana State',
            'Michigan State', 'USC',
            'Marquette', 'Vermont'
        ],
        'West': [
            'Kansas', 'Howard',
            'Arkansas', 'Illinois',
            'Saint Mary\'s', 'VCU',
            'UConn', 'Iona',
            'TCU', 'Arizona State',
            'Gonzaga', 'Grand Canyon',
            'Northwestern', 'Boise State',
            'UCLA', 'UNC Asheville'
        ],
        'South': [
            'Alabama', 'Texas A&M CC',
            'Maryland', 'West Virginia',
            'San Diego State', 'Charleston',
            'Virginia', 'Furman',
            'Creighton', 'NC State',
            'Baylor', 'UC Santa Barbara',
            'Missouri', 'Utah State',
            'Arizona', 'Princeton'
        ],
        'Midwest': [
            'Houston', 'Northern Kentucky',
            'Iowa', 'Auburn',
            'Miami FL', 'Drake',
            'Indiana', 'Kent State',
            'Iowa State', 'Pittsburgh',
            'Xavier', 'Kennesaw State',
            'Texas A&M', 'Penn State',
            'Texas', 'Colgate'
        ]
    }
    
    return regions

def run_simulation(team_data, model, feature_columns):
    """Run the tournament simulation with known matchups"""
    print("\nRunning simulation with known first-round matchups...")
    
    # Get known matchups
    known_matchups = create_known_matchups()
    
    # Get seeds and regions
    seeds = create_team_seeds()
    regions = create_team_regions()
    
    # Get all teams
    all_teams = []
    for region_teams in regions.values():
        all_teams.extend(region_teams)
    
    # Run simulation
    simulation_result = simulate_bracket(
        teams=all_teams,
        seeds=seeds,
        regions=regions,
        model=model,
        team_data=team_data,
        known_matchups=known_matchups
    )
    
    # Print results
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

def main():
    """Main function"""
    print("=== March Madness Prediction with Known Matchups ===")
    
    # Load data
    team_data = load_data(year=2023)
    
    # Train model
    model, feature_columns = train_model(team_data)
    
    # Run simulation
    simulation = run_simulation(team_data, model, feature_columns)
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()