import pandas as pd
import numpy as np
from collections import defaultdict, deque

def manual_merge_basketball_data(year=2023):
    """
    Manually merge basketball data with handling for column inconsistencies.
    
    Args:
        year: The season year to filter data for
        
    Returns:
        A merged DataFrame with comprehensive team data
    """
    print(f"Loading and merging basketball data for {year}...")
    
    # Define file paths
    SUMMARY_PATH = "./data/archive/INT _ KenPom _ Summary.csv"
    EFFICIENCY_PATH = "./data/archive/INT _ KenPom _ Efficiency.csv" 
    DEFENSE_PATH = "./data/archive/INT _ KenPom _ Defense.csv"
    OFFENSE_PATH = "./data/archive/INT _ KenPom _ Offense.csv"
    HEIGHT_PATH = "./data/archive/INT _ KenPom _ Height.csv"
    
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

def merge_basketball_data(kenpom_summary, kenpom_efficiency, kenpom_defense, kenpom_offense, kenpom_height, year=None):
    """
    Merges different KenPom datasets for a comprehensive team analysis.
    
    Args:
        kenpom_summary: Summary KenPom data
        kenpom_efficiency: Efficiency metrics
        kenpom_defense: Defensive metrics
        kenpom_offense: Offensive metrics
        kenpom_height: Team height data
        year: Optional filter for specific season
        
    Returns:
        A merged DataFrame with comprehensive team data
    """
    # Filter for year if specified
    if year is not None:
        summary = kenpom_summary[kenpom_summary['Season'] == year].copy()
        efficiency = kenpom_efficiency[kenpom_efficiency['Season'] == year].copy()
        defense = kenpom_defense[kenpom_defense['Season'] == year].copy()
        offense = kenpom_offense[kenpom_offense['Season'] == year].copy()
        height = kenpom_height[kenpom_height['Season'] == year].copy()
    else:
        summary = kenpom_summary.copy()
        efficiency = kenpom_efficiency.copy()
        defense = kenpom_defense.copy()
        offense = kenpom_offense.copy() 
        height = kenpom_height.copy()
    
    # Merge datasets on common keys
    merged = summary.merge(efficiency, on=['Season', 'TeamName'], how='inner')
    merged = merged.merge(defense, on=['Season', 'TeamName'], how='inner')
    merged = merged.merge(offense, on=['Season', 'TeamName'], how='inner')
    merged = merged.merge(height, on=['Season', 'TeamName'], how='inner')
    
    return merged

def generate_matchups(team_data, teams=None, random_matchups=0):
    """
    Creates matchup combinations from team data.
    
    Args:
        team_data: DataFrame with team information
        teams: Optional list of specific teams to create matchups for
        random_matchups: Number of random matchups to generate
        
    Returns:
        DataFrame with team matchups and their corresponding features
    """
    matchups = []
    
    # Generate specific matchups if teams provided
    if teams:
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                team1_data = team_data[team_data['TeamName'] == team1]
                team2_data = team_data[team_data['TeamName'] == team2]
                
                if not team1_data.empty and not team2_data.empty:
                    matchup = create_matchup_features(team1, team2, team_data)
                    if matchup:
                        matchups.append(matchup)
    
    # Generate random matchups if requested
    if random_matchups > 0:
        all_teams = team_data['TeamName'].unique()
        for _ in range(random_matchups):
            team1, team2 = np.random.choice(all_teams, 2, replace=False)
            matchup = create_matchup_features(team1, team2, team_data)
            if matchup:
                matchups.append(matchup)
    
    return pd.DataFrame(matchups)

def create_matchup_features(team1, team2, team_data):
    """
    Creates feature differences between two teams.
    
    Args:
        team1: First team name
        team2: Second team name
        team_data: DataFrame with team data
        
    Returns:
        Dictionary with differential features or None if teams not found
    """
    # Get team data
    team1_data = team_data[team_data['TeamName'] == team1]
    team2_data = team_data[team_data['TeamName'] == team2]
    
    if team1_data.empty or team2_data.empty:
        return None
    
    # Select key features
    features = [
        # Efficiency metrics
        'AdjEM', 'AdjOE', 'AdjDE', 'Tempo', 'RankAdjEM', 'RankAdjOE', 'RankAdjDE',
        
        # Shooting metrics 
        'eFGPct', 'RankeFGPct', 'FG2Pct', 'FG3Pct', 'FTPct',
        
        # Possession metrics
        'TOPct', 'ORPct', 'DRPct', 'ARate', 'StlRate', 'BlockPct',
        
        # Team composition
        'Experience', 'AvgHeight', 'Bench'
    ]
    
    # Create differentials
    diff_features = {}
    
    for feature in features:
        if feature in team1_data.columns and feature in team2_data.columns:
            try:
                diff_features[f"{feature}_DIFF"] = float(team1_data[feature].values[0]) - float(team2_data[feature].values[0])
            except (ValueError, TypeError):
                # Skip features that can't be converted to float
                continue
    
    # Add team info
    diff_features['TEAM1'] = team1
    diff_features['TEAM2'] = team2
    
    return diff_features

def calculate_team_elos(matchups_data, init_elo=1500, k=32):
    """
    Calculates ELO ratings for teams based on game results.
    
    Args:
        matchups_data: DataFrame with team matchups and results
        init_elo: Initial ELO rating for new teams
        k: K-factor for ELO calculation
        
    Returns:
        Dictionary with calculated ELO ratings per team
    """
    elo_ratings = defaultdict(lambda: init_elo)
    
    for _, row in matchups_data.iterrows():
        team1 = row['TEAM1']
        team2 = row['TEAM2']
        result = row['RESULT']  # 1 if team1 won, -1 if team2 won
        
        # Get current ELOs
        elo1 = elo_ratings[team1]
        elo2 = elo_ratings[team2]
        
        # Calculate expected outcome
        expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        expected2 = 1 - expected1
        
        # Update ELOs based on result
        if result == 1:  # team1 won
            actual1, actual2 = 1, 0
        else:  # team2 won
            actual1, actual2 = 0, 1
        
        elo_ratings[team1] += k * (actual1 - expected1)
        elo_ratings[team2] += k * (actual2 - expected2)
    
    return dict(elo_ratings)

def simulate_bracket(teams, seeds, regions, model, team_data, known_matchups=None):
    """
    Simulates a tournament bracket with the provided model.
    
    Args:
        teams: List of team names
        seeds: Dictionary mapping team names to their seeds
        regions: Dictionary mapping teams to their tournament regions
        model: Trained prediction model
        team_data: DataFrame with team statistics
        known_matchups: Optional dictionary of predefined matchups by region
                        Format: {'region_name': [(team1, team2), ...]}
        
    Returns:
        Dictionary with tournament simulation results
    """
    # Organize first round matchups by region
    first_round = {}
    
    if known_matchups is not None:
        # Use provided matchups
        first_round = known_matchups
    else:
        # Generate traditional matchups based on seeds
        for region_name, region_teams in regions.items():
            region_matchups = []
            # Create traditional 1v16, 8v9, etc. matchups
            seed_pairs = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
            
            for seed1, seed2 in seed_pairs:
                team1 = next((t for t in region_teams if seeds.get(t) == seed1), None)
                team2 = next((t for t in region_teams if seeds.get(t) == seed2), None)
                
                if team1 and team2:
                    region_matchups.append((team1, team2))
            
            first_round[region_name] = region_matchups
    
    # Simulate each region
    region_winners = {}
    for region, matchups in first_round.items():
        current_round = matchups
        while len(current_round) > 1:
            next_round = []
            for team1, team2 in current_round:
                # Create feature differences
                features = create_matchup_features(team1, team2, team_data)
                features_df = pd.DataFrame([features])
                
                if features is None:
                    # If we can't create features, just pick the first team
                    winner = team1
                else:
                    # Use model to predict
                    X = features_df.drop(['TEAM1', 'TEAM2'], axis=1)
                    try:
                        prediction = model.predict(X)[0]
                        winner = team1 if prediction == 1 else team2
                    except:
                        # Fallback if prediction fails
                        winner = team1
                
                next_round.append(winner)
            
            # Pair teams for next round
            current_round = []
            for i in range(0, len(next_round), 2):
                if i + 1 < len(next_round):
                    current_round.append((next_round[i], next_round[i+1]))
                else:
                    # Handle odd number of teams (should not happen in standard bracket)
                    current_round.append((next_round[i], "BYE"))  # Automatic advancement
        
        # Store region winner
        region_winners[region] = current_round[0][0] if isinstance(current_round[0], tuple) else current_round[0]
    
    # Final Four - semifinals
    region_list = list(region_winners.keys())
    
    # Handle case with fewer than 4 regions
    if len(region_list) >= 4:
        semi1 = (region_winners[region_list[0]], region_winners[region_list[1]])
        semi2 = (region_winners[region_list[2]], region_winners[region_list[3]])
    elif len(region_list) == 3:
        semi1 = (region_winners[region_list[0]], region_winners[region_list[1]])
        semi2 = (region_winners[region_list[2]], "BYE")
    elif len(region_list) == 2:
        semi1 = (region_winners[region_list[0]], region_winners[region_list[1]])
        semi2 = ("BYE", "BYE")
    else:
        # Handle case with only one or no regions
        winner = region_winners[region_list[0]] if region_list else "UNKNOWN"
        return {
            'first_round': {},
            'region_winners': region_winners,
            'semifinals': [],
            'finalists': [winner, ""],
            'champion': winner
        }
    
    # Predict semifinals
    semi1_features = create_matchup_features(semi1[0], semi1[1], team_data)
    semi2_features = create_matchup_features(semi2[0], semi2[1], team_data)
    
    if semi1_features:
        semi1_X = pd.DataFrame([semi1_features]).drop(['TEAM1', 'TEAM2'], axis=1)
        semi1_prediction = model.predict(semi1_X)[0]
        semi1_winner = semi1[0] if semi1_prediction == 1 else semi1[1]
    else:
        semi1_winner = semi1[0]
    
    if semi2_features:
        semi2_X = pd.DataFrame([semi2_features]).drop(['TEAM1', 'TEAM2'], axis=1)
        semi2_prediction = model.predict(semi2_X)[0]
        semi2_winner = semi2[0] if semi2_prediction == 1 else semi2[1]
    else:
        semi2_winner = semi2[0]
    
    # Championship game
    final_features = create_matchup_features(semi1_winner, semi2_winner, team_data)
    if final_features:
        final_X = pd.DataFrame([final_features]).drop(['TEAM1', 'TEAM2'], axis=1)
        final_prediction = model.predict(final_X)[0]
        champion = semi1_winner if final_prediction == 1 else semi2_winner
    else:
        champion = semi1_winner
    
    return {
        'first_round': first_round,
        'region_winners': region_winners,
        'semifinals': [semi1, semi2],
        'finalists': [semi1_winner, semi2_winner],
        'champion': champion
    }