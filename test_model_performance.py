"""
Test script to evaluate the performance of our Random Forest implementation
compared to scikit-learn's implementation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Import our custom Random Forest implementation
from RandomForest.RandomForest import RandomForest
from BasketballUtils import manual_merge_basketball_data, create_matchup_features

def load_sample_data():
    """Load sample basketball data and create matchup training data"""
    print("Loading basketball data and generating matchups...")
    
    # Load team data
    team_data = manual_merge_basketball_data(year=2023)
    
    # Generate matchups
    all_teams = team_data['TeamName'].unique()
    
    # Generate random matchups for training/testing
    matchups = []
    for _ in range(1000):  # 1000 matchups for a good test
        team1, team2 = np.random.choice(all_teams, 2, replace=False)
        features = create_matchup_features(team1, team2, team_data)
        
        if features:
            # Generate synthetic result
            # Use a rule based on rank difference to make this realistic
            if 'RankAdjEM_DIFF' in features:
                # Negative means team1 is better
                prob_team1_wins = 1 / (1 + np.exp(features['RankAdjEM_DIFF'] / 10))
            else:
                # Default if no rank info
                prob_team1_wins = 0.6
            
            features['RESULT'] = 1 if np.random.random() < prob_team1_wins else -1
            matchups.append(features)
    
    matchups_df = pd.DataFrame(matchups)
    print(f"Generated {len(matchups_df)} matchups")
    
    # Prepare features and target
    X = matchups_df.drop(['TEAM1', 'TEAM2', 'RESULT'], axis=1).select_dtypes(include=[np.number])
    y = matchups_df['RESULT']
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def evaluate_custom_random_forest(X_train, X_test, y_train, y_test):
    """Evaluate our custom Random Forest implementation"""
    print("\nEvaluating custom Random Forest...")
    
    # Convert data to numpy arrays for our implementation
    X_train_np = X_train.values
    y_train_np = y_train.values.reshape(-1, 1)
    X_test_np = X_test.values
    y_test_np = y_test.values.reshape(-1, 1)
    
    # Combine features and target for training data
    train_data_np = np.hstack((X_train_np, y_train_np))
    test_data_np = np.hstack((X_test_np, y_test_np))
    
    # Record start time
    start_time = time.time()
    
    # Train our custom Random Forest
    n_features = X_train.shape[1]
    my_rf = RandomForest(
        n_features=int(n_features * 0.7),  # Use 70% of features for each tree
        n_estimators=50,  # Use 50 trees
        tree_params=dict(max_depth=10, min_samples_split=5)
    )
    my_rf.build_forest(train_data_np)
    
    # Record training time
    train_time = time.time() - start_time
    
    # Evaluate on test data
    start_time = time.time()
    custom_accuracy = my_rf.calculate_accuracy(test_data_np)
    predict_time = time.time() - start_time
    
    print(f"Custom RF - Accuracy: {custom_accuracy:.4f}")
    print(f"Custom RF - Training time: {train_time:.4f} seconds")
    print(f"Custom RF - Prediction time: {predict_time:.4f} seconds")
    
    return custom_accuracy, train_time, predict_time

def evaluate_sklearn_random_forest(X_train, X_test, y_train, y_test):
    """Evaluate scikit-learn's Random Forest implementation"""
    print("\nEvaluating scikit-learn Random Forest...")
    
    # Record start time
    start_time = time.time()
    
    # Train scikit-learn Random Forest
    sklearn_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        max_features=0.7,
        random_state=42
    )
    sklearn_rf.fit(X_train, y_train)
    
    # Record training time
    train_time = time.time() - start_time
    
    # Evaluate on test data
    start_time = time.time()
    y_pred = sklearn_rf.predict(X_test)
    predict_time = time.time() - start_time
    
    # Calculate metrics
    sklearn_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Sklearn RF - Accuracy: {sklearn_accuracy:.4f}")
    print(f"Sklearn RF - Training time: {train_time:.4f} seconds")
    print(f"Sklearn RF - Prediction time: {predict_time:.4f} seconds")
    
    # Show feature importance
    importance = pd.Series(sklearn_rf.feature_importances_, index=X_train.columns)
    top_features = importance.sort_values(ascending=False).head(10)
    
    print("\nTop 10 important features (scikit-learn):")
    for feature, importance_value in top_features.items():
        print(f"{feature}: {importance_value:.4f}")
    
    return sklearn_accuracy, train_time, predict_time, sklearn_rf

def compare_results(custom_results, sklearn_results):
    """Compare the results between implementations"""
    custom_acc, custom_train_time, custom_predict_time = custom_results
    sklearn_acc, sklearn_train_time, sklearn_predict_time, _ = sklearn_results
    
    print("\n=== Performance Comparison ===")
    print(f"Accuracy: Custom={custom_acc:.4f}, Sklearn={sklearn_acc:.4f}, Diff={sklearn_acc-custom_acc:.4f}")
    print(f"Training Time: Custom={custom_train_time:.4f}s, Sklearn={sklearn_train_time:.4f}s, Ratio={custom_train_time/sklearn_train_time:.2f}x")
    print(f"Prediction Time: Custom={custom_predict_time:.4f}s, Sklearn={sklearn_predict_time:.4f}s, Ratio={custom_predict_time/sklearn_predict_time:.2f}x")
    
    # Create a comparison chart
    metrics = ['Accuracy', 'Training Time (s)', 'Prediction Time (s)']
    custom_values = [custom_acc, custom_train_time, custom_predict_time]
    sklearn_values = [sklearn_acc, sklearn_train_time, sklearn_predict_time]
    
    plt.figure(figsize=(10, 6))
    
    # Accuracy subplot
    plt.subplot(1, 3, 1)
    plt.bar(['Custom RF', 'Sklearn RF'], [custom_acc, sklearn_acc])
    plt.title('Accuracy')
    plt.ylim([0.5, 1.0])  # Accuracy range
    
    # Training time subplot
    plt.subplot(1, 3, 2)
    plt.bar(['Custom RF', 'Sklearn RF'], [custom_train_time, sklearn_train_time])
    plt.title('Training Time (s)')
    
    # Prediction time subplot
    plt.subplot(1, 3, 3)
    plt.bar(['Custom RF', 'Sklearn RF'], [custom_predict_time, sklearn_predict_time])
    plt.title('Prediction Time (s)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Comparison chart saved as 'model_comparison.png'")
    
def main():
    """Main function to evaluate and compare models"""
    print("=== Random Forest Implementation Comparison ===")
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_data()
    
    # Evaluate custom implementation
    custom_results = evaluate_custom_random_forest(X_train, X_test, y_train, y_test)
    
    # Evaluate scikit-learn implementation
    sklearn_results = evaluate_sklearn_random_forest(X_train, X_test, y_train, y_test)
    
    # Compare results
    compare_results(custom_results, sklearn_results)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()