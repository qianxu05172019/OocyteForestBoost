"""
Example usage of OocyteForestBoost for cell-cell interaction prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from oocyteforestboost.predictor import CellInteractionPredictor
import matplotlib.pyplot as plt

def load_example_data():
    """
    Load or generate example data.
    In practice, replace this with your actual data loading code.
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Generate expression data
    cell_type1_expression = np.random.normal(0, 1, (n_samples, n_features))
    cell_type2_expression = np.random.normal(0, 1, (n_samples, n_features))
    
    # Simulate some true interactions based on correlation
    correlations = np.array([
        np.corrcoef(cell_type1_expression[:, i], cell_type2_expression[:, i])[0, 1]
        for i in range(n_features)
    ])
    
    interaction_prob = 1 / (1 + np.exp(-5 * (correlations.mean() - 0.5)))
    y = np.random.binomial(1, interaction_prob, n_samples)
    
    return cell_type1_expression, cell_type2_expression, y

def main():
    # Load data
    print("Loading data...")
    cell_type1_expression, cell_type2_expression, y = load_example_data()
    
    # Initialize predictor
    predictor = CellInteractionPredictor(random_state=42)
    
    # Prepare features
    print("Preparing features...")
    X = predictor.prepare_features(cell_type1_expression, cell_type2_expression)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("Training models...")
    predictor.train_models(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict_proba(X_test)
    
    # Evaluate models
    print("Evaluating models...")
    eval_results = predictor.evaluate_models(X_test, y_test)
    
    # Apply FDR control
    print("Applying FDR control...")
    significant_interactions = predictor.apply_fdr_control(predictions)
    
    # Print results
    print("\nResults:")
    print(f"ROC AUC: {eval_results['roc_auc']:.3f}")
    print(f"PR AUC: {eval_results['pr_auc']:.3f}")
    print(f"Number of significant interactions: {significant_interactions.sum()}")
    
    # Plot evaluation metrics
    predictor.plot_evaluation_metrics(eval_results)
    
    # Show feature importance
    top_features = eval_results['feature_importance'].head(10)
    print("\nTop 10 Most Important Features:")
    print(top_features)

if __name__ == "__main__":
    main()