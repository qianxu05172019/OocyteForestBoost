"""
OocyteForestBoost utility functions.
Author: Qian Xu
This module provides utility functions for the OocyteForestBoost package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def validate_input(cell_type1_expression: np.ndarray, 
                  cell_type2_expression: np.ndarray) -> None:
    """
    Validate input data for feature preparation.
    
    Parameters
    ----------
    cell_type1_expression : np.ndarray
        Gene expression matrix for first cell type
    cell_type2_expression : np.ndarray
        Gene expression matrix for second cell type
        
    Raises
    ------
    ValueError
        If input data is invalid
    """
    # Check if inputs are numpy arrays
    if not isinstance(cell_type1_expression, np.ndarray) or \
       not isinstance(cell_type2_expression, np.ndarray):
        raise ValueError("Input data must be numpy arrays")
    
    # Check if inputs have the same shape
    if cell_type1_expression.shape != cell_type2_expression.shape:
        raise ValueError("Input arrays must have the same shape")
    
    # Check for NaN values
    if np.isnan(cell_type1_expression).any() or np.isnan(cell_type2_expression).any():
        raise ValueError("Input data contains NaN values")
    
    # Check for infinite values
    if np.isinf(cell_type1_expression).any() or np.isinf(cell_type2_expression).any():
        raise ValueError("Input data contains infinite values")

def calculate_feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    """
    Calculate and format feature importance from Random Forest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importance scores
    """
    feature_importance = pd.DataFrame({
        'feature': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    })
    
    return feature_importance.sort_values('importance', ascending=False)

def plot_feature_importance(feature_importance: pd.DataFrame, 
                          ax: plt.Axes,
                          top_n: int = 10) -> None:
    """
    Plot feature importance visualization.
    
    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame containing feature importance scores
    ax : plt.Axes
        Matplotlib axes object for plotting
    top_n : int, optional
        Number of top features to plot
    """
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Create barplot
    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
    ax.set_title('Top Feature Importance')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')

def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate various performance metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    dict
        Dictionary containing various performance metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred > 0.5),
        'precision': precision_score(y_true, y_pred > 0.5),
        'recall': recall_score(y_true, y_pred > 0.5),
        'f1': f1_score(y_true, y_pred > 0.5)
    }
    
    return metrics

def process_gene_expression_data(expression_data: Union[pd.DataFrame, np.ndarray],
                               normalize: bool = True) -> np.ndarray:
    """
    Process gene expression data for model input.
    
    Parameters
    ----------
    expression_data : Union[pd.DataFrame, np.ndarray]
        Raw gene expression data
    normalize : bool, optional
        Whether to perform normalization
        
    Returns
    -------
    np.ndarray
        Processed expression data
    """
    if isinstance(expression_data, pd.DataFrame):
        expression_data = expression_data.values
    
    if normalize:
        # Log transformation for gene expression data
        expression_data = np.log2(expression_data + 1)
        
        # Scale the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        expression_data = scaler.fit_transform(expression_data)
    
    return expression_data

def save_results(results: Dict, filepath: str) -> None:
    """
    Save analysis results to file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing analysis results
    filepath : str
        Path to save the results
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f)

def load_results(filepath: str) -> Dict:
    """
    Load analysis results from file.
    
    Parameters
    ----------
    filepath : str
        Path to the results file
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    for key in ['precision', 'recall', 'fpr', 'tpr']:
        if key in results:
            results[key] = np.array(results[key])
    
    return results