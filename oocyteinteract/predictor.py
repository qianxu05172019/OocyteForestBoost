"""
OocyteForestBoost predictor module.
Author: Qian Xu
This module implements the core cell-cell interaction prediction functionality.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Tuple, List, Union, Optional
from .utils import validate_input, calculate_feature_importance, plot_feature_importance

class CellInteractionPredictor:
    """
    A class for predicting cell-cell interactions during oocyte maturation using
    ensemble learning with Random Forest and XGBoost.
    """
    
    def __init__(self, random_state: int = 42, rf_params: Optional[Dict] = None, 
                 xgb_params: Optional[Dict] = None):
        """
        Initialize the predictor with customizable parameters.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        rf_params : dict, optional
            Parameters for Random Forest classifier
        xgb_params : dict, optional
            Parameters for XGBoost classifier
        """
        self.random_state = random_state
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # Default parameters for Random Forest
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': random_state
        }
        if rf_params:
            self.rf_params.update(rf_params)
            
        # Default parameters for XGBoost
        self.xgb_params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        if xgb_params:
            self.xgb_params.update(xgb_params)

    def prepare_features(self, cell_type1_expression: np.ndarray, 
                        cell_type2_expression: np.ndarray) -> np.ndarray:
        """
        Prepare features for cell-cell interaction prediction.

        Parameters
        ----------
        cell_type1_expression : np.ndarray
            Gene expression matrix for first cell type
        cell_type2_expression : np.ndarray
            Gene expression matrix for second cell type

        Returns
        -------
        np.ndarray
            Combined feature matrix
        """
        # Validate input
        validate_input(cell_type1_expression, cell_type2_expression)
        
        # Basic features
        features = np.hstack([
            cell_type1_expression,
            cell_type2_expression
        ])
        
        # Calculate interaction features
        interaction_features = cell_type1_expression * cell_type2_expression
        
        # Calculate correlation features
        correlation_features = np.array([
            stats.pearsonr(cell_type1_expression[i], cell_type2_expression[i])[0]
            for i in range(len(cell_type1_expression))
        ]).reshape(-1, 1)
        
        # Combine all features
        features = np.hstack([
            features,
            interaction_features,
            correlation_features
        ])
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        return features

    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train both Random Forest and XGBoost models.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        """
        # Initialize models with parameters
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.xgb_model = XGBClassifier(**self.xgb_params)
        
        # Train models
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using both models.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Ensemble probability predictions
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models must be trained before making predictions")
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        
        # Weighted average of predictions
        ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred
        
        return ensemble_pred

    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate models using multiple metrics.

        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test labels

        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Get feature importance
        feature_importance = calculate_feature_importance(self.rf_model)
        
        return {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'feature_importance': feature_importance,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr
        }

    def apply_fdr_control(self, predictions: np.ndarray, 
                         threshold: float = 0.05) -> np.ndarray:
        """
        Apply FDR control to predictions.

        Parameters
        ----------
        predictions : np.ndarray
            Probability predictions
        threshold : float, optional
            FDR threshold

        Returns
        -------
        np.ndarray
            Boolean array indicating significant interactions
        """
        # Convert probabilities to p-values
        pvalues = 1 - predictions
        
        # Apply FDR correction
        _, qvalues, _, _ = multipletests(pvalues, alpha=threshold, method='fdr_bh')
        
        return qvalues < threshold

    def plot_evaluation_metrics(self, eval_results: Dict) -> None:
        """
        Plot evaluation metrics.

        Parameters
        ----------
        eval_results : dict
            Dictionary containing evaluation metrics
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot precision-recall curve
        ax1.plot(eval_results['recall'], eval_results['precision'])
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'Precision-Recall Curve (AUC = {eval_results["pr_auc"]:.3f})')
        
        # Plot ROC curve
        ax2.plot(eval_results['fpr'], eval_results['tpr'])
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'ROC Curve (AUC = {eval_results["roc_auc"]:.3f})')
        
        # Plot feature importance
        plot_feature_importance(eval_results['feature_importance'], ax3)
        
        plt.tight_layout()
        plt.show()