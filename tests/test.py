import unittest
import numpy as np
from oocyteforestboost.predictor import CellInteractionPredictor

class TestCellInteractionPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = CellInteractionPredictor(random_state=42)
        
        # Generate synthetic test data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 20
        
        self.cell_type1_expression = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.cell_type2_expression = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.y = np.random.binomial(1, 0.3, self.n_samples)

    def test_prepare_features(self):
        """Test feature preparation"""
        features = self.predictor.prepare_features(
            self.cell_type1_expression,
            self.cell_type2_expression
        )
        
        # Check feature dimensions
        expected_feature_count = self.n_features * 2 + self.n_features + 1  # basic + interaction + correlation
        self.assertEqual(features.shape[1], expected_feature_count)
        
        # Check for NaN values
        self.assertFalse(np.isnan(features).any())

    def test_train_models(self):
        """Test model training"""
        features = self.predictor.prepare_features(
            self.cell_type1_expression,
            self.cell_type2_expression
        )
        
        self.predictor.train_models(features, self.y)
        
        # Check if models were created
        self.assertIsNotNone(self.predictor.rf_model)
        self.assertIsNotNone(self.predictor.xgb_model)

    def test_predict_proba(self):
        """Test probability predictions"""
        features = self.predictor.prepare_features(
            self.cell_type1_expression,
            self.cell_type2_expression
        )
        
        self.predictor.train_models(features, self.y)
        predictions = self.predictor.predict_proba(features)
        
        # Check prediction shape and values
        self.assertEqual(len(predictions), self.n_samples)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))

    def test_evaluate_models(self):
        """Test model evaluation"""
        features = self.predictor.prepare_features(
            self.cell_type1_expression,
            self.cell_type2_expression
        )
        
        self.predictor.train_models(features, self.y)
        eval_results = self.predictor.evaluate_models(features, self.y)
        
        # Check evaluation metrics
        self.assertIn('pr_auc', eval_results)
        self.assertIn('roc_auc', eval_results)
        self.assertIn('feature_importance', eval_results)
        
        # Check metric values
        self.assertTrue(0 <= eval_results['pr_auc'] <= 1)
        self.assertTrue(0 <= eval_results['roc_auc'] <= 1)

    def test_fdr_control(self):
        """Test FDR control"""
        predictions = np.random.random(100)
        significant = self.predictor.apply_fdr_control(predictions, threshold=0.05)
        
        # Check output type and shape
        self.assertTrue(isinstance(significant, np.ndarray))
        self.assertEqual(len(significant), 100)
        
        # Check if output is boolean
        self.assertTrue(np.issubdtype(significant.dtype, np.bool_))

if __name__ == '__main__':
    unittest.main()