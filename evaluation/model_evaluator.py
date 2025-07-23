"""
ModelEvaluator class for comprehensive model evaluation and comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from models.base_model import BaseModel, calculate_metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation system for network latency prediction models.
    
    This class provides functionality to evaluate individual models, compare multiple models,
    and perform statistical significance testing for model performance differences.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results = {}
        self.comparison_results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for regression models.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MAE, RMSE, and R² Score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate metrics for empty arrays")
        
        # Handle NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(mask):
            raise ValueError("All values are NaN, cannot calculate metrics")
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Additional metrics
        mape = self._calculate_mape(y_true_clean, y_pred_clean)
        residuals = y_true_clean - y_pred_clean
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2,
            'MAPE': mape,
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals),
            'Sample_Size': len(y_true_clean)
        }
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            MAPE value as percentage
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def evaluate_model(self, model: BaseModel, X: Union[np.ndarray, pd.DataFrame], 
                      y: np.ndarray, model_name: str = None) -> Dict[str, float]:
        """
        Evaluate a single model's performance.
        
        Args:
            model: Trained model instance
            X: Feature matrix or DataFrame
            y: True target values
            model_name: Optional name for the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not model.is_trained:
            raise ValueError(f"Model {model_name or 'unnamed'} must be trained before evaluation")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, predictions)
        
        # Store results
        if model_name:
            self.evaluation_results[model_name] = {
                'metrics': metrics,
                'predictions': predictions.copy(),
                'model_type': type(model).__name__
            }
        
        return metrics
    
    def compare_models(self, models: Dict[str, BaseModel], X: Union[np.ndarray, pd.DataFrame], 
                      y: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X: Feature matrix or DataFrame
            y: True target values
            
        Returns:
            DataFrame with comparison results
        """
        if not models:
            raise ValueError("At least one model must be provided for comparison")
        
        comparison_data = []
        
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X, y, model_name)
                
                # Add model name and type to metrics
                metrics['Model_Name'] = model_name
                metrics['Model_Type'] = type(model).__name__
                
                comparison_data.append(metrics)
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate model {model_name}: {str(e)}")
                continue
        
        if not comparison_data:
            raise ValueError("No models could be successfully evaluated")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reorder columns for better readability
        column_order = ['Model_Name', 'Model_Type', 'MAE', 'RMSE', 'R2_Score', 
                       'MAPE', 'Mean_Residual', 'Std_Residual', 'Sample_Size']
        comparison_df = comparison_df[column_order]
        
        # Sort by R² Score (descending) as primary metric
        comparison_df = comparison_df.sort_values('R2_Score', ascending=False).reset_index(drop=True)
        
        # Store comparison results
        self.comparison_results = {
            'comparison_table': comparison_df,
            'best_model': comparison_df.iloc[0]['Model_Name'],
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        return comparison_df
    
    def statistical_significance_test(self, model1_predictions: np.ndarray, 
                                    model2_predictions: np.ndarray, 
                                    y_true: np.ndarray,
                                    test_type: str = 'paired_t_test',
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance testing for model performance differences.
        
        Args:
            model1_predictions: Predictions from first model
            model2_predictions: Predictions from second model
            y_true: True target values
            test_type: Type of statistical test ('paired_t_test', 'wilcoxon', 'mcnemar')
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if len(model1_predictions) != len(model2_predictions) or len(model1_predictions) != len(y_true):
            raise ValueError("All arrays must have the same length")
        
        # Calculate residuals for each model
        residuals1 = np.abs(y_true - model1_predictions)
        residuals2 = np.abs(y_true - model2_predictions)
        
        # Calculate performance metrics for comparison
        mae1 = np.mean(residuals1)
        mae2 = np.mean(residuals2)
        
        # Perform statistical test
        if test_type == 'paired_t_test':
            # Paired t-test on absolute residuals
            statistic, p_value = stats.ttest_rel(residuals1, residuals2)
            test_name = "Paired t-test"
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric alternative)
            statistic, p_value = stats.wilcoxon(residuals1, residuals2, 
                                              alternative='two-sided', zero_method='zsplit')
            test_name = "Wilcoxon signed-rank test"
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Effect size (Cohen's d for paired samples)
        diff = residuals1 - residuals2
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) != 0 else 0
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_significant': is_significant,
            'effect_size': effect_size,
            'mae_model1': mae1,
            'mae_model2': mae2,
            'mae_difference': mae1 - mae2,
            'sample_size': len(y_true),
            'interpretation': self._interpret_significance_test(
                is_significant, mae1, mae2, p_value, alpha
            )
        }
    
    def _interpret_significance_test(self, is_significant: bool, mae1: float, 
                                   mae2: float, p_value: float, alpha: float) -> str:
        """
        Provide interpretation of statistical significance test results.
        
        Args:
            is_significant: Whether the test is statistically significant
            mae1: MAE of first model
            mae2: MAE of second model
            p_value: P-value from the test
            alpha: Significance level
            
        Returns:
            Human-readable interpretation
        """
        if is_significant:
            better_model = "Model 1" if mae1 < mae2 else "Model 2"
            worse_model = "Model 2" if mae1 < mae2 else "Model 1"
            return (f"The performance difference is statistically significant (p={p_value:.4f} < {alpha}). "
                   f"{better_model} performs significantly better than {worse_model}.")
        else:
            return (f"The performance difference is not statistically significant (p={p_value:.4f} >= {alpha}). "
                   f"There is no significant difference between the models.")
    
    def cross_validate_models(self, models: Dict[str, BaseModel], 
                            X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
                            cv: int = 5, scoring: str = 'neg_mean_squared_error') -> pd.DataFrame:
        """
        Perform cross-validation evaluation for multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            X: Feature matrix or DataFrame
            y: Target values
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            DataFrame with cross-validation results
        """
        cv_results = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'model') and model.model is not None:
                    # For models with sklearn estimators
                    if hasattr(model, '_prepare_features') and isinstance(X, pd.DataFrame):
                        # Handle models that need DataFrame preprocessing
                        X_processed = model._prepare_features(X, fit_encoders=False)
                    elif hasattr(model, 'scaler') and hasattr(model.scaler, 'transform'):
                        # Handle models with scalers
                        X_processed = model.scaler.transform(X)
                    else:
                        X_processed = X
                    
                    scores = cross_val_score(model.model, X_processed, y, cv=cv, scoring=scoring)
                    
                    cv_results.append({
                        'Model_Name': model_name,
                        'Model_Type': type(model).__name__,
                        'CV_Mean': np.mean(scores),
                        'CV_Std': np.std(scores),
                        'CV_Min': np.min(scores),
                        'CV_Max': np.max(scores)
                    })
                    
            except Exception as e:
                warnings.warn(f"Cross-validation failed for model {model_name}: {str(e)}")
                continue
        
        if not cv_results:
            raise ValueError("Cross-validation failed for all models")
        
        cv_df = pd.DataFrame(cv_results)
        
        # Sort by mean CV score (higher is better for negative metrics)
        cv_df = cv_df.sort_values('CV_Mean', ascending=False).reset_index(drop=True)
        
        return cv_df
    
    def generate_performance_summary(self, models: Dict[str, BaseModel], 
                                   X: Union[np.ndarray, pd.DataFrame], 
                                   y: np.ndarray) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary for all models.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X: Feature matrix or DataFrame
            y: True target values
            
        Returns:
            Dictionary with comprehensive performance summary
        """
        # Basic comparison
        comparison_df = self.compare_models(models, X, y)
        
        # Cross-validation results
        try:
            cv_results = self.cross_validate_models(models, X, y)
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {str(e)}")
            cv_results = None
        
        # Statistical significance tests between top models
        significance_tests = {}
        if len(models) >= 2:
            model_names = list(models.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1_name = model_names[i]
                    model2_name = model_names[j]
                    
                    try:
                        pred1 = self.evaluation_results[model1_name]['predictions']
                        pred2 = self.evaluation_results[model2_name]['predictions']
                        
                        test_result = self.statistical_significance_test(pred1, pred2, y)
                        significance_tests[f"{model1_name}_vs_{model2_name}"] = test_result
                        
                    except Exception as e:
                        warnings.warn(f"Significance test failed for {model1_name} vs {model2_name}: {str(e)}")
                        continue
        
        # Best model identification
        best_model_name = comparison_df.iloc[0]['Model_Name']
        best_model_metrics = comparison_df.iloc[0].to_dict()
        
        return {
            'performance_comparison': comparison_df,
            'cross_validation_results': cv_results,
            'statistical_significance_tests': significance_tests,
            'best_model': {
                'name': best_model_name,
                'metrics': best_model_metrics
            },
            'summary_statistics': {
                'total_models_evaluated': len(models),
                'evaluation_timestamp': pd.Timestamp.now(),
                'dataset_size': len(y)
            }
        }
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """
        Get stored evaluation results.
        
        Returns:
            Dictionary with all evaluation results
        """
        return {
            'individual_evaluations': self.evaluation_results,
            'comparison_results': self.comparison_results
        }
    
    def clear_results(self) -> None:
        """Clear all stored evaluation results."""
        self.evaluation_results.clear()
        self.comparison_results.clear()