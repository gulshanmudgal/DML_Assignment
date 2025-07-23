"""
Base model abstract class and common evaluation metrics for network latency prediction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseModel(ABC):
    """
    Abstract base class for all network latency prediction models.
    
    This class defines the common interface that all models must implement,
    including training, prediction, and evaluation methods.
    """
    
    def __init__(self):
        self.is_trained = False
        self.model = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True target values of shape (n_samples,)
            
        Returns:
            Dictionary containing evaluation metrics (MAE, RMSE, R² Score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        return calculate_metrics(y, predictions)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common evaluation metrics for regression models.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing MAE, RMSE, and R² Score
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2
    }


def validate_input_data(X: np.ndarray, y: np.ndarray = None) -> None:
    """
    Validate input data for training or prediction.
    
    Args:
        X: Feature matrix
        y: Target vector (optional, for training validation)
        
    Raises:
        ValueError: If data validation fails
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    # Check for NaN values - handle both numeric and mixed data types
    if X.dtype.kind in ['f', 'i', 'u']:  # float, int, unsigned int
        if np.any(np.isnan(X.astype(float))):
            raise ValueError("X contains NaN values")
    else:
        # Mixed data types - check each column individually
        for i in range(X.shape[1]):
            col = X[:, i]
            # Try to convert to float to check for NaN, skip if it's categorical
            try:
                col_numeric = col.astype(float)
                if np.any(np.isnan(col_numeric)):
                    raise ValueError(f"X contains NaN values in column {i}")
            except (ValueError, TypeError):
                # Column contains non-numeric data (categorical), skip NaN check
                continue
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
        
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")