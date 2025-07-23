"""
FusionModel for combining Model A (Infrastructure) and Model B (User Behavior) predictions.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict, Optional
from models.base_model import BaseModel, validate_input_data, calculate_metrics
from models.infrastructure_model import InfrastructureModel
from models.user_behavior_model import UserBehaviorModel


class FusionModel(BaseModel):
    """
    Fusion model that combines predictions from InfrastructureModel (Model A) 
    and UserBehaviorModel (Model B) using either weighted averaging or meta-learning.
    
    This model implements vertical partitioning by training separate models on
    different feature subsets and then combining their predictions.
    """
    
    def __init__(self, fusion_strategy='weighted_average', random_state=42):
        """
        Initialize the FusionModel.
        
        Args:
            fusion_strategy: Either 'weighted_average' or 'meta_learner'
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.random_state = random_state
        
        # Component models
        self.infrastructure_model = InfrastructureModel(random_state=random_state)
        self.user_behavior_model = UserBehaviorModel(random_state=random_state)
        
        # Fusion parameters
        self.weights = None  # For weighted averaging
        self.meta_learner = None  # For meta-learning approach
        
        if fusion_strategy not in ['weighted_average', 'meta_learner']:
            raise ValueError("fusion_strategy must be 'weighted_average' or 'meta_learner'")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the fusion model on the provided data.
        
        Args:
            X: Feature matrix with shape (n_samples, 4) containing:
               - Column 0: Signal Strength (dBm)
               - Column 1: Network Traffic (MB)
               - Column 2: User Count (integer)
               - Column 3: Device Type (string/categorical)
            y: Target vector with network latency values (ms)
        """
        validate_input_data(X, y)
        
        if X.shape[1] != 4:
            raise ValueError("FusionModel expects exactly 4 features: Signal Strength, Network Traffic, User Count, Device Type")
        
        # Split features for each model
        X_infrastructure = X[:, :2]  # Signal Strength, Network Traffic
        X_user_behavior = X[:, 2:]   # User Count, Device Type
        
        # Train individual models
        self.infrastructure_model.train(X_infrastructure, y)
        self.user_behavior_model.train(X_user_behavior, y)
        
        # Train fusion mechanism
        if self.fusion_strategy == 'weighted_average':
            self._train_weighted_average(X_infrastructure, X_user_behavior, y)
        elif self.fusion_strategy == 'meta_learner':
            self._train_meta_learner(X_infrastructure, X_user_behavior, y)
        
        self.is_trained = True
    
    def _train_weighted_average(self, X_infra: np.ndarray, X_user: np.ndarray, y: np.ndarray) -> None:
        """
        Train weighted averaging fusion by determining optimal weights based on validation performance.
        
        Args:
            X_infra: Infrastructure features
            X_user: User behavior features
            y: Target values
        """
        # Use cross-validation to determine model performance
        infra_scores = cross_val_score(
            self.infrastructure_model.model, 
            self.infrastructure_model.scaler.transform(X_infra), 
            y, 
            cv=3, 
            scoring='neg_mean_squared_error'
        )
        
        user_scores = cross_val_score(
            self.user_behavior_model.model,
            self.user_behavior_model.preprocessor.transform(X_user),
            y,
            cv=3,
            scoring='neg_mean_squared_error'
        )
        
        # Convert negative MSE to positive and then to weights (inverse relationship)
        infra_mse = -np.mean(infra_scores)
        user_mse = -np.mean(user_scores)
        
        # Calculate weights inversely proportional to MSE (better models get higher weights)
        total_inverse_mse = (1 / infra_mse) + (1 / user_mse)
        weight_infra = (1 / infra_mse) / total_inverse_mse
        weight_user = (1 / user_mse) / total_inverse_mse
        
        self.weights = {
            'infrastructure': weight_infra,
            'user_behavior': weight_user
        }
    
    def _train_meta_learner(self, X_infra: np.ndarray, X_user: np.ndarray, y: np.ndarray) -> None:
        """
        Train meta-learner fusion using linear regression on individual model predictions.
        
        Args:
            X_infra: Infrastructure features
            X_user: User behavior features
            y: Target values
        """
        # Get predictions from individual models
        infra_pred = self.infrastructure_model.predict(X_infra)
        user_pred = self.user_behavior_model.predict(X_user)
        
        # Create meta-features matrix
        meta_features = np.column_stack([infra_pred, user_pred])
        
        # Train linear regression meta-learner
        self.meta_learner = LinearRegression()
        self.meta_learner.fit(meta_features, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained fusion model.
        
        Args:
            X: Feature matrix with shape (n_samples, 4) containing:
               - Column 0: Signal Strength (dBm)
               - Column 1: Network Traffic (MB)
               - Column 2: User Count (integer)
               - Column 3: Device Type (string/categorical)
               
        Returns:
            Predicted latency values (ms)
        """
        validate_input_data(X)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if X.shape[1] != 4:
            raise ValueError("FusionModel expects exactly 4 features: Signal Strength, Network Traffic, User Count, Device Type")
        
        # Split features for each model
        X_infrastructure = X[:, :2]  # Signal Strength, Network Traffic
        X_user_behavior = X[:, 2:]   # User Count, Device Type
        
        # Get predictions from individual models
        infra_pred = self.infrastructure_model.predict(X_infrastructure)
        user_pred = self.user_behavior_model.predict(X_user_behavior)
        
        # Combine predictions based on fusion strategy
        if self.fusion_strategy == 'weighted_average':
            return self._predict_weighted_average(infra_pred, user_pred)
        elif self.fusion_strategy == 'meta_learner':
            return self._predict_meta_learner(infra_pred, user_pred)
    
    def _predict_weighted_average(self, infra_pred: np.ndarray, user_pred: np.ndarray) -> np.ndarray:
        """
        Combine predictions using weighted averaging.
        
        Args:
            infra_pred: Predictions from infrastructure model
            user_pred: Predictions from user behavior model
            
        Returns:
            Combined predictions
        """
        return (self.weights['infrastructure'] * infra_pred + 
                self.weights['user_behavior'] * user_pred)
    
    def _predict_meta_learner(self, infra_pred: np.ndarray, user_pred: np.ndarray) -> np.ndarray:
        """
        Combine predictions using meta-learner.
        
        Args:
            infra_pred: Predictions from infrastructure model
            user_pred: Predictions from user behavior model
            
        Returns:
            Combined predictions
        """
        meta_features = np.column_stack([infra_pred, user_pred])
        return self.meta_learner.predict(meta_features)
    
    def get_fusion_weights(self) -> Optional[Dict[str, float]]:
        """
        Get the fusion weights (only applicable for weighted_average strategy).
        
        Returns:
            Dictionary with model weights or None if using meta_learner
        """
        if self.fusion_strategy == 'weighted_average' and self.is_trained:
            return self.weights.copy()
        return None
    
    def get_meta_learner_coefficients(self) -> Optional[np.ndarray]:
        """
        Get the meta-learner coefficients (only applicable for meta_learner strategy).
        
        Returns:
            Array of coefficients or None if using weighted_average
        """
        if self.fusion_strategy == 'meta_learner' and self.is_trained and self.meta_learner is not None:
            return self.meta_learner.coef_.copy()
        return None
    
    def get_individual_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from individual models separately.
        
        Args:
            X: Feature matrix with shape (n_samples, 4)
            
        Returns:
            Tuple of (infrastructure_predictions, user_behavior_predictions)
        """
        validate_input_data(X)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if X.shape[1] != 4:
            raise ValueError("FusionModel expects exactly 4 features")
        
        # Split features for each model
        X_infrastructure = X[:, :2]  # Signal Strength, Network Traffic
        X_user_behavior = X[:, 2:]   # User Count, Device Type
        
        # Get predictions from individual models
        infra_pred = self.infrastructure_model.predict(X_infrastructure)
        user_pred = self.user_behavior_model.predict(X_user_behavior)
        
        return infra_pred, user_pred
    
    def compare_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of individual models and fusion model.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with performance metrics for each model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before comparison")
        
        # Get predictions from all models
        infra_pred, user_pred = self.get_individual_predictions(X)
        fusion_pred = self.predict(X)
        
        # Calculate metrics for each model
        results = {
            'infrastructure_model': calculate_metrics(y, infra_pred),
            'user_behavior_model': calculate_metrics(y, user_pred),
            'fusion_model': calculate_metrics(y, fusion_pred)
        }
        
        return results
    
    def get_feature_names(self) -> list:
        """
        Get the names of the features used by this model.
        
        Returns:
            List of feature names
        """
        return ['Signal_Strength_dBm', 'Network_Traffic_MB', 'User_Count', 'Device_Type']