"""
InfrastructureModel (Model A) for network latency prediction using infrastructure features.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel, validate_input_data


class InfrastructureModel(BaseModel):
    """
    Model A: Specialized for infrastructure features (Signal Strength, Network Traffic).
    
    This model uses Random Forest Regressor to predict network latency based on
    infrastructure-related features with appropriate preprocessing and scaling.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the InfrastructureModel.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the infrastructure model on signal strength and network traffic features.
        
        Args:
            X: Feature matrix with shape (n_samples, 2) containing:
               - Column 0: Signal Strength (dBm)
               - Column 1: Network Traffic (MB)
            y: Target vector with network latency values (ms)
        """
        validate_input_data(X, y)
        
        if X.shape[1] != 2:
            raise ValueError("InfrastructureModel expects exactly 2 features: Signal Strength and Network Traffic")
        
        # Scale the features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the Random Forest model
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained infrastructure model.
        
        Args:
            X: Feature matrix with shape (n_samples, 2) containing:
               - Column 0: Signal Strength (dBm)
               - Column 1: Network Traffic (MB)
               
        Returns:
            Predicted latency values (ms)
        """
        validate_input_data(X)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if X.shape[1] != 2:
            raise ValueError("InfrastructureModel expects exactly 2 features: Signal Strength and Network Traffic")
        
        # Scale the features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained Random Forest model.
        
        Returns:
            Array of feature importances [Signal Strength, Network Traffic]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.model.feature_importances_
    
    def get_feature_names(self) -> list:
        """
        Get the names of the features used by this model.
        
        Returns:
            List of feature names
        """
        return ['Signal_Strength_dBm', 'Network_Traffic_MB']