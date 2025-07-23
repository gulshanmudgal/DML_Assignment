"""
MonolithicModel for network latency prediction using all features together.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.base_model import BaseModel, validate_input_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonolithicModel(BaseModel):
    """
    Monolithic model that uses all features together for network latency prediction.
    
    This model serves as a baseline for comparison with partitioned approaches.
    It processes all available features (infrastructure, user behavior, and location)
    in a single unified model without any partitioning strategy.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the MonolithicModel.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.device_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.location_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Track feature names for interpretability
        self.feature_names = []
    
    def _prepare_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> np.ndarray:
        """
        Prepare all features for training or prediction.
        
        Args:
            df: DataFrame with all features
            fit_encoders: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Processed feature matrix
        """
        # Expected feature columns (excluding Tower ID and target)
        expected_features = [
            'Signal Strength (dBm)', 'Network Traffic (MB)', 
            'User Count', 'Device Type', 'Location Type'
        ]
        
        # Check for missing columns
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract numerical features
        numerical_features = df[['Signal Strength (dBm)', 'Network Traffic (MB)', 'User Count']].copy()
        
        # Handle missing values in numerical features
        numerical_features = numerical_features.fillna(numerical_features.median())
        
        # Extract categorical features
        device_features = df[['Device Type']].copy()
        device_features = device_features.fillna('Unknown')
        
        location_features = df[['Location Type']].copy()
        location_features = location_features.fillna('Unknown')
        
        # Process features
        if fit_encoders:
            # Fit and transform
            numerical_scaled = self.scaler.fit_transform(numerical_features)
            device_encoded = self.device_encoder.fit_transform(device_features)
            location_encoded = self.location_encoder.fit_transform(location_features)
            
            # Store feature names
            self.feature_names = list(numerical_features.columns)
            
            if hasattr(self.device_encoder, 'categories_'):
                device_categories = self.device_encoder.categories_[0][1:]  # Skip first due to drop='first'
                self.feature_names.extend([f'Device_Type_{cat}' for cat in device_categories])
            
            if hasattr(self.location_encoder, 'categories_'):
                location_categories = self.location_encoder.categories_[0][1:]  # Skip first due to drop='first'
                self.feature_names.extend([f'Location_Type_{cat}' for cat in location_categories])
            
            logger.info("Fitted encoders for MonolithicModel")
            logger.info(f"Device types found: {self.device_encoder.categories_[0]}")
            logger.info(f"Location types found: {self.location_encoder.categories_[0]}")
        else:
            # Transform only
            if (not hasattr(self.scaler, 'mean_') or 
                not hasattr(self.device_encoder, 'categories_') or
                not hasattr(self.location_encoder, 'categories_')):
                raise ValueError("Encoders not fitted. Call train() first.")
            
            numerical_scaled = self.scaler.transform(numerical_features)
            device_encoded = self.device_encoder.transform(device_features)
            location_encoded = self.location_encoder.transform(location_features)
            
            logger.info("Transformed features for MonolithicModel prediction")
        
        # Combine all features
        X_combined = np.hstack([numerical_scaled, device_encoded, location_encoded])
        
        logger.info(f"Prepared MonolithicModel features with shape {X_combined.shape}")
        
        return X_combined
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train the monolithic model on all available features.
        
        Args:
            X: Feature DataFrame with all features
            y: Target vector with network latency values (ms)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for MonolithicModel")
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Prepare features
        X_processed = self._prepare_features(X, fit_encoders=True)
        
        # Validate processed data
        validate_input_data(X_processed, y)
        
        # Train the model
        self.model.fit(X_processed, y)
        self.is_trained = True
        
        logger.info(f"Successfully trained MonolithicModel on {X_processed.shape[0]} samples with {X_processed.shape[1]} features")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained monolithic model.
        
        Args:
            X: Feature DataFrame with all features
            
        Returns:
            Predicted latency values (ms)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for MonolithicModel")
        
        if not self.is_trained:
            raise ValueError("MonolithicModel must be trained before prediction")
        
        # Prepare features
        X_processed = self._prepare_features(X, fit_encoders=False)
        
        # Validate processed data
        validate_input_data(X_processed)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        logger.info(f"Made {len(predictions)} predictions using MonolithicModel")
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained Random Forest model.
        
        Returns:
            Array of feature importances
        """
        if not self.is_trained:
            raise ValueError("MonolithicModel must be trained before getting feature importance")
        
        return self.model.feature_importances_
    
    def get_feature_names(self) -> list:
        """
        Get the names of the features used by this model.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def get_feature_importance_dict(self) -> dict:
        """
        Get feature importance as a dictionary with feature names.
        
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_trained:
            raise ValueError("MonolithicModel must be trained before getting feature importance")
        
        importance_values = self.get_feature_importance()
        return dict(zip(self.feature_names, importance_values))
    
    def get_model_info(self) -> dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            raise ValueError("MonolithicModel must be trained before getting model info")
        
        return {
            'model_type': 'MonolithicModel',
            'algorithm': 'RandomForestRegressor',
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'is_trained': self.is_trained
        }