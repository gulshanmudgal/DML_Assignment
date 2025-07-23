"""
Geographical models for horizontal partitioning in network latency prediction.
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


class GeographicalModel(BaseModel):
    """
    Base class for geographical models (Urban and Rural).
    
    This class provides common functionality for models that are specialized
    for specific geographical contexts in horizontal partitioning.
    """
    
    def __init__(self, location_type: str, n_estimators=100, random_state=42, min_samples=5):
        """
        Initialize the GeographicalModel.
        
        Args:
            location_type: Type of location ('Urban' or 'Rural')
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
            min_samples: Minimum number of samples required for training
        """
        super().__init__()
        self.location_type = location_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_samples = min_samples
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.device_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Track feature names for interpretability
        self.feature_names = []
    
    def _validate_sample_size(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate that there are enough samples for meaningful training.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Raises:
            ValueError: If sample size is insufficient
        """
        if X.shape[0] < self.min_samples:
            raise ValueError(
                f"Insufficient samples for {self.location_type} model training. "
                f"Got {X.shape[0]} samples, minimum required: {self.min_samples}"
            )
        
        logger.info(f"{self.location_type} model has {X.shape[0]} samples for training")
    
    def _prepare_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> np.ndarray:
        """
        Prepare features for training or prediction.
        
        Args:
            df: DataFrame with all features
            fit_encoders: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Processed feature matrix
        """
        # Expected feature columns (excluding Tower ID and target)
        expected_features = [
            'Signal Strength (dBm)', 'Network Traffic (MB)', 
            'User Count', 'Device Type'
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
        categorical_features = df[['Device Type']].copy()
        categorical_features = categorical_features.fillna('Unknown')
        
        # Process features
        if fit_encoders:
            # Fit and transform
            numerical_scaled = self.scaler.fit_transform(numerical_features)
            categorical_encoded = self.device_encoder.fit_transform(categorical_features)
            
            # Store feature names
            self.feature_names = list(numerical_features.columns)
            if hasattr(self.device_encoder, 'categories_'):
                device_categories = self.device_encoder.categories_[0][1:]  # Skip first due to drop='first'
                self.feature_names.extend([f'Device_Type_{cat}' for cat in device_categories])
            
            logger.info(f"Fitted encoders for {self.location_type} model")
            logger.info(f"Device types found: {self.device_encoder.categories_[0]}")
        else:
            # Transform only
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.device_encoder, 'categories_'):
                raise ValueError("Encoders not fitted. Call train() first.")
            
            numerical_scaled = self.scaler.transform(numerical_features)
            categorical_encoded = self.device_encoder.transform(categorical_features)
            
            logger.info(f"Transformed features for {self.location_type} model prediction")
        
        # Combine features
        X_combined = np.hstack([numerical_scaled, categorical_encoded])
        
        logger.info(f"Prepared {self.location_type} features with shape {X_combined.shape}")
        
        return X_combined
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train the geographical model on location-specific data.
        
        Args:
            X: Feature DataFrame with all features
            y: Target vector with network latency values (ms)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for geographical models")
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Validate sample size
        self._validate_sample_size(X.values, y)
        
        # Prepare features
        X_processed = self._prepare_features(X, fit_encoders=True)
        
        # Validate processed data
        validate_input_data(X_processed, y)
        
        # Train the model
        self.model.fit(X_processed, y)
        self.is_trained = True
        
        logger.info(f"Successfully trained {self.location_type} model on {X_processed.shape[0]} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained geographical model.
        
        Args:
            X: Feature DataFrame with all features
            
        Returns:
            Predicted latency values (ms)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for geographical models")
        
        if not self.is_trained:
            raise ValueError(f"{self.location_type} model must be trained before prediction")
        
        # Prepare features
        X_processed = self._prepare_features(X, fit_encoders=False)
        
        # Validate processed data
        validate_input_data(X_processed)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        logger.info(f"Made {len(predictions)} predictions using {self.location_type} model")
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained Random Forest model.
        
        Returns:
            Array of feature importances
        """
        if not self.is_trained:
            raise ValueError(f"{self.location_type} model must be trained before getting feature importance")
        
        return self.model.feature_importances_
    
    def get_feature_names(self) -> list:
        """
        Get the names of the features used by this model.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()


class UrbanModel(GeographicalModel):
    """
    Specialized model for urban geographical contexts.
    
    This model is trained specifically on urban network data and optimized
    for predicting latency in urban environments with higher user density
    and different infrastructure characteristics.
    """
    
    def __init__(self, n_estimators=100, random_state=42, min_samples=5):
        """
        Initialize the UrbanModel.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
            min_samples: Minimum number of samples required for training
        """
        super().__init__(
            location_type='Urban',
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples=min_samples
        )


class RuralModel(GeographicalModel):
    """
    Specialized model for rural geographical contexts.
    
    This model is trained specifically on rural network data and optimized
    for predicting latency in rural environments with lower user density
    and different infrastructure characteristics.
    """
    
    def __init__(self, n_estimators=100, random_state=42, min_samples=5):
        """
        Initialize the RuralModel.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
            min_samples: Minimum number of samples required for training
        """
        super().__init__(
            location_type='Rural',
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples=min_samples
        )