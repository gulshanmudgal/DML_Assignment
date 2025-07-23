"""
UserBehaviorModel (Model B) for network latency prediction using user behavior features.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from models.base_model import BaseModel, validate_input_data


class UserBehaviorModel(BaseModel):
    """
    Model B: Specialized for user behavior features (User Count, Device Type).
    
    This model uses Random Forest Regressor to predict network latency based on
    user behavior features with appropriate preprocessing including categorical encoding.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the UserBehaviorModel.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.preprocessor = None
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.device_types_seen = None
    
    def _create_preprocessor(self, X: np.ndarray) -> ColumnTransformer:
        """
        Create preprocessor for user behavior features.
        
        Args:
            X: Feature matrix with User Count and Device Type
            
        Returns:
            Configured ColumnTransformer
        """
        # Column 0: User Count (numerical) - needs scaling
        # Column 1: Device Type (categorical) - needs one-hot encoding
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [0]),  # User Count
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), [1])  # Device Type
            ]
        )
        
        return preprocessor
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the user behavior model on user count and device type features.
        
        Args:
            X: Feature matrix with shape (n_samples, 2) containing:
               - Column 0: User Count (integer)
               - Column 1: Device Type (string/categorical)
            y: Target vector with network latency values (ms)
        """
        validate_input_data(X, y)
        
        if X.shape[1] != 2:
            raise ValueError("UserBehaviorModel expects exactly 2 features: User Count and Device Type")
        
        # Store unique device types for validation
        self.device_types_seen = set(X[:, 1])
        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Train the Random Forest model
        self.model.fit(X_processed, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained user behavior model.
        
        Args:
            X: Feature matrix with shape (n_samples, 2) containing:
               - Column 0: User Count (integer)
               - Column 1: Device Type (string/categorical)
               
        Returns:
            Predicted latency values (ms)
        """
        validate_input_data(X)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if X.shape[1] != 2:
            raise ValueError("UserBehaviorModel expects exactly 2 features: User Count and Device Type")
        
        # Process features using the fitted preprocessor
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        return self.model.predict(X_processed)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained Random Forest model.
        
        Note: The importance array will have more elements than input features
        due to one-hot encoding of Device Type.
        
        Returns:
            Array of feature importances for all processed features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.model.feature_importances_
    
    def get_feature_names(self) -> list:
        """
        Get the names of the original features used by this model.
        
        Returns:
            List of original feature names
        """
        return ['User_Count', 'Device_Type']
    
    def get_processed_feature_names(self) -> list:
        """
        Get the names of all processed features after preprocessing.
        
        Returns:
            List of processed feature names including one-hot encoded categories
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting processed feature names")
        
        # Get feature names from the preprocessor
        feature_names = []
        
        # Numerical features (User Count)
        feature_names.extend(['User_Count_scaled'])
        
        # Categorical features (Device Type one-hot encoded)
        if hasattr(self.preprocessor.named_transformers_['cat'], 'categories_'):
            categories = self.preprocessor.named_transformers_['cat'].categories_[0]
            # OneHotEncoder with drop='first' drops the first category
            for cat in categories[1:]:  # Skip first category due to drop='first'
                feature_names.append(f'Device_Type_{cat}')
        
        return feature_names
    
    def get_device_types_seen(self) -> set:
        """
        Get the device types seen during training.
        
        Returns:
            Set of device types encountered during training
        """
        if self.device_types_seen is None:
            raise ValueError("Model must be trained before getting device types")
        
        return self.device_types_seen.copy()