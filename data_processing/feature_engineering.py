"""
Feature engineering module for network latency prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    FeatureEngineer class for handling feature engineering tasks
    for network latency prediction dataset.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.infra_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self.device_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.is_fitted = False
    
    def split_vertical_features(self, df: pd.DataFrame, include_target: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split features vertically into infrastructure and user behavior features.
        
        Args:
            df (pd.DataFrame): Input dataset
            include_target (bool): Whether to include target variable in both splits
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Infrastructure features, User behavior features
        """
        logger.info("Splitting features vertically for Model A (Infrastructure) and Model B (User Behavior)")
        
        # Validate required columns exist
        required_infra_cols = ['Tower ID', 'Signal Strength (dBm)', 'Network Traffic (MB)']
        required_user_cols = ['Tower ID', 'User Count', 'Device Type']
        
        missing_infra = [col for col in required_infra_cols if col not in df.columns]
        missing_user = [col for col in required_user_cols if col not in df.columns]
        
        if missing_infra:
            raise ValueError(f"Missing infrastructure columns: {missing_infra}")
        if missing_user:
            raise ValueError(f"Missing user behavior columns: {missing_user}")
        
        # Infrastructure features (Model A) - Signal Strength and Network Traffic
        infra_columns = ['Tower ID', 'Signal Strength (dBm)', 'Network Traffic (MB)']
        if include_target and 'Latency (ms)' in df.columns:
            infra_columns.append('Latency (ms)')
        
        infra_features = df[infra_columns].copy()
        
        # User behavior features (Model B) - User Count and Device Type
        user_columns = ['Tower ID', 'User Count', 'Device Type']
        if include_target and 'Latency (ms)' in df.columns:
            user_columns.append('Latency (ms)')
        
        user_features = df[user_columns].copy()
        
        logger.info(f"Created infrastructure features (Model A) with shape {infra_features.shape}")
        logger.info(f"Infrastructure features: {[col for col in infra_features.columns if col != 'Tower ID']}")
        logger.info(f"Created user behavior features (Model B) with shape {user_features.shape}")
        logger.info(f"User behavior features: {[col for col in user_features.columns if col != 'Tower ID']}")
        
        return infra_features, user_features
    
    def prepare_infrastructure_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare infrastructure features for Model A training/prediction.
        
        Args:
            df (pd.DataFrame): Infrastructure features dataframe
            fit_scaler (bool): Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled features (X), Target values (y) if available
        """
        logger.info("Preparing infrastructure features for Model A")
        
        # Extract features (exclude Tower ID and target)
        feature_columns = ['Signal Strength (dBm)', 'Network Traffic (MB)']
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X_df = df[feature_columns].copy()
        
        # Handle any remaining missing values
        X_df = X_df.fillna(X_df.median())
        
        # Scale features
        if fit_scaler:
            X_scaled = self.infra_scaler.fit_transform(X_df)
            logger.info("Fitted and transformed infrastructure features with StandardScaler")
        else:
            if not hasattr(self.infra_scaler, 'mean_'):
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            X_scaled = self.infra_scaler.transform(X_df)
            logger.info("Transformed infrastructure features with fitted StandardScaler")
        
        # Extract target if available
        y = None
        if 'Latency (ms)' in df.columns:
            y = df['Latency (ms)'].values
            logger.info(f"Extracted target variable with shape {y.shape}")
        
        logger.info(f"Prepared infrastructure features with shape {X_scaled.shape}")
        
        return X_scaled, y
    
    def prepare_user_behavior_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare user behavior features for Model B training/prediction.
        
        Args:
            df (pd.DataFrame): User behavior features dataframe
            fit_encoders (bool): Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed features (X), Target values (y) if available
        """
        logger.info("Preparing user behavior features for Model B")
        
        # Extract features (exclude Tower ID and target)
        if 'User Count' not in df.columns or 'Device Type' not in df.columns:
            raise ValueError("Missing required columns: User Count or Device Type")
        
        # Handle User Count (numerical)
        user_count = df['User Count'].fillna(df['User Count'].median()).values.reshape(-1, 1)
        
        # Handle Device Type (categorical)
        device_type = df['Device Type'].fillna('Unknown').values.reshape(-1, 1)
        
        # Encode categorical features
        if fit_encoders:
            device_encoded = self.device_encoder.fit_transform(device_type)
            user_count_scaled = self.user_scaler.fit_transform(user_count)
            logger.info("Fitted and transformed user behavior features")
            logger.info(f"Device types found: {self.device_encoder.categories_[0]}")
        else:
            if not hasattr(self.device_encoder, 'categories_'):
                raise ValueError("Encoders not fitted. Call with fit_encoders=True first.")
            device_encoded = self.device_encoder.transform(device_type)
            user_count_scaled = self.user_scaler.transform(user_count)
            logger.info("Transformed user behavior features with fitted encoders")
        
        # Combine features
        X_combined = np.hstack([user_count_scaled, device_encoded])
        
        # Extract target if available
        y = None
        if 'Latency (ms)' in df.columns:
            y = df['Latency (ms)'].values
            logger.info(f"Extracted target variable with shape {y.shape}")
        
        logger.info(f"Prepared user behavior features with shape {X_combined.shape}")
        logger.info(f"Features: User Count (scaled) + Device Type (one-hot encoded: {device_encoded.shape[1]} categories)")
        
        return X_combined, y
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get feature names for both model types.
        
        Returns:
            Dict[str, List[str]]: Dictionary with feature names for each model
        """
        feature_names = {
            'infrastructure': ['Signal Strength (dBm)', 'Network Traffic (MB)'],
            'user_behavior': ['User Count']
        }
        
        # Add device type feature names if encoder is fitted
        if hasattr(self.device_encoder, 'categories_'):
            device_features = [f'Device Type_{cat}' for cat in self.device_encoder.categories_[0][1:]]  # Skip first due to drop='first'
            feature_names['user_behavior'].extend(device_features)
        
        return feature_names
    
    def validate_vertical_split(self, infra_df: pd.DataFrame, user_df: pd.DataFrame) -> bool:
        """
        Validate that vertical split was performed correctly.
        
        Args:
            infra_df (pd.DataFrame): Infrastructure features dataframe
            user_df (pd.DataFrame): User behavior features dataframe
            
        Returns:
            bool: True if split is valid
        """
        logger.info("Validating vertical feature split")
        
        # Check that both dataframes have the same number of rows
        if len(infra_df) != len(user_df):
            logger.error(f"Row count mismatch: Infrastructure {len(infra_df)} vs User Behavior {len(user_df)}")
            return False
        
        # Check that Tower IDs match
        if not infra_df['Tower ID'].equals(user_df['Tower ID']):
            logger.error("Tower ID mismatch between infrastructure and user behavior features")
            return False
        
        # Check required columns
        required_infra = ['Signal Strength (dBm)', 'Network Traffic (MB)']
        required_user = ['User Count', 'Device Type']
        
        missing_infra = [col for col in required_infra if col not in infra_df.columns]
        missing_user = [col for col in required_user if col not in user_df.columns]
        
        if missing_infra:
            logger.error(f"Missing infrastructure columns: {missing_infra}")
            return False
        
        if missing_user:
            logger.error(f"Missing user behavior columns: {missing_user}")
            return False
        
        # Check for feature overlap (should not have overlapping features except Tower ID and target)
        infra_features = set(infra_df.columns) - {'Tower ID', 'Latency (ms)'}
        user_features = set(user_df.columns) - {'Tower ID', 'Latency (ms)'}
        
        overlap = infra_features.intersection(user_features)
        if overlap:
            logger.error(f"Feature overlap detected: {overlap}")
            return False
        
        logger.info("Vertical split validation passed")
        return True
    
    def split_horizontal_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data horizontally based on geographical location.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Urban data, Rural data
        """
        logger.info("Splitting data horizontally by location")
        
        # Check if Location Type column exists
        if 'Location Type' not in df.columns:
            logger.error("Location Type column not found in dataset")
            raise ValueError("Location Type column not found in dataset")
        
        # Split by location
        urban_data = df[df['Location Type'] == 'Urban'].copy()
        rural_data = df[df['Location Type'] == 'Rural'].copy()
        
        # Handle unknown locations using heuristics if needed
        unknown_data = df[~df['Location Type'].isin(['Urban', 'Rural'])]
        if len(unknown_data) > 0:
            logger.warning(f"Found {len(unknown_data)} rows with unknown location type")
            # Apply heuristic: Assign based on User Count and Signal Strength
            # Higher user count and stronger signal typically indicates urban areas
            unknown_assignments = []
            for idx, row in unknown_data.iterrows():
                if row['User Count'] > 50 and row['Signal Strength (dBm)'] > -85:
                    unknown_assignments.append(('urban', idx))
                else:
                    unknown_assignments.append(('rural', idx))
            
            # Assign unknown data to appropriate datasets
            for assignment, idx in unknown_assignments:
                if assignment == 'urban':
                    urban_data = pd.concat([urban_data, unknown_data.loc[[idx]]], ignore_index=True)
                else:
                    rural_data = pd.concat([rural_data, unknown_data.loc[[idx]]], ignore_index=True)
            
            logger.info(f"Assigned {len(unknown_assignments)} unknown location rows using heuristics")
        
        logger.info(f"Created urban dataset with shape {urban_data.shape}")
        logger.info(f"Created rural dataset with shape {rural_data.shape}")
        
        return urban_data, rural_data
    
    def validate_horizontal_split(self, urban_df: pd.DataFrame, rural_df: pd.DataFrame, original_df: pd.DataFrame) -> bool:
        """
        Validate that horizontal split was performed correctly.
        
        Args:
            urban_df (pd.DataFrame): Urban data subset
            rural_df (pd.DataFrame): Rural data subset
            original_df (pd.DataFrame): Original dataset
            
        Returns:
            bool: True if split is valid
        """
        logger.info("Validating horizontal data split")
        
        # Check that total rows match original dataset
        total_split_rows = len(urban_df) + len(rural_df)
        if total_split_rows != len(original_df):
            logger.error(f"Row count mismatch: Original {len(original_df)} vs Split {total_split_rows}")
            return False
        
        # Check that no Tower IDs are duplicated across splits
        urban_towers = set(urban_df['Tower ID']) if 'Tower ID' in urban_df.columns else set()
        rural_towers = set(rural_df['Tower ID']) if 'Tower ID' in rural_df.columns else set()
        
        overlap = urban_towers.intersection(rural_towers)
        if overlap:
            logger.error(f"Tower ID overlap between urban and rural splits: {overlap}")
            return False
        
        # Check that all Tower IDs from original are present in splits
        original_towers = set(original_df['Tower ID']) if 'Tower ID' in original_df.columns else set()
        split_towers = urban_towers.union(rural_towers)
        
        missing_towers = original_towers - split_towers
        if missing_towers:
            logger.error(f"Missing Tower IDs in splits: {missing_towers}")
            return False
        
        # Check minimum sample size requirements
        min_samples = 5  # Minimum samples for meaningful model training
        if len(urban_df) < min_samples:
            logger.warning(f"Urban dataset has only {len(urban_df)} samples (minimum recommended: {min_samples})")
        
        if len(rural_df) < min_samples:
            logger.warning(f"Rural dataset has only {len(rural_df)} samples (minimum recommended: {min_samples})")
        
        logger.info("Horizontal split validation passed")
        return True
    
    def apply_location_heuristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply heuristics to assign location types to rows with missing location data.
        
        Args:
            df (pd.DataFrame): Input dataset with potentially missing location data
            
        Returns:
            pd.DataFrame: Dataset with location types assigned using heuristics
        """
        logger.info("Applying location heuristics for missing location data")
        
        processed_df = df.copy()
        
        # Find rows with missing or unknown location data
        missing_location_mask = (
            processed_df['Location Type'].isnull() | 
            (processed_df['Location Type'] == 'Unknown') |
            (~processed_df['Location Type'].isin(['Urban', 'Rural']))
        )
        
        missing_count = missing_location_mask.sum()
        if missing_count == 0:
            logger.info("No missing location data found")
            return processed_df
        
        logger.info(f"Found {missing_count} rows with missing location data")
        
        # Apply heuristics based on multiple factors
        for idx in processed_df[missing_location_mask].index:
            row = processed_df.loc[idx]
            
            # Heuristic factors
            user_count = row.get('User Count', 0)
            signal_strength = row.get('Signal Strength (dBm)', -100)
            network_traffic = row.get('Network Traffic (MB)', 0)
            
            # Urban indicators:
            # - Higher user count (more people in urban areas)
            # - Stronger signal (better infrastructure in cities)
            # - Higher network traffic (more activity in urban areas)
            urban_score = 0
            
            if user_count > 40:  # High user count
                urban_score += 2
            elif user_count > 25:  # Medium user count
                urban_score += 1
            
            if signal_strength > -80:  # Strong signal
                urban_score += 2
            elif signal_strength > -90:  # Medium signal
                urban_score += 1
            
            if network_traffic > 200:  # High traffic
                urban_score += 1
            
            # Assign location based on score
            if urban_score >= 3:
                processed_df.loc[idx, 'Location Type'] = 'Urban'
            else:
                processed_df.loc[idx, 'Location Type'] = 'Rural'
        
        # Log assignment results
        assigned_urban = (processed_df.loc[missing_location_mask, 'Location Type'] == 'Urban').sum()
        assigned_rural = (processed_df.loc[missing_location_mask, 'Location Type'] == 'Rural').sum()
        
        logger.info(f"Assigned {assigned_urban} rows to Urban and {assigned_rural} rows to Rural using heuristics")
        
        return processed_df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Latency (ms)') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target column name
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features array, Target array
        """
        logger.info("Preparing features for model training")
        
        # Check if target column exists
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Extract target
        y = df[target_col].values
        
        # Extract features (exclude Tower ID and target)
        X_df = df.drop(columns=['Tower ID', target_col])
        
        # Handle categorical features
        cat_columns = X_df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            # One-hot encode categorical columns
            dummies = pd.get_dummies(X_df[col], prefix=col, drop_first=True)
            X_df = pd.concat([X_df.drop(columns=[col]), dummies], axis=1)
        
        # Convert to numpy array
        X = X_df.values
        
        logger.info(f"Prepared features with shape {X.shape} and target with shape {y.shape}")
        
        return X, y