"""
Data loading and validation module for network latency prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkData:
    """Data schema for network dataset."""
    tower_id: str
    signal_strength: float  # dBm
    network_traffic: float  # MB
    latency: float  # ms (target variable)
    user_count: int
    device_type: str  # categorical
    location_type: str  # Urban/Rural


class DataLoader:
    """
    DataLoader class for handling Excel file reading and data validation
    for network latency prediction dataset.
    """
    
    REQUIRED_COLUMNS = [
        'Tower ID', 'Signal Strength (dBm)', 'Network Traffic (MB)', 
        'Latency (ms)', 'User Count', 'Device Type', 'Location Type'
    ]
    
    COLUMN_TYPES = {
        'Tower ID': str,
        'Signal Strength (dBm)': float,
        'Network Traffic (MB)': float,
        'Latency (ms)': float,
        'User Count': int,
        'Device Type': str,
        'Location Type': str
    }
    
    def __init__(self):
        """Initialize DataLoader."""
        self.data = None
        self.validation_errors = []
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from Excel file.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Store original data
            self.data = df.copy()
            
            logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate dataset schema and data quality.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        self.validation_errors = []
        is_valid = True
        
        # Check required columns
        missing_columns = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            is_valid = False
        
        # Check data types and ranges
        for column, expected_type in self.COLUMN_TYPES.items():
            if column in df.columns:
                # Check for missing values
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    warning_msg = f"Column '{column}' has {missing_count} missing values"
                    self.validation_errors.append(warning_msg)
                    logger.warning(warning_msg)
                
                # Validate data types and ranges
                if not self._validate_column(df, column, expected_type):
                    is_valid = False
        
        # Check for duplicate Tower IDs
        if 'Tower ID' in df.columns:
            duplicate_count = df['Tower ID'].duplicated().sum()
            if duplicate_count > 0:
                error_msg = f"Found {duplicate_count} duplicate Tower IDs"
                self.validation_errors.append(error_msg)
                logger.error(error_msg)
                is_valid = False
        
        if is_valid:
            logger.info("Data validation passed successfully")
        else:
            logger.error(f"Data validation failed with {len(self.validation_errors)} errors")
        
        return is_valid
    
    def _validate_column(self, df: pd.DataFrame, column: str, expected_type: type) -> bool:
        """
        Validate individual column data type and range.
        
        Args:
            df (pd.DataFrame): Dataset
            column (str): Column name
            expected_type (type): Expected data type
            
        Returns:
            bool: True if column is valid
        """
        is_valid = True
        
        try:
            # Skip validation for rows with missing values
            non_null_data = df[column].dropna()
            
            if len(non_null_data) == 0:
                return True  # All values are null, handled separately
            
            # Validate specific columns with range checks
            if column == 'Signal Strength (dBm)':
                # Signal strength should be negative (dBm values)
                invalid_values = non_null_data[(non_null_data > 0) | (non_null_data < -150)]
                if len(invalid_values) > 0:
                    error_msg = f"Signal Strength values should be between -150 and 0 dBm. Found {len(invalid_values)} invalid values"
                    self.validation_errors.append(error_msg)
                    logger.error(error_msg)
                    is_valid = False
            
            elif column == 'Network Traffic (MB)':
                # Network traffic should be positive
                invalid_values = non_null_data[non_null_data < 0]
                if len(invalid_values) > 0:
                    error_msg = f"Network Traffic should be positive. Found {len(invalid_values)} negative values"
                    self.validation_errors.append(error_msg)
                    logger.error(error_msg)
                    is_valid = False
            
            elif column == 'Latency (ms)':
                # Latency should be positive
                invalid_values = non_null_data[non_null_data <= 0]
                if len(invalid_values) > 0:
                    error_msg = f"Latency should be positive. Found {len(invalid_values)} non-positive values"
                    self.validation_errors.append(error_msg)
                    logger.error(error_msg)
                    is_valid = False
            
            elif column == 'User Count':
                # User count should be non-negative integer
                invalid_values = non_null_data[non_null_data < 0]
                if len(invalid_values) > 0:
                    error_msg = f"User Count should be non-negative. Found {len(invalid_values)} negative values"
                    self.validation_errors.append(error_msg)
                    logger.error(error_msg)
                    is_valid = False
            
            elif column == 'Device Type':
                # Check for valid device types (can be extended)
                valid_device_types = ['Mobile', 'Tablet', 'Laptop', 'Desktop', 'IoT', 
                                     'IoT Device', 'Feature Phone', 'Smartphone']
                invalid_devices = non_null_data[~non_null_data.isin(valid_device_types)]
                if len(invalid_devices) > 0:
                    unique_invalid = invalid_devices.unique()
                    warning_msg = f"Found potentially invalid device types: {unique_invalid}"
                    self.validation_errors.append(warning_msg)
                    logger.warning(warning_msg)
            
            elif column == 'Location Type':
                # Location should be Urban or Rural
                valid_locations = ['Urban', 'Rural']
                invalid_locations = non_null_data[~non_null_data.isin(valid_locations)]
                if len(invalid_locations) > 0:
                    error_msg = f"Location Type should be 'Urban' or 'Rural'. Found {len(invalid_locations)} invalid values"
                    self.validation_errors.append(error_msg)
                    logger.error(error_msg)
                    is_valid = False
        
        except Exception as e:
            error_msg = f"Error validating column '{column}': {str(e)}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            is_valid = False
        
        return is_valid
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive preprocessing of the dataset including missing value handling,
        outlier detection and removal, and data type conversion.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Starting comprehensive data preprocessing")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        initial_rows = len(processed_df)
        
        # Step 1: Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Step 2: Handle outliers
        processed_df = self._handle_outliers(processed_df)
        
        # Step 3: Convert and validate data types
        processed_df = self._convert_data_types(processed_df)
        
        # Step 4: Remove rows where target variable (Latency) is missing
        if 'Latency (ms)' in processed_df.columns:
            processed_df = processed_df.dropna(subset=['Latency (ms)'])
            removed_rows = initial_rows - len(processed_df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing latency values")
        
        final_rows = len(processed_df)
        logger.info(f"Preprocessing completed. Dataset shape: {initial_rows} -> {final_rows} rows")
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies for each column type.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        logger.info("Handling missing values")
        processed_df = df.copy()
        
        # Numerical columns - use median imputation
        numerical_columns = ['Signal Strength (dBm)', 'Network Traffic (MB)', 'User Count']
        for col in numerical_columns:
            if col in processed_df.columns:
                missing_count = processed_df[col].isnull().sum()
                if missing_count > 0:
                    median_value = processed_df[col].median()
                    processed_df[col] = processed_df[col].fillna(median_value)
                    logger.info(f"Filled {missing_count} missing values in '{col}' with median: {median_value}")
        
        # Categorical columns - use mode or 'Unknown'
        if 'Device Type' in processed_df.columns:
            missing_count = processed_df['Device Type'].isnull().sum()
            if missing_count > 0:
                # Use mode if available, otherwise 'Unknown'
                if len(processed_df['Device Type'].dropna()) > 0:
                    mode_value = processed_df['Device Type'].mode().iloc[0]
                    processed_df['Device Type'] = processed_df['Device Type'].fillna(mode_value)
                    logger.info(f"Filled {missing_count} missing values in 'Device Type' with mode: {mode_value}")
                else:
                    processed_df['Device Type'] = processed_df['Device Type'].fillna('Unknown')
                    logger.info(f"Filled {missing_count} missing values in 'Device Type' with 'Unknown'")
        
        if 'Location Type' in processed_df.columns:
            missing_count = processed_df['Location Type'].isnull().sum()
            if missing_count > 0:
                # For location, we'll use heuristics in horizontal partitioning
                processed_df['Location Type'] = processed_df['Location Type'].fillna('Unknown')
                logger.info(f"Filled {missing_count} missing values in 'Location Type' with 'Unknown'")
        
        return processed_df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method for numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        logger.info("Detecting and handling outliers")
        processed_df = df.copy()
        
        # Define columns to check for outliers
        outlier_columns = ['Signal Strength (dBm)', 'Network Traffic (MB)', 'Latency (ms)', 'User Count']
        
        for col in outlier_columns:
            if col in processed_df.columns:
                # Calculate IQR
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = processed_df[(processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them to preserve data
                    processed_df.loc[processed_df[col] < lower_bound, col] = lower_bound
                    processed_df.loc[processed_df[col] > upper_bound, col] = upper_bound
                    logger.info(f"Capped {outlier_count} outliers in '{col}' to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return processed_df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert and validate data types for all columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with proper data types
        """
        logger.info("Converting and validating data types")
        processed_df = df.copy()
        
        # Convert numerical columns
        numerical_conversions = {
            'Signal Strength (dBm)': float,
            'Network Traffic (MB)': float,
            'Latency (ms)': float,
            'User Count': int
        }
        
        for col, dtype in numerical_conversions.items():
            if col in processed_df.columns:
                try:
                    if dtype == int:
                        processed_df[col] = processed_df[col].round().astype(dtype)
                    else:
                        processed_df[col] = processed_df[col].astype(dtype)
                    logger.info(f"Converted '{col}' to {dtype.__name__}")
                except Exception as e:
                    logger.warning(f"Could not convert '{col}' to {dtype.__name__}: {e}")
        
        # Convert string columns
        string_columns = ['Tower ID', 'Device Type', 'Location Type']
        for col in string_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str)
                logger.info(f"Converted '{col}' to string")
        
        return processed_df
    
    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors from last validation.
        
        Returns:
            List[str]: List of validation error messages
        """
        return self.validation_errors.copy()
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to summarize
            
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Add numerical column statistics
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            summary['numerical_stats'] = df[numerical_columns].describe().to_dict()
        
        # Add categorical column statistics
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            summary['categorical_stats'] = {}
            for col in categorical_columns:
                summary['categorical_stats'][col] = df[col].value_counts().to_dict()
        
        return summary