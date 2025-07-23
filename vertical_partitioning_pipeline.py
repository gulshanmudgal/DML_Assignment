#!/usr/bin/env python3
"""
End-to-end vertical partitioning pipeline for network latency prediction.

This script integrates data loading, preprocessing, Model A/B training, and fusion
to create a complete vertical partitioning workflow with comprehensive error handling
and logging.
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import project modules
from data_processing.data_loader import DataLoader
from data_processing.feature_engineering import FeatureEngineer
from models.infrastructure_model import InfrastructureModel
from models.user_behavior_model import UserBehaviorModel
from models.fusion_model import FusionModel
from models.monolithic_model import MonolithicModel
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_reporter import PerformanceReporter


class VerticalPartitioningPipeline:
    """
    Complete end-to-end pipeline for vertical partitioning approach to network latency prediction.
    
    This pipeline implements the vertical partitioning strategy by:
    1. Loading and preprocessing data
    2. Splitting features into infrastructure and user behavior subsets
    3. Training specialized models (Model A and Model B)
    4. Fusing model predictions
    5. Evaluating performance against baseline monolithic model
    """
    
    def __init__(self, data_file: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the vertical partitioning pipeline.
        
        Args:
            data_file: Path to the Excel data file
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
        """
        self.data_file = data_file
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.reporter = PerformanceReporter()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        
        # Models
        self.infrastructure_model = None
        self.user_behavior_model = None
        self.fusion_model = None
        self.monolithic_model = None
        
        # Results
        self.results = {}
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vertical_partitioning_pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute the complete vertical partitioning pipeline.
        
        Returns:
            Dictionary containing all results and performance metrics
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING VERTICAL PARTITIONING PIPELINE")
            self.logger.info("=" * 80)
            
            # Step 1: Load and validate data
            self._load_and_validate_data()
            
            # Step 2: Preprocess data
            self._preprocess_data()
            
            # Step 3: Split data into train/test sets
            self._split_train_test()
            
            # Step 4: Prepare features for vertical partitioning
            self._prepare_vertical_features()
            
            # Step 5: Train individual models (Model A and Model B)
            self._train_individual_models()
            
            # Step 6: Train fusion model
            self._train_fusion_model()
            
            # Step 7: Train baseline monolithic model
            self._train_monolithic_model()
            
            # Step 8: Evaluate all models
            self._evaluate_models()
            
            # Step 9: Generate performance report
            self._generate_performance_report()
            
            self.logger.info("=" * 80)
            self.logger.info("VERTICAL PARTITIONING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_and_validate_data(self):
        """Load data from file and validate its structure."""
        try:
            self.logger.info("Step 1: Loading and validating data")
            
            # Check if file exists
            if not Path(self.data_file).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            # Load data
            self.raw_data = self.data_loader.load_dataset(self.data_file)
            self.logger.info(f"Loaded dataset with shape: {self.raw_data.shape}")
            
            # Validate data structure
            if not self.data_loader.validate_data(self.raw_data):
                validation_errors = self.data_loader.get_validation_errors()
                self.logger.warning("Data validation issues found:")
                for error in validation_errors:
                    self.logger.warning(f"  - {error}")
                
                # Continue with warnings but stop on critical errors
                critical_errors = [e for e in validation_errors if 'Missing required columns' in e]
                if critical_errors:
                    raise ValueError(f"Critical validation errors: {critical_errors}")
            
            # Log data summary
            summary = self.data_loader.get_data_summary(self.raw_data)
            self.logger.info(f"Data summary: {summary['total_rows']} rows, {summary['total_columns']} columns")
            
        except Exception as e:
            self.logger.error(f"Failed to load and validate data: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """Preprocess the raw data for model training."""
        try:
            self.logger.info("Step 2: Preprocessing data")
            
            # Apply comprehensive preprocessing
            self.processed_data = self.data_loader.preprocess_data(self.raw_data)
            self.logger.info(f"Preprocessed data shape: {self.processed_data.shape}")
            
            # Check for sufficient data after preprocessing
            if len(self.processed_data) < 10:
                raise ValueError(f"Insufficient data after preprocessing: {len(self.processed_data)} rows")
            
            # Log preprocessing results
            removed_rows = len(self.raw_data) - len(self.processed_data)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} rows during preprocessing")
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def _split_train_test(self):
        """Split the processed data into training and testing sets."""
        try:
            self.logger.info("Step 3: Splitting data into train/test sets")
            
            # Split data
            self.train_data, self.test_data = train_test_split(
                self.processed_data,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None  # Can't stratify on continuous target
            )
            
            self.logger.info(f"Training set shape: {self.train_data.shape}")
            self.logger.info(f"Testing set shape: {self.test_data.shape}")
            
            # Validate split
            if len(self.train_data) < 5:
                raise ValueError(f"Insufficient training data: {len(self.train_data)} samples")
            if len(self.test_data) < 2:
                raise ValueError(f"Insufficient testing data: {len(self.test_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to split train/test data: {str(e)}")
            raise
    
    def _prepare_vertical_features(self):
        """Prepare features for vertical partitioning (Model A and Model B)."""
        try:
            self.logger.info("Step 4: Preparing features for vertical partitioning")
            
            # Split features vertically for training data
            self.train_infra, self.train_user = self.feature_engineer.split_vertical_features(
                self.train_data, include_target=True
            )
            
            # Split features vertically for testing data
            self.test_infra, self.test_user = self.feature_engineer.split_vertical_features(
                self.test_data, include_target=True
            )
            
            # Validate vertical split
            if not self.feature_engineer.validate_vertical_split(self.train_infra, self.train_user):
                raise ValueError("Vertical feature split validation failed for training data")
            
            if not self.feature_engineer.validate_vertical_split(self.test_infra, self.test_user):
                raise ValueError("Vertical feature split validation failed for testing data")
            
            self.logger.info("Vertical feature split completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare vertical features: {str(e)}")
            raise
    
    def _train_individual_models(self):
        """Train individual models (Model A: Infrastructure, Model B: User Behavior)."""
        try:
            self.logger.info("Step 5: Training individual models")
            
            # Prepare features for Model A (Infrastructure)
            self.logger.info("Training Model A (Infrastructure Model)")
            X_train_infra, y_train_infra = self.feature_engineer.prepare_infrastructure_features(
                self.train_infra, fit_scaler=True
            )
            
            # Train Infrastructure Model
            self.infrastructure_model = InfrastructureModel(random_state=self.random_state)
            self.infrastructure_model.train(X_train_infra, y_train_infra)
            self.logger.info("Model A (Infrastructure) training completed")
            
            # Prepare features for Model B (User Behavior) - use raw features
            self.logger.info("Training Model B (User Behavior Model)")
            # Extract raw features for UserBehaviorModel (it handles preprocessing internally)
            user_count = self.train_user['User Count'].values.reshape(-1, 1)
            device_type = self.train_user['Device Type'].values.reshape(-1, 1)
            X_train_user = np.hstack([user_count, device_type])
            y_train_user = self.train_user['Latency (ms)'].values
            
            # Train User Behavior Model
            self.user_behavior_model = UserBehaviorModel(random_state=self.random_state)
            self.user_behavior_model.train(X_train_user, y_train_user)
            self.logger.info("Model B (User Behavior) training completed")
            
        except Exception as e:
            self.logger.error(f"Failed to train individual models: {str(e)}")
            raise
    
    def _train_fusion_model(self):
        """Train fusion model to combine Model A and Model B predictions."""
        try:
            self.logger.info("Step 6: Training fusion model")
            
            # Prepare combined features for fusion model
            # Combine infrastructure and user behavior features
            X_train_combined = self._prepare_fusion_features(self.train_data)
            y_train = self.train_data['Latency (ms)'].values
            
            # Train fusion model with weighted average strategy
            self.fusion_model = FusionModel(
                fusion_strategy='weighted_average',
                random_state=self.random_state
            )
            self.fusion_model.train(X_train_combined, y_train)
            
            # Log fusion weights
            weights = self.fusion_model.get_fusion_weights()
            if weights:
                self.logger.info(f"Fusion weights: Infrastructure={weights['infrastructure']:.3f}, "
                               f"User Behavior={weights['user_behavior']:.3f}")
            
            self.logger.info("Fusion model training completed")
            
        except Exception as e:
            self.logger.error(f"Failed to train fusion model: {str(e)}")
            raise
    
    def _train_monolithic_model(self):
        """Train baseline monolithic model using all features."""
        try:
            self.logger.info("Step 7: Training baseline monolithic model")
            
            # Prepare features for monolithic model - pass DataFrame directly
            X_train_mono = self.train_data.drop(columns=['Tower ID', 'Latency (ms)'])
            y_train_mono = self.train_data['Latency (ms)'].values
            
            # Train monolithic model
            self.monolithic_model = MonolithicModel(random_state=self.random_state)
            self.monolithic_model.train(X_train_mono, y_train_mono)
            
            self.logger.info("Monolithic model training completed")
            
        except Exception as e:
            self.logger.error(f"Failed to train monolithic model: {str(e)}")
            raise
    
    def _evaluate_models(self):
        """Evaluate all trained models on the test set."""
        try:
            self.logger.info("Step 8: Evaluating all models")
            
            # Prepare test features for each model type
            X_test_infra, y_test_infra = self.feature_engineer.prepare_infrastructure_features(
                self.test_infra, fit_scaler=False
            )
            
            # Prepare user behavior features - use raw features for UserBehaviorModel
            user_count = self.test_user['User Count'].values.reshape(-1, 1)
            device_type = self.test_user['Device Type'].values.reshape(-1, 1)
            X_test_user = np.hstack([user_count, device_type])
            
            X_test_combined = self._prepare_fusion_features(self.test_data)
            
            # For monolithic model, pass DataFrame directly
            X_test_mono = self.test_data.drop(columns=['Tower ID', 'Latency (ms)'])
            y_test_mono = self.test_data['Latency (ms)'].values
            
            # Get predictions from all models
            pred_infra = self.infrastructure_model.predict(X_test_infra)
            pred_user = self.user_behavior_model.predict(X_test_user)
            pred_fusion = self.fusion_model.predict(X_test_combined)
            pred_mono = self.monolithic_model.predict(X_test_mono)
            
            # Calculate metrics for all models
            y_true = self.test_data['Latency (ms)'].values
            
            self.results = {
                'infrastructure_model': self.evaluator.calculate_metrics(y_true, pred_infra),
                'user_behavior_model': self.evaluator.calculate_metrics(y_true, pred_user),
                'fusion_model': self.evaluator.calculate_metrics(y_true, pred_fusion),
                'monolithic_model': self.evaluator.calculate_metrics(y_true, pred_mono)
            }
            
            # Log results summary
            self.logger.info("Model evaluation completed:")
            for model_name, metrics in self.results.items():
                self.logger.info(f"  {model_name}: MAE={metrics['MAE']:.3f}, "
                               f"RMSE={metrics['RMSE']:.3f}, R²={metrics['R2_Score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate models: {str(e)}")
            raise
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        try:
            self.logger.info("Step 9: Generating performance report")
            
            # Create a simple performance table from results
            comparison_data = []
            for model_name, metrics in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R²': metrics['R2_Score'],
                    'MAPE': metrics.get('MAPE', 0)
                })
            
            comparison_table = pd.DataFrame(comparison_data)
            comparison_table = comparison_table.sort_values('R²', ascending=False).reset_index(drop=True)
            
            # Generate summary findings
            summary = self._generate_simple_summary(self.results)
            
            # Save results to files
            comparison_table.to_csv('vertical_partitioning_results.csv', index=False)
            
            with open('vertical_partitioning_summary.txt', 'w') as f:
                f.write("VERTICAL PARTITIONING PIPELINE RESULTS\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary)
                f.write("\n\nDETAILED RESULTS:\n")
                f.write(comparison_table.to_string(index=False))
            
            # Store in results
            self.results['comparison_table'] = comparison_table
            self.results['summary'] = summary
            
            self.logger.info("Performance report generated successfully")
            self.logger.info(f"Summary: {summary}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
            raise
    
    def _prepare_fusion_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for fusion model (combines infrastructure and user behavior features).
        
        Args:
            data: Input dataframe
            
        Returns:
            Combined feature array for fusion model
        """
        # Extract required columns
        required_cols = ['Signal Strength (dBm)', 'Network Traffic (MB)', 'User Count', 'Device Type']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for fusion model: {missing_cols}")
        
        # Create feature array
        features = []
        features.append(data['Signal Strength (dBm)'].values)
        features.append(data['Network Traffic (MB)'].values)
        features.append(data['User Count'].values)
        features.append(data['Device Type'].values)
        
        # Stack features (transpose to get correct shape)
        X_combined = np.column_stack(features)
        
        return X_combined
    
    def _generate_simple_summary(self, results: Dict) -> str:
        """
        Generate a simple summary of the results.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Summary string
        """
        # Find best model based on R² Score
        best_model_name = None
        best_r2 = -float('inf')
        
        for model_name, metrics in results.items():
            if metrics['R2_Score'] > best_r2:
                best_r2 = metrics['R2_Score']
                best_model_name = model_name
        
        summary = f"""VERTICAL PARTITIONING PIPELINE RESULTS SUMMARY

Best Performing Model: {best_model_name}
- R² Score: {results[best_model_name]['R2_Score']:.4f}
- MAE: {results[best_model_name]['MAE']:.4f}
- RMSE: {results[best_model_name]['RMSE']:.4f}

Model Performance Comparison:
"""
        
        for model_name, metrics in results.items():
            summary += f"- {model_name}: R²={metrics['R2_Score']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}\n"
        
        # Add interpretation
        if best_r2 > 0.8:
            summary += "\nInterpretation: Excellent model performance achieved (R² > 0.8)"
        elif best_r2 > 0.6:
            summary += "\nInterpretation: Good model performance achieved (R² > 0.6)"
        elif best_r2 > 0.4:
            summary += "\nInterpretation: Moderate model performance achieved (R² > 0.4)"
        else:
            summary += "\nInterpretation: Model performance needs improvement (R² < 0.4)"
        
        # Compare fusion vs individual models
        if 'fusion_model' in results and 'infrastructure_model' in results and 'user_behavior_model' in results:
            fusion_r2 = results['fusion_model']['R2_Score']
            infra_r2 = results['infrastructure_model']['R2_Score']
            user_r2 = results['user_behavior_model']['R2_Score']
            
            if fusion_r2 > max(infra_r2, user_r2):
                summary += "\n\nVertical Partitioning Analysis: Fusion model outperforms individual models, indicating successful feature complementarity."
            else:
                summary += "\n\nVertical Partitioning Analysis: Individual models perform better than fusion, suggesting potential overfitting or suboptimal fusion strategy."
        
        return summary
    
    def get_model_predictions(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all trained models for given data.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with predictions from each model
        """
        if not all([self.infrastructure_model, self.user_behavior_model, 
                   self.fusion_model, self.monolithic_model]):
            raise ValueError("All models must be trained before getting predictions")
        
        # Prepare features for each model
        infra_data, user_data = self.feature_engineer.split_vertical_features(data, include_target=False)
        
        X_infra, _ = self.feature_engineer.prepare_infrastructure_features(infra_data, fit_scaler=False)
        
        # Prepare user behavior features - use raw features
        user_count = user_data['User Count'].values.reshape(-1, 1)
        device_type = user_data['Device Type'].values.reshape(-1, 1)
        X_user = np.hstack([user_count, device_type])
        
        X_combined = self._prepare_fusion_features(data)
        X_mono = data.drop(columns=['Tower ID', 'Latency (ms)'] if 'Latency (ms)' in data.columns else ['Tower ID'])
        
        # Get predictions
        predictions = {
            'infrastructure_model': self.infrastructure_model.predict(X_infra),
            'user_behavior_model': self.user_behavior_model.predict(X_user),
            'fusion_model': self.fusion_model.predict(X_combined),
            'monolithic_model': self.monolithic_model.predict(X_mono)
        }
        
        return predictions


def main():
    """Main function to run the vertical partitioning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run vertical partitioning pipeline for network latency prediction')
    parser.add_argument('--data', required=True, help='Path to the Excel data file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = VerticalPartitioningPipeline(
            data_file=args.data,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to:")
        print("  - vertical_partitioning_results.csv")
        print("  - vertical_partitioning_summary.txt")
        print("  - vertical_partitioning_pipeline.log")
        
        return 0
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())