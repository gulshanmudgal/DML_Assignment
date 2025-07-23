#!/usr/bin/env python3
"""
End-to-end horizontal partitioning pipeline for network latency prediction.

This script integrates geographical data splitting, specialized model training, and evaluation
to create a complete horizontal partitioning workflow with comprehensive error handling
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
from models.geographical_model import GeographicalModel
from models.monolithic_model import MonolithicModel
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_reporter import PerformanceReporter


class HorizontalPartitioningPipeline:
    """
    Complete end-to-end pipeline for horizontal partitioning approach to network latency prediction.
    
    This pipeline implements the horizontal partitioning strategy by:
    1. Loading and preprocessing data
    2. Splitting data geographically into urban and rural subsets
    3. Training specialized models for each geographical context
    4. Training a global baseline model
    5. Evaluating performance comparison between specialized and global models
    """
    
    def __init__(self, data_file: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the horizontal partitioning pipeline.
        
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
        
        # Geographical splits
        self.urban_train = None
        self.urban_test = None
        self.rural_train = None
        self.rural_test = None
        
        # Models
        self.urban_model = None
        self.rural_model = None
        self.global_model = None
        
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
                logging.FileHandler('horizontal_partitioning_pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute the complete horizontal partitioning pipeline.
        
        Returns:
            Dictionary containing all results and performance metrics
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING HORIZONTAL PARTITIONING PIPELINE")
            self.logger.info("=" * 80)
            
            # Step 1: Load and validate data
            self._load_and_validate_data()
            
            # Step 2: Preprocess data
            self._preprocess_data()
            
            # Step 3: Apply location heuristics if needed
            self._apply_location_heuristics()
            
            # Step 4: Split data into train/test sets
            self._split_train_test()
            
            # Step 5: Split data geographically
            self._split_geographical_data()
            
            # Step 6: Train specialized geographical models
            self._train_geographical_models()
            
            # Step 7: Train global baseline model
            self._train_global_model()
            
            # Step 8: Evaluate all models
            self._evaluate_models()
            
            # Step 9: Generate performance report
            self._generate_performance_report()
            
            self.logger.info("=" * 80)
            self.logger.info("HORIZONTAL PARTITIONING PIPELINE COMPLETED SUCCESSFULLY")
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
    
    def _apply_location_heuristics(self):
        """Apply location heuristics for missing location data."""
        try:
            self.logger.info("Step 3: Applying location heuristics")
            
            # Check if location heuristics are needed
            missing_location_count = (
                self.processed_data['Location Type'].isnull() | 
                (self.processed_data['Location Type'] == 'Unknown') |
                (~self.processed_data['Location Type'].isin(['Urban', 'Rural']))
            ).sum()
            
            if missing_location_count > 0:
                self.logger.info(f"Applying heuristics for {missing_location_count} rows with missing location data")
                self.processed_data = self.feature_engineer.apply_location_heuristics(self.processed_data)
            else:
                self.logger.info("No location heuristics needed - all location data is valid")
            
        except Exception as e:
            self.logger.error(f"Failed to apply location heuristics: {str(e)}")
            raise
    
    def _split_train_test(self):
        """Split the processed data into training and testing sets."""
        try:
            self.logger.info("Step 4: Splitting data into train/test sets")
            
            # Split data
            self.train_data, self.test_data = train_test_split(
                self.processed_data,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.processed_data['Location Type']  # Stratify by location to ensure balanced splits
            )
            
            self.logger.info(f"Training set shape: {self.train_data.shape}")
            self.logger.info(f"Testing set shape: {self.test_data.shape}")
            
            # Log location distribution
            train_location_dist = self.train_data['Location Type'].value_counts()
            test_location_dist = self.test_data['Location Type'].value_counts()
            
            self.logger.info(f"Training set location distribution: {train_location_dist.to_dict()}")
            self.logger.info(f"Testing set location distribution: {test_location_dist.to_dict()}")
            
            # Validate split
            if len(self.train_data) < 5:
                raise ValueError(f"Insufficient training data: {len(self.train_data)} samples")
            if len(self.test_data) < 2:
                raise ValueError(f"Insufficient testing data: {len(self.test_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to split train/test data: {str(e)}")
            raise
    
    def _split_geographical_data(self):
        """Split training and testing data by geographical location."""
        try:
            self.logger.info("Step 5: Splitting data geographically")
            
            # Split training data geographically
            self.urban_train, self.rural_train = self.feature_engineer.split_horizontal_data(self.train_data)
            
            # Split testing data geographically
            self.urban_test, self.rural_test = self.feature_engineer.split_horizontal_data(self.test_data)
            
            # Validate geographical splits
            if not self.feature_engineer.validate_horizontal_split(self.urban_train, self.rural_train, self.train_data):
                raise ValueError("Geographical split validation failed for training data")
            
            if not self.feature_engineer.validate_horizontal_split(self.urban_test, self.rural_test, self.test_data):
                raise ValueError("Geographical split validation failed for testing data")
            
            # Log split results
            self.logger.info(f"Urban training data: {self.urban_train.shape}")
            self.logger.info(f"Rural training data: {self.rural_train.shape}")
            self.logger.info(f"Urban testing data: {self.urban_test.shape}")
            self.logger.info(f"Rural testing data: {self.rural_test.shape}")
            
            # Check minimum sample requirements
            min_samples = 3
            if len(self.urban_train) < min_samples:
                self.logger.warning(f"Urban training data has only {len(self.urban_train)} samples (minimum: {min_samples})")
            if len(self.rural_train) < min_samples:
                self.logger.warning(f"Rural training data has only {len(self.rural_train)} samples (minimum: {min_samples})")
            
        except Exception as e:
            self.logger.error(f"Failed to split geographical data: {str(e)}")
            raise
    
    def _train_geographical_models(self):
        """Train specialized models for urban and rural geographical contexts."""
        try:
            self.logger.info("Step 6: Training specialized geographical models")
            
            # Train Urban Model
            self.logger.info("Training Urban Model")
            if len(self.urban_train) >= 3:  # Minimum samples for training
                X_urban = self.urban_train.drop(columns=['Tower ID', 'Latency (ms)'])
                y_urban = self.urban_train['Latency (ms)'].values
                
                self.urban_model = GeographicalModel(location_type='Urban', random_state=self.random_state)
                self.urban_model.train(X_urban, y_urban)
                self.logger.info(f"Urban model trained on {len(self.urban_train)} samples")
            else:
                self.logger.warning(f"Insufficient urban training data ({len(self.urban_train)} samples), skipping urban model")
                self.urban_model = None
            
            # Train Rural Model
            self.logger.info("Training Rural Model")
            if len(self.rural_train) >= 3:  # Minimum samples for training
                X_rural = self.rural_train.drop(columns=['Tower ID', 'Latency (ms)'])
                y_rural = self.rural_train['Latency (ms)'].values
                
                self.rural_model = GeographicalModel(location_type='Rural', random_state=self.random_state)
                self.rural_model.train(X_rural, y_rural)
                self.logger.info(f"Rural model trained on {len(self.rural_train)} samples")
            else:
                self.logger.warning(f"Insufficient rural training data ({len(self.rural_train)} samples), skipping rural model")
                self.rural_model = None
            
            # Check if at least one model was trained
            if self.urban_model is None and self.rural_model is None:
                raise ValueError("No geographical models could be trained due to insufficient data")
            
        except Exception as e:
            self.logger.error(f"Failed to train geographical models: {str(e)}")
            raise
    
    def _train_global_model(self):
        """Train global baseline model using all training data."""
        try:
            self.logger.info("Step 7: Training global baseline model")
            
            # Prepare features for global model
            X_global = self.train_data.drop(columns=['Tower ID', 'Latency (ms)'])
            y_global = self.train_data['Latency (ms)'].values
            
            # Train global model
            self.global_model = MonolithicModel(random_state=self.random_state)
            self.global_model.train(X_global, y_global)
            
            self.logger.info(f"Global model trained on {len(self.train_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to train global model: {str(e)}")
            raise
    
    def _evaluate_models(self):
        """Evaluate all trained models on their respective test sets."""
        try:
            self.logger.info("Step 8: Evaluating all models")
            
            self.results = {}
            
            # Evaluate Urban Model
            if self.urban_model is not None and len(self.urban_test) > 0:
                X_urban_test = self.urban_test.drop(columns=['Tower ID', 'Latency (ms)'])
                y_urban_test = self.urban_test['Latency (ms)'].values
                
                pred_urban = self.urban_model.predict(X_urban_test)
                self.results['urban_model'] = self.evaluator.calculate_metrics(y_urban_test, pred_urban)
                self.logger.info(f"Urban model evaluated on {len(self.urban_test)} samples")
            else:
                self.logger.warning("Urban model evaluation skipped (no model or no test data)")
            
            # Evaluate Rural Model
            if self.rural_model is not None and len(self.rural_test) > 0:
                X_rural_test = self.rural_test.drop(columns=['Tower ID', 'Latency (ms)'])
                y_rural_test = self.rural_test['Latency (ms)'].values
                
                pred_rural = self.rural_model.predict(X_rural_test)
                self.results['rural_model'] = self.evaluator.calculate_metrics(y_rural_test, pred_rural)
                self.logger.info(f"Rural model evaluated on {len(self.rural_test)} samples")
            else:
                self.logger.warning("Rural model evaluation skipped (no model or no test data)")
            
            # Evaluate Global Model on full test set
            X_global_test = self.test_data.drop(columns=['Tower ID', 'Latency (ms)'])
            y_global_test = self.test_data['Latency (ms)'].values
            
            pred_global = self.global_model.predict(X_global_test)
            self.results['global_model'] = self.evaluator.calculate_metrics(y_global_test, pred_global)
            self.logger.info(f"Global model evaluated on {len(self.test_data)} samples")
            
            # Evaluate Global Model on Urban subset for comparison
            if len(self.urban_test) > 0:
                X_urban_test = self.urban_test.drop(columns=['Tower ID', 'Latency (ms)'])
                y_urban_test = self.urban_test['Latency (ms)'].values
                
                pred_global_urban = self.global_model.predict(X_urban_test)
                self.results['global_model_urban'] = self.evaluator.calculate_metrics(y_urban_test, pred_global_urban)
            
            # Evaluate Global Model on Rural subset for comparison
            if len(self.rural_test) > 0:
                X_rural_test = self.rural_test.drop(columns=['Tower ID', 'Latency (ms)'])
                y_rural_test = self.rural_test['Latency (ms)'].values
                
                pred_global_rural = self.global_model.predict(X_rural_test)
                self.results['global_model_rural'] = self.evaluator.calculate_metrics(y_rural_test, pred_global_rural)
            
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
                    'MAPE': metrics.get('MAPE', 0),
                    'Sample_Size': metrics.get('Sample_Size', 0)
                })
            
            comparison_table = pd.DataFrame(comparison_data)
            comparison_table = comparison_table.sort_values('R²', ascending=False).reset_index(drop=True)
            
            # Generate summary findings
            summary = self._generate_horizontal_summary(self.results)
            
            # Save results to files
            comparison_table.to_csv('horizontal_partitioning_results.csv', index=False)
            
            with open('horizontal_partitioning_summary.txt', 'w') as f:
                f.write("HORIZONTAL PARTITIONING PIPELINE RESULTS\n")
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
    
    def _generate_horizontal_summary(self, results: Dict) -> str:
        """
        Generate a summary of the horizontal partitioning results.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Summary string
        """
        # Find best model based on R² Score
        best_model_name = None
        best_r2 = -float('inf')
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'R2_Score' in metrics:
                if metrics['R2_Score'] > best_r2:
                    best_r2 = metrics['R2_Score']
                    best_model_name = model_name
        
        if best_model_name is None:
            return "No valid model results found for summary generation."
        
        summary = f"""HORIZONTAL PARTITIONING PIPELINE RESULTS SUMMARY

Best Performing Model: {best_model_name}
- R² Score: {results[best_model_name]['R2_Score']:.4f}
- MAE: {results[best_model_name]['MAE']:.4f}
- RMSE: {results[best_model_name]['RMSE']:.4f}

Model Performance Comparison:
"""
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'R2_Score' in metrics:
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
        
        # Compare specialized vs global models
        urban_r2 = results.get('urban_model', {}).get('R2_Score', None)
        rural_r2 = results.get('rural_model', {}).get('R2_Score', None)
        global_r2 = results.get('global_model', {}).get('R2_Score', None)
        global_urban_r2 = results.get('global_model_urban', {}).get('R2_Score', None)
        global_rural_r2 = results.get('global_model_rural', {}).get('R2_Score', None)
        
        summary += "\n\nHorizontal Partitioning Analysis:"
        
        # Urban comparison
        if urban_r2 is not None and global_urban_r2 is not None:
            if urban_r2 > global_urban_r2:
                improvement = ((urban_r2 - global_urban_r2) / abs(global_urban_r2)) * 100 if global_urban_r2 != 0 else 0
                summary += f"\n- Urban specialized model outperforms global model on urban data (R² improvement: {improvement:.1f}%)"
            else:
                summary += "\n- Global model performs better than urban specialized model on urban data"
        
        # Rural comparison
        if rural_r2 is not None and global_rural_r2 is not None:
            if rural_r2 > global_rural_r2:
                improvement = ((rural_r2 - global_rural_r2) / abs(global_rural_r2)) * 100 if global_rural_r2 != 0 else 0
                summary += f"\n- Rural specialized model outperforms global model on rural data (R² improvement: {improvement:.1f}%)"
            else:
                summary += "\n- Global model performs better than rural specialized model on rural data"
        
        # Overall horizontal partitioning effectiveness
        if urban_r2 is not None and rural_r2 is not None and global_r2 is not None:
            specialized_avg = (urban_r2 + rural_r2) / 2
            if specialized_avg > global_r2:
                summary += f"\n- Horizontal partitioning is effective: Average specialized model performance (R²={specialized_avg:.4f}) > Global model (R²={global_r2:.4f})"
            else:
                summary += f"\n- Horizontal partitioning shows limited benefit: Global model (R²={global_r2:.4f}) performs similarly to specialized models (avg R²={specialized_avg:.4f})"
        
        return summary
    
    def get_model_predictions(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all trained models for given data.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        # Prepare features
        X = data.drop(columns=['Tower ID', 'Latency (ms)'] if 'Latency (ms)' in data.columns else ['Tower ID'])
        
        # Get predictions from global model
        if self.global_model is not None:
            predictions['global_model'] = self.global_model.predict(X)
        
        # Get predictions from geographical models based on location
        if 'Location Type' in data.columns:
            urban_mask = data['Location Type'] == 'Urban'
            rural_mask = data['Location Type'] == 'Rural'
            
            # Urban predictions
            if self.urban_model is not None and urban_mask.any():
                urban_data = data[urban_mask]
                X_urban = urban_data.drop(columns=['Tower ID', 'Latency (ms)'] if 'Latency (ms)' in urban_data.columns else ['Tower ID'])
                urban_pred = self.urban_model.predict(X_urban)
                
                # Create full prediction array
                full_urban_pred = np.full(len(data), np.nan)
                full_urban_pred[urban_mask] = urban_pred
                predictions['urban_model'] = full_urban_pred
            
            # Rural predictions
            if self.rural_model is not None and rural_mask.any():
                rural_data = data[rural_mask]
                X_rural = rural_data.drop(columns=['Tower ID', 'Latency (ms)'] if 'Latency (ms)' in rural_data.columns else ['Tower ID'])
                rural_pred = self.rural_model.predict(X_rural)
                
                # Create full prediction array
                full_rural_pred = np.full(len(data), np.nan)
                full_rural_pred[rural_mask] = rural_pred
                predictions['rural_model'] = full_rural_pred
        
        return predictions


def main():
    """Main function to run the horizontal partitioning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run horizontal partitioning pipeline for network latency prediction')
    parser.add_argument('--data', required=True, help='Path to the Excel data file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = HorizontalPartitioningPipeline(
            data_file=args.data,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to:")
        print("  - horizontal_partitioning_results.csv")
        print("  - horizontal_partitioning_summary.txt")
        print("  - horizontal_partitioning_pipeline.log")
        
        return 0
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())