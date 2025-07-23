#!/usr/bin/env python3
"""
Production-ready script for vertical partitioning approach to network latency prediction.

This script provides a command-line interface for running the complete vertical partitioning
pipeline with configurable parameters and comprehensive output.

Usage:
    python run_vertical_partitioning.py --data test_data.xlsx
    python run_vertical_partitioning.py --data test_data.xlsx --test-size 0.3 --output results/
    python run_vertical_partitioning.py --help

Author: Network Latency Prediction System
Version: 1.0
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Import the vertical partitioning pipeline
from vertical_partitioning_pipeline import VerticalPartitioningPipeline


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Set up comprehensive logging for the script.
    
    Args:
        output_dir: Directory for log files
        verbose: Enable verbose logging
        
    Returns:
        Configured logger
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = output_dir / f"vertical_partitioning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file '{args.data}' not found.")
        return False
    
    # Validate test size
    if not 0.1 <= args.test_size <= 0.5:
        print(f"Error: Test size must be between 0.1 and 0.5, got {args.test_size}")
        return False
    
    # Validate random state
    if args.random_state < 0:
        print(f"Error: Random state must be non-negative, got {args.random_state}")
        return False
    
    # Check output directory
    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_dir():
        print(f"Error: Output path '{args.output}' exists but is not a directory.")
        return False
    
    return True


def save_results(results: Dict, output_dir: Path, logger: logging.Logger) -> None:
    """
    Save pipeline results to files.
    
    Args:
        results: Pipeline results dictionary
        output_dir: Output directory
        logger: Logger instance
    """
    try:
        # Save performance comparison table
        if 'comparison_table' in results:
            comparison_file = output_dir / 'vertical_partitioning_comparison.csv'
            results['comparison_table'].to_csv(comparison_file, index=False)
            logger.info(f"Performance comparison saved to: {comparison_file}")
        
        # Save summary
        if 'summary' in results:
            summary_file = output_dir / 'vertical_partitioning_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("VERTICAL PARTITIONING RESULTS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(results['summary'])
            logger.info(f"Summary saved to: {summary_file}")
        
        # Save detailed metrics as JSON
        metrics_file = output_dir / 'vertical_partitioning_metrics.json'
        metrics_data = {
            model_name: metrics for model_name, metrics in results.items()
            if isinstance(metrics, dict) and 'MAE' in metrics
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        logger.info(f"Detailed metrics saved to: {metrics_file}")
        
        # Save configuration
        config_file = output_dir / 'vertical_partitioning_config.json'
        config_data = {
            'timestamp': datetime.now().isoformat(),
            'script_version': '1.0',
            'approach': 'vertical_partitioning',
            'data_file': str(Path(args.data).absolute()),
            'test_size': args.test_size,
            'random_state': args.random_state,
            'output_directory': str(output_dir.absolute())
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to: {config_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def print_results_summary(results: Dict, logger: logging.Logger) -> None:
    """
    Print a summary of results to console.
    
    Args:
        results: Pipeline results dictionary
        logger: Logger instance
    """
    print("\n" + "=" * 80)
    print("VERTICAL PARTITIONING RESULTS SUMMARY")
    print("=" * 80)
    
    if 'comparison_table' in results:
        print("\nModel Performance Comparison:")
        print(results['comparison_table'].to_string(index=False, float_format='%.4f'))
    
    # Find best model
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'R2_Score' in metrics:
            if metrics['R2_Score'] > best_r2:
                best_r2 = metrics['R2_Score']
                best_model = model_name
    
    if best_model:
        print(f"\nBest Performing Model: {best_model}")
        print(f"  R² Score: {results[best_model]['R2_Score']:.4f}")
        print(f"  MAE: {results[best_model]['MAE']:.4f}")
        print(f"  RMSE: {results[best_model]['RMSE']:.4f}")
    
    if 'summary' in results:
        print(f"\nDetailed Analysis:")
        print(results['summary'])
    
    print("\n" + "=" * 80)


def main():
    """Main function to run the vertical partitioning script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run vertical partitioning pipeline for network latency prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data test_data.xlsx
  %(prog)s --data test_data.xlsx --test-size 0.3 --output results/
  %(prog)s --data test_data.xlsx --verbose --random-state 123
        """
    )
    
    parser.add_argument(
        '--data', 
        required=True, 
        help='Path to the Excel data file'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2, 
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42, 
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output', 
        default='vertical_partitioning_results', 
        help='Output directory for results (default: vertical_partitioning_results)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 1.0'
    )
    
    # Parse arguments
    global args
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Set up output directory and logging
    output_dir = Path(args.output)
    logger = setup_logging(output_dir, args.verbose)
    
    try:
        # Log execution parameters
        logger.info("Starting vertical partitioning pipeline")
        logger.info(f"Data file: {args.data}")
        logger.info(f"Test size: {args.test_size}")
        logger.info(f"Random state: {args.random_state}")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize and run pipeline
        pipeline = VerticalPartitioningPipeline(
            data_file=args.data,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Execute pipeline
        results = pipeline.run_complete_pipeline()
        
        # Save results
        save_results(results, output_dir, logger)
        
        # Print summary
        print_results_summary(results, logger)
        
        logger.info("Vertical partitioning pipeline completed successfully")
        print(f"\nResults saved to: {output_dir.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\nPipeline interrupted by user.")
        return 130
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        print(f"\nPipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())