# Network Latency Prediction System

A comprehensive machine learning system for predicting network latency using both vertical and horizontal partitioning strategies.

## Overview

This system implements two distinct approaches to network latency prediction:

1. **Vertical Partitioning**: Splits features into infrastructure-related (Model A) and user behavior-related (Model B) subsets, then fuses their predictions.
2. **Horizontal Partitioning**: Splits data geographically into urban and rural subsets, training specialized models for each context.

## Features

- **Multiple Modeling Approaches**: Vertical partitioning, horizontal partitioning, and monolithic baseline models
- **Comprehensive Evaluation**: Performance comparison with statistical metrics (MAE, RMSE, R²)
- **Interactive Notebooks**: Jupyter notebooks for exploratory analysis and visualization
- **Production Scripts**: Command-line tools for automated pipeline execution
- **Robust Data Processing**: Data validation, preprocessing, and feature engineering
- **Detailed Logging**: Comprehensive logging and error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
jupyter>=1.0.0
```

## Project Structure

```
network-latency-prediction/
├── data_processing/           # Data loading and preprocessing modules
│   ├── data_loader.py
│   └── feature_engineering.py
├── models/                    # Model implementations
│   ├── base_model.py
│   ├── infrastructure_model.py
│   ├── user_behavior_model.py
│   ├── fusion_model.py
│   ├── geographical_model.py
│   └── monolithic_model.py
├── evaluation/                # Model evaluation and reporting
│   ├── model_evaluator.py
│   └── performance_reporter.py
├── notebooks/                 # Interactive Jupyter notebooks
│   ├── vertical_partitioning_notebook.ipynb
│   └── horizontal_partitioning_notebook.ipynb
├── scripts/                   # Production-ready scripts
│   ├── run_vertical_partitioning.py
│   └── run_horizontal_partitioning.py
├── tests/                     # Unit and integration tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Interactive Analysis (Jupyter Notebooks)

#### Vertical Partitioning Analysis

```bash
jupyter notebook vertical_partitioning_notebook.ipynb
```

This notebook provides:
- Data exploration and visualization
- Feature partitioning demonstration
- Model training and evaluation
- Performance comparison and analysis

#### Horizontal Partitioning Analysis

```bash
jupyter notebook horizontal_partitioning_notebook.ipynb
```

This notebook provides:
- Geographical data analysis
- Urban vs rural context comparison
- Specialized model training
- Performance evaluation and insights

### Production Scripts

#### Vertical Partitioning Pipeline

```bash
# Basic usage
python run_vertical_partitioning.py --data test_data.xlsx

# Advanced usage with custom parameters
python run_vertical_partitioning.py \
    --data test_data.xlsx \
    --test-size 0.3 \
    --random-state 123 \
    --output vertical_results/ \
    --verbose

# Get help
python run_vertical_partitioning.py --help
```

#### Horizontal Partitioning Pipeline

```bash
# Basic usage
python run_horizontal_partitioning.py --data test_data.xlsx

# Advanced usage with custom parameters
python run_horizontal_partitioning.py \
    --data test_data.xlsx \
    --test-size 0.25 \
    --random-state 456 \
    --output horizontal_results/ \
    --verbose

# Get help
python run_horizontal_partitioning.py --help
```

### Command-Line Options

Both scripts support the following options:

- `--data`: Path to the Excel data file (required)
- `--test-size`: Test set size (default: 0.2, range: 0.1-0.5)
- `--random-state`: Random state for reproducibility (default: 42)
- `--output`: Output directory for results (default: approach-specific directory)
- `--verbose`: Enable verbose logging
- `--version`: Show script version
- `--help`: Show help message

## Data Format

The system expects an Excel file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Tower ID | String | Unique identifier for network tower |
| Signal Strength (dBm) | Float | Signal strength in decibels |
| Network Traffic (MB) | Float | Network traffic in megabytes |
| Latency (ms) | Float | Network latency in milliseconds (target variable) |
| User Count | Integer | Number of users connected |
| Device Type | String | Type of device (categorical) |
| Location Type | String | Urban or Rural location |

### Example Data

```csv
Tower ID,Signal Strength (dBm),Network Traffic (MB),Latency (ms),User Count,Device Type,Location Type
T001,-65.2,150.5,25.3,45,Smartphone,Urban
T002,-78.1,89.2,42.7,23,Tablet,Rural
T003,-70.5,200.1,18.9,67,Laptop,Urban
```

## Output Files

### Vertical Partitioning Results

- `vertical_partitioning_comparison.csv`: Performance comparison table
- `vertical_partitioning_summary.txt`: Detailed analysis summary
- `vertical_partitioning_metrics.json`: Raw performance metrics
- `vertical_partitioning_config.json`: Execution configuration
- `vertical_partitioning_YYYYMMDD_HHMMSS.log`: Execution log

### Horizontal Partitioning Results

- `horizontal_partitioning_comparison.csv`: Performance comparison table
- `horizontal_partitioning_summary.txt`: Detailed analysis summary
- `horizontal_partitioning_metrics.json`: Raw performance metrics
- `horizontal_partitioning_config.json`: Execution configuration
- `horizontal_partitioning_YYYYMMDD_HHMMSS.log`: Execution log

## Model Performance Metrics

The system evaluates models using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **R² Score**: Coefficient of determination (proportion of variance explained)
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

## Approaches Comparison

### Vertical Partitioning

**Advantages:**
- Specialized models for different feature types
- Can capture feature-specific patterns
- Fusion strategy combines complementary information

**Use Cases:**
- When features have distinct characteristics
- Infrastructure vs user behavior analysis
- Feature complementarity exploration

### Horizontal Partitioning

**Advantages:**
- Context-specific model specialization
- Geographical pattern recognition
- Localized prediction optimization

**Use Cases:**
- Geographical context matters
- Urban vs rural network differences
- Location-based service optimization

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_vertical_partitioning.py

# Run with coverage
python -m pytest tests/ --cov=.
```

### Code Style

The project follows PEP 8 style guidelines. Use the following tools:

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

## Troubleshooting

### Common Issues

1. **Data File Not Found**
   - Ensure the data file path is correct
   - Check file permissions

2. **Memory Issues**
   - Reduce dataset size for testing
   - Increase system memory allocation

3. **Model Training Failures**
   - Check data quality and completeness
   - Verify minimum sample size requirements

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration

### Getting Help

1. Check the log files for detailed error messages
2. Run scripts with `--verbose` flag for more information
3. Verify data format matches expected schema
4. Ensure sufficient data for model training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Network Latency Prediction System Team
- Version 1.0

## Changelog

### Version 1.0 (Current)
- Initial implementation of vertical and horizontal partitioning
- Interactive Jupyter notebooks
- Production-ready command-line scripts
- Comprehensive evaluation framework
- Documentation and examples