"""
Performance comparison and reporting utilities for network latency prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import warnings
from evaluation.model_evaluator import ModelEvaluator


class PerformanceReporter:
    """
    Performance comparison and reporting system for model evaluation results.
    
    This class provides functionality to generate performance evaluation tables,
    create visualizations for model comparison results, and generate summary
    findings and interpretations.
    """
    
    def __init__(self, evaluator: ModelEvaluator = None):
        """
        Initialize the PerformanceReporter.
        
        Args:
            evaluator: Optional ModelEvaluator instance with evaluation results
        """
        self.evaluator = evaluator or ModelEvaluator()
        self.report_data = {}
    
    def generate_performance_table(self, models: Dict[str, Any], 
                                 X: Union[np.ndarray, pd.DataFrame], 
                                 y: np.ndarray,
                                 include_cv: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive performance evaluation table.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X: Feature matrix or DataFrame
            y: True target values
            include_cv: Whether to include cross-validation results
            
        Returns:
            DataFrame with comprehensive performance comparison
        """
        # Get basic comparison results
        comparison_df = self.evaluator.compare_models(models, X, y)
        
        # Add cross-validation results if requested
        if include_cv:
            try:
                cv_results = self.evaluator.cross_validate_models(models, X, y)
                
                # Merge CV results with comparison results
                cv_metrics = cv_results[['Model_Name', 'CV_Mean', 'CV_Std']].rename(columns={
                    'CV_Mean': 'CV_Score_Mean',
                    'CV_Std': 'CV_Score_Std'
                })
                
                comparison_df = comparison_df.merge(cv_metrics, on='Model_Name', how='left')
                
            except Exception as e:
                warnings.warn(f"Cross-validation failed, skipping CV metrics: {str(e)}")
        
        # Add ranking based on R² Score
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        # Reorder columns for better presentation
        column_order = ['Rank', 'Model_Name', 'Model_Type', 'R2_Score', 'MAE', 'RMSE', 'MAPE']
        if include_cv and 'CV_Score_Mean' in comparison_df.columns:
            column_order.extend(['CV_Score_Mean', 'CV_Score_Std'])
        column_order.extend(['Mean_Residual', 'Std_Residual', 'Sample_Size'])
        
        # Filter to only include existing columns
        available_columns = [col for col in column_order if col in comparison_df.columns]
        comparison_df = comparison_df[available_columns]
        
        # Store for reporting
        self.report_data['performance_table'] = comparison_df
        
        return comparison_df
    
    def create_performance_visualization(self, performance_df: pd.DataFrame, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of model performance comparison.
        
        Args:
            performance_df: DataFrame with performance metrics
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. R² Score comparison (bar plot)
        ax1 = axes[0, 0]
        bars1 = ax1.bar(performance_df['Model_Name'], performance_df['R2_Score'], 
                       color=sns.color_palette("husl", len(performance_df)))
        ax1.set_title('R² Score Comparison', fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. MAE comparison (bar plot)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(performance_df['Model_Name'], performance_df['MAE'],
                       color=sns.color_palette("husl", len(performance_df)))
        ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(performance_df['MAE']) * 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. RMSE comparison (bar plot)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(performance_df['Model_Name'], performance_df['RMSE'],
                       color=sns.color_palette("husl", len(performance_df)))
        ax3.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        ax3.set_ylabel('RMSE')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(performance_df['RMSE']) * 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. Multi-metric comparison (radar chart or grouped bar)
        ax4 = axes[1, 1]
        
        # Normalize metrics for comparison (0-1 scale)
        metrics_to_plot = ['R2_Score', 'MAE', 'RMSE']
        normalized_data = performance_df[metrics_to_plot].copy()
        
        # For MAE and RMSE, lower is better, so we invert them
        if 'MAE' in normalized_data.columns:
            normalized_data['MAE'] = 1 - (normalized_data['MAE'] / normalized_data['MAE'].max())
        if 'RMSE' in normalized_data.columns:
            normalized_data['RMSE'] = 1 - (normalized_data['RMSE'] / normalized_data['RMSE'].max())
        
        # Create grouped bar chart
        x = np.arange(len(performance_df))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            ax4.bar(x + i * width, normalized_data[metric], width, 
                   label=metric, alpha=0.8)
        
        ax4.set_title('Normalized Performance Metrics', fontweight='bold')
        ax4.set_ylabel('Normalized Score (Higher is Better)')
        ax4.set_xlabel('Models')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(performance_df['Model_Name'], rotation=45)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_residual_analysis_plot(self, models: Dict[str, Any], 
                                    X: Union[np.ndarray, pd.DataFrame], 
                                    y: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create residual analysis plots for model comparison.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X: Feature matrix or DataFrame
            y: True target values
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_models = len(models)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
        
        for i, (model_name, model) in enumerate(models.items()):
            try:
                predictions = model.predict(X)
                residuals = y - predictions
                
                # Residuals vs Predicted plot
                ax1 = axes[0, i]
                ax1.scatter(predictions, residuals, alpha=0.6)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_title(f'{model_name}\nResiduals vs Predicted')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Residuals')
                
                # Q-Q plot for residuals
                ax2 = axes[1, i]
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=ax2)
                ax2.set_title(f'{model_name}\nQ-Q Plot')
                
            except Exception as e:
                warnings.warn(f"Failed to create residual plot for {model_name}: {str(e)}")
                continue
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_summary_findings(self, performance_df: pd.DataFrame, 
                                significance_tests: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate summary findings and interpretation from performance results.
        
        Args:
            performance_df: DataFrame with performance metrics
            significance_tests: Optional dictionary with significance test results
            
        Returns:
            Dictionary with summary findings and interpretations
        """
        if performance_df.empty:
            return {"error": "No performance data available for analysis"}
        
        # Best performing model
        best_model = performance_df.iloc[0]
        worst_model = performance_df.iloc[-1]
        
        # Performance statistics
        r2_stats = {
            'mean': performance_df['R2_Score'].mean(),
            'std': performance_df['R2_Score'].std(),
            'range': performance_df['R2_Score'].max() - performance_df['R2_Score'].min()
        }
        
        mae_stats = {
            'mean': performance_df['MAE'].mean(),
            'std': performance_df['MAE'].std(),
            'range': performance_df['MAE'].max() - performance_df['MAE'].min()
        }
        
        # Generate findings
        findings = {
            'best_model': {
                'name': best_model['Model_Name'],
                'type': best_model['Model_Type'],
                'r2_score': best_model['R2_Score'],
                'mae': best_model['MAE'],
                'rmse': best_model['RMSE']
            },
            'worst_model': {
                'name': worst_model['Model_Name'],
                'type': worst_model['Model_Type'],
                'r2_score': worst_model['R2_Score'],
                'mae': worst_model['MAE'],
                'rmse': worst_model['RMSE']
            },
            'performance_statistics': {
                'r2_score_stats': r2_stats,
                'mae_stats': mae_stats,
                'total_models': len(performance_df)
            },
            'interpretations': self._generate_interpretations(performance_df, significance_tests),
            'recommendations': self._generate_recommendations(performance_df)
        }
        
        return findings
    
    def _generate_interpretations(self, performance_df: pd.DataFrame, 
                                significance_tests: Dict[str, Any] = None) -> List[str]:
        """
        Generate human-readable interpretations of the results.
        
        Args:
            performance_df: DataFrame with performance metrics
            significance_tests: Optional significance test results
            
        Returns:
            List of interpretation strings
        """
        interpretations = []
        
        # Overall performance interpretation
        best_r2 = performance_df['R2_Score'].max()
        if best_r2 > 0.8:
            interpretations.append("Excellent model performance achieved with R² > 0.8")
        elif best_r2 > 0.6:
            interpretations.append("Good model performance achieved with R² > 0.6")
        elif best_r2 > 0.4:
            interpretations.append("Moderate model performance achieved with R² > 0.4")
        else:
            interpretations.append("Model performance is below expectations with R² < 0.4")
        
        # Performance variation interpretation
        r2_range = performance_df['R2_Score'].max() - performance_df['R2_Score'].min()
        if r2_range > 0.2:
            interpretations.append("Significant performance variation between models suggests model choice is critical")
        else:
            interpretations.append("Similar performance across models suggests consistent predictive capability")
        
        # Model type analysis
        if 'Model_Type' in performance_df.columns:
            model_types = performance_df['Model_Type'].value_counts()
            if len(model_types) > 1:
                best_type = performance_df.iloc[0]['Model_Type']
                interpretations.append(f"Best performing model type: {best_type}")
        
        # Significance test interpretation
        if significance_tests:
            significant_count = sum(1 for test in significance_tests.values() 
                                  if test.get('is_significant', False))
            total_tests = len(significance_tests)
            
            if significant_count > 0:
                interpretations.append(f"{significant_count}/{total_tests} model comparisons show statistically significant differences")
            else:
                interpretations.append("No statistically significant differences found between models")
        
        return interpretations
    
    def _generate_recommendations(self, performance_df: pd.DataFrame) -> List[str]:
        """
        Generate actionable recommendations based on performance results.
        
        Args:
            performance_df: DataFrame with performance metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Best model recommendation
        best_model = performance_df.iloc[0]
        recommendations.append(f"Recommended model: {best_model['Model_Name']} ({best_model['Model_Type']})")
        
        # Performance improvement suggestions
        best_r2 = performance_df['R2_Score'].max()
        if best_r2 < 0.7:
            recommendations.append("Consider feature engineering or ensemble methods to improve performance")
        
        # Error analysis recommendations
        if 'MAPE' in performance_df.columns:
            high_mape_models = performance_df[performance_df['MAPE'] > 20]
            if not high_mape_models.empty:
                recommendations.append("High MAPE values detected - review prediction accuracy for business impact")
        
        # Model complexity recommendations
        if len(performance_df) > 3:
            recommendations.append("Multiple models show similar performance - consider computational efficiency in final selection")
        
        return recommendations
    
    def create_comprehensive_report(self, models: Dict[str, Any], 
                                  X: Union[np.ndarray, pd.DataFrame], 
                                  y: np.ndarray,
                                  output_dir: str = "performance_report") -> Dict[str, Any]:
        """
        Create a comprehensive performance report with tables, visualizations, and findings.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X: Feature matrix or DataFrame
            y: True target values
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with complete report data
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate performance table
        performance_table = self.generate_performance_table(models, X, y)
        
        # Generate comprehensive summary
        summary = self.evaluator.generate_performance_summary(models, X, y)
        
        # Create visualizations
        try:
            perf_fig = self.create_performance_visualization(
                performance_table, 
                save_path=os.path.join(output_dir, "performance_comparison.png")
            )
            plt.close(perf_fig)
        except Exception as e:
            warnings.warn(f"Failed to create performance visualization: {str(e)}")
        
        try:
            residual_fig = self.create_residual_analysis_plot(
                models, X, y,
                save_path=os.path.join(output_dir, "residual_analysis.png")
            )
            plt.close(residual_fig)
        except Exception as e:
            warnings.warn(f"Failed to create residual analysis: {str(e)}")
        
        # Generate findings
        findings = self.generate_summary_findings(
            performance_table, 
            summary.get('statistical_significance_tests', {})
        )
        
        # Save performance table
        performance_table.to_csv(os.path.join(output_dir, "performance_table.csv"), index=False)
        
        # Create text report
        self._create_text_report(performance_table, findings, 
                               os.path.join(output_dir, "performance_report.txt"))
        
        # Compile complete report
        complete_report = {
            'performance_table': performance_table,
            'summary': summary,
            'findings': findings,
            'output_directory': output_dir,
            'files_created': [
                "performance_table.csv",
                "performance_comparison.png",
                "residual_analysis.png",
                "performance_report.txt"
            ]
        }
        
        return complete_report
    
    def _create_text_report(self, performance_table: pd.DataFrame, 
                          findings: Dict[str, Any], output_path: str) -> None:
        """
        Create a text-based performance report.
        
        Args:
            performance_table: DataFrame with performance metrics
            findings: Dictionary with summary findings
            output_path: Path to save the text report
        """
        with open(output_path, 'w') as f:
            f.write("NETWORK LATENCY PREDICTION MODEL PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            best_model = findings['best_model']
            f.write(f"Best Performing Model: {best_model['name']} ({best_model['type']})\n")
            f.write(f"R² Score: {best_model['r2_score']:.4f}\n")
            f.write(f"MAE: {best_model['mae']:.4f}\n")
            f.write(f"RMSE: {best_model['rmse']:.4f}\n\n")
            
            # Performance Table
            f.write("DETAILED PERFORMANCE COMPARISON\n")
            f.write("-" * 35 + "\n")
            f.write(performance_table.to_string(index=False))
            f.write("\n\n")
            
            # Interpretations
            f.write("KEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            for i, interpretation in enumerate(findings['interpretations'], 1):
                f.write(f"{i}. {interpretation}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 18 + "\n")
            for i, recommendation in enumerate(findings['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # Statistics
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 25 + "\n")
            stats = findings['performance_statistics']
            f.write(f"Total Models Evaluated: {stats['total_models']}\n")
            f.write(f"R² Score Range: {stats['r2_score_stats']['range']:.4f}\n")
            f.write(f"Average R² Score: {stats['r2_score_stats']['mean']:.4f}\n")
            f.write(f"MAE Range: {stats['mae_stats']['range']:.4f}\n")
            f.write(f"Average MAE: {stats['mae_stats']['mean']:.4f}\n")