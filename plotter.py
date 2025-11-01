# Plotter Module

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional
import matplotlib.patches as patches
import warnings


class Plotter:

    def __init__(self, style: str = 'default'):

        self.style = style
        self._setup_style()

    def _setup_style(self):

        if self.style == 'seaborn':
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif self.style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')
        else:
            plt.style.use('default')

    def plot_bootstrap_results(self,
                               bootstrap_results: np.ndarray,
                               time_interval: int,
                               output_file: str = 'bootstrap_analysis.pdf',
                               title: str = 'Bootstrap Analysis Results',
                               xlabel: str = 'Age (Ma)',
                               ylabel: str = 'Probability Ratio (%)',
                               confidence_level: float = 0.95,
                               min_age: float = None,
                               max_age: float = None,
                               figsize: tuple = (12, 8),
                               dpi: int = 500) -> bool:

        # Plot Bootstrap analysis results
        try:
            # If age range is not specified, auto-detect
            if min_age is None or max_age is None:
                # Estimate age range based on bootstrap_results shape
                n_bins = bootstrap_results.shape[0]
                if min_age is None:
                    min_age = 0.0
                if max_age is None:
                    max_age = n_bins * time_interval
                
                warnings.warn(f"Age range not fully specified for plotting. Using calculated range: {plot_min_age}-{plot_max_age} (based on number of bins). Ensure this is correct.", UserWarning)
            else:
                plot_min_age = min_age
                plot_max_age = max_age
            
            # Generate bin edges
            bin_edges = np.arange(plot_min_age, plot_max_age + time_interval, time_interval)

            # Calculate statistics
            stats = self._calculate_plot_statistics(bootstrap_results, confidence_level)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Generate X-axis data
            n_bins = bootstrap_results.shape[0]
            age_x = np.arange(min_age + time_interval / 2, max_age, time_interval)

            # Plot error bars
            if confidence_level == 0.95:
                yerr = stats['confidence_95_std']
            elif confidence_level == 0.90:
                yerr = stats['confidence_90_std']
            elif confidence_level == 0.68:
                yerr = stats['confidence_68_std']
            else:
                yerr = stats['std'] * 2  # Default to 2x standard deviation

            # Plot main curve
            ax.errorbar(age_x, stats['mean'], yerr=yerr,
                        ecolor='red', capsize=4, capthick=2,
                        linewidth=2, marker='o', markersize=6,
                        label='Bootstrap Mean')

            # Plot confidence intervals
            if confidence_level == 0.95:
                ax.fill_between(age_x, stats['confidence_95_lower'],
                                stats['confidence_95_upper'],
                                alpha=0.3, color='blue',
                                label=f'{int(confidence_level * 100)}% Confidence Interval')
            elif confidence_level == 0.90:
                ax.fill_between(age_x, stats['confidence_90_lower'],
                                stats['confidence_90_upper'],
                                alpha=0.3, color='blue',
                                label=f'{int(confidence_level * 100)}% Confidence Interval')
            elif confidence_level == 0.68:
                ax.fill_between(age_x, stats['confidence_68_lower'],
                                stats['confidence_68_upper'],
                                alpha=0.3, color='blue',
                                label=f'{int(confidence_level * 100)}% Confidence Interval')

            # Set plot attributes
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)

            # Set axis limits
            ax.set_xlim(min_age, max_age)
            ax.set_ylim(0, 100)

            # Invert X-axis (geological time from old to new)
            ax.invert_xaxis()

            # Save figure
            plt.tight_layout()
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()

            print(f"Plot saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Plotting failed: {str(e)}")
            return False

    def plot_bootstrap_results_detailed(self,
                                        bootstrap_results: np.ndarray,
                                        time_interval: int,
                                        output_file: str = 'bootstrap_analysis_detailed.pdf',
                                        min_age: float = None,
                                        max_age: float = None,
                                        figsize: tuple = (15, 10)) -> bool:

        # Plot detailed Bootstrap analysis results (with multiple subplots)
        try:
            # If age range is not specified, auto-detect
            if min_age is None or max_age is None:
                # Estimate age range based on bootstrap_results shape
                n_bins = bootstrap_results.shape[0]
                if min_age is None:
                    min_age = 0.0
                if max_age is None:
                    max_age = n_bins * time_interval
            
            # Calculate statistical information
            stats = self._calculate_plot_statistics(bootstrap_results, 0.95)

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # Generate X-axis data
            n_bins = bootstrap_results.shape[0]
            age_x = np.arange(min_age + time_interval / 2, max_age, time_interval)

            # Subplot 1: Main Results
            ax1.errorbar(age_x, stats['mean'], yerr=stats['confidence_95_std'],
                         ecolor='red', capsize=4, linewidth=2, marker='o')
            ax1.fill_between(age_x, stats['confidence_95_lower'],
                             stats['confidence_95_upper'], alpha=0.3, color='blue')
            ax1.set_xlabel('Age (Ma)')
            ax1.set_ylabel('Probability (%)')
            ax1.set_title('Bootstrap Analysis Results')
            ax1.grid(True, alpha=0.3)
            ax1.invert_xaxis()

            # Subplot 2: Boxplot
            ax2.boxplot([bootstrap_results[i, :] for i in range(n_bins)],
                        positions=age_x, widths=time_interval * 0.8)
            ax2.set_xlabel('Age (Ma)')
            ax2.set_ylabel('Probability (%)')
            ax2.set_title('Distribution Boxplot')
            ax2.grid(True, alpha=0.3)
            ax2.invert_xaxis()

            # Subplot 3: Heatmap
            im = ax3.imshow(bootstrap_results, aspect='auto', cmap='viridis',
                            extent=[0, bootstrap_results.shape[1], max_age, min_age])
            ax3.set_xlabel('Bootstrap Iteration')
            ax3.set_ylabel('Age (Ma)')
            ax3.set_title('Bootstrap Results Heatmap')
            plt.colorbar(im, ax=ax3, label='Probability (%)')

            # Subplot 4: Statistics Summary
            ax4.plot(age_x, stats['mean'], 'b-', label='Mean', linewidth=2)
            ax4.plot(age_x, stats['median'], 'r--', label='Median', linewidth=2)
            ax4.fill_between(age_x, stats['confidence_68_lower'],
                             stats['confidence_68_upper'], alpha=0.3, color='green',
                             label='68% Confidence Interval')
            ax4.set_xlabel('Age (Ma)')
            ax4.set_ylabel('Probability (%)')
            ax4.set_title('Statistics Summary')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.invert_xaxis()

            # Adjust layout
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Detailed plot saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Failed to plot detailed chart: {str(e)}")
            return False

    def plot_weight_distribution(self,
                                 weights: np.ndarray,
                                 output_file: str = 'weight_distribution.pdf',
                                 title: str = 'Weight Distribution') -> bool:

        # Plot weight distribution
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Histogram
            ax1.hist(weights.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Weight Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Weight Distribution Histogram')
            ax1.grid(True, alpha=0.3)

            # Boxplot
            ax2.boxplot(weights.flatten())
            ax2.set_ylabel('Weight Value')
            ax2.set_title('Weight Distribution Boxplot')
            ax2.grid(True, alpha=0.3)

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Weight distribution plot saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Failed to plot weight distribution: {str(e)}")
            return False

    def plot_data_summary(self,
                          data: Dict[str, np.ndarray],
                          output_file: str = 'data_summary.pdf') -> bool:

        # Plot data summary
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Age Distribution
            ax1.hist(data['age'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_xlabel('Age (Ma)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Age Distribution')
            ax1.grid(True, alpha=0.3)

            # Probability Distribution
            ax2.hist(data['probability'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Probability Distribution')
            ax2.grid(True, alpha=0.3)

            # Spatial Distribution
            scatter = ax3.scatter(data['lon'], data['lat'], c=data['age'],
                                  cmap='viridis', alpha=0.6, s=10)
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('Spatial Distribution (Color represents Age)')
            plt.colorbar(scatter, ax=ax3, label='Age (Ma)')
            ax3.grid(True, alpha=0.3)

            # Age vs Probability Scatter Plot
            ax4.scatter(data['age'], data['probability'], alpha=0.5, s=10)
            ax4.set_xlabel('Age (Ma)')
            ax4.set_ylabel('Probability')
            ax4.set_title('Age vs Probability')
            ax4.grid(True, alpha=0.3)
            ax4.invert_xaxis()

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Data summary plot saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Failed to plot data summary: {str(e)}")
            return False

    def _calculate_plot_statistics(self,
                                   bootstrap_results: np.ndarray,
                                   confidence_level: float) -> Dict[str, np.ndarray]:

        # Basic statistics
        mean_values = np.nanmean(bootstrap_results, axis=1)
        std_values = np.nanstd(bootstrap_results, axis=1)
        median_values = np.nanmedian(bootstrap_results, axis=1)

        # Confidence interval
        confidence_95 = np.percentile(bootstrap_results, [2.5, 97.5], axis=1)
        confidence_90 = np.percentile(bootstrap_results, [5, 95], axis=1)
        confidence_68 = np.percentile(bootstrap_results, [16, 84], axis=1)

        return {
            'mean': mean_values,
            'std': std_values,
            'median': median_values,
            'confidence_95_lower': confidence_95[0, :],
            'confidence_95_upper': confidence_95[1, :],
            'confidence_95_std': (confidence_95[1, :] - confidence_95[0, :]) / 2,
            'confidence_90_lower': confidence_90[0, :],
            'confidence_90_upper': confidence_90[1, :],
            'confidence_90_std': (confidence_90[1, :] - confidence_90[0, :]) / 2,
            'confidence_68_lower': confidence_68[0, :],
            'confidence_68_upper': confidence_68[1, :],
            'confidence_68_std': (confidence_68[1, :] - confidence_68[0, :]) / 2
        }