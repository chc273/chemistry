"""
Publication-Quality Visualization for Quantum Chemistry Benchmarking

This module provides comprehensive visualization capabilities for benchmarking
results including error analysis plots, method comparisons, convergence studies,
and interactive dashboards suitable for publication and presentations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Optional advanced plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plots will be limited.")

try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256, RdYlBu11
    from bokeh.layouts import gridplot
    from bokeh.io import curdoc
    HAS_BOKEH = True
except ImportError:
    HAS_BOKEH = False
    warnings.warn("Bokeh not available. Interactive web plots will be limited.")

from .statistical_analysis import ErrorMetrics

logger = logging.getLogger(__name__)


class PublicationQualityPlotter:
    """Publication-quality plotting for quantum chemistry benchmarking."""
    
    def __init__(self, style: str = "publication"):
        """Initialize plotter with publication-ready settings."""
        
        self.style = style
        self.setup_matplotlib_style()
        
        # Color palettes
        self.method_colors = {
            'avas': '#1f77b4',
            'apc': '#ff7f0e', 
            'dmet': '#2ca02c',
            'boys': '#d62728',
            'casscf': '#9467bd',
            'nevpt2': '#8c564b',
            'caspt2': '#e377c2'
        }
        
        self.database_colors = {
            'w4_11': '#2E86AB',
            'g2_97': '#A23B72',
            'questdb': '#F18F01',
            'tmc_151': '#C73E1D'
        }
        
        # Figure parameters
        self.figsize = (10, 8)
        self.dpi = 300
        self.font_size = 12
        
    def setup_matplotlib_style(self):
        """Setup matplotlib for publication-quality figures."""
        
        plt.style.use('default')  # Start fresh
        
        # Set publication parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'patch.linewidth': 1,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_error_distribution(
        self,
        method_errors: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        title: str = "Error Distribution Analysis"
    ) -> plt.Figure:
        """Create comprehensive error distribution plots."""
        
        n_methods = len(method_errors)
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Histogram with KDE
        ax1 = fig.add_subplot(gs[0, 0])
        for method, errors in method_errors.items():
            color = self.method_colors.get(method, '#333333')
            ax1.hist(errors, bins=30, alpha=0.6, density=True, 
                    label=method, color=color, edgecolor='white', linewidth=0.5)
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax1.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        ax1.set_xlabel('Error (Hartree)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution with KDE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = fig.add_subplot(gs[0, 1])
        data_for_box = [errors for errors in method_errors.values()]
        labels = list(method_errors.keys())
        colors = [self.method_colors.get(method, '#333333') for method in labels]
        
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Error (Hartree)')
        ax2.set_title('Error Distribution (Box Plot)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot for normality
        ax3 = fig.add_subplot(gs[0, 2])
        from scipy import stats
        
        for method, errors in method_errors.items():
            color = self.method_colors.get(method, '#333333')
            stats.probplot(errors, dist="norm", plot=ax3)
            # Modify the last plotted line to have the correct color and label
            ax3.get_lines()[-1].set_color(color)
            ax3.get_lines()[-1].set_label(method)
            ax3.get_lines()[-2].set_color(color)  # The points
        
        ax3.set_title('Q-Q Plot (Normality Check)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Violin plot
        ax4 = fig.add_subplot(gs[1, 0])
        positions = range(1, len(method_errors) + 1)
        violin_parts = ax4.violinplot(data_for_box, positions=positions, showmeans=True)
        
        for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.set_ylabel('Error (Hartree)')
        ax4.set_title('Error Distribution (Violin Plot)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative distribution
        ax5 = fig.add_subplot(gs[1, 1])
        for method, errors in method_errors.items():
            color = self.method_colors.get(method, '#333333')
            sorted_errors = np.sort(np.abs(errors))
            y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax5.plot(sorted_errors, y_vals, label=method, color=color, linewidth=2)
        
        ax5.set_xlabel('|Error| (Hartree)')
        ax5.set_ylabel('Cumulative Probability')
        ax5.set_title('Cumulative Error Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        
        # 6. Error statistics table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Prepare statistics table
        stats_data = []
        for method, errors in method_errors.items():
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            max_err = np.max(np.abs(errors))
            stats_data.append([method, f'{mae:.4f}', f'{rmse:.4f}', f'{max_err:.4f}'])
        
        table = ax6.table(cellText=stats_data,
                         colLabels=['Method', 'MAE', 'RMSE', 'Max |Error|'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title('Error Statistics', pad=20)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {output_path}")
        
        return fig
    
    def plot_method_comparison(
        self,
        method_metrics: Dict[str, ErrorMetrics],
        output_path: Optional[Path] = None,
        title: str = "Method Performance Comparison"
    ) -> plt.Figure:
        """Create comprehensive method comparison plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, y=0.98)
        
        methods = list(method_metrics.keys())
        colors = [self.method_colors.get(method, '#333333') for method in methods]
        
        # 1. MAE comparison with error bars
        ax1 = axes[0, 0]
        mae_values = [metrics.mae for metrics in method_metrics.values()]
        mae_errors = [(metrics.mae_ci[1] - metrics.mae_ci[0])/2 for metrics in method_metrics.values()]
        
        bars1 = ax1.bar(methods, mae_values, yerr=mae_errors, capsize=5,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Mean Absolute Error (Hartree)')
        ax1.set_title('Mean Absolute Error Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mae in zip(bars1, mae_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(mae_errors)*0.1,
                    f'{mae:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. RMSE comparison
        ax2 = axes[0, 1]
        rmse_values = [metrics.rmse for metrics in method_metrics.values()]
        rmse_errors = [(metrics.rmse_ci[1] - metrics.rmse_ci[0])/2 for metrics in method_metrics.values()]
        
        bars2 = ax2.bar(methods, rmse_values, yerr=rmse_errors, capsize=5,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Root Mean Square Error (Hartree)')
        ax2.set_title('Root Mean Square Error Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rmse in zip(bars2, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(rmse_errors)*0.1,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 3. R² comparison
        ax3 = axes[1, 0]
        r2_values = [metrics.r2 for metrics in method_metrics.values()]
        
        bars3 = ax3.bar(methods, r2_values, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        ax3.set_ylabel('R² Score')
        ax3.set_title('Correlation Quality (R²)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, r2 in zip(bars3, r2_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Performance radar chart
        ax4 = axes[1, 1]
        
        # Normalize metrics for radar chart (lower is better for errors)
        mae_norm = 1 - np.array(mae_values) / max(mae_values)
        rmse_norm = 1 - np.array(rmse_values) / max(rmse_values)
        r2_norm = np.array(r2_values)
        
        # Calculate angles for radar chart
        categories = ['MAE\n(inverted)', 'RMSE\n(inverted)', 'R²']
        n_cats = len(categories)
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax4.set_theta_offset(np.pi / 2)
        ax4.set_theta_direction(-1)
        ax4.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax4.set_ylim(0, 1)
        ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax4.grid(True)
        
        for i, method in enumerate(methods):
            values = [mae_norm[i], rmse_norm[i], r2_norm[i]]
            values += values[:1]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=method, 
                    color=colors[i], markersize=6)
            ax4.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax4.set_title('Performance Profile', y=1.08)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Method comparison plot saved to {output_path}")
        
        return fig
    
    def plot_parity_plots(
        self,
        method_data: Dict[str, Dict[str, np.ndarray]],
        output_path: Optional[Path] = None,
        title: str = "Parity Plot Analysis"
    ) -> plt.Figure:
        """Create parity plots (computed vs reference) for methods."""
        
        n_methods = len(method_data)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=18, y=0.98)
        
        for i, (method, data) in enumerate(method_data.items()):
            ax = axes[i]
            
            computed = data['computed']
            reference = data['reference']
            
            # Calculate statistics
            mae = np.mean(np.abs(computed - reference))
            rmse = np.sqrt(np.mean((computed - reference)**2))
            r2 = np.corrcoef(computed, reference)[0, 1]**2
            
            # Scatter plot
            color = self.method_colors.get(method, '#333333')
            ax.scatter(reference, computed, alpha=0.6, color=color, s=50, edgecolors='white', linewidth=0.5)
            
            # Perfect correlation line
            min_val = min(reference.min(), computed.min())
            max_val = max(reference.max(), computed.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Perfect correlation')
            
            # Linear fit
            coeffs = np.polyfit(reference, computed, 1)
            fit_line = np.poly1d(coeffs)
            ax.plot(reference, fit_line(reference), 'r-', linewidth=2, alpha=0.8, label=f'Linear fit (R² = {r2:.3f})')
            
            # Confidence bands
            residuals = computed - fit_line(reference)
            rmse_fit = np.sqrt(np.mean(residuals**2))
            ax.fill_between(reference, fit_line(reference) - 2*rmse_fit, 
                           fit_line(reference) + 2*rmse_fit, alpha=0.2, color='red', label='95% confidence')
            
            ax.set_xlabel('Reference Energy (Hartree)')
            ax.set_ylabel('Computed Energy (Hartree)')
            ax.set_title(f'{method}\nMAE = {mae:.4f}, RMSE = {rmse:.4f}')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Force square aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Parity plots saved to {output_path}")
        
        return fig
    
    def plot_convergence_study(
        self,
        convergence_data: Dict[str, Dict[str, np.ndarray]],
        parameter_name: str = "Basis Set",
        output_path: Optional[Path] = None,
        title: str = "Convergence Analysis"
    ) -> plt.Figure:
        """Plot convergence with respect to a parameter."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{title}: {parameter_name}', fontsize=16)
        
        # 1. Energy convergence
        for method, data in convergence_data.items():
            param_values = data['parameter_values']
            energies = data['energies']
            
            color = self.method_colors.get(method, '#333333')
            ax1.plot(param_values, energies, 'o-', label=method, color=color, 
                    linewidth=2, markersize=8, markerfacecolor='white', 
                    markeredgecolor=color, markeredgewidth=2)
        
        ax1.set_xlabel(parameter_name)
        ax1.set_ylabel('Energy (Hartree)')
        ax1.set_title('Energy Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error convergence (if reference available)
        for method, data in convergence_data.items():
            if 'reference_energy' in data:
                param_values = data['parameter_values']
                energies = data['energies']
                reference = data['reference_energy']
                
                errors = np.abs(energies - reference)
                color = self.method_colors.get(method, '#333333')
                
                ax2.semilogy(param_values, errors, 'o-', label=method, color=color,
                           linewidth=2, markersize=8, markerfacecolor='white',
                           markeredgecolor=color, markeredgewidth=2)
        
        ax2.set_xlabel(parameter_name)
        ax2.set_ylabel('|Error| (Hartree)')
        ax2.set_title('Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add convergence threshold line
        ax2.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Chemical accuracy (1 μH)')
        ax2.axhline(y=1e-3, color='orange', linestyle='--', alpha=0.7, label='Loose threshold (1 mH)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {output_path}")
        
        return fig
    
    def plot_active_space_analysis(
        self,
        active_space_data: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None,
        title: str = "Active Space Analysis"
    ) -> plt.Figure:
        """Plot active space selection analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18)
        
        # Extract data
        methods = list(active_space_data.keys())
        n_electrons = [data['n_active_electrons'] for data in active_space_data.values()]
        n_orbitals = [data['n_active_orbitals'] for data in active_space_data.values()]
        selection_times = [data.get('selection_time', 0) for data in active_space_data.values()]
        
        colors = [self.method_colors.get(method, '#333333') for method in methods]
        
        # 1. Active space size comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, n_electrons, width, label='Electrons', 
                       color=colors, alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x_pos + width/2, n_orbitals, width, label='Orbitals',
                       color=colors, alpha=0.6, edgecolor='black')
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Count')
        ax1.set_title('Active Space Size')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, n_electrons):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val}', ha='center', va='bottom', fontsize=10)
        
        for bar, val in zip(bars2, n_orbitals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val}', ha='center', va='bottom', fontsize=10)
        
        # 2. Active space scaling
        ax2 = axes[0, 1]
        ax2.scatter(n_electrons, n_orbitals, s=200, c=colors, alpha=0.8, 
                   edgecolors='black', linewidth=2)
        
        for i, method in enumerate(methods):
            ax2.annotate(method, (n_electrons[i], n_orbitals[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('Number of Active Electrons')
        ax2.set_ylabel('Number of Active Orbitals')
        ax2.set_title('Active Space Scaling')
        ax2.grid(True, alpha=0.3)
        
        # Add diagonal lines for common ratios
        max_val = max(max(n_electrons), max(n_orbitals))
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 ratio')
        ax2.plot([0, max_val/2], [0, max_val], 'gray', linestyle=':', alpha=0.7, label='1:2 ratio')
        ax2.legend()
        
        # 3. Selection time comparison
        ax3 = axes[1, 0]
        if any(t > 0 for t in selection_times):
            bars3 = ax3.bar(methods, selection_times, color=colors, alpha=0.8, 
                           edgecolor='black', linewidth=1)
            ax3.set_ylabel('Selection Time (s)')
            ax3.set_title('Active Space Selection Time')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, time_val in zip(bars3, selection_times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(selection_times)*0.01,
                        f'{time_val:.2f}', ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No timing data available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Selection Time (No Data)')
        
        # 4. Configuration space size (combinatorial analysis)
        ax4 = axes[1, 1]
        from math import comb
        
        config_sizes = []
        for ne, no in zip(n_electrons, n_orbitals):
            if ne <= no and ne > 0 and no > 0:
                # Number of ways to place ne electrons in no orbitals
                try:
                    size = comb(no, ne)
                    config_sizes.append(min(size, 1e12))  # Cap at 1e12 for visualization
                except (ValueError, OverflowError):
                    config_sizes.append(1e12)
            else:
                config_sizes.append(0)
        
        bars4 = ax4.bar(methods, config_sizes, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        ax4.set_ylabel('Configuration Space Size')
        ax4.set_title('Computational Complexity')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars4, config_sizes):
            height = bar.get_height()
            if size > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                        f'{size:.0e}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Active space analysis plot saved to {output_path}")
        
        return fig
    
    def create_interactive_dashboard(
        self,
        benchmark_data: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """Create interactive dashboard using Plotly."""
        
        if not HAS_PLOTLY:
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return None
        
        # Extract data
        method_errors = benchmark_data.get('method_errors', {})
        method_metrics = benchmark_data.get('method_metrics', {})
        timing_data = benchmark_data.get('timing_data', {})
        
        if not method_errors:
            logger.warning("No method error data available for dashboard.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Distribution', 'Method Comparison', 
                          'Performance vs Time', 'Error Correlation'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        methods = list(method_errors.keys())
        colors_plotly = [self.method_colors.get(method, '#333333') for method in methods]
        
        # 1. Error distribution histograms
        for i, (method, errors) in enumerate(method_errors.items()):
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=method,
                    opacity=0.7,
                    marker_color=colors_plotly[i],
                    nbinsx=30
                ),
                row=1, col=1
            )
        
        # 2. Method comparison bar chart
        if method_metrics:
            mae_values = [metrics.mae for metrics in method_metrics.values()]
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=mae_values,
                    name='MAE',
                    marker_color=colors_plotly,
                    text=[f'{mae:.4f}' for mae in mae_values],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Performance vs Time scatter
        if timing_data and method_metrics:
            times = [timing_data.get(method, {}).get('mean_time', 0) for method in methods]
            mae_vals = [method_metrics[method].mae for method in methods]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=mae_vals,
                    mode='markers+text',
                    text=methods,
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=colors_plotly,
                        opacity=0.8,
                        line=dict(width=2, color='black')
                    ),
                    name='Methods'
                ),
                row=2, col=1
            )
        
        # 4. Error correlation heatmap
        if len(methods) > 1:
            # Calculate correlation matrix
            error_matrix = np.array([method_errors[method] for method in methods])
            min_length = min(len(errors) for errors in error_matrix)
            error_matrix_trimmed = np.array([errors[:min_length] for errors in error_matrix])
            
            corr_matrix = np.corrcoef(error_matrix_trimmed)
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=methods,
                    y=methods,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Quantum Chemistry Benchmarking Dashboard",
            title_font_size=20,
            showlegend=True,
            height=800,
            font=dict(size=12)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Error (Hartree)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_yaxes(title_text="MAE (Hartree)", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="MAE (Hartree)", row=2, col=1)
        
        # Save interactive plot
        if output_path:
            html_file = output_path.with_suffix('.html')
            fig.write_html(str(html_file))
            logger.info(f"Interactive dashboard saved to {html_file}")
            return str(html_file)
        
        return fig.to_html()
    
    def create_publication_figure_set(
        self,
        benchmark_data: Dict[str, Any],
        output_dir: Path,
        study_name: str = "benchmark_study"
    ) -> List[Path]:
        """Create a complete set of publication-ready figures."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # 1. Error distribution analysis
        if 'method_errors' in benchmark_data:
            error_fig = self.plot_error_distribution(
                benchmark_data['method_errors'],
                output_dir / f"{study_name}_error_distribution.png",
                f"{study_name}: Error Distribution Analysis"
            )
            created_files.append(output_dir / f"{study_name}_error_distribution.png")
            plt.close(error_fig)
        
        # 2. Method comparison
        if 'method_metrics' in benchmark_data:
            comparison_fig = self.plot_method_comparison(
                benchmark_data['method_metrics'],
                output_dir / f"{study_name}_method_comparison.png",
                f"{study_name}: Method Performance Comparison"
            )
            created_files.append(output_dir / f"{study_name}_method_comparison.png")
            plt.close(comparison_fig)
        
        # 3. Parity plots
        if 'parity_data' in benchmark_data:
            parity_fig = self.plot_parity_plots(
                benchmark_data['parity_data'],
                output_dir / f"{study_name}_parity_plots.png",
                f"{study_name}: Parity Plot Analysis"
            )
            created_files.append(output_dir / f"{study_name}_parity_plots.png")
            plt.close(parity_fig)
        
        # 4. Convergence study
        if 'convergence_data' in benchmark_data:
            conv_fig = self.plot_convergence_study(
                benchmark_data['convergence_data'],
                benchmark_data.get('parameter_name', 'Parameter'),
                output_dir / f"{study_name}_convergence.png",
                f"{study_name}: Convergence Analysis"
            )
            created_files.append(output_dir / f"{study_name}_convergence.png")
            plt.close(conv_fig)
        
        # 5. Active space analysis
        if 'active_space_data' in benchmark_data:
            as_fig = self.plot_active_space_analysis(
                benchmark_data['active_space_data'],
                output_dir / f"{study_name}_active_space.png",
                f"{study_name}: Active Space Analysis"
            )
            created_files.append(output_dir / f"{study_name}_active_space.png")
            plt.close(as_fig)
        
        # 6. Interactive dashboard
        dashboard_file = self.create_interactive_dashboard(
            benchmark_data,
            output_dir / f"{study_name}_dashboard.html"
        )
        if dashboard_file:
            created_files.append(Path(dashboard_file))
        
        # 7. Combined PDF report
        pdf_file = output_dir / f"{study_name}_figures.pdf"
        self._create_pdf_report(created_files[:-1], pdf_file)  # Exclude HTML dashboard
        created_files.append(pdf_file)
        
        logger.info(f"Created {len(created_files)} publication figures in {output_dir}")
        return created_files
    
    def _create_pdf_report(self, figure_paths: List[Path], output_path: Path) -> None:
        """Combine all figures into a single PDF report."""
        
        with PdfPages(output_path) as pdf:
            for fig_path in figure_paths:
                if fig_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Read and display image
                    import matplotlib.image as mpimg
                    
                    img = mpimg.imread(fig_path)
                    fig, ax = plt.subplots(figsize=(11, 8.5))  # Letter size
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(fig_path.stem.replace('_', ' ').title(), fontsize=16, pad=20)
                    
                    pdf.savefig(fig, bbox_inches='tight', dpi=150)
                    plt.close(fig)
        
        logger.info(f"Combined PDF report saved to {output_path}")