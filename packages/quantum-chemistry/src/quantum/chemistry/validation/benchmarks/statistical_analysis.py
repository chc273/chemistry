"""
Advanced Statistical Analysis for Quantum Chemistry Benchmarking

This module provides sophisticated statistical analysis tools including:
- Uncertainty quantification with error propagation
- Bootstrap confidence intervals
- Bayesian analysis for method comparison
- Advanced error metrics and model validation
- Publication-quality statistical reporting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import Bootstrap
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson

# Optional advanced statistical packages
try:
    import pymc3 as pm
    HAS_PYMC3 = True
except ImportError:
    HAS_PYMC3 = False
    warnings.warn("PyMC3 not available. Bayesian analysis will be limited.")

try:
    from uncertainties import ufloat, unumpy
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    warnings.warn("Uncertainties package not available. Error propagation will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class ErrorMetrics:
    """Comprehensive error metrics for method evaluation."""
    
    # Basic metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    max_error: float  # Maximum error
    median_error: float  # Median error
    r2: float  # R-squared
    
    # Advanced metrics
    mape: float  # Mean Absolute Percentage Error
    smape: float  # Symmetric Mean Absolute Percentage Error
    mase: Optional[float]  # Mean Absolute Scaled Error
    
    # Statistical significance
    mean_error: float  # Mean error (bias)
    std_error: float  # Standard deviation of errors
    
    # Confidence intervals
    mae_ci: Tuple[float, float]  # Bootstrap CI for MAE
    rmse_ci: Tuple[float, float]  # Bootstrap CI for RMSE
    bias_ci: Tuple[float, float]  # Bootstrap CI for bias
    
    # Distribution tests
    normality_p_value: float  # Shapiro-Wilk test p-value
    is_normal: bool  # Whether errors are normally distributed
    
    # Regression diagnostics
    durbin_watson_stat: Optional[float]  # Autocorrelation test
    white_test_p_value: Optional[float]  # Heteroscedasticity test
    
    # Sample size
    n_samples: int
    
    @property
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "max_error": self.max_error,
            "median_error": self.median_error, 
            "r2": self.r2,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "is_normal": self.is_normal,
            "n_samples": self.n_samples
        }


@dataclass
class UncertaintyAnalysis:
    """Uncertainty analysis results."""
    
    # Propagated uncertainties
    total_uncertainty: float
    systematic_uncertainty: float
    statistical_uncertainty: float
    model_uncertainty: float
    
    # Coverage analysis
    coverage_probability: float  # Fraction of reference values within uncertainty
    mean_coverage_width: float  # Average width of uncertainty intervals
    
    # Uncertainty correlations
    uncertainty_vs_error_correlation: float
    uncertainty_reliability: float  # How well uncertainties predict actual errors


class UncertaintyQuantifier:
    """Advanced uncertainty quantification for quantum chemistry calculations."""
    
    def __init__(self):
        self.bootstrap_samples = 1000
        self.confidence_level = 0.95
    
    def propagate_uncertainties(
        self,
        computed_values: np.ndarray,
        reference_values: np.ndarray,
        computed_uncertainties: Optional[np.ndarray] = None,
        reference_uncertainties: Optional[np.ndarray] = None
    ) -> UncertaintyAnalysis:
        """Propagate uncertainties through error calculation."""
        
        if not HAS_UNCERTAINTIES:
            logger.warning("Uncertainties package not available. Using simplified analysis.")
            return self._simplified_uncertainty_analysis(
                computed_values, reference_values, computed_uncertainties, reference_uncertainties
            )
        
        # Create uncertain numbers
        if computed_uncertainties is not None:
            computed_uncertain = unumpy.uarray(computed_values, computed_uncertainties)
        else:
            computed_uncertain = unumpy.uarray(computed_values, 0)
        
        if reference_uncertainties is not None:
            reference_uncertain = unumpy.uarray(reference_values, reference_uncertainties)
        else:
            reference_uncertain = unumpy.uarray(reference_values, 0)
        
        # Calculate errors with uncertainty propagation
        errors_uncertain = computed_uncertain - reference_uncertain
        
        # Extract nominal values and uncertainties
        error_values = unumpy.nominal_values(errors_uncertain)
        error_uncertainties = unumpy.std_devs(errors_uncertain)
        
        # Calculate uncertainty components
        systematic_unc = np.mean(computed_uncertainties) if computed_uncertainties is not None else 0
        statistical_unc = np.std(error_values) / np.sqrt(len(error_values))
        model_unc = np.std(error_uncertainties)
        total_unc = np.sqrt(systematic_unc**2 + statistical_unc**2 + model_unc**2)
        
        # Coverage analysis
        if error_uncertainties is not None and np.any(error_uncertainties > 0):
            # Check if actual errors fall within predicted uncertainties
            z_scores = np.abs(error_values) / error_uncertainties
            coverage_prob = np.mean(z_scores <= 1.96)  # 95% coverage
            mean_coverage_width = 2 * 1.96 * np.mean(error_uncertainties)
        else:
            coverage_prob = 0.0
            mean_coverage_width = 0.0
        
        # Uncertainty reliability
        if error_uncertainties is not None and np.any(error_uncertainties > 0):
            unc_error_corr = np.corrcoef(error_uncertainties, np.abs(error_values))[0, 1]
            unc_reliability = 1.0 - np.mean((np.abs(error_values) - error_uncertainties)**2) / np.var(error_values)
        else:
            unc_error_corr = 0.0
            unc_reliability = 0.0
        
        return UncertaintyAnalysis(
            total_uncertainty=total_unc,
            systematic_uncertainty=systematic_unc,
            statistical_uncertainty=statistical_unc,
            model_uncertainty=model_unc,
            coverage_probability=coverage_prob,
            mean_coverage_width=mean_coverage_width,
            uncertainty_vs_error_correlation=unc_error_corr,
            uncertainty_reliability=unc_reliability
        )
    
    def _simplified_uncertainty_analysis(
        self,
        computed_values: np.ndarray,
        reference_values: np.ndarray,
        computed_uncertainties: Optional[np.ndarray] = None,
        reference_uncertainties: Optional[np.ndarray] = None
    ) -> UncertaintyAnalysis:
        """Simplified uncertainty analysis without uncertainties package."""
        
        errors = computed_values - reference_values
        
        # Estimate uncertainty components
        systematic_unc = np.mean(computed_uncertainties) if computed_uncertainties is not None else 0
        statistical_unc = np.std(errors) / np.sqrt(len(errors))
        model_unc = 0.0  # Cannot calculate without uncertainties package
        total_unc = np.sqrt(systematic_unc**2 + statistical_unc**2)
        
        return UncertaintyAnalysis(
            total_uncertainty=total_unc,
            systematic_uncertainty=systematic_unc,
            statistical_uncertainty=statistical_unc,
            model_uncertainty=model_unc,
            coverage_probability=0.0,
            mean_coverage_width=0.0,
            uncertainty_vs_error_correlation=0.0,
            uncertainty_reliability=0.0
        )
    
    def bootstrap_confidence_intervals(
        self,
        errors: np.ndarray,
        metric_func: callable,
        confidence_level: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for error metrics."""
        
        n_samples = len(errors)
        bootstrap_stats = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_errors = errors[boot_indices]
            
            # Calculate metric
            boot_stat = metric_func(boot_errors)
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        # Point estimate
        point_estimate = metric_func(errors)
        
        return point_estimate, (ci_lower, ci_upper)


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analyzer with publication-quality metrics."""
    
    def __init__(self):
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.significance_level = 0.05
        self.bootstrap_samples = 1000
    
    def comprehensive_error_analysis(
        self,
        computed_values: np.ndarray,
        reference_values: np.ndarray,
        computed_uncertainties: Optional[np.ndarray] = None,
        reference_uncertainties: Optional[np.ndarray] = None
    ) -> ErrorMetrics:
        """Perform comprehensive error analysis with advanced metrics."""
        
        # Remove any NaN or infinite values
        mask = np.isfinite(computed_values) & np.isfinite(reference_values)
        computed = computed_values[mask]
        reference = reference_values[mask]
        
        if len(computed) == 0:
            raise ValueError("No valid data points for analysis")
        
        # Calculate errors
        errors = computed - reference
        abs_errors = np.abs(errors)
        relative_errors = errors / reference
        
        # Basic metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(abs_errors)
        median_error = np.median(abs_errors)
        r2 = r2_score(reference, computed)
        
        # Advanced metrics
        mape = np.mean(np.abs(relative_errors)) * 100  # Percentage
        smape = np.mean(2 * abs_errors / (np.abs(computed) + np.abs(reference))) * 100
        
        # MASE (Mean Absolute Scaled Error) - requires naive forecast
        # For now, use mean of reference values as naive forecast
        naive_errors = np.abs(reference - np.mean(reference))
        mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else None
        
        # Statistical properties
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Bootstrap confidence intervals
        mae_est, mae_ci = self.uncertainty_quantifier.bootstrap_confidence_intervals(
            abs_errors, np.mean
        )
        rmse_est, rmse_ci = self.uncertainty_quantifier.bootstrap_confidence_intervals(
            errors, lambda x: np.sqrt(np.mean(x**2))
        )
        bias_est, bias_ci = self.uncertainty_quantifier.bootstrap_confidence_intervals(
            errors, np.mean
        )
        
        # Normality test
        if len(errors) >= 3:
            normality_stat, normality_p = stats.shapiro(errors)
            is_normal = normality_p > self.significance_level
        else:
            normality_p = 0.0
            is_normal = False
        
        # Regression diagnostics (if enough data)
        durbin_watson_stat = None
        white_test_p = None
        
        if len(errors) >= 10:
            try:
                # Fit simple linear regression
                X = sm.add_constant(reference)
                model = sm.OLS(computed, X).fit()
                
                # Durbin-Watson test for autocorrelation
                durbin_watson_stat = durbin_watson(model.resid)
                
                # White test for heteroscedasticity
                white_stat, white_p, _, _ = het_white(model.resid, X)
                white_test_p = white_p
                
            except Exception as e:
                logger.warning(f"Regression diagnostics failed: {e}")
        
        return ErrorMetrics(
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            median_error=median_error,
            r2=r2,
            mape=mape,
            smape=smape,
            mase=mase,
            mean_error=mean_error,
            std_error=std_error,
            mae_ci=mae_ci,
            rmse_ci=rmse_ci,
            bias_ci=bias_ci,
            normality_p_value=normality_p,
            is_normal=is_normal,
            durbin_watson_stat=durbin_watson_stat,
            white_test_p_value=white_test_p,
            n_samples=len(errors)
        )
    
    def bayesian_method_comparison(
        self,
        method_errors: Dict[str, np.ndarray],
        prior_params: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """Bayesian comparison of methods using hierarchical modeling."""
        
        if not HAS_PYMC3:
            logger.warning("PyMC3 not available. Using frequentist comparison.")
            return self._frequentist_method_comparison(method_errors)
        
        # Prepare data
        methods = list(method_errors.keys())
        n_methods = len(methods)
        
        # Flatten data
        all_errors = []
        method_indices = []
        
        for i, method in enumerate(methods):
            errors = method_errors[method]
            all_errors.extend(errors)
            method_indices.extend([i] * len(errors))
        
        all_errors = np.array(all_errors)
        method_indices = np.array(method_indices)
        
        # Bayesian hierarchical model
        with pm.Model() as model:
            # Hyperpriors for method means and precisions
            mu_mu = pm.Normal('mu_mu', 0, 1)  # Grand mean
            sigma_mu = pm.HalfNormal('sigma_mu', 1)  # Between-method variation
            
            # Method-specific parameters
            method_means = pm.Normal('method_means', mu_mu, sigma_mu, shape=n_methods)
            method_precisions = pm.Gamma('method_precisions', 2, 1, shape=n_methods)
            
            # Likelihood
            obs = pm.Normal('obs', 
                          method_means[method_indices], 
                          1/pm.math.sqrt(method_precisions[method_indices]),
                          observed=all_errors)
            
            # Sample
            trace = pm.sample(2000, tune=1000, return_inferencedata=False)
        
        # Extract results
        method_summaries = {}
        for i, method in enumerate(methods):
            method_summaries[method] = {
                'posterior_mean': np.mean(trace['method_means'][:, i]),
                'posterior_std': np.std(trace['method_means'][:, i]),
                'hdi_95': pm.hdi(trace['method_means'][:, i], hdi_prob=0.95).tolist()
            }
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                diff = trace['method_means'][:, i] - trace['method_means'][:, j]
                prob_better = np.mean(diff < 0)  # Probability method1 is better (smaller error)
                
                pairwise_comparisons[f"{method1}_vs_{method2}"] = {
                    'prob_method1_better': prob_better,
                    'mean_difference': np.mean(diff),
                    'hdi_difference': pm.hdi(diff, hdi_prob=0.95).tolist()
                }
        
        return {
            'method_summaries': method_summaries,
            'pairwise_comparisons': pairwise_comparisons,
            'model_trace': trace,
            'analysis_type': 'bayesian'
        }
    
    def _frequentist_method_comparison(
        self, 
        method_errors: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Frequentist method comparison as fallback."""
        
        methods = list(method_errors.keys())
        
        # ANOVA
        error_groups = [method_errors[method] for method in methods]
        f_stat, anova_p = stats.f_oneway(*error_groups)
        
        # Pairwise t-tests with Bonferroni correction
        n_comparisons = len(methods) * (len(methods) - 1) // 2
        alpha_corrected = self.significance_level / n_comparisons
        
        pairwise_results = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                t_stat, p_val = stats.ttest_ind(
                    method_errors[method1], 
                    method_errors[method2]
                )
                
                pairwise_results[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < alpha_corrected,
                    'bonferroni_corrected_alpha': alpha_corrected
                }
        
        return {
            'anova_f_statistic': f_stat,
            'anova_p_value': anova_p,
            'pairwise_comparisons': pairwise_results,
            'analysis_type': 'frequentist'
        }
    
    def outlier_detection(
        self, 
        errors: np.ndarray, 
        method: str = "iqr"
    ) -> Dict[str, Any]:
        """Detect outliers in error distribution."""
        
        if method == "iqr":
            Q1 = np.percentile(errors, 25)
            Q3 = np.percentile(errors, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (errors < lower_bound) | (errors > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(errors))
            outliers = z_scores > 3
            
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(errors.reshape(-1, 1))
            outliers = outlier_labels == -1
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return {
            'outlier_indices': np.where(outliers)[0].tolist(),
            'n_outliers': np.sum(outliers),
            'outlier_fraction': np.mean(outliers),
            'outlier_values': errors[outliers].tolist(),
            'method': method
        }
    
    def convergence_analysis(
        self,
        parameter_values: np.ndarray,
        energies: np.ndarray,
        parameter_name: str = "basis_set_size"
    ) -> Dict[str, Any]:
        """Analyze convergence with respect to a parameter."""
        
        # Sort by parameter values
        sort_indices = np.argsort(parameter_values)
        sorted_params = parameter_values[sort_indices]
        sorted_energies = energies[sort_indices]
        
        # Fit exponential convergence: E(x) = E_inf + A * exp(-alpha * x)
        def exponential_model(params, x):
            E_inf, A, alpha = params
            return E_inf + A * np.exp(-alpha * x)
        
        def objective(params):
            predicted = exponential_model(params, sorted_params)
            return np.sum((sorted_energies - predicted)**2)
        
        # Initial guess
        E_inf_guess = sorted_energies[-1]
        A_guess = sorted_energies[0] - E_inf_guess
        alpha_guess = 1.0
        
        try:
            result = minimize(
                objective, 
                [E_inf_guess, A_guess, alpha_guess],
                method='L-BFGS-B'
            )
            
            E_inf, A, alpha = result.x
            converged_value = E_inf
            convergence_rate = alpha
            
            # Calculate R²
            predicted = exponential_model(result.x, sorted_params)
            ss_res = np.sum((sorted_energies - predicted)**2)
            ss_tot = np.sum((sorted_energies - np.mean(sorted_energies))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            fit_success = True
            
        except Exception as e:
            logger.warning(f"Convergence fitting failed: {e}")
            fit_success = False
            converged_value = None
            convergence_rate = None
            r_squared = 0
        
        # Simple convergence check
        energy_diffs = np.abs(np.diff(sorted_energies))
        is_converged = energy_diffs[-1] < 1e-6 if len(energy_diffs) > 0 else False
        
        return {
            'parameter_name': parameter_name,
            'converged_value': converged_value,
            'convergence_rate': convergence_rate,
            'fit_r_squared': r_squared,
            'fit_success': fit_success,
            'is_converged': is_converged,
            'final_energy_difference': energy_diffs[-1] if len(energy_diffs) > 0 else None,
            'parameter_values': sorted_params.tolist(),
            'energy_values': sorted_energies.tolist()
        }
    
    def generate_statistical_report(
        self,
        error_metrics: Dict[str, ErrorMetrics],
        uncertainty_analysis: Optional[UncertaintyAnalysis] = None,
        method_comparison: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive statistical report."""
        
        report_sections = []
        
        # Header
        report_sections.append("# Statistical Analysis Report")
        report_sections.append("")
        
        # Summary table
        report_sections.append("## Error Metrics Summary")
        report_sections.append("")
        
        # Create table
        headers = ["Method", "MAE", "RMSE", "Max Error", "R²", "N Samples"]
        rows = []
        
        for method, metrics in error_metrics.items():
            row = [
                method,
                f"{metrics.mae:.6f}",
                f"{metrics.rmse:.6f}", 
                f"{metrics.max_error:.6f}",
                f"{metrics.r2:.4f}",
                str(metrics.n_samples)
            ]
            rows.append(row)
        
        # Format table
        col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
        
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = " | ".join("-" * w for w in col_widths)
        
        report_sections.append(header_row)
        report_sections.append(separator)
        
        for row in rows:
            row_str = " | ".join(str(item).ljust(w) for item, w in zip(row, col_widths))
            report_sections.append(row_str)
        
        report_sections.append("")
        
        # Detailed analysis for each method
        for method, metrics in error_metrics.items():
            report_sections.append(f"## Detailed Analysis: {method}")
            report_sections.append("")
            
            report_sections.append(f"- **Mean Absolute Error**: {metrics.mae:.6f} ± {(metrics.mae_ci[1] - metrics.mae_ci[0])/2:.6f}")
            report_sections.append(f"- **Root Mean Square Error**: {metrics.rmse:.6f} ± {(metrics.rmse_ci[1] - metrics.rmse_ci[0])/2:.6f}")
            report_sections.append(f"- **Mean Error (Bias)**: {metrics.mean_error:.6f} ± {(metrics.bias_ci[1] - metrics.bias_ci[0])/2:.6f}")
            report_sections.append(f"- **Standard Deviation**: {metrics.std_error:.6f}")
            report_sections.append(f"- **Error Distribution**: {'Normal' if metrics.is_normal else 'Non-normal'} (p = {metrics.normality_p_value:.4f})")
            
            if metrics.durbin_watson_stat is not None:
                dw_interpretation = "No autocorrelation" if 1.5 <= metrics.durbin_watson_stat <= 2.5 else "Possible autocorrelation"
                report_sections.append(f"- **Durbin-Watson Statistic**: {metrics.durbin_watson_stat:.3f} ({dw_interpretation})")
            
            if metrics.white_test_p_value is not None:
                heteroscedasticity = "Homoscedastic" if metrics.white_test_p_value > 0.05 else "Heteroscedastic"
                report_sections.append(f"- **Heteroscedasticity Test**: {heteroscedasticity} (p = {metrics.white_test_p_value:.4f})")
            
            report_sections.append("")
        
        # Uncertainty analysis
        if uncertainty_analysis:
            report_sections.append("## Uncertainty Analysis")
            report_sections.append("")
            report_sections.append(f"- **Total Uncertainty**: {uncertainty_analysis.total_uncertainty:.6f}")
            report_sections.append(f"- **Systematic Component**: {uncertainty_analysis.systematic_uncertainty:.6f}")
            report_sections.append(f"- **Statistical Component**: {uncertainty_analysis.statistical_uncertainty:.6f}")
            report_sections.append(f"- **Model Component**: {uncertainty_analysis.model_uncertainty:.6f}")
            report_sections.append(f"- **Coverage Probability**: {uncertainty_analysis.coverage_probability:.3f}")
            report_sections.append(f"- **Uncertainty Reliability**: {uncertainty_analysis.uncertainty_reliability:.3f}")
            report_sections.append("")
        
        # Method comparison
        if method_comparison:
            report_sections.append("## Method Comparison")
            report_sections.append("")
            
            if method_comparison['analysis_type'] == 'bayesian':
                report_sections.append("### Bayesian Analysis Results")
                report_sections.append("")
                
                for comparison, results in method_comparison['pairwise_comparisons'].items():
                    method1, method2 = comparison.split('_vs_')
                    prob = results['prob_method1_better']
                    report_sections.append(f"- **{method1} vs {method2}**: {prob:.3f} probability that {method1} is better")
            
            else:
                report_sections.append("### Frequentist Analysis Results")
                report_sections.append("")
                report_sections.append(f"- **ANOVA F-statistic**: {method_comparison['anova_f_statistic']:.4f}")
                report_sections.append(f"- **ANOVA p-value**: {method_comparison['anova_p_value']:.6f}")
                
                significant_comparisons = [
                    comp for comp, results in method_comparison['pairwise_comparisons'].items()
                    if results['significant']
                ]
                
                if significant_comparisons:
                    report_sections.append(f"- **Significant differences found**: {len(significant_comparisons)}")
                else:
                    report_sections.append("- **No significant differences found**")
        
        return "\n".join(report_sections)
    
    def export_statistical_data(
        self,
        output_dir: Path,
        error_metrics: Dict[str, ErrorMetrics],
        method_comparison: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export statistical data in various formats."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export error metrics as CSV
        metrics_data = []
        for method, metrics in error_metrics.items():
            row = {
                'method': method,
                **metrics.summary_dict
            }
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(output_dir / "error_metrics.csv", index=False)
        
        # Export method comparison results
        if method_comparison:
            comparison_file = output_dir / "method_comparison.json"
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            exportable_comparison = {}
            for key, value in method_comparison.items():
                if key == 'model_trace':
                    continue  # Skip trace data
                exportable_comparison[key] = value
            
            with open(comparison_file, 'w') as f:
                json.dump(exportable_comparison, f, indent=2, default=str)
        
        logger.info(f"Statistical data exported to {output_dir}")