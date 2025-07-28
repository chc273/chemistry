"""
Statistical analysis and convergence tracking for quantum chemistry validation.

This module provides tools for analyzing validation results statistically
and tracking convergence patterns across different methods and systems.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .comparison import MethodComparison, ValidationResult


@dataclass
class ConvergencePoint:
    """A single point in a convergence analysis."""

    parameter_value: (
        float  # Parameter being varied (basis set size, bond dimension, etc.)
    )
    energy: float
    error: Optional[float] = None  # Error vs reference if available
    uncertainty: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceAnalysis:
    """Analysis of convergence behavior for a method/system."""

    system_name: str
    method: str
    parameter_name: str  # What parameter is being varied

    points: List[ConvergencePoint]

    # Fitted convergence parameters
    converged_value: Optional[float] = None
    convergence_rate: Optional[float] = None
    fit_quality: Optional[float] = None  # R²

    # Convergence assessment
    is_converged: bool = False
    convergence_threshold: float = 1e-6  # Hartree

    def __post_init__(self):
        self._analyze_convergence()

    def _analyze_convergence(self):
        """Analyze convergence behavior and extract parameters."""
        if len(self.points) < 3:
            return

        # Sort points by parameter value
        self.points.sort(key=lambda p: p.parameter_value)

        energies = np.array([p.energy for p in self.points])
        params = np.array([p.parameter_value for p in self.points])

        # Check for convergence by looking at energy differences
        if len(energies) >= 2:
            energy_diffs = np.abs(np.diff(energies))
            if len(energy_diffs) >= 2:
                recent_diff = energy_diffs[-1]
                self.is_converged = recent_diff < self.convergence_threshold

        # Try exponential convergence fit: E(x) = E_∞ + A * exp(-α * x)
        if len(self.points) >= 4:
            try:
                # Initial guess for parameters
                E_inf_guess = energies[-1]  # Last point as convergence estimate
                A_guess = energies[0] - E_inf_guess
                alpha_guess = 1.0

                def exp_convergence(x, E_inf, A, alpha):
                    return E_inf + A * np.exp(-alpha * x)

                from scipy.optimize import curve_fit

                popt, pcov = curve_fit(
                    exp_convergence,
                    params,
                    energies,
                    p0=[E_inf_guess, A_guess, alpha_guess],
                    maxfev=5000,
                )

                self.converged_value = popt[0]
                self.convergence_rate = popt[2]

                # Calculate R²
                y_pred = exp_convergence(params, *popt)
                ss_res = np.sum((energies - y_pred) ** 2)
                ss_tot = np.sum((energies - np.mean(energies)) ** 2)
                self.fit_quality = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            except Exception:
                # Fallback to linear extrapolation using last few points
                if len(energies) >= 3:
                    recent_energies = energies[-3:]
                    recent_params = params[-3:]

                    # Linear fit to 1/x vs E for basis set convergence
                    if self.parameter_name.lower() in ["basis_size", "cardinal_number"]:
                        try:
                            inv_params = 1.0 / recent_params
                            fit = np.polyfit(inv_params, recent_energies, 1)
                            self.converged_value = fit[1]  # Intercept at 1/x = 0
                        except (np.linalg.LinAlgError, ValueError):
                            self.converged_value = recent_energies[-1]
                    else:
                        self.converged_value = recent_energies[-1]


class StatisticalAnalyzer:
    """Statistical analysis of quantum chemistry validation results."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.method_comparisons: List[MethodComparison] = []
        self.convergence_analyses: List[ConvergenceAnalysis] = []

    def add_validation_results(self, results: List[ValidationResult]):
        """Add validation results for analysis."""
        self.validation_results.extend(results)

    def add_method_comparisons(self, comparisons: List[MethodComparison]):
        """Add method comparisons for analysis."""
        self.method_comparisons.extend(comparisons)

    def add_convergence_analysis(self, analysis: ConvergenceAnalysis):
        """Add convergence analysis."""
        self.convergence_analyses.append(analysis)

    def analyze_method_accuracy(self, method: str) -> Dict[str, Any]:
        """Analyze accuracy statistics for a specific method."""
        method_results = [
            r for r in self.validation_results if r.method.lower() == method.lower()
        ]

        if not method_results:
            return {}

        errors = np.array([r.energy_error for r in method_results])
        abs_errors = np.array([r.absolute_error for r in method_results])
        relative_errors = np.array(
            [
                abs(r.relative_error)
                for r in method_results
                if abs(r.relative_error) != float("inf")
            ]
        )

        # Basic statistics
        stats_dict = {
            "method": method,
            "n_systems": len(method_results),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mean_absolute_error": float(np.mean(abs_errors)),
            "max_absolute_error": float(np.max(abs_errors)),
            "min_absolute_error": float(np.min(abs_errors)),
            "rms_error": float(np.sqrt(np.mean(errors**2))),
            "median_absolute_error": float(np.median(abs_errors)),
            "q75_absolute_error": float(np.percentile(abs_errors, 75)),
            "q95_absolute_error": float(np.percentile(abs_errors, 95)),
        }

        # Relative error statistics
        if len(relative_errors) > 0:
            stats_dict.update(
                {
                    "mean_relative_error": float(np.mean(relative_errors)),
                    "max_relative_error": float(np.max(relative_errors)),
                    "median_relative_error": float(np.median(relative_errors)),
                }
            )

        # Error distribution
        error_mhartree = abs_errors * 1000
        stats_dict["error_distribution"] = {
            "excellent": int(np.sum(error_mhartree < 0.1)),
            "good": int(np.sum((error_mhartree >= 0.1) & (error_mhartree < 1.0))),
            "acceptable": int(np.sum((error_mhartree >= 1.0) & (error_mhartree < 5.0))),
            "poor": int(np.sum(error_mhartree >= 5.0)),
        }

        # Statistical tests
        # Test for systematic bias (one-sample t-test against zero)
        if len(errors) > 1:
            t_stat, p_value = stats.ttest_1samp(errors, 0)
            stats_dict["bias_test"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_bias": p_value < 0.05,
            }

        # Normality test for errors
        if len(errors) >= 8:  # Minimum for Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(errors)
            stats_dict["normality_test"] = {
                "shapiro_statistic": float(shapiro_stat),
                "shapiro_p_value": float(shapiro_p),
                "errors_normal": shapiro_p > 0.05,
            }

        return stats_dict

    def compare_method_performance(self, methods: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple methods."""
        method_stats = {}
        for method in methods:
            method_stats[method] = self.analyze_method_accuracy(method)

        # Cross-method comparisons
        comparisons = {}

        # Statistical significance tests between methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                results1 = [
                    r
                    for r in self.validation_results
                    if r.method.lower() == method1.lower()
                ]
                results2 = [
                    r
                    for r in self.validation_results
                    if r.method.lower() == method2.lower()
                ]

                # Find common systems
                systems1 = {r.system_name for r in results1}
                systems2 = {r.system_name for r in results2}
                common_systems = systems1.intersection(systems2)

                if len(common_systems) >= 3:  # Minimum for paired test
                    # Get paired results
                    paired_errors1 = []
                    paired_errors2 = []

                    for system in common_systems:
                        r1 = next(r for r in results1 if r.system_name == system)
                        r2 = next(r for r in results2 if r.system_name == system)
                        paired_errors1.append(r1.absolute_error)
                        paired_errors2.append(r2.absolute_error)

                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(paired_errors1, paired_errors2)

                    comparisons[f"{method1}_vs_{method2}"] = {
                        "common_systems": len(common_systems),
                        "mean_error_diff": float(
                            np.mean(np.array(paired_errors1) - np.array(paired_errors2))
                        ),
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significantly_different": p_value < 0.05,
                    }

        return {
            "method_statistics": method_stats,
            "pairwise_comparisons": comparisons,
            "ranking": self._rank_methods_by_accuracy(method_stats),
        }

    def _rank_methods_by_accuracy(
        self, method_stats: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Rank methods by accuracy metrics."""
        ranking = []

        for method, method_stats_data in method_stats.items():
            if not method_stats_data:  # Skip empty stats
                continue

            # Composite score based on multiple metrics
            mae = method_stats_data.get("mean_absolute_error", float("inf"))
            rms = method_stats_data.get("rms_error", float("inf"))
            max_err = method_stats_data.get("max_absolute_error", float("inf"))

            # Weighted score (lower is better)
            composite_score = 0.4 * mae + 0.3 * rms + 0.3 * max_err

            ranking.append(
                {
                    "method": method,
                    "composite_score": float(composite_score),
                    "mean_absolute_error": float(mae),
                    "rms_error": float(rms),
                    "max_absolute_error": float(max_err),
                }
            )

        # Sort by composite score (ascending - lower is better)
        ranking.sort(key=lambda x: x["composite_score"])

        # Add rank numbers
        for i, entry in enumerate(ranking):
            entry["rank"] = i + 1

        return ranking

    def analyze_system_difficulty(self) -> Dict[str, Any]:
        """Analyze which systems are most challenging for different methods."""
        system_stats = defaultdict(
            lambda: {
                "methods": {},
                "mean_error": 0,
                "std_error": 0,
                "difficulty_score": 0,
            }
        )

        # Group results by system
        for result in self.validation_results:
            system = result.system_name
            method = result.method

            if "methods" not in system_stats[system]:
                system_stats[system]["methods"] = {}

            system_stats[system]["methods"][method] = {
                "error": result.energy_error,
                "absolute_error": result.absolute_error,
                "error_magnitude": result.error_magnitude,
            }

        # Calculate difficulty metrics for each system
        for system, system_stats_data in system_stats.items():
            method_errors = [
                data["absolute_error"] for data in system_stats_data["methods"].values()
            ]

            if method_errors:
                system_stats_data["mean_error"] = float(np.mean(method_errors))
                system_stats_data["std_error"] = float(np.std(method_errors))
                system_stats_data["max_error"] = float(np.max(method_errors))
                system_stats_data["n_methods"] = len(method_errors)

                # Difficulty score: higher mean error and higher variability = more difficult
                system_stats_data["difficulty_score"] = float(
                    system_stats_data["mean_error"] * (1 + system_stats_data["std_error"])
                )

        # Rank systems by difficulty
        system_ranking = []
        for system, system_stats_data in system_stats.items():
            system_ranking.append(
                {
                    "system": system,
                    "difficulty_score": system_stats_data["difficulty_score"],
                    "mean_error": system_stats_data["mean_error"],
                    "std_error": system_stats_data["std_error"],
                    "max_error": system_stats_data.get("max_error", 0),
                    "n_methods": system_stats_data.get("n_methods", 0),
                }
            )

        system_ranking.sort(key=lambda x: x["difficulty_score"], reverse=True)

        for i, entry in enumerate(system_ranking):
            entry["difficulty_rank"] = i + 1

        return {
            "system_statistics": dict(system_stats),
            "difficulty_ranking": system_ranking,
        }

    def plot_error_distributions(
        self, methods: List[str], save_path: Optional[Path] = None
    ):
        """Plot error distributions for different methods."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, method in enumerate(methods[:4]):  # Limit to 4 methods
            method_results = [
                r for r in self.validation_results if r.method.lower() == method.lower()
            ]

            if not method_results:
                continue

            errors_mhartree = np.array([r.energy_error * 1000 for r in method_results])

            axes[i].hist(errors_mhartree, bins=20, alpha=0.7, edgecolor="black")
            axes[i].axvline(0, color="red", linestyle="--", alpha=0.8)
            axes[i].set_xlabel("Energy Error (mHartree)")
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{method} Error Distribution")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_method_comparison(
        self, methods: List[str], save_path: Optional[Path] = None
    ):
        """Plot method comparison showing accuracy vs computational cost."""
        method_stats = {}
        for method in methods:
            stats = self.analyze_method_accuracy(method)
            if stats:
                method_stats[method] = stats

        if not method_stats:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        x_data = []  # Mean absolute error
        y_data = []  # Max absolute error
        labels = []

        for method, stats in method_stats.items():
            mae = stats.get("mean_absolute_error", 0) * 1000  # Convert to mHartree
            max_err = stats.get("max_absolute_error", 0) * 1000

            x_data.append(mae)
            y_data.append(max_err)
            labels.append(method)

        ax.scatter(x_data, y_data, s=100, alpha=0.7)

        # Add method labels
        for i, label in enumerate(labels):
            ax.annotate(
                label, (x_data[i], y_data[i]), xytext=(5, 5), textcoords="offset points"
            )

        ax.set_xlabel("Mean Absolute Error (mHartree)")
        ax.set_ylabel("Maximum Absolute Error (mHartree)")
        ax.set_title("Method Comparison: Accuracy Profile")
        ax.grid(True, alpha=0.3)

        # Add diagonal reference lines
        max_val = max(max(x_data), max(y_data))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="MAE = MAX")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def export_statistical_report(self, output_path: Union[str, Path]):
        """Export comprehensive statistical analysis report."""
        output_path = Path(output_path)

        # Get all unique methods
        methods = list(set(r.method for r in self.validation_results))

        report = {
            "timestamp": str(np.datetime64("now")),
            "summary": {
                "total_validation_results": len(self.validation_results),
                "total_method_comparisons": len(self.method_comparisons),
                "unique_methods": len(methods),
                "unique_systems": len(
                    set(r.system_name for r in self.validation_results)
                ),
            },
            "method_performance": self.compare_method_performance(methods),
            "system_difficulty": self.analyze_system_difficulty(),
            "convergence_analyses": [
                {
                    "system": ca.system_name,
                    "method": ca.method,
                    "parameter": ca.parameter_name,
                    "converged_value": ca.converged_value,
                    "convergence_rate": ca.convergence_rate,
                    "fit_quality": ca.fit_quality,
                    "is_converged": ca.is_converged,
                }
                for ca in self.convergence_analyses
            ],
        }

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


class ConvergenceTracker:
    """Track convergence behavior of quantum chemistry methods."""

    def __init__(self):
        self.convergence_data: Dict[str, List[ConvergencePoint]] = defaultdict(list)

    def add_convergence_point(
        self,
        system: str,
        method: str,
        parameter_name: str,
        parameter_value: float,
        energy: float,
        reference_energy: Optional[float] = None,
        **additional_data,
    ):
        """Add a convergence data point."""
        key = f"{system}_{method}_{parameter_name}"

        error = None
        if reference_energy is not None:
            error = energy - reference_energy

        point = ConvergencePoint(
            parameter_value=parameter_value,
            energy=energy,
            error=error,
            additional_data=additional_data,
        )

        self.convergence_data[key].append(point)

    def analyze_convergence(
        self, system: str, method: str, parameter_name: str
    ) -> Optional[ConvergenceAnalysis]:
        """Analyze convergence for a specific system/method/parameter combination."""
        key = f"{system}_{method}_{parameter_name}"

        if key not in self.convergence_data or len(self.convergence_data[key]) < 2:
            return None

        analysis = ConvergenceAnalysis(
            system_name=system,
            method=method,
            parameter_name=parameter_name,
            points=self.convergence_data[key].copy(),
        )

        return analysis

    def plot_convergence(
        self,
        system: str,
        method: str,
        parameter_name: str,
        save_path: Optional[Path] = None,
    ):
        """Plot convergence behavior."""
        analysis = self.analyze_convergence(system, method, parameter_name)

        if analysis is None:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Sort points by parameter value
        points = sorted(analysis.points, key=lambda p: p.parameter_value)
        params = [p.parameter_value for p in points]
        energies = [p.energy for p in points]
        errors = [p.error for p in points if p.error is not None]

        # Energy convergence plot
        ax1.plot(params, energies, "bo-", markersize=6, linewidth=2)
        ax1.set_xlabel(parameter_name)
        ax1.set_ylabel("Energy (Hartree)")
        ax1.set_title(f"{system} - {method}: Energy Convergence")
        ax1.grid(True, alpha=0.3)

        # Add converged value if available
        if analysis.converged_value is not None:
            ax1.axhline(
                analysis.converged_value,
                color="red",
                linestyle="--",
                alpha=0.8,
                label=f"Converged: {analysis.converged_value:.6f}",
            )
            ax1.legend()

        # Error plot (if reference available)
        if errors and len(errors) == len(params):
            ax2.semilogy(params, np.abs(errors), "ro-", markersize=6, linewidth=2)
            ax2.axhline(
                analysis.convergence_threshold,
                color="green",
                linestyle="--",
                alpha=0.8,
                label=f"Threshold: {analysis.convergence_threshold}",
            )
            ax2.set_xlabel(parameter_name)
            ax2.set_ylabel("|Error| (Hartree)")
            ax2.set_title("Error vs Reference")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # Energy differences plot
            if len(energies) >= 2:
                energy_diffs = np.abs(np.diff(energies))
                ax2.semilogy(params[1:], energy_diffs, "go-", markersize=6, linewidth=2)
                ax2.axhline(
                    analysis.convergence_threshold,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label=f"Threshold: {analysis.convergence_threshold}",
                )
                ax2.set_xlabel(parameter_name)
                ax2.set_ylabel("|ΔE| (Hartree)")
                ax2.set_title("Energy Differences")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
