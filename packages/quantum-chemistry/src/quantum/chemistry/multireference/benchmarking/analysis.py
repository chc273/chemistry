"""
Statistical analysis and validation tools for benchmarking results.

This module provides comprehensive analysis tools for evaluating
multireference method accuracy and performance.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .datasets import BenchmarkDataset, BenchmarkEntry, SystemType


class BenchmarkAnalyzer:
    """
    Comprehensive analysis toolkit for benchmark validation.
    
    This class provides statistical analysis, comparison tools,
    and validation metrics for multireference method benchmarking.
    """
    
    def __init__(self, dataset: BenchmarkDataset):
        """
        Initialize analyzer with benchmark dataset.
        
        Args:
            dataset: BenchmarkDataset to analyze
        """
        self.dataset = dataset
        self.entries_with_reference = [
            e for e in dataset.entries 
            if e.reference_energy is not None and e.absolute_error is not None
        ]
    
    def calculate_error_statistics(self, 
                                 method: Optional[str] = None,
                                 system_type: Optional[SystemType] = None) -> Dict[str, float]:
        """
        Calculate comprehensive error statistics.
        
        Args:
            method: Filter by specific method
            system_type: Filter by system type
            
        Returns:
            Dict with error statistics
        """
        entries = self.entries_with_reference
        
        # Apply filters
        if method:
            entries = [e for e in entries if e.method.lower() == method.lower()]
        if system_type:
            entries = [e for e in entries if e.system.system_type == system_type]
        
        if not entries:
            return {"error": "No entries match criteria"}
        
        errors = [e.absolute_error for e in entries if e.absolute_error is not None]
        if not errors:
            return {"error": "No valid errors found"}
        
        errors = np.array(errors)
        
        # Convert to eV if energies are in Hartree
        # Assume Hartree if errors are < 1.0, otherwise assume eV
        if np.mean(errors) < 1.0:
            errors_ev = errors * 27.2114  # Hartree to eV conversion
            unit = "eV"
        else:
            errors_ev = errors
            unit = "eV"
        
        return {
            "n_points": len(errors),
            "mean_absolute_error": float(np.mean(errors_ev)),
            "rmse": float(np.sqrt(np.mean(errors_ev**2))),
            "max_error": float(np.max(errors_ev)),
            "min_error": float(np.min(errors_ev)),
            "median_error": float(np.median(errors_ev)),
            "std_error": float(np.std(errors_ev)),
            "q75_error": float(np.percentile(errors_ev, 75)),
            "q25_error": float(np.percentile(errors_ev, 25)),
            "unit": unit
        }
    
    def compare_methods(self, 
                       methods: List[str],
                       system_type: Optional[SystemType] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare error statistics across different methods.
        
        Args:
            methods: List of method names to compare
            system_type: Optional system type filter
            
        Returns:
            Dict mapping method names to error statistics
        """
        comparison = {}
        
        for method in methods:
            stats = self.calculate_error_statistics(method=method, system_type=system_type)
            comparison[method] = stats
        
        return comparison
    
    def method_ranking(self, 
                      methods: List[str],
                      metric: str = "mean_absolute_error",
                      system_type: Optional[SystemType] = None) -> List[Tuple[str, float]]:
        """
        Rank methods by specified error metric.
        
        Args:
            methods: List of method names
            metric: Error metric for ranking
            system_type: Optional system type filter
            
        Returns:
            List of (method, metric_value) tuples sorted by performance
        """
        comparison = self.compare_methods(methods, system_type)
        
        ranking = []
        for method, stats in comparison.items():
            if "error" not in stats and metric in stats:
                ranking.append((method, stats[metric]))
        
        # Sort by metric value (lower is better)
        ranking.sort(key=lambda x: x[1])
        
        return ranking
    
    def systematic_error_analysis(self, 
                                 method: str,
                                 system_type: Optional[SystemType] = None) -> Dict[str, Any]:
        """
        Analyze systematic errors and biases for a specific method.
        
        Args:
            method: Method name to analyze
            system_type: Optional system type filter
            
        Returns:
            Dict with systematic error analysis
        """
        entries = self.entries_with_reference
        
        # Apply filters
        entries = [e for e in entries if e.method.lower() == method.lower()]
        if system_type:
            entries = [e for e in entries if e.system.system_type == system_type]
        
        if not entries:
            return {"error": f"No entries found for method {method}"}
        
        # Get signed errors (calculated - reference)
        signed_errors = []
        absolute_errors = []
        
        for entry in entries:
            if entry.reference_energy is not None:
                signed_error = entry.energy - entry.reference_energy
                signed_errors.append(signed_error)
                absolute_errors.append(abs(signed_error))
        
        if not signed_errors:
            return {"error": "No valid errors found"}
        
        signed_errors = np.array(signed_errors)
        absolute_errors = np.array(absolute_errors)
        
        # Convert to eV if needed
        if np.mean(absolute_errors) < 1.0:
            signed_errors_ev = signed_errors * 27.2114
            unit = "eV"
        else:
            signed_errors_ev = signed_errors
            unit = "eV"
        
        # Statistical tests
        # Test for systematic bias (mean significantly different from zero)
        t_stat, p_value = stats.ttest_1samp(signed_errors_ev, 0.0)
        
        # Test for normality of errors
        shapiro_stat, shapiro_p = stats.shapiro(signed_errors_ev)
        
        analysis = {
            "n_points": len(signed_errors_ev),
            "mean_signed_error": float(np.mean(signed_errors_ev)),
            "std_signed_error": float(np.std(signed_errors_ev)),
            "systematic_bias": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_bias": p_value < 0.05,
                "bias_direction": "overestimate" if np.mean(signed_errors_ev) > 0 else "underestimate"
            },
            "error_distribution": {
                "shapiro_wilk_stat": float(shapiro_stat),
                "shapiro_wilk_p": float(shapiro_p),
                "normally_distributed": shapiro_p > 0.05,
                "skewness": float(stats.skew(signed_errors_ev)),
                "kurtosis": float(stats.kurtosis(signed_errors_ev))
            },
            "outlier_analysis": self._identify_outliers(signed_errors_ev),
            "unit": unit
        }
        
        return analysis
    
    def _identify_outliers(self, errors: np.ndarray) -> Dict[str, Any]:
        """Identify outliers using IQR method."""
        q25, q75 = np.percentile(errors, [25, 75])
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
        
        return {
            "n_outliers": len(outliers),
            "outlier_fraction": len(outliers) / len(errors),
            "outlier_values": outliers.tolist(),
            "iqr_bounds": [float(lower_bound), float(upper_bound)]
        }
    
    def convergence_analysis(self, 
                           active_space_sizes: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Analyze convergence with respect to active space size.
        
        Args:
            active_space_sizes: List of (n_electrons, n_orbitals) to analyze
            
        Returns:
            Dict with convergence analysis
        """
        if not active_space_sizes:
            # Extract all active space sizes from dataset
            sizes = set()
            for entry in self.entries_with_reference:
                sizes.add((entry.n_active_electrons, entry.n_active_orbitals))
            active_space_sizes = sorted(list(sizes))
        
        convergence_data = {}
        
        for n_elec, n_orb in active_space_sizes:
            size_entries = [
                e for e in self.entries_with_reference
                if e.n_active_electrons == n_elec and e.n_active_orbitals == n_orb
            ]
            
            if size_entries:
                errors = [e.absolute_error for e in size_entries if e.absolute_error is not None]
                if errors:
                    convergence_data[f"({n_elec},{n_orb})"] = {
                        "n_points": len(errors),
                        "mean_error": float(np.mean(errors)),
                        "std_error": float(np.std(errors)),
                        "active_space_size": n_elec * n_orb
                    }
        
        return convergence_data
    
    def basis_set_dependence(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze method performance dependence on basis set.
        
        Returns:
            Dict mapping basis sets to error statistics
        """
        basis_sets = set(entry.system.basis_set for entry in self.entries_with_reference)
        
        basis_analysis = {}
        for basis in basis_sets:
            basis_entries = [
                e for e in self.entries_with_reference
                if e.system.basis_set == basis
            ]
            
            if basis_entries:
                errors = [e.absolute_error for e in basis_entries if e.absolute_error is not None]
                if errors:
                    errors = np.array(errors)
                    
                    # Convert to eV if needed
                    if np.mean(errors) < 1.0:
                        errors_ev = errors * 27.2114
                    else:
                        errors_ev = errors
                    
                    basis_analysis[basis] = {
                        "n_points": len(errors_ev),
                        "mean_error": float(np.mean(errors_ev)),
                        "std_error": float(np.std(errors_ev)),
                        "rmse": float(np.sqrt(np.mean(errors_ev**2)))
                    }
        
        return basis_analysis
    
    def generate_validation_report(self, 
                                 methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            methods: Optional list of methods to include
            
        Returns:
            Dict with complete validation analysis
        """
        if methods is None:
            methods = list(set(entry.method for entry in self.entries_with_reference))
        
        report = {
            "dataset_info": {
                "name": self.dataset.name,
                "description": self.dataset.description,
                "total_entries": len(self.dataset.entries),
                "entries_with_reference": len(self.entries_with_reference),
                "methods_analyzed": methods
            },
            "overall_statistics": self.calculate_error_statistics(),
            "method_comparison": self.compare_methods(methods),
            "method_ranking": {
                "by_mae": self.method_ranking(methods, "mean_absolute_error"),
                "by_rmse": self.method_ranking(methods, "rmse"),
                "by_max_error": self.method_ranking(methods, "max_error")
            },
            "systematic_analysis": {},
            "convergence_analysis": self.convergence_analysis(),
            "basis_set_analysis": self.basis_set_dependence()
        }
        
        # Add systematic error analysis for each method
        for method in methods:
            report["systematic_analysis"][method] = self.systematic_error_analysis(method)
        
        return report
    
    def identify_problem_cases(self, 
                             error_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Identify benchmark cases with large errors.
        
        Args:
            error_threshold: Error threshold in eV
            
        Returns:
            List of problematic benchmark cases
        """
        problem_cases = []
        
        for entry in self.entries_with_reference:
            if entry.absolute_error is not None:
                error_ev = entry.absolute_error
                if error_ev < 1.0:  # Assume Hartree, convert to eV
                    error_ev *= 27.2114
                
                if error_ev > error_threshold:
                    problem_cases.append({
                        "system_name": entry.system.name,
                        "method": entry.method,
                        "error_ev": error_ev,
                        "system_type": entry.system.system_type.value,
                        "active_space": f"({entry.n_active_electrons},{entry.n_active_orbitals})",
                        "basis_set": entry.system.basis_set,
                        "convergence_info": entry.convergence_info
                    })
        
        # Sort by error magnitude
        problem_cases.sort(key=lambda x: x["error_ev"], reverse=True)
        
        return problem_cases