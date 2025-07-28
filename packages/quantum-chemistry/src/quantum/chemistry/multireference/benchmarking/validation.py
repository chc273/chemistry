"""
Cross-method validation and literature comparison tools.

This module provides tools for validating multireference implementations
against literature benchmarks and cross-checking between different codes.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult, find_active_space_avas
from ..base import MultireferenceResult
from ..methods import CASSCFMethod, NEVPT2Method
from ..workflows import MultireferenceWorkflow
from .datasets import BenchmarkDataset, BenchmarkEntry, BenchmarkMolecule


class ValidationRunner:
    """
    Automated validation runner for multireference methods.
    
    This class orchestrates validation calculations, compares results
    against literature benchmarks, and identifies potential issues.
    """
    
    def __init__(self, 
                 tolerance_tight: float = 0.001,
                 tolerance_loose: float = 0.01,
                 max_scf_cycles: int = 100):
        """
        Initialize validation runner.
        
        Args:
            tolerance_tight: Tight convergence tolerance (Hartree)
            tolerance_loose: Loose convergence tolerance (Hartree) 
            max_scf_cycles: Maximum SCF iterations
        """
        self.tolerance_tight = tolerance_tight
        self.tolerance_loose = tolerance_loose
        self.max_scf_cycles = max_scf_cycles
        
        self.workflow = MultireferenceWorkflow()
        self.results_cache = {}
    
    def validate_molecule(self,
                         molecule: BenchmarkMolecule,
                         methods: List[str] = None,
                         active_space_methods: List[str] = None) -> Dict[str, Any]:
        """
        Run validation calculations for a single molecule.
        
        Args:
            molecule: BenchmarkMolecule to validate
            methods: List of MR methods to test
            active_space_methods: List of active space selection methods
            
        Returns:
            Dict with validation results
        """
        if methods is None:
            methods = ["casscf", "nevpt2"]
        if active_space_methods is None:
            active_space_methods = ["avas"]
        
        # Create cache key
        cache_key = f"{molecule.get_system_hash()}_{'-'.join(methods)}_{'-'.join(active_space_methods)}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        validation_results = {
            "molecule": molecule.name,
            "system_hash": molecule.get_system_hash(),
            "scf_results": {},
            "active_space_results": {},
            "multireference_results": {},
            "validation_status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: SCF calculation
            mol = molecule.to_pyscf_molecule()
            
            if molecule.multiplicity == 1:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)
            
            mf.max_cycle = self.max_scf_cycles
            mf.conv_tol = 1e-8
            
            start_time = time.time()
            scf_energy = mf.kernel()
            scf_time = time.time() - start_time
            
            validation_results["scf_results"] = {
                "converged": mf.converged,
                "energy": float(scf_energy),
                "iterations": getattr(mf, 'niter', None),
                "time_seconds": scf_time
            }
            
            if not mf.converged:
                validation_results["errors"].append("SCF did not converge")
                validation_results["validation_status"] = "scf_failed"
                return validation_results
            
            # Step 2: Active space selection
            for as_method in active_space_methods:
                try:
                    start_time = time.time()
                    
                    if as_method == "avas":
                        active_space = find_active_space_avas(mf, threshold=0.2)
                    else:
                        # Use workflow's active space finder for other methods
                        active_space = self.workflow.active_space_finder.find_active_space(
                            as_method, mf
                        )
                    
                    as_time = time.time() - start_time
                    
                    validation_results["active_space_results"][as_method] = {
                        "n_active_electrons": active_space.n_active_electrons,
                        "n_active_orbitals": active_space.n_active_orbitals,  
                        "selection_method": active_space.method.value,
                        "time_seconds": as_time
                    }
                    
                    # Step 3: Multireference calculations
                    for mr_method in methods:
                        method_key = f"{as_method}_{mr_method}"
                        
                        try:
                            start_time = time.time()
                            
                            # Run MR calculation
                            if mr_method.lower() == "casscf":
                                method_instance = CASSCFMethod()
                            elif mr_method.lower() == "nevpt2":
                                method_instance = NEVPT2Method()
                            else:
                                validation_results["warnings"].append(
                                    f"Unknown method: {mr_method}"
                                )
                                continue
                            
                            mr_result = method_instance.calculate(mf, active_space)
                            mr_time = time.time() - start_time
                            
                            # Store results
                            validation_results["multireference_results"][method_key] = {
                                "method": mr_method,
                                "active_space_method": as_method,
                                "energy": mr_result.energy,
                                "correlation_energy": mr_result.correlation_energy,
                                "converged": mr_result.convergence_info.get("converged", False),
                                "time_seconds": mr_time,
                                "n_active_electrons": mr_result.n_active_electrons,
                                "n_active_orbitals": mr_result.n_active_orbitals
                            }
                            
                        except Exception as e:
                            validation_results["errors"].append(
                                f"MR calculation failed ({method_key}): {str(e)}"
                            )
                
                except Exception as e:
                    validation_results["errors"].append(
                        f"Active space selection failed ({as_method}): {str(e)}"
                    )
            
            # Determine overall validation status
            if validation_results["multireference_results"]:
                validation_results["validation_status"] = "success"
            elif validation_results["active_space_results"]:
                validation_results["validation_status"] = "partial_success"
            else:
                validation_results["validation_status"] = "failed"
        
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            validation_results["validation_status"] = "failed"
        
        # Cache results
        self.results_cache[cache_key] = validation_results
        
        return validation_results
    
    def validate_dataset(self,
                        dataset: BenchmarkDataset,
                        methods: List[str] = None,
                        max_systems: Optional[int] = None) -> Dict[str, Any]:
        """
        Run validation on entire benchmark dataset.
        
        Args:
            dataset: BenchmarkDataset to validate
            methods: List of MR methods to test
            max_systems: Maximum number of systems to test (for large datasets)
            
        Returns:
            Dict with complete validation results
        """
        if methods is None:
            methods = ["casscf", "nevpt2"]
        
        validation_summary = {
            "dataset_name": dataset.name,
            "methods_tested": methods,
            "total_systems": len(dataset.entries),
            "systems_tested": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "partial_successes": 0,
            "results": {},
            "summary_statistics": {},
            "common_errors": {}
        }
        
        # Get unique molecules from dataset
        unique_molecules = {}
        for entry in dataset.entries:
            mol_hash = entry.system.get_system_hash()
            if mol_hash not in unique_molecules:
                unique_molecules[mol_hash] = entry.system
        
        molecules_to_test = list(unique_molecules.values())
        if max_systems:
            molecules_to_test = molecules_to_test[:max_systems]
        
        validation_summary["systems_tested"] = len(molecules_to_test)
        
        # Run validation for each molecule
        for i, molecule in enumerate(molecules_to_test):
            print(f"Validating {molecule.name} ({i+1}/{len(molecules_to_test)})")
            
            mol_results = self.validate_molecule(molecule, methods)
            validation_summary["results"][molecule.name] = mol_results
            
            # Update counters
            status = mol_results["validation_status"]
            if status == "success":
                validation_summary["successful_calculations"] += 1
            elif status == "partial_success":
                validation_summary["partial_successes"] += 1
            else:
                validation_summary["failed_calculations"] += 1
        
        # Analyze common errors and generate statistics
        validation_summary["common_errors"] = self._analyze_common_errors(validation_summary)
        validation_summary["summary_statistics"] = self._calculate_validation_statistics(validation_summary)
        
        return validation_summary
    
    def _analyze_common_errors(self, validation_summary: Dict[str, Any]) -> Dict[str, int]:
        """Analyze and count common error types."""
        error_counts = {}
        
        for mol_name, results in validation_summary["results"].items():
            for error in results.get("errors", []):
                error_type = self._classify_error(error)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error message into categories."""
        error_lower = error_message.lower()
        
        if "scf" in error_lower and "converge" in error_lower:
            return "SCF convergence failure"
        elif "active space" in error_lower:
            return "Active space selection failure"
        elif "casscf" in error_lower or "nevpt2" in error_lower:
            return "Multireference calculation failure"
        elif "memory" in error_lower:
            return "Memory error"
        elif "basis" in error_lower:
            return "Basis set error"
        else:
            return "Other error"
    
    def _calculate_validation_statistics(self, validation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for validation run."""
        total_tested = validation_summary["systems_tested"]
        if total_tested == 0:
            return {"error": "No systems tested"}
        
        stats = {
            "success_rate": validation_summary["successful_calculations"] / total_tested,
            "failure_rate": validation_summary["failed_calculations"] / total_tested,
            "partial_success_rate": validation_summary["partial_successes"] / total_tested
        }
        
        # Calculate timing statistics
        all_times = []
        scf_times = []
        mr_times = []
        
        for mol_results in validation_summary["results"].values():
            if "scf_results" in mol_results:
                scf_time = mol_results["scf_results"].get("time_seconds", 0)
                scf_times.append(scf_time)
            
            for mr_key, mr_data in mol_results.get("multireference_results", {}).items():
                mr_time = mr_data.get("time_seconds", 0)
                mr_times.append(mr_time)
                all_times.append(mr_time)
        
        if all_times:
            stats["timing"] = {
                "mean_total_time": float(np.mean(all_times)),
                "mean_scf_time": float(np.mean(scf_times)) if scf_times else 0.0,
                "mean_mr_time": float(np.mean(mr_times)) if mr_times else 0.0,
                "max_time": float(np.max(all_times)),
                "min_time": float(np.min(all_times))
            }
        
        return stats
    
    def compare_with_literature(self,
                              validation_results: Dict[str, Any],
                              literature_dataset: BenchmarkDataset,
                              tolerance: float = 0.05) -> Dict[str, Any]:
        """
        Compare validation results with literature benchmarks.
        
        Args:
            validation_results: Results from validate_dataset
            literature_dataset: Dataset with literature reference values
            tolerance: Tolerance for agreement (in eV)
            
        Returns:
            Dict with comparison analysis
        """
        comparison = {
            "total_comparisons": 0,
            "agreements": 0,
            "disagreements": 0,
            "tolerance_ev": tolerance,
            "detailed_comparisons": {},
            "statistical_summary": {}
        }
        
        # Create lookup for literature values
        lit_lookup = {}
        for entry in literature_dataset.entries:
            if entry.reference_energy is not None:
                key = f"{entry.system.name}_{entry.method}"
                lit_lookup[key] = entry.reference_energy
        
        # Compare calculated vs literature values
        agreements = []
        disagreements = []
        
        for mol_name, mol_results in validation_results["results"].items():
            for method_key, method_data in mol_results.get("multireference_results", {}).items():
                lit_key = f"{mol_name}_{method_data['method']}"
                
                if lit_key in lit_lookup:
                    calculated = method_data["energy"]
                    literature = lit_lookup[lit_key]
                    
                    # Convert to eV for comparison
                    if abs(calculated) > 10:  # Assume already in eV
                        diff_ev = abs(calculated - literature)
                    else:  # Assume Hartree, convert to eV
                        diff_ev = abs(calculated - literature) * 27.2114
                    
                    comparison["total_comparisons"] += 1
                    
                    if diff_ev <= tolerance:
                        agreements.append(diff_ev)
                        comparison["agreements"] += 1
                    else:
                        disagreements.append(diff_ev)
                        comparison["disagreements"] += 1
                    
                    comparison["detailed_comparisons"][f"{mol_name}_{method_key}"] = {
                        "calculated": calculated,
                        "literature": literature,
                        "difference_ev": diff_ev,
                        "agreement": diff_ev <= tolerance
                    }
        
        # Statistical summary
        if comparison["total_comparisons"] > 0:
            all_diffs = agreements + disagreements
            comparison["statistical_summary"] = {
                "agreement_rate": comparison["agreements"] / comparison["total_comparisons"],
                "mean_difference_ev": float(np.mean(all_diffs)),
                "std_difference_ev": float(np.std(all_diffs)),
                "max_difference_ev": float(np.max(all_diffs)),
                "median_difference_ev": float(np.median(all_diffs))
            }
        
        return comparison
    
    def generate_validation_report(self,
                                 dataset: BenchmarkDataset,
                                 methods: List[str] = None,
                                 max_systems: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            dataset: BenchmarkDataset to validate
            methods: List of methods to test
            max_systems: Maximum systems to test
            
        Returns:
            Complete validation report
        """
        print("Starting comprehensive validation...")
        
        # Run validation
        validation_results = self.validate_dataset(dataset, methods, max_systems)
        
        # Prepare comprehensive report
        report = {
            "validation_metadata": {
                "dataset_name": dataset.name,
                "methods_tested": methods or ["casscf", "nevpt2"],
                "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tolerance_settings": {
                    "tight": self.tolerance_tight,
                    "loose": self.tolerance_loose
                }
            },
            "validation_results": validation_results,
            "quality_assessment": self._assess_implementation_quality(validation_results),
            "recommendations": self._generate_recommendations(validation_results)
        }
        
        return report
    
    def _assess_implementation_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall implementation quality based on validation results."""
        assessment = {
            "overall_grade": "unknown",
            "strengths": [],
            "weaknesses": [],
            "critical_issues": []
        }
        
        success_rate = validation_results["summary_statistics"].get("success_rate", 0)
        
        # Determine overall grade
        if success_rate >= 0.95:
            assessment["overall_grade"] = "excellent"
        elif success_rate >= 0.85:
            assessment["overall_grade"] = "good"
        elif success_rate >= 0.70:
            assessment["overall_grade"] = "fair"
        else:
            assessment["overall_grade"] = "poor"
        
        # Identify strengths and weaknesses
        if success_rate >= 0.90:
            assessment["strengths"].append("High success rate in calculations")
        
        if validation_results["failed_calculations"] == 0:
            assessment["strengths"].append("No complete calculation failures")
        
        common_errors = validation_results.get("common_errors", {})
        if "SCF convergence failure" in common_errors:
            assessment["weaknesses"].append("SCF convergence issues present")
        
        if "Multireference calculation failure" in common_errors:
            assessment["critical_issues"].append("Multireference method implementation issues")
        
        return assessment
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        success_rate = validation_results["summary_statistics"].get("success_rate", 0)
        common_errors = validation_results.get("common_errors", {})
        
        if success_rate < 0.90:
            recommendations.append("Improve overall calculation success rate")
        
        if "SCF convergence failure" in common_errors:
            recommendations.append("Implement more robust SCF convergence algorithms")
        
        if "Active space selection failure" in common_errors:
            recommendations.append("Add fallback methods for active space selection")
        
        if "Memory error" in common_errors:
            recommendations.append("Optimize memory usage for large systems")
        
        if not recommendations:
            recommendations.append("Implementation shows good stability and accuracy")
        
        return recommendations