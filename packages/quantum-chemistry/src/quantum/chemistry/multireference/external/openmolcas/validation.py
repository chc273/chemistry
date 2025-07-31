"""
Cross-method validation utilities for OpenMolcas CASPT2 calculations.

This module provides comprehensive validation of CASPT2 results against
other multireference methods and established benchmarks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from ...base import MultireferenceResult
from ...methods.casscf import CASSCFMethod
from .caspt2_method import CASPT2Method


class ValidationResult(BaseModel):
    """
    Container for cross-method validation results.
    
    Provides structured storage of validation metrics and comparison data
    between different multireference methods.
    """
    
    # Method comparison
    reference_method: str = Field(..., description="Reference method name")
    test_method: str = Field(..., description="Method being validated")
    
    # Energy comparisons
    reference_energy: float = Field(..., description="Reference energy in Hartree")
    test_energy: float = Field(..., description="Test method energy in Hartree")
    energy_difference: float = Field(..., description="Energy difference (test - ref)")
    relative_error: float = Field(..., description="Relative error in energy")
    
    # Correlation energy analysis
    reference_correlation: Optional[float] = Field(
        None, description="Reference correlation energy"
    )
    test_correlation: Optional[float] = Field(
        None, description="Test correlation energy"
    )
    correlation_recovery: Optional[float] = Field(
        None, description="Fraction of correlation energy recovered"
    )
    
    # Statistical analysis
    validation_status: str = Field(..., description="Pass/Fail/Warning status")
    validation_message: str = Field(..., description="Human-readable validation summary")
    
    # Detailed metrics
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Additional validation metrics"
    )
    
    # Calculation details
    system_info: Dict[str, Any] = Field(
        default_factory=dict, description="System and calculation details"
    )


class OpenMolcasValidator:
    """
    Comprehensive validation framework for OpenMolcas CASPT2 calculations.
    
    This class provides automated validation against reference methods,
    literature benchmarks, and internal consistency checks.
    """
    
    def __init__(self):
        """Initialize validator with tolerance settings."""
        # Default validation tolerances
        self.tolerances = {
            "energy_absolute": 1e-3,      # 1 mHartree absolute tolerance
            "energy_relative": 1e-4,      # 0.01% relative tolerance
            "correlation_recovery": 0.85,  # Minimum correlation recovery
            "convergence_threshold": 1e-6, # Convergence requirement
        }
        
        # Benchmark data for common systems
        self._load_benchmark_data()
    
    def _load_benchmark_data(self):
        """Load reference benchmark data for validation."""
        # Literature benchmarks for small molecules
        # These values should be updated with high-quality reference data
        self.benchmarks = {
            "h2": {
                "geometry": "H 0 0 0; H 0 0 0.74",
                "basis": "cc-pVDZ",
                "caspt2_energy": -1.173498,  # Example value
                "correlation_energy": -0.0401,
            },
            "h2o": {
                "geometry": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
                "basis": "cc-pVDZ", 
                "caspt2_energy": -76.237412,  # Example value
                "correlation_energy": -0.2156,
            },
            "n2": {
                "geometry": "N 0 0 0; N 0 0 1.098",
                "basis": "cc-pVDZ",
                "caspt2_energy": -109.281532,  # Example value
                "correlation_energy": -0.4123,
            }
        }
    
    def validate_against_casscf(
        self,
        caspt2_result: MultireferenceResult,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        run_casscf: bool = True,
        casscf_result: Optional[MultireferenceResult] = None
    ) -> ValidationResult:
        """
        Validate CASPT2 results against CASSCF reference.
        
        Args:
            caspt2_result: CASPT2 calculation result
            scf_obj: SCF object used for calculation
            active_space: Active space used
            run_casscf: Whether to run CASSCF for reference (if casscf_result not provided)
            casscf_result: Pre-computed CASSCF result (optional)
            
        Returns:
            ValidationResult with comparison metrics
        """
        # Get CASSCF reference if not provided
        if casscf_result is None and run_casscf:
            casscf_method = CASSCFMethod()
            casscf_result = casscf_method.calculate(scf_obj, active_space)
        elif casscf_result is None:
            raise ValueError("Either run_casscf=True or casscf_result must be provided")
        
        # Calculate validation metrics
        energy_diff = caspt2_result.energy - casscf_result.energy
        relative_error = abs(energy_diff) / abs(casscf_result.energy)
        
        # Correlation energy analysis
        casscf_correlation = casscf_result.correlation_energy or 0.0
        caspt2_correlation = caspt2_result.correlation_energy or 0.0
        additional_correlation = caspt2_correlation - casscf_correlation
        
        # Validation status
        status, message = self._assess_casscf_comparison(
            energy_diff, additional_correlation, caspt2_result, casscf_result
        )
        
        # Build system information
        system_info = {
            "molecule": scf_obj.mol.atom,
            "basis_set": scf_obj.mol.basis,
            "active_space": (active_space.n_active_electrons, active_space.n_active_orbitals),
            "casscf_converged": casscf_result.convergence_info.get("converged", False),
            "caspt2_converged": caspt2_result.convergence_info.get("converged", False),
        }
        
        # Additional metrics
        metrics = {
            "energy_lowering": -energy_diff,  # CASPT2 should lower energy
            "correlation_enhancement": additional_correlation,
            "casscf_wall_time": casscf_result.computational_cost.get("wall_time", 0),
            "caspt2_wall_time": caspt2_result.computational_cost.get("wall_time", 0),
            "speedup_ratio": (caspt2_result.computational_cost.get("wall_time", 1) / 
                            max(casscf_result.computational_cost.get("wall_time", 1), 1e-6)),
        }
        
        return ValidationResult(
            reference_method="CASSCF",
            test_method=caspt2_result.method,
            reference_energy=casscf_result.energy,
            test_energy=caspt2_result.energy,
            energy_difference=energy_diff,
            relative_error=relative_error,
            reference_correlation=casscf_correlation,
            test_correlation=caspt2_correlation,
            correlation_recovery=None,  # Not applicable for CASSCF comparison
            validation_status=status,
            validation_message=message,
            metrics=metrics,
            system_info=system_info
        )
    
    def _assess_casscf_comparison(
        self,
        energy_diff: float,
        additional_correlation: float,
        caspt2_result: MultireferenceResult,
        casscf_result: MultireferenceResult
    ) -> Tuple[str, str]:
        """Assess CASSCF vs CASPT2 comparison and return status."""
        issues = []
        
        # CASPT2 should lower the energy
        if energy_diff > 1e-6:
            issues.append(f"CASPT2 energy higher than CASSCF by {energy_diff:.6f} Hartree")
        
        # Check for reasonable energy lowering
        if energy_diff < -0.1:
            issues.append(f"Very large energy lowering: {-energy_diff:.6f} Hartree")
        elif energy_diff > -1e-5:
            issues.append(f"Very small energy lowering: {-energy_diff:.6f} Hartree")
        
        # Check convergence
        if not caspt2_result.convergence_info.get("converged", False):
            issues.append("CASPT2 calculation did not converge")
        
        if not casscf_result.convergence_info.get("converged", False):
            issues.append("CASSCF reference calculation did not converge")
        
        # Determine status
        if not issues:
            return "PASS", "CASPT2 result is consistent with CASSCF reference"
        elif len(issues) == 1 and "small energy lowering" in issues[0]:
            return "WARNING", f"Validation passed with warning: {issues[0]}"
        else:
            return "FAIL", f"Validation failed: {'; '.join(issues)}"
    
    def validate_against_benchmark(
        self,
        caspt2_result: MultireferenceResult,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        benchmark_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate CASPT2 results against literature benchmarks.
        
        Args:
            caspt2_result: CASPT2 calculation result
            scf_obj: SCF object used
            active_space: Active space used
            benchmark_name: Specific benchmark to use (auto-detect if None)
            
        Returns:
            ValidationResult with benchmark comparison
        """
        # Auto-detect benchmark if not specified
        if benchmark_name is None:
            benchmark_name = self._detect_benchmark_system(scf_obj)
        
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"No benchmark data available for system: {benchmark_name}")
        
        benchmark = self.benchmarks[benchmark_name]
        
        # Calculate validation metrics
        ref_energy = benchmark["caspt2_energy"]
        energy_diff = caspt2_result.energy - ref_energy
        relative_error = abs(energy_diff) / abs(ref_energy)
        
        # Correlation energy comparison
        ref_correlation = benchmark.get("correlation_energy")
        test_correlation = caspt2_result.correlation_energy
        correlation_recovery = None
        
        if ref_correlation and test_correlation:
            correlation_recovery = test_correlation / ref_correlation
        
        # Validation assessment
        status, message = self._assess_benchmark_comparison(
            energy_diff, relative_error, correlation_recovery, benchmark_name
        )
        
        # System information
        system_info = {
            "benchmark_system": benchmark_name,
            "benchmark_geometry": benchmark["geometry"],
            "benchmark_basis": benchmark["basis"],
            "actual_basis": scf_obj.mol.basis,
            "active_space": (active_space.n_active_electrons, active_space.n_active_orbitals),
        }
        
        # Additional metrics
        metrics = {
            "absolute_error_hartree": abs(energy_diff),
            "absolute_error_kcal_mol": abs(energy_diff) * 627.509,
            "benchmark_correlation": ref_correlation or 0.0,
        }
        
        return ValidationResult(
            reference_method=f"Benchmark ({benchmark_name})",
            test_method=caspt2_result.method,
            reference_energy=ref_energy,
            test_energy=caspt2_result.energy,
            energy_difference=energy_diff,
            relative_error=relative_error,
            reference_correlation=ref_correlation,
            test_correlation=test_correlation,
            correlation_recovery=correlation_recovery,
            validation_status=status,
            validation_message=message,
            metrics=metrics,
            system_info=system_info
        )
    
    def _detect_benchmark_system(self, scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]) -> str:
        """Auto-detect benchmark system based on molecular structure."""
        mol = scf_obj.mol
        natm = mol.natm
        elements = [mol.atom_symbol(i) for i in range(natm)]
        
        # Simple pattern matching for common systems
        if natm == 2:
            if elements == ['H', 'H']:
                return "h2"
            elif sorted(elements) == ['N', 'N']:
                return "n2"
        elif natm == 3 and sorted(elements) == ['H', 'H', 'O']:
            return "h2o"
        
        raise ValueError(f"No benchmark available for molecular system: {elements}")
    
    def _assess_benchmark_comparison(
        self,
        energy_diff: float,
        relative_error: float,
        correlation_recovery: Optional[float],
        benchmark_name: str
    ) -> Tuple[str, str]:
        """Assess benchmark comparison and return validation status."""
        issues = []
        
        # Check absolute energy accuracy
        if abs(energy_diff) > self.tolerances["energy_absolute"]:
            issues.append(
                f"Energy differs from benchmark by {abs(energy_diff):.6f} Hartree "
                f"(tolerance: {self.tolerances['energy_absolute']:.6f})"
            )
        
        # Check relative energy accuracy
        if relative_error > self.tolerances["energy_relative"]:
            issues.append(
                f"Relative energy error {relative_error:.6f} exceeds tolerance "
                f"{self.tolerances['energy_relative']:.6f}"
            )
        
        # Check correlation recovery if available
        if correlation_recovery is not None:
            if correlation_recovery < self.tolerances["correlation_recovery"]:
                issues.append(
                    f"Low correlation recovery: {correlation_recovery:.3f} "
                    f"(minimum: {self.tolerances['correlation_recovery']:.3f})"
                )
        
        # Determine status
        if not issues:
            return "PASS", f"CASPT2 result matches {benchmark_name} benchmark within tolerances"
        elif len(issues) == 1 and "correlation recovery" in issues[0]:
            return "WARNING", f"Validation passed with warning: {issues[0]}"
        else:
            return "FAIL", f"Benchmark validation failed: {'; '.join(issues)}"
    
    def validate_internal_consistency(
        self,
        caspt2_result: MultireferenceResult
    ) -> ValidationResult:
        """
        Perform internal consistency checks on CASPT2 results.
        
        Args:
            caspt2_result: CASPT2 calculation result
            
        Returns:
            ValidationResult with consistency analysis
        """
        issues = []
        warnings = []
        
        # Check energy components
        if caspt2_result.correlation_energy is not None:
            if caspt2_result.correlation_energy > 0:
                issues.append("Positive correlation energy (should be negative)")
            elif caspt2_result.correlation_energy < -2.0:
                warnings.append(f"Very large correlation energy: {caspt2_result.correlation_energy:.6f}")
        
        # Check convergence
        conv_info = caspt2_result.convergence_info
        if not conv_info.get("converged", False):
            issues.append("Calculation did not converge")
        
        # Check for reasonable computation time
        wall_time = caspt2_result.computational_cost.get("wall_time", 0)
        if wall_time > 86400:  # 24 hours
            warnings.append(f"Very long computation time: {wall_time:.1f} seconds")
        
        # Check active space info for warnings/errors
        active_info = caspt2_result.active_space_info
        if "errors" in active_info and active_info["errors"]:
            issues.extend(active_info["errors"])
        if "warnings" in active_info and active_info["warnings"]:
            warnings.extend(active_info["warnings"])
        
        # Check for parameter issues
        ipea_shift = conv_info.get("ipea_shift", 0)
        if ipea_shift > 1.0:
            warnings.append(f"Very large IPEA shift: {ipea_shift}")
        
        imaginary_shift = conv_info.get("imaginary_shift", 0)
        if imaginary_shift > 0.5:
            warnings.append(f"Very large imaginary shift: {imaginary_shift}")
        
        # Determine validation status
        if issues:
            status = "FAIL"
            message = f"Internal consistency check failed: {'; '.join(issues)}"
        elif warnings:
            status = "WARNING" 
            message = f"Internal consistency check passed with warnings: {'; '.join(warnings)}"
        else:
            status = "PASS"
            message = "All internal consistency checks passed"
        
        # Build metrics
        metrics = {
            "num_errors": len(issues),
            "num_warnings": len(warnings),
            "wall_time_hours": wall_time / 3600,
            "correlation_energy": caspt2_result.correlation_energy or 0.0,
            "ipea_shift": ipea_shift,
            "imaginary_shift": imaginary_shift,
        }
        
        # System info
        system_info = {
            "method": caspt2_result.method,
            "active_space": (
                caspt2_result.n_active_electrons,
                caspt2_result.n_active_orbitals
            ),
            "basis_set": caspt2_result.basis_set,
            "software_version": caspt2_result.software_version,
        }
        
        return ValidationResult(
            reference_method="Internal Consistency",
            test_method=caspt2_result.method,
            reference_energy=0.0,  # Not applicable
            test_energy=caspt2_result.energy,
            energy_difference=0.0,  # Not applicable
            relative_error=0.0,     # Not applicable
            validation_status=status,
            validation_message=message,
            metrics=metrics,
            system_info=system_info
        )
    
    def comprehensive_validation(
        self,
        caspt2_result: MultireferenceResult,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        run_casscf: bool = True,
        include_benchmark: bool = True
    ) -> Dict[str, ValidationResult]:
        """
        Perform comprehensive validation including all available checks.
        
        Args:
            caspt2_result: CASPT2 calculation result
            scf_obj: SCF object used
            active_space: Active space used
            run_casscf: Whether to run CASSCF for comparison
            include_benchmark: Whether to include benchmark validation
            
        Returns:
            Dict of validation results for each check performed
        """
        validation_results = {}
        
        # Internal consistency check (always performed)
        validation_results["internal"] = self.validate_internal_consistency(caspt2_result)
        
        # CASSCF comparison
        if run_casscf:
            try:
                validation_results["casscf"] = self.validate_against_casscf(
                    caspt2_result, scf_obj, active_space, run_casscf=True
                )
            except Exception as e:
                print(f"CASSCF validation failed: {e}")
        
        # Benchmark validation
        if include_benchmark:
            try:
                validation_results["benchmark"] = self.validate_against_benchmark(
                    caspt2_result, scf_obj, active_space
                )
            except Exception as e:
                print(f"Benchmark validation failed: {e}")
        
        return validation_results
    
    def set_tolerances(self, **tolerances):
        """
        Update validation tolerances.
        
        Args:
            **tolerances: Tolerance values to update
        """
        for key, value in tolerances.items():
            if key in self.tolerances:
                self.tolerances[key] = value
            else:
                print(f"Warning: Unknown tolerance key '{key}' ignored")
    
    def add_benchmark(self, name: str, benchmark_data: Dict[str, Any]):
        """
        Add custom benchmark data.
        
        Args:
            name: Benchmark name
            benchmark_data: Dict with benchmark values
        """
        required_keys = ["caspt2_energy"]
        if not all(key in benchmark_data for key in required_keys):
            raise ValueError(f"Benchmark data must contain keys: {required_keys}")
        
        self.benchmarks[name] = benchmark_data