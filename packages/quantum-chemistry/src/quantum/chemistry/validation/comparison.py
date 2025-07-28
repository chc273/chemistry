"""
Cross-method comparison framework for quantum chemistry validation.

This module provides tools for comparing results between different
quantum chemistry methods and validating against reference data.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..external import ExternalMethodRunner
from ..multireference import MultireferenceWorkflow
from ..multireference.base import MultireferenceMethodType
from .benchmarks import BenchmarkSuite
from .reference_data import ReferenceDatabase


@dataclass
class ValidationResult:
    """Result of a method validation against reference data."""

    system_name: str
    method: str
    basis_set: str

    # Calculated vs reference
    calculated_energy: float
    reference_energy: float
    energy_error: float  # calculated - reference
    relative_error: float  # (calculated - reference) / |reference|

    # Additional properties comparison
    property_errors: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    absolute_error: float = field(init=False)
    error_magnitude: str = field(
        init=False
    )  # 'excellent', 'good', 'acceptable', 'poor'

    # Reference information
    reference_source: str = ""
    reference_uncertainty: Optional[float] = None
    reference_quality: str = "unknown"

    # Calculation details
    calculation_time: Optional[float] = None
    convergence_info: Dict[str, Any] = field(default_factory=dict)

    # Statistical measures
    z_score: Optional[float] = None  # For uncertainty analysis

    def __post_init__(self):
        self.absolute_error = abs(self.energy_error)

        # Classify error magnitude (in mHartree)
        error_mhartree = self.absolute_error * 1000
        if error_mhartree < 0.1:
            self.error_magnitude = "excellent"
        elif error_mhartree < 1.0:
            self.error_magnitude = "good"
        elif error_mhartree < 5.0:
            self.error_magnitude = "acceptable"
        else:
            self.error_magnitude = "poor"

        # Calculate z-score if uncertainty is available
        if self.reference_uncertainty and self.reference_uncertainty > 0:
            self.z_score = self.energy_error / self.reference_uncertainty


@dataclass
class MethodComparison:
    """Comparison between two quantum chemistry methods."""

    system_name: str
    method1: str
    method2: str
    basis_set: str

    energy1: float
    energy2: float
    energy_difference: float  # method1 - method2

    # Additional properties
    property_differences: Dict[str, float] = field(default_factory=dict)

    # Statistical measures
    relative_difference: float = field(init=False)
    agreement_level: str = field(init=False)

    def __post_init__(self):
        # Avoid division by zero
        if abs(self.energy2) > 1e-10:
            self.relative_difference = self.energy_difference / abs(self.energy2)
        else:
            self.relative_difference = float("inf")

        # Classify agreement level
        abs_diff_mhartree = abs(self.energy_difference) * 1000
        if abs_diff_mhartree < 0.1:
            self.agreement_level = "excellent"
        elif abs_diff_mhartree < 1.0:
            self.agreement_level = "good"
        elif abs_diff_mhartree < 5.0:
            self.agreement_level = "acceptable"
        else:
            self.agreement_level = "poor"


class MethodComparator:
    """Framework for comparing quantum chemistry methods."""

    def __init__(
        self,
        benchmark_suite: Optional[BenchmarkSuite] = None,
        reference_db: Optional[ReferenceDatabase] = None,
    ):
        """Initialize the method comparator.

        Args:
            benchmark_suite: Suite of benchmark systems
            reference_db: Database of reference values
        """
        self.benchmark_suite = benchmark_suite or BenchmarkSuite()
        self.reference_db = reference_db or ReferenceDatabase()
        self.workflow = MultireferenceWorkflow()
        self.external_runner = ExternalMethodRunner()

        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.method_comparisons: List[MethodComparison] = []

    def validate_method(
        self,
        system_name: str,
        method: Union[str, MultireferenceMethodType],
        basis_set: str = None,
        reference_method: str = None,
        external_method: bool = False,
        **calculation_kwargs,
    ) -> ValidationResult:
        """Validate a method against reference data for a specific system.

        Args:
            system_name: Name of the benchmark system
            method: Method to validate
            basis_set: Basis set (if None, uses system default)
            reference_method: Reference method to compare against
            external_method: Whether to use external method runner
            **calculation_kwargs: Additional calculation parameters

        Returns:
            ValidationResult object
        """
        # Get benchmark system
        system = self.benchmark_suite.get_system(system_name)
        if system is None:
            raise ValueError(f"System '{system_name}' not found in benchmark suite")

        # Use system's basis set if not specified
        if basis_set is None:
            basis_set = system.basis_set

        # Run calculation
        start_time = datetime.now()

        try:
            if external_method:
                # Run external method calculation
                input_data = {
                    "geometry": system.geometry,
                    "basis_set": basis_set,
                    "charge": system.charge,
                    "spin": system.spin,
                    **calculation_kwargs,
                }

                # Add active space if available
                if system.recommended_active_space:
                    input_data["active_space"] = system.recommended_active_space

                external_result = self.external_runner.run_calculation(
                    str(method), input_data
                )
                calculated_energy = external_result["energy"]
                convergence_info = external_result.get("convergence_info", {})

            else:
                # Run internal multireference calculation
                mf = system.run_scf()

                if isinstance(method, str):
                    method_type = MultireferenceMethodType(method.lower())
                else:
                    method_type = method

                results = self.workflow.run_calculation(
                    mf,
                    active_space_method="avas",
                    mr_method=method_type,
                    **calculation_kwargs,
                )

                calculated_energy = results["multireference_result"].energy
                convergence_info = (
                    results["multireference_result"].convergence_info or {}
                )

        except Exception as e:
            raise RuntimeError(f"Calculation failed for {system_name}/{method}: {e}")

        end_time = datetime.now()
        calculation_time = (end_time - start_time).total_seconds()

        # Get reference data
        if reference_method is None:
            # Try to find the best available reference
            reference_entry = None
            for ref_method in ["fci", "ccsd(t)", "mrci", "experimental"]:
                reference_entry = self.reference_db.get_reference_energy(
                    system_name, ref_method, basis_set
                )
                if reference_entry:
                    reference_method = ref_method
                    break
        else:
            reference_entry = self.reference_db.get_reference_energy(
                system_name, reference_method, basis_set
            )

        if reference_entry is None:
            raise ValueError(
                f"No reference data found for {system_name}/{reference_method}/{basis_set}"
            )

        # Calculate errors
        reference_energy = reference_entry.energy
        energy_error = calculated_energy - reference_energy
        relative_error = (
            energy_error / abs(reference_energy)
            if abs(reference_energy) > 1e-10
            else float("inf")
        )

        # Create validation result
        validation_result = ValidationResult(
            system_name=system_name,
            method=str(method_type.value)
            if isinstance(method_type, MultireferenceMethodType)
            else str(method),
            basis_set=basis_set,
            calculated_energy=calculated_energy,
            reference_energy=reference_energy,
            energy_error=energy_error,
            relative_error=relative_error,
            reference_source=reference_entry.source,
            reference_uncertainty=reference_entry.uncertainty,
            reference_quality=reference_entry.quality_level,
            calculation_time=calculation_time,
            convergence_info=convergence_info,
        )

        self.validation_results.append(validation_result)
        return validation_result

    def validate_external_method(
        self,
        system_name: str,
        external_method: str,
        method_type: str = None,
        basis_set: str = None,
        reference_method: str = None,
        **calculation_kwargs,
    ) -> ValidationResult:
        """Validate an external quantum chemistry method.

        Args:
            system_name: Name of the benchmark system
            external_method: External software package name (e.g., 'molpro', 'orca')
            method_type: Specific method within the package (e.g., 'caspt2', 'ccsd(t)')
            basis_set: Basis set (if None, uses system default)
            reference_method: Reference method to compare against
            **calculation_kwargs: Additional calculation parameters

        Returns:
            ValidationResult object
        """
        # Get benchmark system
        system = self.benchmark_suite.get_system(system_name)
        if system is None:
            raise ValueError(f"System '{system_name}' not found in benchmark suite")

        # Use system's basis set if not specified
        if basis_set is None:
            basis_set = system.basis_set

        # Check external method availability
        if not self.external_runner.is_method_available(external_method):
            raise RuntimeError(f"External method '{external_method}' is not available")

        # Run external calculation
        start_time = datetime.now()

        try:
            input_data = {
                "geometry": system.geometry,
                "basis_set": basis_set,
                "charge": system.charge,
                "spin": system.spin,
                "method_type": method_type,
                **calculation_kwargs,
            }

            # Add active space information if available
            if system.recommended_active_space:
                input_data["active_space"] = system.recommended_active_space

            external_result = self.external_runner.run_calculation(
                external_method, input_data
            )
            calculated_energy = external_result["energy"]
            convergence_info = external_result.get("convergence_info", {})

        except Exception as e:
            raise RuntimeError(
                f"External calculation failed for {system_name}/{external_method}: {e}"
            )

        end_time = datetime.now()
        calculation_time = (end_time - start_time).total_seconds()

        # Get reference data
        method_string = (
            f"{external_method}_{method_type}" if method_type else external_method
        )

        if reference_method is None:
            # Try to find the best available reference
            reference_entry = None
            for ref_method in ["fci", "ccsd(t)", "mrci", "experimental"]:
                reference_entry = self.reference_db.get_reference_energy(
                    system_name, ref_method, basis_set
                )
                if reference_entry:
                    reference_method = ref_method
                    break
        else:
            reference_entry = self.reference_db.get_reference_energy(
                system_name, reference_method, basis_set
            )

        if reference_entry is None:
            raise ValueError(
                f"No reference data found for {system_name}/{reference_method}/{basis_set}"
            )

        # Calculate errors
        reference_energy = reference_entry.energy
        energy_error = calculated_energy - reference_energy
        relative_error = (
            energy_error / abs(reference_energy)
            if abs(reference_energy) > 1e-10
            else float("inf")
        )

        # Create validation result
        validation_result = ValidationResult(
            system_name=system_name,
            method=method_string,
            basis_set=basis_set,
            calculated_energy=calculated_energy,
            reference_energy=reference_energy,
            energy_error=energy_error,
            relative_error=relative_error,
            reference_source=reference_entry.source,
            reference_uncertainty=reference_entry.uncertainty,
            reference_quality=reference_entry.quality_level,
            calculation_time=calculation_time,
            convergence_info=convergence_info,
        )

        self.validation_results.append(validation_result)
        return validation_result

    def compare_methods(
        self,
        system_name: str,
        method1: Union[str, MultireferenceMethodType],
        method2: Union[str, MultireferenceMethodType],
        basis_set: str = None,
        **calculation_kwargs,
    ) -> MethodComparison:
        """Compare two methods on the same system.

        Args:
            system_name: Name of the benchmark system
            method1: First method to compare
            method2: Second method to compare
            basis_set: Basis set (if None, uses system default)
            **calculation_kwargs: Additional calculation parameters

        Returns:
            MethodComparison object
        """
        # Get benchmark system
        system = self.benchmark_suite.get_system(system_name)
        if system is None:
            raise ValueError(f"System '{system_name}' not found in benchmark suite")

        # Use system's basis set if not specified
        if basis_set is None:
            basis_set = system.basis_set

        # Run calculations for both methods
        mf = system.run_scf()

        # Method 1
        if isinstance(method1, str):
            method1_type = MultireferenceMethodType(method1.lower())
        else:
            method1_type = method1

        results1 = self.workflow.run_calculation(
            mf, active_space_method="avas", mr_method=method1_type, **calculation_kwargs
        )
        energy1 = results1["multireference_result"].energy

        # Method 2
        if isinstance(method2, str):
            method2_type = MultireferenceMethodType(method2.lower())
        else:
            method2_type = method2

        results2 = self.workflow.run_calculation(
            mf, active_space_method="avas", mr_method=method2_type, **calculation_kwargs
        )
        energy2 = results2["multireference_result"].energy

        # Create comparison
        comparison = MethodComparison(
            system_name=system_name,
            method1=str(method1_type.value),
            method2=str(method2_type.value),
            basis_set=basis_set,
            energy1=energy1,
            energy2=energy2,
            energy_difference=energy1 - energy2,
        )

        self.method_comparisons.append(comparison)
        return comparison

    def run_validation_suite(
        self,
        methods: List[Union[str, MultireferenceMethodType]],
        systems: List[str] = None,
        difficulty_levels: List[str] = None,
        **calculation_kwargs,
    ) -> Dict[str, List[ValidationResult]]:
        """Run a comprehensive validation suite across multiple methods and systems.

        Args:
            methods: List of methods to validate
            systems: List of system names (if None, uses default validation suite)
            difficulty_levels: System difficulty levels to include
            **calculation_kwargs: Additional calculation parameters

        Returns:
            Dictionary mapping method names to validation results
        """
        if systems is None:
            if difficulty_levels is None:
                difficulty_levels = ["easy", "medium"]
            benchmark_systems = self.benchmark_suite.get_validation_suite(
                difficulty_levels
            )
            systems = [sys.name for sys in benchmark_systems]

        results = {}

        for method in methods:
            method_results = []
            method_name = (
                str(method.value)
                if isinstance(method, MultireferenceMethodType)
                else str(method)
            )

            print(f"Validating {method_name}...")

            for system_name in systems:
                try:
                    result = self.validate_method(
                        system_name=system_name, method=method, **calculation_kwargs
                    )
                    method_results.append(result)

                    print(
                        f"  {system_name}: {result.error_magnitude} "
                        f"(error: {result.energy_error*1000:.2f} mH)"
                    )

                except Exception as e:
                    print(f"  {system_name}: FAILED ({str(e)[:50]}...)")

            results[method_name] = method_results

        return results

    def run_method_comparison_suite(
        self,
        method_pairs: List[Tuple[str, str]],
        systems: List[str] = None,
        difficulty_levels: List[str] = None,
        **calculation_kwargs,
    ) -> List[MethodComparison]:
        """Run method comparisons across multiple systems.

        Args:
            method_pairs: List of (method1, method2) tuples to compare
            systems: List of system names
            difficulty_levels: System difficulty levels to include
            **calculation_kwargs: Additional calculation parameters

        Returns:
            List of MethodComparison objects
        """
        if systems is None:
            if difficulty_levels is None:
                difficulty_levels = ["easy", "medium"]
            benchmark_systems = self.benchmark_suite.get_validation_suite(
                difficulty_levels
            )
            systems = [sys.name for sys in benchmark_systems]

        comparisons = []

        for method1, method2 in method_pairs:
            print(f"Comparing {method1} vs {method2}...")

            for system_name in systems:
                try:
                    comparison = self.compare_methods(
                        system_name=system_name,
                        method1=method1,
                        method2=method2,
                        **calculation_kwargs,
                    )
                    comparisons.append(comparison)

                    print(
                        f"  {system_name}: {comparison.agreement_level} "
                        f"(diff: {comparison.energy_difference*1000:.2f} mH)"
                    )

                except Exception as e:
                    print(f"  {system_name}: FAILED ({str(e)[:50]}...)")

        return comparisons

    def run_external_validation_suite(
        self,
        external_methods: List[str],
        method_types: Dict[str, List[str]] = None,
        systems: List[str] = None,
        difficulty_levels: List[str] = None,
        **calculation_kwargs,
    ) -> Dict[str, List[ValidationResult]]:
        """Run validation suite for external quantum chemistry methods.

        Args:
            external_methods: List of external software packages
            method_types: Dictionary mapping external methods to their specific methods
            systems: List of system names
            difficulty_levels: System difficulty levels to include
            **calculation_kwargs: Additional calculation parameters

        Returns:
            Dictionary mapping method names to validation results
        """
        if method_types is None:
            method_types = {
                "molpro": ["casscf", "caspt2", "mrci"],
                "orca": ["casscf", "nevpt2", "ccsd(t)"],
                "psi4": ["casscf", "caspt2", "ccsd(t)"],
                "gaussian": ["casscf", "mp2", "ccsd(t)"],
            }

        # Get suitable systems for external methods
        if systems is None:
            external_suite = self.benchmark_suite.get_external_method_suite(
                external_methods, difficulty_levels
            )
        else:
            external_suite = {
                method: [
                    self.benchmark_suite.get_system(s)
                    for s in systems
                    if self.benchmark_suite.get_system(s)
                ]
                for method in external_methods
            }

        all_results = {}

        for external_method in external_methods:
            if not self.external_runner.is_method_available(external_method):
                print(
                    f"⚠️  External method '{external_method}' is not available - skipping"
                )
                continue

            method_suite_results = []
            available_method_types = method_types.get(external_method, ["default"])
            suitable_systems = external_suite.get(external_method, [])

            print(f"\n🧪 Validating {external_method.upper()} methods...")

            for method_type in available_method_types:
                print(f"\n  Testing {method_type}:")

                for system in suitable_systems:
                    if system is None:
                        continue

                    try:
                        result = self.validate_external_method(
                            system_name=system.name.lower(),
                            external_method=external_method,
                            method_type=method_type,
                            **calculation_kwargs,
                        )
                        method_suite_results.append(result)

                        print(
                            f"    • {system.name:>8}: {result.error_magnitude:>10} "
                            f"(error: {result.energy_error*1000:>6.2f} mH)"
                        )

                    except Exception as e:
                        print(
                            f"    • {system.name:>8}: {'FAILED':>10} ({str(e)[:40]}...)"
                        )

            if method_suite_results:
                all_results[external_method] = method_suite_results

        return all_results

    def compare_external_methods(
        self,
        system_name: str,
        external_method1: str,
        external_method2: str,
        method_type1: str = None,
        method_type2: str = None,
        basis_set: str = None,
        **calculation_kwargs,
    ) -> MethodComparison:
        """Compare two external quantum chemistry methods.

        Args:
            system_name: Name of the benchmark system
            external_method1: First external software package
            external_method2: Second external software package
            method_type1: Method type for first package
            method_type2: Method type for second package
            basis_set: Basis set (if None, uses system default)
            **calculation_kwargs: Additional calculation parameters

        Returns:
            MethodComparison object
        """
        # Get benchmark system
        system = self.benchmark_suite.get_system(system_name)
        if system is None:
            raise ValueError(f"System '{system_name}' not found in benchmark suite")

        if basis_set is None:
            basis_set = system.basis_set

        # Check availability
        for ext_method in [external_method1, external_method2]:
            if not self.external_runner.is_method_available(ext_method):
                raise RuntimeError(f"External method '{ext_method}' is not available")

        # Run first external method
        input_data1 = {
            "geometry": system.geometry,
            "basis_set": basis_set,
            "charge": system.charge,
            "spin": system.spin,
            "method_type": method_type1,
            **calculation_kwargs,
        }

        if system.recommended_active_space:
            input_data1["active_space"] = system.recommended_active_space

        result1 = self.external_runner.run_calculation(external_method1, input_data1)
        energy1 = result1["energy"]

        # Run second external method
        input_data2 = {
            "geometry": system.geometry,
            "basis_set": basis_set,
            "charge": system.charge,
            "spin": system.spin,
            "method_type": method_type2,
            **calculation_kwargs,
        }

        if system.recommended_active_space:
            input_data2["active_space"] = system.recommended_active_space

        result2 = self.external_runner.run_calculation(external_method2, input_data2)
        energy2 = result2["energy"]

        # Create method names
        method1_name = (
            f"{external_method1}_{method_type1}" if method_type1 else external_method1
        )
        method2_name = (
            f"{external_method2}_{method_type2}" if method_type2 else external_method2
        )

        # Create comparison
        comparison = MethodComparison(
            system_name=system_name,
            method1=method1_name,
            method2=method2_name,
            basis_set=basis_set,
            energy1=energy1,
            energy2=energy2,
            energy_difference=energy1 - energy2,
        )

        self.method_comparisons.append(comparison)
        return comparison

    def get_method_statistics(self, method: str) -> Dict[str, Any]:
        """Get statistical summary for a method's validation results.

        Args:
            method: Method name

        Returns:
            Dictionary with statistical information
        """
        method_results = [
            r for r in self.validation_results if r.method.lower() == method.lower()
        ]

        if not method_results:
            return {}

        errors = [r.energy_error for r in method_results]
        abs_errors = [r.absolute_error for r in method_results]
        relative_errors = [r.relative_error for r in method_results]

        # Error magnitude distribution
        error_dist = {}
        for level in ["excellent", "good", "acceptable", "poor"]:
            error_dist[level] = len(
                [r for r in method_results if r.error_magnitude == level]
            )

        return {
            "method": method,
            "total_systems": len(method_results),
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "mean_absolute_error": np.mean(abs_errors),
            "max_absolute_error": np.max(abs_errors),
            "rms_error": np.sqrt(np.mean([e**2 for e in errors])),
            "mean_relative_error": np.mean(
                [abs(re) for re in relative_errors if abs(re) != float("inf")]
            ),
            "error_distribution": error_dist,
            "success_rate": len(method_results) / len(self.validation_results)
            if self.validation_results
            else 0,
        }

    def export_results(self, output_path: Union[str, Path]):
        """Export validation and comparison results to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to dictionaries for JSON serialization
        validation_data = []
        for result in self.validation_results:
            validation_data.append(
                {
                    "system_name": result.system_name,
                    "method": result.method,
                    "basis_set": result.basis_set,
                    "calculated_energy": result.calculated_energy,
                    "reference_energy": result.reference_energy,
                    "energy_error": result.energy_error,
                    "relative_error": result.relative_error,
                    "absolute_error": result.absolute_error,
                    "error_magnitude": result.error_magnitude,
                    "reference_source": result.reference_source,
                    "reference_uncertainty": result.reference_uncertainty,
                    "reference_quality": result.reference_quality,
                    "calculation_time": result.calculation_time,
                    "convergence_info": result.convergence_info,
                    "z_score": result.z_score,
                }
            )

        comparison_data = []
        for comp in self.method_comparisons:
            comparison_data.append(
                {
                    "system_name": comp.system_name,
                    "method1": comp.method1,
                    "method2": comp.method2,
                    "basis_set": comp.basis_set,
                    "energy1": comp.energy1,
                    "energy2": comp.energy2,
                    "energy_difference": comp.energy_difference,
                    "relative_difference": comp.relative_difference,
                    "agreement_level": comp.agreement_level,
                }
            )

        data = {
            "timestamp": datetime.now().isoformat(),
            "validation_results": validation_data,
            "method_comparisons": comparison_data,
            "summary": self.get_summary_statistics(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get overall summary statistics for all validation results."""
        if not self.validation_results:
            return {}

        methods = list(set(r.method for r in self.validation_results))
        systems = list(set(r.system_name for r in self.validation_results))

        method_stats = {}
        for method in methods:
            method_stats[method] = self.get_method_statistics(method)

        return {
            "total_validations": len(self.validation_results),
            "total_comparisons": len(self.method_comparisons),
            "unique_methods": len(methods),
            "unique_systems": len(systems),
            "methods": methods,
            "systems": systems,
            "method_statistics": method_stats,
        }
