"""Computation engine for managing quantum chemistry calculations."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .base import BaseCalculator, BaseSystem, OptimizationResult
from .quantum_system import QuantumSystem


class ComputationEngine(BaseModel):
    """Main computation engine for quantum chemistry calculations."""

    default_method: str = Field(default="hf", description="Default calculation method")
    default_basis: str = Field(default="sto-3g", description="Default basis set")
    parallel_jobs: int = Field(default=1, description="Number of parallel jobs")
    memory_limit: str | None = Field(default=None, description="Memory limit")

    _calculators: dict[str, type[BaseCalculator]] = {}
    _results_cache: dict[str, dict[str, Any]] = {}

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def register_calculator(
        self, name: str, calculator_class: type[BaseCalculator]
    ) -> None:
        """Register a new calculator type."""
        self._calculators[name] = calculator_class

    def get_available_calculators(self) -> list[str]:
        """Get list of available calculator names."""
        return list(self._calculators.keys())

    def create_calculator(
        self, method: str, basis_set: str | None = None, **kwargs
    ) -> BaseCalculator:
        """Create a calculator instance."""
        if method not in self._calculators:
            raise ValueError(f"Unknown method: {method}")

        basis = basis_set or self.default_basis
        calculator_class = self._calculators[method]

        return calculator_class(method=method, basis_set=basis, **kwargs)

    def run_single_point(
        self,
        system: QuantumSystem | BaseSystem,
        method: str | None = None,
        basis_set: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run single-point energy calculation."""
        calc_method = method or self.default_method
        calculator = self.create_calculator(calc_method, basis_set, **kwargs)

        if isinstance(system, QuantumSystem):
            base_system = system.system
        else:
            base_system = system

        results = calculator.run_calculation(base_system)

        # Cache results
        cache_key = self._generate_cache_key(system, calc_method, basis_set)
        self._results_cache[cache_key] = results

        return results

    def optimize_geometry(
        self,
        system: QuantumSystem | BaseSystem,
        method: str | None = None,
        basis_set: str | None = None,
        optimizer: str = "bfgs",
        max_steps: int = 100,
        convergence_threshold: float = 1e-6,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize molecular/crystal geometry."""
        calc_method = method or self.default_method
        calculator = self.create_calculator(calc_method, basis_set, **kwargs)

        if isinstance(system, QuantumSystem):
            base_system = system.system
        else:
            base_system = system

        # Simple optimization loop (would use scipy.optimize in practice)
        current_geometry = base_system.get_geometry()
        converged = False
        step = 0
        energies = []

        for step in range(max_steps):
            # Calculate energy and forces
            results = calculator.run_calculation(base_system)
            energy = results["energy"]
            energies.append(energy)

            # Check convergence
            if step > 0 and abs(energies[-1] - energies[-2]) < convergence_threshold:
                converged = True
                break

            # Update geometry (simplified)
            # In practice, this would use proper optimization algorithms
            current_geometry += np.random.normal(0, 0.01, current_geometry.shape)

        return OptimizationResult(
            final_energy=energies[-1],
            final_geometry=current_geometry,
            converged=converged,
            num_iterations=step + 1,
            force_rms=0.001,  # Placeholder
        )

    def run_property_calculation(
        self,
        system: QuantumSystem | BaseSystem,
        properties: list[str],
        method: str | None = None,
        basis_set: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate molecular/material properties."""
        # First run single-point calculation
        sp_results = self.run_single_point(system, method, basis_set, **kwargs)

        property_results = {"energy": sp_results["energy"]}

        # Calculate requested properties
        for prop in properties:
            if prop == "dipole":
                property_results["dipole"] = self._calculate_dipole(system)
            elif prop == "polarizability":
                property_results["polarizability"] = self._calculate_polarizability(
                    system
                )
            elif prop == "vibrational_frequencies":
                property_results["frequencies"] = self._calculate_frequencies(system)
            else:
                raise ValueError(f"Unknown property: {prop}")

        return property_results

    def get_cached_results(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve cached calculation results."""
        return self._results_cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear results cache."""
        self._results_cache.clear()

    def _generate_cache_key(
        self,
        system: QuantumSystem | BaseSystem,
        method: str,
        basis_set: str | None,
    ) -> str:
        """Generate cache key for results."""
        if isinstance(system, QuantumSystem):
            system_hash = hash(str(system.system.get_geometry().tobytes()))
        else:
            system_hash = hash(str(system.get_geometry().tobytes()))

        return f"{method}_{basis_set}_{system_hash}"

    def _calculate_dipole(self, system) -> np.ndarray:
        """Calculate dipole moment (placeholder)."""
        return np.array([0.0, 0.0, 0.0])

    def _calculate_polarizability(self, system) -> np.ndarray:
        """Calculate polarizability tensor (placeholder)."""
        return np.eye(3) * 10.0

    def _calculate_frequencies(self, system) -> np.ndarray:
        """Calculate vibrational frequencies (placeholder)."""
        num_atoms = len(system.get_atomic_numbers())
        return np.random.random(3 * num_atoms - 6) * 3000  # cm⁻¹
