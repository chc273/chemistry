"""Base classes and abstract interfaces for the quantchem package."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class BaseSystem(BaseModel, ABC):
    """Abstract base class for all quantum systems."""

    name: str = Field(description="Human-readable name for the system")
    charge: int = Field(default=0, description="Total charge of the system")
    multiplicity: int = Field(default=1, description="Spin multiplicity")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        validate_assignment = True

    @abstractmethod
    def get_geometry(self) -> np.ndarray:
        """Return the geometry of the system as a numpy array."""
        pass

    @abstractmethod
    def get_atomic_numbers(self) -> list[int]:
        """Return list of atomic numbers."""
        pass

    @abstractmethod
    def compute_nuclear_repulsion(self) -> float:
        """Compute nuclear repulsion energy."""
        pass

    def get_num_electrons(self) -> int:
        """Calculate number of electrons in the system."""
        return sum(self.get_atomic_numbers()) - self.charge


class BaseCalculator(ABC):
    """Abstract base class for all quantum chemistry calculators."""

    def __init__(
        self,
        method: str,
        basis_set: str,
        convergence_threshold: float = 1e-8,
        max_iterations: int = 100,
    ):
        self.method = method
        self.basis_set = basis_set
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self._results: dict[str, Any] = {}

    @abstractmethod
    def run_calculation(self, system: BaseSystem) -> dict[str, Any]:
        """Run the quantum chemistry calculation."""
        pass

    @abstractmethod
    def get_energy(self) -> float:
        """Get the total energy from the last calculation."""
        pass

    def get_results(self) -> dict[str, Any]:
        """Return all calculation results."""
        return self._results.copy()

    def is_converged(self) -> bool:
        """Check if the last calculation converged."""
        return self._results.get("converged", False)


class PropertyCalculator(ABC):
    """Abstract base class for property calculations."""

    @abstractmethod
    def calculate_property(
        self, system: BaseSystem, calculator: BaseCalculator
    ) -> dict[str, Any]:
        """Calculate molecular/material properties."""
        pass


class OptimizationResult(BaseModel):
    """Results from geometry optimization."""

    final_energy: float = Field(description="Final optimized energy")
    final_geometry: np.ndarray = Field(description="Final optimized geometry")
    converged: bool = Field(description="Whether optimization converged")
    num_iterations: int = Field(description="Number of optimization steps")
    force_rms: float = Field(description="RMS force in final structure")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
