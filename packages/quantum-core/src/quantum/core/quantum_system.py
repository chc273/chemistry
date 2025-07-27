"""General quantum system interface for unified calculations."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .crystal import Crystal
from .molecule import Molecule


class QuantumSystem(BaseModel):
    """Unified interface for quantum systems (molecules, crystals, etc.)."""

    system: Molecule | Crystal = Field(description="The underlying system")
    environment: dict[str, Any] | None = Field(
        default=None,
        description="Environmental conditions (temperature, pressure, etc.)",
    )
    constraints: dict[str, Any] | None = Field(
        default=None, description="Constraints for calculations"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @property
    def name(self) -> str:
        """Get system name."""
        return self.system.name

    @property
    def charge(self) -> int:
        """Get system charge."""
        return self.system.charge

    @property
    def multiplicity(self) -> int:
        """Get system multiplicity."""
        return self.system.multiplicity

    def is_molecular(self) -> bool:
        """Check if system is molecular."""
        return isinstance(self.system, Molecule)

    def is_periodic(self) -> bool:
        """Check if system is periodic (crystal)."""
        return isinstance(self.system, Crystal)

    def get_geometry(self) -> np.ndarray:
        """Get system geometry."""
        return self.system.get_geometry()

    def get_atomic_numbers(self) -> list[int]:
        """Get atomic numbers."""
        return self.system.get_atomic_numbers()

    def get_num_electrons(self) -> int:
        """Get number of electrons."""
        return self.system.get_num_electrons()

    def compute_nuclear_repulsion(self) -> float:
        """Compute nuclear repulsion energy."""
        return self.system.compute_nuclear_repulsion()

    def add_environment(self, **kwargs) -> "QuantumSystem":
        """Add environmental conditions."""
        env = self.environment or {}
        env.update(kwargs)
        return self.model_copy(update={"environment": env})

    def add_constraints(self, **kwargs) -> "QuantumSystem":
        """Add calculation constraints."""
        constraints = self.constraints or {}
        constraints.update(kwargs)
        return self.model_copy(update={"constraints": constraints})

    def get_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "name": self.name,
            "type": "molecular" if self.is_molecular() else "periodic",
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "num_atoms": len(self.system.atoms),
            "num_electrons": self.get_num_electrons(),
            "nuclear_repulsion": self.compute_nuclear_repulsion(),
        }

        if self.is_periodic():
            crystal = self.system
            info.update(
                {
                    "volume": crystal.get_volume(),
                    "density": crystal.get_density(),
                    "lattice_parameters": crystal.get_lattice_parameters(),
                    "space_group": crystal.space_group,
                }
            )

        if self.environment:
            info["environment"] = self.environment

        if self.constraints:
            info["constraints"] = self.constraints

        return info
