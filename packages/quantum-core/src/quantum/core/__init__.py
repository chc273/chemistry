"""Core classes and interfaces for quantum chemistry computations with ASE/PyMatGen integration."""

from .base import BaseCalculator, BaseSystem
from .computation_engine import ComputationEngine
from .converters import (
    from_file,
    get_supported_formats,
    to_ase_atoms,
    to_file,
    to_pymatgen_molecule,
    to_pymatgen_structure,
    to_qcschema,
    to_quantum_molecule,
)
from .crystal import Crystal
from .molecule import Molecule
from .quantum_system import QuantumSystem

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Molecule",
    "Crystal",
    "QuantumSystem",
    "ComputationEngine",
    "BaseSystem",
    "BaseCalculator",
    # Conversion utilities
    "to_ase_atoms",
    "to_pymatgen_molecule",
    "to_pymatgen_structure",
    "to_qcschema",
    "to_quantum_molecule",
    "from_file",
    "to_file",
    "get_supported_formats",
]
