"""Quantum chemistry calculation methods using PySCF backend."""

from .active_space import (
    ActiveSpaceMethod,
    ActiveSpaceResult,
    UnifiedActiveSpaceFinder,
    auto_find_active_space,
    find_active_space_apc,
    find_active_space_avas,
    find_active_space_dmet_cas,
    find_active_space_energy_window,
    find_active_space_iao,
    find_active_space_ibo,
    find_active_space_natural_orbitals,
)
from .dft import (
    B3LYPCalculator,
    DFTCalculator,
    M06Calculator,
    PBECalculator,
    wB97XDCalculator,
)
from .hartree_fock import (
    HartreeFockCalculator,
    RestrictedHF,
    RestrictedOpenHF,
    UnrestrictedHF,
)

__version__ = "0.1.0"

__all__ = [
    # Hartree-Fock methods
    "HartreeFockCalculator",
    "RestrictedHF",
    "UnrestrictedHF",
    "RestrictedOpenHF",
    # DFT methods
    "DFTCalculator",
    "B3LYPCalculator",
    "PBECalculator",
    "M06Calculator",
    "wB97XDCalculator",
    # Active space methods
    "ActiveSpaceMethod",
    "ActiveSpaceResult",
    "UnifiedActiveSpaceFinder",
    "find_active_space_avas",
    "find_active_space_apc",
    "find_active_space_dmet_cas",
    "find_active_space_energy_window",
    "find_active_space_iao",
    "find_active_space_ibo",
    "find_active_space_natural_orbitals",
    "auto_find_active_space",
]
