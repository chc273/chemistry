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
from .multireference import (
    MultireferenceMethod,
    MultireferenceResult,
    MethodSelector,
)
from .multireference.methods import (
    CASSCFMethod,
    NEVPT2Method,
    CASPT2Method,
)
from .multireference.workflows import (
    MultireferenceWorkflow,
)
from .diagnostics import (
    MultireferenceDiagnostics,
    IntelligentMethodSelector,
    DiagnosticConfig,
    MultireferenceCharacter,
    DiagnosticResult,
    DiagnosticMethod,
    SystemClassification,
    calculate_homo_lumo_gap,
    calculate_spin_contamination,
    calculate_natural_orbital_occupations,
    calculate_fractional_occupation_density,
    calculate_bond_order_fluctuation,
    calculate_t1_diagnostic,
    calculate_d1_diagnostic,
    calculate_correlation_recovery,
    calculate_s_diagnostic,
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
    # Multireference methods
    "MultireferenceMethod",
    "MultireferenceResult",
    "MethodSelector",
    "CASSCFMethod",
    "NEVPT2Method",
    "CASPT2Method",
    "MultireferenceWorkflow",
    # Multireference diagnostics
    "MultireferenceDiagnostics",
    "IntelligentMethodSelector", 
    "DiagnosticConfig",
    "MultireferenceCharacter",
    "DiagnosticResult",
    "DiagnosticMethod",
    "SystemClassification",
    "calculate_homo_lumo_gap",
    "calculate_spin_contamination",
    "calculate_natural_orbital_occupations",
    "calculate_fractional_occupation_density",
    "calculate_bond_order_fluctuation",
    "calculate_t1_diagnostic", 
    "calculate_d1_diagnostic",
    "calculate_correlation_recovery",
    "calculate_s_diagnostic",
]
