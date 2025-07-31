"""
Multireference diagnostics module for quantum chemistry calculations.

This module provides comprehensive diagnostics to determine whether a molecular system
requires multireference treatment. It implements both fast screening methods and
accurate reference diagnostics to enable automated method selection.
"""

from .core import MultireferenceDiagnostics
from .fast_screening import (
    calculate_homo_lumo_gap,
    calculate_spin_contamination,
    calculate_natural_orbital_occupations,
    calculate_fractional_occupation_density,
    calculate_bond_order_fluctuation,
)
from .reference_methods import (
    calculate_t1_diagnostic,
    calculate_d1_diagnostic,
    calculate_correlation_recovery,
    calculate_s_diagnostic,
)
from .decision_tree import IntelligentMethodSelector, ComputationalConstraint, AccuracyTarget
from .models.core_models import (
    MultireferenceCharacter,
    DiagnosticResult,
    DiagnosticMethod,
    DiagnosticConfig,
    SystemClassification,
    ComprehensiveDiagnosticResult,
)

__all__ = [
    # Main orchestrator
    "MultireferenceDiagnostics",
    # Fast screening methods
    "calculate_homo_lumo_gap",
    "calculate_spin_contamination", 
    "calculate_natural_orbital_occupations",
    "calculate_fractional_occupation_density",
    "calculate_bond_order_fluctuation",
    # Reference methods
    "calculate_t1_diagnostic",
    "calculate_d1_diagnostic",
    "calculate_correlation_recovery",
    "calculate_s_diagnostic",
    # Decision logic
    "IntelligentMethodSelector",
    # Data models
    "MultireferenceCharacter",
    "DiagnosticResult",
    "DiagnosticMethod",
    "DiagnosticConfig",
    "SystemClassification",
    "ComprehensiveDiagnosticResult",
    # Decision tree classes
    "ComputationalConstraint",
    "AccuracyTarget",
]