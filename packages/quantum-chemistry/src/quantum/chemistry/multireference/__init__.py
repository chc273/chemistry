"""
Multireference quantum chemistry methods with automated benchmarking.

This module provides a unified interface for various multireference methods including:
- CASSCF/NEVPT2/CASPT2 
- Selected CI (SHCI/CIPSI)
- AF-QMC (Auxiliary Field Quantum Monte Carlo)
- DMRG (Density Matrix Renormalization Group)

Features:
- Seamless integration with active space selection methods
- Automated method selection and parameter optimization
- Comprehensive benchmarking and validation infrastructure
- Cross-method comparison and analysis tools
"""

from .base import (
    MultireferenceMethod,
    MultireferenceMethodType,
    MultireferenceResult,
    MethodSelector,
)
from .methods import (
    CASSCFMethod,
    NEVPT2Method,
    CASPT2Method,
)
from .workflows import (
    MultireferenceWorkflow,
)

__version__ = "0.1.0"

__all__ = [
    "MultireferenceMethod",
    "MultireferenceMethodType",
    "MultireferenceResult", 
    "MethodSelector",
    "CASSCFMethod",
    "NEVPT2Method",
    "CASPT2Method",
    "MultireferenceWorkflow",
]