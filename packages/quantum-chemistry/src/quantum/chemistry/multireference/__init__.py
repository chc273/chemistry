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

# External method integrations (optional imports)
try:
    from .external import (
        DMRGMethod,
        AFQMCMethod,
        SelectedCIMethod,
        ExternalMethodInterface,
        ExternalSoftwareError,
    )
    _EXTERNAL_METHODS_AVAILABLE = True
except ImportError as e:
    # External methods require additional dependencies
    _EXTERNAL_METHODS_AVAILABLE = False
    _EXTERNAL_IMPORT_ERROR = str(e)

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

# Add external methods to __all__ if available
if _EXTERNAL_METHODS_AVAILABLE:
    __all__.extend([
        "DMRGMethod",
        "AFQMCMethod", 
        "SelectedCIMethod",
        "ExternalMethodInterface",
        "ExternalSoftwareError",
    ])