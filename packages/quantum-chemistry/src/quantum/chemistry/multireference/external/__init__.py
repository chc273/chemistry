"""
External multireference method integrations.

This module provides interfaces to external quantum chemistry software
for advanced multireference methods including:

- DMRG (Density Matrix Renormalization Group) via block2
- CASPT2/MS-CASPT2 via OpenMolcas
- AF-QMC (Auxiliary Field Quantum Monte Carlo) via ipie/QMCPACK
- Selected CI (SHCI/CIPSI) via Dice/Quantum Package
"""

from .base import (
    ExternalMethodInterface,
    ExternalMethodResult,
    ExternalSoftwareError,
    SoftwareNotFoundError,
)

from .dmrg import DMRGMethod
from .openmolcas import CASPT2Method, OpenMolcasInterface
from .afqmc import AFQMCMethod
from .selected_ci import SelectedCIMethod, SHCIInterface, CIPSIInterface

# Import Docker-based external method runner
from ...external import ExternalMethodRunner

__all__ = [
    "ExternalMethodInterface",
    "ExternalMethodResult", 
    "ExternalSoftwareError",
    "SoftwareNotFoundError",
    "DMRGMethod",
    "CASPT2Method",
    "OpenMolcasInterface",
    "AFQMCMethod",
    "SelectedCIMethod",
    "SHCIInterface",
    "CIPSIInterface",
    "ExternalMethodRunner",
]