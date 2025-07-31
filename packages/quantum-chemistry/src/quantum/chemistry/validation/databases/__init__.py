"""
Database integration for quantum chemistry benchmarking.

This module provides access to standard quantum chemistry databases including:
- W4-11: High-accuracy thermochemistry dataset
- G2/97: Gaussian-2 test set
- HEAT: High-accuracy extrapolated ab initio thermochemistry
- TMC-151: Transition metal complexes dataset
- QUESTDB: Excited state benchmark database

All databases are accessed through a unified interface with automatic
data curation, validation, and standardization.
"""

from .base import DatabaseInterface, MolecularEntry, ReferenceDataEntry, PropertyType, BasisSetType
from .w4_11 import W4_11Database
from .g2_97 import G2_97Database
from .tmc_151 import TMC151Database
from .questdb import QuestDBDatabase
from .database_manager import DatabaseManager

__all__ = [
    "DatabaseInterface",
    "MolecularEntry", 
    "ReferenceDataEntry",
    "PropertyType",
    "BasisSetType",
    "W4_11Database",
    "G2_97Database",
    "TMC151Database",
    "QuestDBDatabase",
    "DatabaseManager",
]