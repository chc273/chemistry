"""
Base classes and interfaces for multireference methods.

This module defines the core abstractions that all multireference methods
must implement, providing a unified interface for method selection,
execution, and result handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult


class MultireferenceMethodType(str, Enum):
    """Available multireference method types."""
    
    CASSCF = "casscf"
    NEVPT2 = "nevpt2" 
    CASPT2 = "caspt2"
    SHCI = "shci"
    CIPSI = "cipsi"
    AFQMC = "afqmc"
    DMRG = "dmrg"
    DMRG_NEVPT2 = "dmrg_nevpt2"
    DMRG_CASPT2 = "dmrg_caspt2"


class MultireferenceResult(BaseModel):
    """
    Standardized result container for multireference calculations.
    
    This class provides a unified interface for storing and accessing
    results from different multireference methods.
    """
    
    method: str = Field(..., description="Method used for calculation")
    energy: float = Field(..., description="Total energy in Hartree")
    correlation_energy: Optional[float] = Field(None, description="Correlation energy contribution")
    
    # Active space information
    active_space_info: Dict[str, Any] = Field(..., description="Active space details")
    n_active_electrons: int = Field(..., description="Number of active electrons")
    n_active_orbitals: int = Field(..., description="Number of active orbitals")
    
    # Calculation properties
    properties: Optional[Dict[str, float]] = Field(None, description="Additional molecular properties")
    convergence_info: Dict[str, Any] = Field(..., description="Convergence details")
    
    # Computational details
    computational_cost: Dict[str, float] = Field(..., description="Resource usage statistics")
    uncertainty: Optional[float] = Field(None, description="Statistical uncertainty estimate")
    
    # Method-specific data
    reference_weights: Optional[np.ndarray] = Field(None, description="Reference configuration weights")
    natural_orbitals: Optional[np.ndarray] = Field(None, description="Natural orbital coefficients")
    occupation_numbers: Optional[np.ndarray] = Field(None, description="Natural orbital occupations")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Calculation timestamp")
    software_version: Optional[str] = Field(None, description="Software version used")
    basis_set: Optional[str] = Field(None, description="Basis set used")
    
    class Config:
        arbitrary_types_allowed = True


class MultireferenceMethod(ABC):
    """
    Abstract base class for multireference quantum chemistry methods.
    
    All multireference method implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize method with configuration parameters."""
        self.config = kwargs
        self.method_type = self._get_method_type()
    
    @abstractmethod
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return the method type identifier."""
        pass
    
    @abstractmethod
    def calculate(self, 
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform multireference calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Method-specific parameters
            
        Returns:
            MultireferenceResult: Calculation results
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, 
                     n_electrons: int,
                     n_orbitals: int, 
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for given system size.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional system parameters
            
        Returns:
            Dict with cost estimates (time, memory, disk)
        """
        pass
    
    def validate_input(self, 
                      scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                      active_space: ActiveSpaceResult) -> bool:
        """
        Validate input parameters for calculation.
        
        Args:
            scf_obj: SCF object to validate
            active_space: Active space to validate
            
        Returns:
            bool: True if inputs are valid
        """
        # Basic validation - can be overridden by subclasses
        if not scf_obj.converged:
            return False
        
        if active_space.n_active_orbitals <= 0:
            return False
            
        if active_space.n_active_electrons <= 0:
            return False
            
        return True
    
    def get_recommended_parameters(self, 
                                 system_type: str,
                                 active_space_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Get recommended parameters for given system type and active space.
        
        Args:
            system_type: Type of chemical system ('organic', 'transition_metal', etc.)
            active_space_size: (n_electrons, n_orbitals) tuple
            
        Returns:
            Dict of recommended parameters
        """
        # Default implementation - should be overridden by subclasses
        return {}


class MethodSelector:
    """
    Automated method selection based on system characteristics.
    
    This class provides intelligent method recommendation based on
    molecular system properties, active space characteristics, and
    computational constraints.
    """
    
    def __init__(self):
        """Initialize method selector."""
        self._method_registry = {}
        self._system_classifiers = {}
    
    def register_method(self, 
                       method_type: MultireferenceMethodType,
                       method_class: type,
                       **metadata):
        """
        Register a multireference method implementation.
        
        Args:
            method_type: Method type identifier
            method_class: Method implementation class
            **metadata: Additional method metadata
        """
        self._method_registry[method_type] = {
            'class': method_class,
            'metadata': metadata
        }
    
    def recommend_method(self,
                        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                        active_space: ActiveSpaceResult,
                        system_type: Optional[str] = None,
                        accuracy_target: str = "standard",
                        cost_constraint: str = "moderate") -> MultireferenceMethodType:
        """
        Recommend optimal method based on system characteristics.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            system_type: Chemical system type hint
            accuracy_target: Desired accuracy level
            cost_constraint: Computational cost constraint
            
        Returns:
            Recommended method type
        """
        # System classification
        if system_type is None:
            system_type = self._classify_system(scf_obj, active_space)
        
        # Method selection logic
        n_active = active_space.n_active_orbitals
        
        if system_type == "organic" and n_active <= 12:
            if accuracy_target == "high":
                return MultireferenceMethodType.NEVPT2
            else:
                return MultireferenceMethodType.CASSCF
                
        elif system_type == "transition_metal":
            if n_active <= 10:
                return MultireferenceMethodType.NEVPT2
            else:
                return MultireferenceMethodType.DMRG_NEVPT2
                
        elif n_active > 16:
            if cost_constraint == "high":
                return MultireferenceMethodType.DMRG
            else:
                return MultireferenceMethodType.SHCI
                
        else:
            return MultireferenceMethodType.NEVPT2
    
    def _classify_system(self,
                        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                        active_space: ActiveSpaceResult) -> str:
        """
        Classify chemical system type based on molecular properties.
        
        Args:
            scf_obj: SCF object
            active_space: Active space result
            
        Returns:
            System type classification
        """
        mol = scf_obj.mol
        
        # Check for transition metals
        transition_metals = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'}
        
        elements = {mol.atom_symbol(i) for i in range(mol.natm)}
        
        if elements & transition_metals:
            return "transition_metal"
        elif len(elements & {'C', 'H', 'N', 'O'}) >= 2:
            return "organic"
        else:
            return "general"
    
    def get_available_methods(self) -> List[MultireferenceMethodType]:
        """Return list of registered methods."""
        return list(self._method_registry.keys())