"""
CASSCF, NEVPT2, and CASPT2 method implementations.

This module provides implementations of Complete Active Space Self-Consistent Field
(CASSCF) and its post-SCF corrections NEVPT2 and CASPT2 using PySCF as the backend.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pyscf import mcscf, mrpt
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from ..base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult


class CASSCFMethod(MultireferenceMethod):
    """
    Complete Active Space Self-Consistent Field (CASSCF) implementation.
    
    This class provides a unified interface for CASSCF calculations using PySCF,
    with automatic integration with active space selection results.
    """
    
    def __init__(self, 
                 max_cycle: int = 100,
                 conv_tol: float = 1e-7,
                 conv_tol_grad: float = 1e-4,
                 **kwargs):
        """
        Initialize CASSCF method.
        
        Args:
            max_cycle: Maximum number of CASSCF iterations
            conv_tol: Energy convergence threshold
            conv_tol_grad: Gradient convergence threshold
            **kwargs: Additional PySCF CASSCF parameters
        """
        super().__init__(**kwargs)
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.CASSCF
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform CASSCF calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with CASSCF results
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for CASSCF calculation")
        
        # Set up CASSCF calculation
        cas = mcscf.CASSCF(scf_obj, 
                          active_space.n_active_orbitals,
                          active_space.n_active_electrons)
        
        # Configure CASSCF parameters
        cas.max_cycle_macro = self.max_cycle
        cas.conv_tol = self.conv_tol
        cas.conv_tol_grad = self.conv_tol_grad
        
        # Apply any additional configuration
        for key, value in kwargs.items():
            if hasattr(cas, key):
                setattr(cas, key, value)
        
        # Run CASSCF calculation
        cas.kernel(active_space.orbital_coefficients)
        
        # Extract results
        cas_energy = cas.e_tot
        correlation_energy = cas_energy - scf_obj.e_tot
        
        # Natural orbitals and occupation numbers
        natural_orbitals, occupation_numbers = mcscf.addons.make_natural_orbitals(cas)
        
        # Convergence information
        convergence_info = {
            'converged': cas.converged,
            'iterations': getattr(cas, 'niter', None),
            'energy_gradient': getattr(cas, 'de', None),
            'orbital_gradient': getattr(cas, 'max_orb_grad', None)
        }
        
        # Computational cost (basic timing would need to be added)
        computational_cost = {
            'wall_time': 0.0,  # Would need actual timing
            'cpu_time': 0.0,
            'memory_mb': 0.0
        }
        
        return MultireferenceResult(
            method="CASSCF",
            energy=cas_energy,
            correlation_energy=correlation_energy,
            active_space_info={
                'n_electrons': active_space.n_active_electrons,
                'n_orbitals': active_space.n_active_orbitals,
                'selection_method': active_space.method
            },
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            convergence_info=convergence_info,
            computational_cost=computational_cost,
            natural_orbitals=natural_orbitals,
            occupation_numbers=occupation_numbers,
            basis_set=scf_obj.mol.basis,
            software_version="PySCF"
        )
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for CASSCF calculation.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates
        """
        # Rough scaling estimates for CASSCF
        # Memory scales as O(N^4) for active space, O(M^2) for basis
        active_memory = (n_orbitals ** 4) * 8e-6  # MB
        basis_memory = (basis_size ** 2) * 8e-6   # MB
        total_memory = active_memory + basis_memory
        
        # Time scales roughly as O(N^6) for active space
        time_estimate = (n_orbitals ** 6) * 1e-6  # seconds
        
        # Disk usage for intermediates
        disk_usage = total_memory * 2  # rough estimate
        
        return {
            'memory_mb': total_memory,
            'time_seconds': time_estimate,
            'disk_mb': disk_usage
        }
    
    def get_recommended_parameters(self,
                                 system_type: str,
                                 active_space_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Get recommended CASSCF parameters for given system.
        
        Args:
            system_type: Type of chemical system
            active_space_size: (n_electrons, n_orbitals) tuple
            
        Returns:
            Dict of recommended parameters
        """
        n_electrons, n_orbitals = active_space_size
        
        params = {
            'max_cycle': 100,
            'conv_tol': 1e-7,
            'conv_tol_grad': 1e-4
        }
        
        # Adjust for difficult systems
        if system_type == "transition_metal":
            params['max_cycle'] = 150
            params['conv_tol'] = 1e-6
            
        if n_orbitals > 10:
            params['max_cycle'] = 200
            
        return params


class NEVPT2Method(CASSCFMethod):
    """
    N-Electron Valence State Perturbation Theory (NEVPT2) implementation.
    
    This class extends CASSCF to include NEVPT2 post-SCF correction
    for dynamic correlation recovery.
    """
    
    def __init__(self, 
                 nevpt2_type: str = "sc",
                 **kwargs):
        """
        Initialize NEVPT2 method.
        
        Args:
            nevpt2_type: Type of NEVPT2 ("sc" for strongly contracted, 
                        "pc" for partially contracted)
            **kwargs: Additional CASSCF parameters
        """
        super().__init__(**kwargs)
        self.nevpt2_type = nevpt2_type
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.NEVPT2
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform CASSCF+NEVPT2 calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with NEVPT2 results
        """
        # First perform CASSCF calculation
        casscf_result = super().calculate(scf_obj, active_space, **kwargs)
        
        # Set up CASSCF object for NEVPT2
        cas = mcscf.CASSCF(scf_obj,
                          active_space.n_active_orbitals,
                          active_space.n_active_electrons)
        
        # Configure and run CASSCF
        cas.max_cycle_macro = self.max_cycle
        cas.conv_tol = self.conv_tol
        cas.conv_tol_grad = self.conv_tol_grad
        cas.kernel(active_space.orbital_coefficients)
        
        # Run NEVPT2 calculation
        if self.nevpt2_type.lower() == "sc":
            pt2_correction = mrpt.NEVPT(cas).kernel()
        else:
            # Partially contracted NEVPT2 (if available)
            pt2_correction = mrpt.NEVPT(cas).kernel()
        
        # Calculate total NEVPT2 energy and correlation energy
        nevpt2_energy = cas.e_tot + pt2_correction
        nevpt2_correlation = nevpt2_energy - scf_obj.e_tot
        
        # Update result with NEVPT2 information
        result = MultireferenceResult(
            method="CASSCF+NEVPT2",
            energy=nevpt2_energy,
            correlation_energy=nevpt2_correlation,
            active_space_info={
                'n_electrons': active_space.n_active_electrons,
                'n_orbitals': active_space.n_active_orbitals,
                'selection_method': active_space.method,
                'pt2_correction': pt2_correction,
                'nevpt2_type': self.nevpt2_type
            },
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            convergence_info=casscf_result.convergence_info,
            computational_cost=casscf_result.computational_cost,
            natural_orbitals=casscf_result.natural_orbitals,
            occupation_numbers=casscf_result.occupation_numbers,
            basis_set=scf_obj.mol.basis,
            software_version="PySCF"
        )
        
        return result
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for NEVPT2 calculation.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates
        """
        # CASSCF cost
        casscf_cost = super().estimate_cost(n_electrons, n_orbitals, basis_size, **kwargs)
        
        # NEVPT2 additional cost (roughly O(N^5) scaling)
        nevpt2_time = (n_orbitals ** 5) * (basis_size ** 2) * 1e-9
        nevpt2_memory = (n_orbitals ** 3) * (basis_size ** 2) * 8e-6
        
        return {
            'memory_mb': casscf_cost['memory_mb'] + nevpt2_memory,
            'time_seconds': casscf_cost['time_seconds'] + nevpt2_time,
            'disk_mb': casscf_cost['disk_mb'] * 1.5
        }


class CASPT2Method(CASSCFMethod):
    """
    Complete Active Space Second-order Perturbation Theory (CASPT2) implementation.
    
    This class extends CASSCF to include CASPT2 post-SCF correction.
    Note: This is a placeholder for future implementation as PySCF has limited CASPT2 support.
    """
    
    def __init__(self, 
                 ipea_shift: float = 0.0,
                 **kwargs):
        """
        Initialize CASPT2 method.
        
        Args:
            ipea_shift: IPEA shift parameter for CASPT2
            **kwargs: Additional CASSCF parameters
        """
        super().__init__(**kwargs)
        self.ipea_shift = ipea_shift
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.CASPT2
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform CASSCF+CASPT2 calculation.
        
        Note: This is a placeholder implementation. Full CASPT2 would require
        external software integration (OpenMolcas, ORCA).
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with CASPT2 results
        """
        # For now, return CASSCF result with CASPT2 placeholder
        casscf_result = super().calculate(scf_obj, active_space, **kwargs)
        
        # Placeholder for CASPT2 energy calculation
        # In practice, this would interface with OpenMolcas or ORCA
        caspt2_energy = casscf_result.energy  # Placeholder
        
        result = MultireferenceResult(
            method="CASSCF+CASPT2",
            energy=caspt2_energy,
            correlation_energy=casscf_result.correlation_energy,
            active_space_info={
                'n_electrons': active_space.n_active_electrons,
                'n_orbitals': active_space.n_active_orbitals,
                'selection_method': active_space.method,
                'ipea_shift': self.ipea_shift,
                'note': 'CASPT2 requires external software integration'
            },
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            convergence_info=casscf_result.convergence_info,
            computational_cost=casscf_result.computational_cost,
            natural_orbitals=casscf_result.natural_orbitals,
            occupation_numbers=casscf_result.occupation_numbers,
            basis_set=scf_obj.mol.basis,
            software_version="PySCF"
        )
        
        return result