"""
Auxiliary Field Quantum Monte Carlo (AF-QMC) integration.

This module provides interfaces to AF-QMC software packages including
ipie (modern Python-based AFQMC) and QMCPACK for high-accuracy
multireference calculations.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from ..base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult
from .base import ExternalMethodInterface, ExternalMethodResult, ExternalSoftwareError


class AFQMCMethod(MultireferenceMethod):
    """
    Auxiliary Field Quantum Monte Carlo (AF-QMC) method implementation.
    
    This class provides phaseless AF-QMC calculations using ipie or QMCPACK
    backends with automatic trial wavefunction generation and statistical
    error analysis.
    """
    
    def __init__(self,
                 backend: str = "ipie",
                 n_walkers: int = 100,
                 n_steps: int = 1000,
                 timestep: float = 0.01,
                 trial_wavefunction: str = "casscf",
                 phaseless_constraint: bool = True,
                 target_error: float = 0.001,  # Hartree
                 max_walltime: int = 3600,  # seconds
                 seed: Optional[int] = None,
                 gpu_enabled: bool = False,
                 **kwargs):
        """
        Initialize AF-QMC method.
        
        Args:
            backend: QMC backend ("ipie" or "qmcpack")
            n_walkers: Number of random walkers
            n_steps: Number of QMC steps
            timestep: QMC timestep
            trial_wavefunction: Trial wavefunction type ("casscf", "uhf", "ghf")
            phaseless_constraint: Whether to use phaseless constraint
            target_error: Target statistical error in Hartree
            max_walltime: Maximum wall time in seconds
            seed: Random seed for reproducibility
            gpu_enabled: Whether to use GPU acceleration
            **kwargs: Additional parameters
        """
        # Set attributes before calling super().__init__()
        self.backend = backend.lower()
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.timestep = timestep
        self.trial_wavefunction = trial_wavefunction
        self.phaseless_constraint = phaseless_constraint
        self.target_error = target_error
        self.max_walltime = max_walltime
        self.seed = seed
        self.gpu_enabled = gpu_enabled
        
        # Validate backend
        self._validate_backend()
        
        # Call parent initialization after setting attributes
        super().__init__(**kwargs)
    
    def _validate_backend(self):
        """Validate that the requested QMC backend is available."""
        if self.backend == "ipie":
            try:
                import ipie
                self.qmc_module = ipie
            except ImportError:
                raise ExternalSoftwareError(
                    "ipie not found. Please install with: pip install ipie"
                )
        elif self.backend == "qmcpack":
            # QMCPACK requires external executable
            import shutil
            qmcpack_exe = shutil.which("qmcpack")
            if qmcpack_exe is None:
                raise ExternalSoftwareError(
                    "QMCPACK executable not found in PATH. "
                    "Please install QMCPACK and ensure it's in your PATH."
                )
            self.qmcpack_path = qmcpack_exe
        else:
            raise ValueError(f"Unknown AF-QMC backend: {self.backend}")
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.AFQMC
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform AF-QMC calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with AF-QMC results
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for AF-QMC calculation")
        
        start_time = time.time()
        
        try:
            if self.backend == "ipie":
                qmc_result = self._run_ipie_calculation(scf_obj, active_space, **kwargs)
            elif self.backend == "qmcpack":
                qmc_result = self._run_qmcpack_calculation(scf_obj, active_space, **kwargs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            wall_time = time.time() - start_time
            
            # Extract results
            energy = qmc_result['energy']
            energy_error = qmc_result.get('energy_error', 0.0)
            correlation_energy = energy - scf_obj.e_tot
            
            # Build convergence info
            convergence_info = {
                'converged': qmc_result.get('converged', True),
                'method': f"AF-QMC ({self.backend})",
                'n_walkers': self.n_walkers,
                'n_steps': self.n_steps,
                'timestep': self.timestep,
                'phaseless': self.phaseless_constraint,
                'statistical_error': energy_error
            }
            
            # Build active space info with QMC-specific data
            active_space_info = {
                'n_electrons': active_space.n_active_electrons,
                'n_orbitals': active_space.n_active_orbitals,
                'selection_method': active_space.method,
                'trial_wavefunction': self.trial_wavefunction,
                'backend': self.backend,
                'qmc_data': qmc_result.get('qmc_data', {})
            }
            
            # Computational cost
            computational_cost = {
                'wall_time': wall_time,
                'cpu_time': qmc_result.get('cpu_time', wall_time),
                'memory_mb': qmc_result.get('memory_mb', 0.0),
                'n_walkers': self.n_walkers,
                'n_steps': self.n_steps
            }
            
            return MultireferenceResult(
                method=f"AF-QMC ({self.backend})",
                energy=energy,
                correlation_energy=correlation_energy,
                active_space_info=active_space_info,
                n_active_electrons=active_space.n_active_electrons,
                n_active_orbitals=active_space.n_active_orbitals,
                convergence_info=convergence_info,
                computational_cost=computational_cost,
                natural_orbitals=None,  # Not typically computed in AF-QMC
                occupation_numbers=None,  # Not typically computed in AF-QMC
                basis_set=scf_obj.mol.basis,
                software_version=f"{self.backend}-{self._get_backend_version()}"
            )
            
        except Exception as e:
            raise ExternalSoftwareError(f"AF-QMC calculation failed: {e}")
    
    def _run_ipie_calculation(self,
                             scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                             active_space: ActiveSpaceResult,
                             **kwargs) -> Dict[str, Any]:
        """
        Run AF-QMC calculation using ipie.
        
        Args:
            scf_obj: SCF object
            active_space: Active space result
            **kwargs: Additional parameters
            
        Returns:
            Dict with QMC calculation results
        """
        try:
            from ipie.systems.generic import Generic
            from ipie.hamiltonians.generic import Generic as GenericHamiltonian
            from ipie.trial_wavefunction.single_det import SingleDet
            from ipie.qmc.afqmc import AFQMC
            from ipie.utils.mpi import get_shared_comm
            
            # Prepare system and Hamiltonian
            system = self._prepare_ipie_system(scf_obj, active_space)
            hamiltonian = self._prepare_ipie_hamiltonian(scf_obj, active_space)
            
            # Prepare trial wavefunction
            trial = self._prepare_ipie_trial(scf_obj, active_space, system)
            
            # Set up QMC parameters
            qmc_params = {
                'dt': self.timestep,
                'n_walkers': self.n_walkers,
                'n_steps_per_block': 10,
                'n_blocks': self.n_steps // 10,
                'stabilize_freq': 5,
                'seed': self.seed or np.random.randint(0, 2**31),
                'pop_control_freq': 5,
                'pop_control_method': 'pair_branch'
            }
            
            # Run AFQMC
            comm = get_shared_comm()
            qmc = AFQMC(comm, system, hamiltonian, trial, **qmc_params)
            
            # Execute calculation
            qmc.run()
            
            # Extract results
            energies = qmc.estimators.estimators['energy'].data
            energy = np.mean(energies)
            energy_error = np.std(energies) / np.sqrt(len(energies))
            
            # Check convergence
            converged = energy_error < self.target_error
            
            result = {
                'energy': energy,
                'energy_error': energy_error,
                'converged': converged,
                'n_samples': len(energies),
                'qmc_data': {
                    'energy_history': energies.tolist(),
                    'statistical_error': energy_error,
                    'acceptance_rate': getattr(qmc, 'acceptance_rate', 0.0)
                }
            }
            
            return result
            
        except ImportError as e:
            raise ExternalSoftwareError(f"ipie import failed: {e}")
        except Exception as e:
            raise ExternalSoftwareError(f"ipie calculation failed: {e}")
    
    def _prepare_ipie_system(self,
                            scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                            active_space: ActiveSpaceResult) -> Any:
        """Prepare ipie system object."""
        from ipie.systems.generic import Generic
        
        # Use full system for now - could be optimized for active space
        nup = (scf_obj.mol.nelectron + scf_obj.mol.spin) // 2
        ndown = scf_obj.mol.nelectron - nup
        
        system = Generic(nelec=(nup, ndown))
        return system
    
    def _prepare_ipie_hamiltonian(self,
                                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                                 active_space: ActiveSpaceResult) -> Any:
        """Prepare ipie Hamiltonian object."""
        from ipie.hamiltonians.generic import Generic as GenericHamiltonian
        
        # Get integrals
        h1e = scf_obj.get_hcore()
        h2e = scf_obj.mol.intor('int2e')
        
        # Transform to MO basis
        mo_coeff = scf_obj.mo_coeff
        h1e_mo = np.einsum('pi,pq,qj->ij', mo_coeff, h1e, mo_coeff)
        h2e_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', mo_coeff, mo_coeff, h2e, mo_coeff, mo_coeff)
        
        hamiltonian = GenericHamiltonian(h1e_mo, h2e_mo)
        return hamiltonian
    
    def _prepare_ipie_trial(self,
                           scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                           active_space: ActiveSpaceResult,
                           system: Any) -> Any:
        """Prepare ipie trial wavefunction."""
        from ipie.trial_wavefunction.single_det import SingleDet
        
        if self.trial_wavefunction.lower() == "casscf":
            # Use CASSCF orbitals if available
            mo_coeff = active_space.orbital_coefficients
        else:
            # Use SCF orbitals
            mo_coeff = scf_obj.mo_coeff
        
        # Create single determinant trial
        trial = SingleDet(mo_coeff, system.nup, system.ndown)
        return trial
    
    def _run_qmcpack_calculation(self,
                                scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                                active_space: ActiveSpaceResult,
                                **kwargs) -> Dict[str, Any]:
        """
        Run AF-QMC calculation using QMCPACK.
        
        This is a placeholder implementation. Full QMCPACK integration
        would require generating input files and parsing output.
        """
        # Placeholder implementation
        raise NotImplementedError(
            "QMCPACK integration not yet implemented. "
            "Use backend='ipie' for now."
        )
    
    def _get_backend_version(self) -> str:
        """Get backend version string."""
        try:
            if self.backend == "ipie":
                import ipie
                return getattr(ipie, '__version__', 'unknown')
            elif self.backend == "qmcpack":
                # Would need to query QMCPACK version
                return 'external'
        except:
            return 'unknown'
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for AF-QMC calculation.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates
        """
        # AF-QMC scaling estimates
        # Memory: O(N_walkers * N_orbitals^2)
        memory_per_walker = (basis_size ** 2) * 8e-6  # MB per walker
        total_memory = memory_per_walker * self.n_walkers
        
        # Time: Linear in n_steps, quadratic in basis size, linear in walkers
        time_per_step = (basis_size ** 2) * self.n_walkers * 1e-9  # seconds
        total_time = time_per_step * self.n_steps
        
        # Statistical error scales as 1/sqrt(n_steps * n_walkers)
        n_samples = self.n_steps * self.n_walkers
        statistical_error = 0.01 / np.sqrt(n_samples)  # Rough estimate in Hartree
        
        return {
            'memory_mb': total_memory,
            'time_seconds': total_time,
            'disk_mb': total_memory * 0.1,  # Minimal disk usage
            'n_walkers': self.n_walkers,
            'n_steps': self.n_steps,
            'estimated_statistical_error': statistical_error,
            'scaling_note': 'AF-QMC scaling: O(N^2*Nw*Ns) time, O(N^2*Nw) memory'
        }
    
    def get_recommended_parameters(self,
                                  system_type: str,
                                  active_space_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Get recommended AF-QMC parameters for given system.
        
        Args:
            system_type: Type of chemical system
            active_space_size: (n_electrons, n_orbitals) tuple
            
        Returns:
            Dict of recommended parameters
        """
        n_electrons, n_orbitals = active_space_size
        
        # Base parameters
        params = {
            'n_walkers': 100,
            'n_steps': 1000,
            'timestep': 0.01,
            'trial_wavefunction': 'casscf'
        }
        
        # Adjust for system size
        if n_orbitals <= 10:
            params['n_walkers'] = 50
            params['n_steps'] = 500
        elif n_orbitals <= 20:
            params['n_walkers'] = 100
            params['n_steps'] = 1000
        else:
            params['n_walkers'] = 200
            params['n_steps'] = 2000
        
        # System-specific adjustments
        if system_type == "transition_metal":
            params['n_walkers'] *= 2  # Need more statistics for TM systems
            params['timestep'] = 0.005  # Smaller timestep for stability
            params['trial_wavefunction'] = 'casscf'  # Better trial for TM
        
        if "bond_breaking" in system_type.lower():
            params['n_steps'] *= 2  # Need more statistics for difficult cases
            params['trial_wavefunction'] = 'casscf'
        
        return params


# AF-QMC method type is already defined in base.py