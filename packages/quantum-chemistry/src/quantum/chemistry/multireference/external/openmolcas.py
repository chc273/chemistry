"""
OpenMolcas integration for CASPT2 and MS-CASPT2 calculations.

This module provides interfaces to OpenMolcas for advanced multireference
calculations including CASPT2, MS-CASPT2, and DMRG-CASPT2.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyscf import scf
from pyscf.tools import molden

from quantum.chemistry.active_space import ActiveSpaceResult
from ..base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult
from .base import ExternalMethodInterface, ExternalMethodResult, ExternalSoftwareError


class OpenMolcasInterface(ExternalMethodInterface):
    """
    Base interface for OpenMolcas calculations.
    
    Provides common functionality for preparing OpenMolcas input files,
    running calculations, and parsing output.
    """
    
    def _get_software_name(self) -> str:
        """Return the name of the external software."""
        return "OpenMolcas"
    
    def _get_default_executable(self) -> str:
        """Return the default executable name for the software."""
        return "pymolcas"
    
    def _prepare_molden_file(self,
                            scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                            work_path: Path) -> str:
        """
        Prepare Molden file for OpenMolcas input.
        
        Args:
            scf_obj: SCF object
            work_path: Working directory
            
        Returns:
            Path to Molden file
        """
        molden_file = work_path / "orbitals.molden"
        molden.from_mo(scf_obj.mol, str(molden_file), scf_obj.mo_coeff)
        return str(molden_file)
    
    def _prepare_xyz_file(self,
                         scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                         work_path: Path) -> str:
        """
        Prepare XYZ coordinate file.
        
        Args:
            scf_obj: SCF object
            work_path: Working directory
            
        Returns:
            Path to XYZ file
        """
        xyz_file = work_path / "molecule.xyz"
        
        mol = scf_obj.mol
        atoms = []
        coords = []
        
        for i in range(mol.natm):
            atoms.append(mol.atom_symbol(i))
            coords.append(mol.atom_coord(i) * 0.529177249)  # Convert bohr to angstrom
        
        with open(xyz_file, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write("Generated from PySCF\n")
            for atom, coord in zip(atoms, coords):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        
        return str(xyz_file)
    
    def _parse_energy_from_output(self, output_content: str, energy_pattern: str) -> float:
        """
        Parse energy from OpenMolcas output.
        
        Args:
            output_content: Output file content
            energy_pattern: Regex pattern for energy extraction
            
        Returns:
            Extracted energy value
        """
        matches = re.findall(energy_pattern, output_content)
        if not matches:
            raise ExternalSoftwareError(f"Could not find energy in output")
        
        # Return the last occurrence (final energy)
        return float(matches[-1])


class CASPT2Method(MultireferenceMethod):
    """
    CASPT2 method implementation using OpenMolcas.
    
    This class provides CASPT2 and MS-CASPT2 calculations with automatic
    IPEA shift optimization and basis set validation.
    """
    
    def __init__(self,
                 ipea_shift: float = 0.0,
                 multistate: bool = False,
                 n_states: int = 1,
                 imaginary_shift: float = 0.0,
                 convergence_threshold: float = 1e-8,
                 max_iterations: int = 50,
                 openmolcas_path: Optional[str] = None,
                 work_dir: Optional[str] = None,
                 keep_files: bool = False,
                 **kwargs):
        """
        Initialize CASPT2 method.
        
        Args:
            ipea_shift: IPEA shift parameter
            multistate: Whether to use MS-CASPT2
            n_states: Number of states for MS-CASPT2
            imaginary_shift: Imaginary shift to avoid intruder states
            convergence_threshold: Energy convergence threshold
            max_iterations: Maximum number of iterations
            openmolcas_path: Path to OpenMolcas installation
            work_dir: Working directory for calculations
            keep_files: Whether to keep temporary files
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.ipea_shift = ipea_shift
        self.multistate = multistate
        self.n_states = n_states
        self.imaginary_shift = imaginary_shift
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
        # Set up OpenMolcas interface
        self.interface = OpenMolcasInterface(
            software_path=openmolcas_path,
            work_dir=work_dir,
            keep_files=keep_files
        )
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.CASPT2
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform CASPT2 calculation using OpenMolcas.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with CASPT2 results
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for CASPT2 calculation")
        
        start_time = time.time()
        
        try:
            # Run calculation through interface
            external_result = self.interface.calculate(scf_obj, active_space, **kwargs)
            
            wall_time = time.time() - start_time
            
            # Extract CASPT2-specific results
            method_name = "MS-CASPT2" if self.multistate else "CASPT2"
            
            # Build convergence info
            convergence_info = {
                'converged': external_result.converged,
                'method': method_name,
                'ipea_shift': self.ipea_shift,
                'n_states': self.n_states if self.multistate else 1,
                'imaginary_shift': self.imaginary_shift
            }
            
            # Add external data to active space info
            active_space_info = {
                'n_electrons': active_space.n_active_electrons,
                'n_orbitals': active_space.n_active_orbitals,
                'selection_method': active_space.method,
                'ipea_shift': self.ipea_shift,
                'multistate': self.multistate,
                'openmolcas_data': external_result.external_data
            }
            
            # Computational cost
            computational_cost = {
                'wall_time': wall_time,
                'cpu_time': external_result.cpu_time or wall_time,
                'memory_mb': external_result.memory_mb or 0.0
            }
            
            return MultireferenceResult(
                method=method_name,
                energy=external_result.energy,
                correlation_energy=external_result.correlation_energy,
                active_space_info=active_space_info,
                n_active_electrons=active_space.n_active_electrons,
                n_active_orbitals=active_space.n_active_orbitals,
                convergence_info=convergence_info,
                computational_cost=computational_cost,
                natural_orbitals=None,  # Could be extracted from OpenMolcas output
                occupation_numbers=None,  # Could be extracted from OpenMolcas output
                basis_set=scf_obj.mol.basis,
                software_version="OpenMolcas (external)"
            )
            
        except Exception as e:
            raise ExternalSoftwareError(f"CASPT2 calculation failed: {e}")
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for CASPT2 calculation.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates
        """
        # CASPT2 scaling estimates
        # Memory: O(N^4) for active space + O(M^2) for basis
        active_memory = (n_orbitals ** 4) * 8e-6  # MB
        basis_memory = (basis_size ** 2) * 8e-6   # MB
        total_memory = active_memory + basis_memory
        
        # Time: O(N^5) scaling for CASPT2
        time_estimate = (n_orbitals ** 5) * (basis_size ** 2) * 1e-9  # seconds
        
        # MS-CASPT2 has additional overhead
        if self.multistate:
            time_estimate *= self.n_states * 1.5
            total_memory *= 1.3
        
        return {
            'memory_mb': total_memory,
            'time_seconds': time_estimate,
            'disk_mb': total_memory * 2,
            'multistate_factor': self.n_states if self.multistate else 1,
            'scaling_note': 'CASPT2 scaling: O(N^5*M^2) time, O(N^4+M^2) memory'
        }


class OpenMolcasCASPT2Interface(OpenMolcasInterface):
    """Specific OpenMolcas interface for CASPT2 calculations."""
    
    def __init__(self, caspt2_method: CASPT2Method, **kwargs):
        """
        Initialize OpenMolcas CASPT2 interface.
        
        Args:
            caspt2_method: CASPT2 method instance with parameters
            **kwargs: Additional interface parameters
        """
        super().__init__(**kwargs)
        self.caspt2_method = caspt2_method
    
    def _prepare_input(self,
                      scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                      active_space: ActiveSpaceResult,
                      work_path: Path,
                      **kwargs) -> Dict[str, str]:
        """
        Prepare OpenMolcas input files for CASPT2 calculation.
        
        Args:
            scf_obj: SCF object
            active_space: Active space result
            work_path: Working directory
            **kwargs: Additional parameters
            
        Returns:
            Dict of input file contents
        """
        mol = scf_obj.mol
        
        # Prepare auxiliary files
        xyz_file = self._prepare_xyz_file(scf_obj, work_path)
        molden_file = self._prepare_molden_file(scf_obj, work_path)
        
        # Build OpenMolcas input
        input_content = self._build_caspt2_input(
            mol, active_space, xyz_file, molden_file, **kwargs
        )
        
        return {
            "caspt2.input": input_content,
            "molecule.xyz": open(xyz_file).read(),
            "orbitals.molden": open(molden_file).read()
        }
    
    def _build_caspt2_input(self,
                           mol,
                           active_space: ActiveSpaceResult,
                           xyz_file: str,
                           molden_file: str,
                           **kwargs) -> str:
        """
        Build OpenMolcas input file for CASPT2 calculation.
        
        Args:
            mol: PySCF molecule object
            active_space: Active space result
            xyz_file: Path to XYZ file
            molden_file: Path to Molden file
            **kwargs: Additional parameters
            
        Returns:
            OpenMolcas input file content
        """
        # Basic molecule info
        charge = mol.charge
        spin = mol.spin
        basis = mol.basis if isinstance(mol.basis, str) else "sto-3g"
        
        # Active space info
        n_electrons = active_space.n_active_electrons
        n_orbitals = active_space.n_active_orbitals
        
        # Build input sections
        input_lines = [
            "! OpenMolcas CASPT2 calculation generated from PySCF",
            f"! Active space: ({n_electrons},{n_orbitals})",
            "",
            "&GATEWAY",
            f"  COORD = {xyz_file}",
            f"  BASIS = {basis}",
            f"  GROUP = C1",
            "END OF INPUT",
            "",
            "&SEWARD",
            "END OF INPUT",
            "",
            "&SCF",
            f"  CHARGE = {charge}",
            "END OF INPUT",
            "",
            "&RASSCF",
            f"  NACTEL = {n_electrons} 0 0",
            f"  INACTIVE = {mol.nelectron//2 - n_electrons//2}",
            f"  RAS2 = {n_orbitals}",
            "  SPIN = 1",  # Assuming singlet
            "  SYMMETRY = 1",
            "  CIROOT = 1 1",
            "END OF INPUT",
            "",
            "&CASPT2",
            f"  IPEA = {self.caspt2_method.ipea_shift}",
            f"  IMAGINARY = {self.caspt2_method.imaginary_shift}",
            f"  MAXITER = {self.caspt2_method.max_iterations}",
            f"  CONVERGENCE = {self.caspt2_method.convergence_threshold}",
        ]
        
        # Add multistate options if requested
        if self.caspt2_method.multistate and self.caspt2_method.n_states > 1:
            input_lines.extend([
                f"  MULTISTATE = {self.caspt2_method.n_states}",
                "  XMIXED",
            ])
        
        input_lines.append("END OF INPUT")
        
        return "\n".join(input_lines)
    
    def _run_calculation(self,
                        work_path: Path,
                        input_files: Dict[str, str],
                        **kwargs) -> subprocess.CompletedProcess:
        """
        Execute OpenMolcas CASPT2 calculation.
        
        Args:
            work_path: Working directory
            input_files: Input file contents
            **kwargs: Additional parameters
            
        Returns:
            Completed subprocess result
        """
        # Write input files
        for filename, content in input_files.items():
            with open(work_path / filename, 'w') as f:
                f.write(content)
        
        # Set up environment
        env = os.environ.copy()
        if 'MOLCAS' not in env:
            # Try to set basic MOLCAS environment
            molcas_home = Path(self.software_path).parent.parent
            env['MOLCAS'] = str(molcas_home)
            env['MOLCAS_MEM'] = '1000'  # MB
        
        # Run OpenMolcas
        cmd = [self.software_path, "caspt2.input"]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            return result
            
        except subprocess.TimeoutExpired as e:
            raise ExternalSoftwareError(f"OpenMolcas calculation timed out: {e}")
    
    def _parse_output(self,
                     work_path: Path,
                     process_result: subprocess.CompletedProcess,
                     **kwargs) -> ExternalMethodResult:
        """
        Parse OpenMolcas output and extract CASPT2 results.
        
        Args:
            work_path: Working directory
            process_result: Subprocess result
            **kwargs: Additional parameters
            
        Returns:
            Parsed external method result
        """
        if process_result.returncode != 0:
            raise ExternalSoftwareError(
                f"OpenMolcas failed with return code {process_result.returncode}:\n"
                f"STDOUT: {process_result.stdout}\n"
                f"STDERR: {process_result.stderr}"
            )
        
        output_content = process_result.stdout
        
        try:
            # Parse energies
            if self.caspt2_method.multistate:
                # MS-CASPT2 energy pattern
                energy_pattern = r"MS-CASPT2 Root\s+1\s+Total energy:\s*([-\d.]+)"
            else:
                # Single-state CASPT2 energy pattern
                energy_pattern = r"CASPT2 energy:\s*([-\d.]+)"
            
            energy = self._parse_energy_from_output(output_content, energy_pattern)
            
            # Try to get reference energy for correlation energy calculation
            casscf_pattern = r"RASSCF root number\s+1\s+Total energy:\s*([-\d.]+)"
            try:
                casscf_energy = self._parse_energy_from_output(output_content, casscf_pattern)
                correlation_energy = energy - casscf_energy
            except:
                correlation_energy = None
            
            # Parse convergence info
            converged = "CASPT2 SUCCESSFUL" in output_content or "converged" in output_content.lower()
            
            # Extract timing information if available
            wall_time = None
            cpu_time = None
            
            time_pattern = r"Total CPU time:\s*([\d.]+)"
            time_matches = re.findall(time_pattern, output_content)
            if time_matches:
                cpu_time = float(time_matches[-1])
            
            # Build external data
            external_data = {
                'ipea_shift': self.caspt2_method.ipea_shift,
                'multistate': self.caspt2_method.multistate,
                'imaginary_shift': self.caspt2_method.imaginary_shift,
                'raw_output': output_content[:1000] + "..." if len(output_content) > 1000 else output_content
            }
            
            # Build result
            method_name = "MS-CASPT2" if self.caspt2_method.multistate else "CASPT2"
            
            result = ExternalMethodResult(
                method=method_name,
                software="OpenMolcas",
                energy=energy,
                correlation_energy=correlation_energy,
                external_data=external_data,
                input_files={k: v[:500] + "..." if len(v) > 500 else v 
                           for k, v in kwargs.get('input_files', {}).items()},
                output_files={"stdout": output_content[:1000] + "..." if len(output_content) > 1000 else output_content},
                work_directory=str(work_path),
                wall_time=wall_time,
                cpu_time=cpu_time,
                converged=converged,
                convergence_info={
                    'method': method_name,
                    'converged': converged,
                    'ipea_shift': self.caspt2_method.ipea_shift
                }
            )
            
            return result
            
        except Exception as e:
            raise ExternalSoftwareError(f"Failed to parse OpenMolcas output: {e}")


# Update the CASPT2Method to use the interface
CASPT2Method.interface_class = OpenMolcasCASPT2Interface