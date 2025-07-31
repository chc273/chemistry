"""
Production-ready CASPT2 method implementation using OpenMolcas.

This module provides a complete CASPT2 implementation with Docker support,
comprehensive error handling, and validation against other methods.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from quantum.chemistry.fcidump import to_openmolcas_fcidump, detect_orbital_symmetries
from ..base import ExternalMethodInterface, ExternalMethodResult, ExternalSoftwareError
from ...base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult
from .input_generator import OpenMolcasInputGenerator, OpenMolcasParameters
from .output_parser import OpenMolcasOutputParser


class CASPT2Method(MultireferenceMethod):
    """
    Production-ready CASPT2 method using OpenMolcas.
    
    This implementation provides:
    - Automatic parameter optimization for different system types
    - Docker-based execution with fallback to native installation
    - Comprehensive error handling and diagnostics
    - Cross-validation with other multireference methods
    - Support for both single-state and multi-state CASPT2
    """
    
    def __init__(
        self,
        ipea_shift: Optional[float] = None,
        multistate: bool = False,
        n_states: int = 1,
        imaginary_shift: Optional[float] = None,
        convergence_threshold: float = 1e-8,
        max_iterations: int = 50,
        openmolcas_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        keep_files: bool = False,
        use_container: Optional[bool] = None,
        template_dir: Optional[str] = None,
        auto_optimize_parameters: bool = True,
        **kwargs
    ):
        """
        Initialize CASPT2 method with comprehensive parameter control.
        
        Args:
            ipea_shift: IPEA shift parameter (None for automatic selection)
            multistate: Whether to use MS-CASPT2
            n_states: Number of states for MS-CASPT2
            imaginary_shift: Imaginary shift to avoid intruder states (None for automatic)
            convergence_threshold: Energy convergence threshold
            max_iterations: Maximum number of iterations
            openmolcas_path: Path to OpenMolcas installation
            work_dir: Working directory for calculations
            keep_files: Whether to keep temporary files
            use_container: Force container usage (None for auto-detect)
            template_dir: Directory containing custom templates
            auto_optimize_parameters: Automatically optimize parameters for system type
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Core calculation parameters
        self.ipea_shift = ipea_shift
        self.multistate = multistate
        self.n_states = n_states
        self.imaginary_shift = imaginary_shift
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.auto_optimize_parameters = auto_optimize_parameters
        
        # Initialize OpenMolcas interface
        self.interface = OpenMolcasCASPT2Interface(
            software_path=openmolcas_path,
            work_dir=work_dir,
            keep_files=keep_files,
            use_container=use_container,
            caspt2_method=self
        )
        
        # Initialize input generator and output parser
        self.input_generator = OpenMolcasInputGenerator(template_dir)
        self.output_parser = OpenMolcasOutputParser()
    
    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        return MultireferenceMethodType.CASPT2
    
    def calculate(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs
    ) -> MultireferenceResult:
        """
        Perform CASPT2 calculation with automatic parameter optimization.
        
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
            # Optimize parameters if requested
            calc_params = self._prepare_calculation_parameters(
                scf_obj, active_space, **kwargs
            )
            
            # Run calculation through interface
            external_result = self.interface.calculate(
                scf_obj, active_space, **calc_params
            )
            
            wall_time = time.time() - start_time
            
            # Build standardized result
            result = self._build_multireference_result(
                external_result, scf_obj, active_space, wall_time
            )
            
            # Add calculation diagnostics
            self._add_calculation_diagnostics(result, external_result, calc_params)
            
            return result
            
        except Exception as e:
            raise ExternalSoftwareError(f"CASPT2 calculation failed: {e}")
    
    def _prepare_calculation_parameters(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare and optimize calculation parameters."""
        mol = scf_obj.mol
        
        # Start with provided parameters
        params = {
            "multistate": self.multistate,
            "n_states": self.n_states,
            "caspt2_convergence": self.convergence_threshold,
            "caspt2_max_iter": self.max_iterations,
        }
        
        # Add explicit parameters if provided
        if self.ipea_shift is not None:
            params["ipea_shift"] = self.ipea_shift
        if self.imaginary_shift is not None:
            params["imaginary_shift"] = self.imaginary_shift
        
        # Auto-optimize parameters if requested
        if self.auto_optimize_parameters:
            optimized_params = self._optimize_parameters_for_system(
                mol, active_space
            )
            # Only use optimized values if not explicitly set
            for key, value in optimized_params.items():
                if key not in params or params[key] is None:
                    params[key] = value
        
        # Override with user-provided kwargs
        params.update(kwargs)
        
        return params
    
    def _optimize_parameters_for_system(
        self,
        mol,
        active_space: ActiveSpaceResult
    ) -> Dict[str, Any]:
        """
        Optimize CASPT2 parameters based on molecular system characteristics.
        
        Args:
            mol: PySCF molecule object
            active_space: Active space selection result
            
        Returns:
            Dict of optimized parameters
        """
        # Classify system type
        system_type = self._classify_system(mol)
        n_active = active_space.n_active_orbitals
        
        # Base parameters
        params = {
            "ipea_shift": 0.0,
            "imaginary_shift": 0.0,
            "memory_mb": 2000,
        }
        
        # System-specific optimizations
        if system_type == "transition_metal":
            # Transition metal complexes often need IPEA shift and intruder state handling
            params.update({
                "ipea_shift": 0.25,  # Standard IPEA shift for TM systems
                "imaginary_shift": 0.1 if n_active > 8 else 0.0,
                "memory_mb": 4000,
                "caspt2_convergence": 1e-7,  # Slightly looser for difficult convergence
            })
        elif system_type == "organic":
            # Organic molecules typically converge well without shifts
            params.update({
                "ipea_shift": 0.0,
                "imaginary_shift": 0.0,
                "memory_mb": 2000,
                "caspt2_convergence": 1e-8,
            })
        elif system_type == "biradical":
            # Biradical systems may need careful parameter tuning
            params.update({
                "ipea_shift": 0.0,  # Start without IPEA for biradicals
                "imaginary_shift": 0.05,  # Small shift to handle near-degeneracies
                "memory_mb": 3000,
            })
        
        # Active space size adjustments
        if n_active > 12:
            params["memory_mb"] *= 2
            params["caspt2_max_iter"] = 100
            if n_active > 16:
                params["imaginary_shift"] = max(params["imaginary_shift"], 0.1)
        
        # Basis set considerations
        basis = mol.basis if isinstance(mol.basis, str) else "sto-3g"
        if any(b in basis.lower() for b in ["cc-", "aug-", "def2-"]):
            # Larger basis sets need more memory and may have convergence issues
            params["memory_mb"] *= 1.5
            if "aug-" in basis.lower():
                params["imaginary_shift"] = max(params["imaginary_shift"], 0.05)
        
        return params
    
    def _classify_system(self, mol) -> str:
        """Classify molecular system for parameter optimization."""
        elements = {mol.atom_symbol(i) for i in range(mol.natm)}
        
        # Transition metals
        transition_metals = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'
        }
        
        if elements & transition_metals:
            return "transition_metal"
        
        # Check for potential biradical character (high spin)
        if mol.spin > 0 and len(elements & {'C', 'N', 'O'}) >= 2:
            return "biradical"
        
        # Organic systems
        if len(elements & {'C', 'H', 'N', 'O'}) >= 2:
            return "organic"
        
        return "general"
    
    def _build_multireference_result(
        self,
        external_result: ExternalMethodResult,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        wall_time: float
    ) -> MultireferenceResult:
        """Build standardized MultireferenceResult from external result."""
        method_name = "MS-CASPT2" if self.multistate else "CASPT2"
        
        # Build convergence info
        convergence_info = external_result.convergence_info.copy()
        convergence_info.update({
            "method": method_name,
            "ipea_shift": external_result.external_data.get("ipea_shift", 0.0),
            "imaginary_shift": external_result.external_data.get("imaginary_shift", 0.0),
            "multistate": self.multistate,
            "n_states": self.n_states if self.multistate else 1,
        })
        
        # Build active space info
        active_space_info = {
            "n_electrons": active_space.n_active_electrons,
            "n_orbitals": active_space.n_active_orbitals,
            "selection_method": active_space.method,
            "openmolcas_data": external_result.external_data,
        }
        
        # Add multi-state information if available
        if "state_energies" in external_result.external_data:
            active_space_info["state_energies"] = external_result.external_data["state_energies"]
            active_space_info["state_weights"] = external_result.external_data.get("state_weights")
        
        # Computational cost
        computational_cost = {
            "wall_time": wall_time,
            "cpu_time": external_result.cpu_time or wall_time,
            "memory_mb": external_result.memory_mb or 0.0,
        }
        
        # Extract natural orbitals and occupations if available
        natural_occupations = external_result.external_data.get("natural_occupations")
        occupation_numbers = np.array(natural_occupations) if natural_occupations else None
        
        return MultireferenceResult(
            method=method_name,
            energy=external_result.energy,
            correlation_energy=external_result.correlation_energy,
            active_space_info=active_space_info,
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            convergence_info=convergence_info,
            computational_cost=computational_cost,
            natural_orbitals=None,  # Could be extracted from checkpoint files
            occupation_numbers=occupation_numbers,
            basis_set=scf_obj.mol.basis,
            software_version=f"OpenMolcas (external)",
            uncertainty=external_result.error_bars.get("energy") if external_result.error_bars else None
        )
    
    def _add_calculation_diagnostics(
        self,
        result: MultireferenceResult,
        external_result: ExternalMethodResult,
        calc_params: Dict[str, Any]
    ):
        """Add diagnostic information to the result."""
        # Add warnings from OpenMolcas output
        warnings = external_result.external_data.get("warnings", [])
        errors = external_result.external_data.get("errors", [])
        
        # Add parameter-specific diagnostics
        if calc_params.get("ipea_shift", 0.0) > 0.5:
            warnings.append(f"Large IPEA shift used: {calc_params['ipea_shift']}")
        
        if calc_params.get("imaginary_shift", 0.0) > 0.2:
            warnings.append(f"Large imaginary shift used: {calc_params['imaginary_shift']}")
        
        # Check for convergence issues
        if not external_result.converged:
            errors.append("CASPT2 calculation did not converge")
        
        # Store diagnostics in active space info
        if warnings:
            result.active_space_info["warnings"] = warnings
        if errors:
            result.active_space_info["errors"] = errors
    
    def estimate_cost(
        self,
        n_electrons: int,
        n_orbitals: int,
        basis_size: int,
        **kwargs
    ) -> Dict[str, float]:
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
        # CASPT2 scaling: O(N^5 * M^2) for active space size N, basis size M
        active_memory = (n_orbitals ** 4) * 8e-6  # MB for active space integrals
        basis_memory = (basis_size ** 2) * 8e-6   # MB for basis transformations
        total_memory = active_memory + basis_memory
        
        # Time scaling estimate
        time_estimate = (n_orbitals ** 5) * (basis_size ** 2) * 1e-9  # seconds
        
        # MS-CASPT2 has additional computational overhead
        if self.multistate and self.n_states > 1:
            time_estimate *= self.n_states * 1.2
            total_memory *= 1.1
        
        # IPEA shift and imaginary shift add minimal cost
        if kwargs.get("ipea_shift", 0.0) > 0:
            time_estimate *= 1.05
        if kwargs.get("imaginary_shift", 0.0) > 0:
            time_estimate *= 1.02
        
        return {
            "memory_mb": total_memory,
            "time_seconds": time_estimate,
            "disk_mb": total_memory * 2.5,  # Temporary files and checkpoints
            "multistate_factor": self.n_states if self.multistate else 1,
            "scaling_info": {
                "active_space_scaling": "O(N^5)",
                "basis_scaling": "O(M^2)",
                "combined_scaling": "O(N^5 * M^2)",
                "memory_scaling": "O(N^4 + M^2)"
            }
        }
    
    def validate_input(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult
    ) -> bool:
        """
        Validate input parameters for CASPT2 calculation.
        
        Args:
            scf_obj: SCF object to validate
            active_space: Active space to validate
            
        Returns:
            bool: True if inputs are valid
        """
        # Basic validation from parent class
        if not super().validate_input(scf_obj, active_space):
            return False
        
        # CASPT2-specific validations
        if active_space.n_active_orbitals > 20:
            # Very large active spaces may be problematic
            print(f"Warning: Large active space ({active_space.n_active_orbitals} orbitals) "
                  f"may lead to very expensive CASPT2 calculation")
        
        # Check for reasonable system size
        mol = scf_obj.mol
        if mol.natm > 50:
            print(f"Warning: Large molecule ({mol.natm} atoms) may require "
                  f"significant computational resources")
        
        # Validate multistate parameters
        if self.multistate:
            if self.n_states < 2:
                print("Warning: Multistate CASPT2 requested but n_states < 2")
                return False
            if self.n_states > 10:
                print("Warning: Very large number of states may be computationally expensive")
        
        return True
    
    def get_recommended_parameters(
        self,
        system_type: str,
        active_space_size: tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Get recommended parameters for given system type and active space.
        
        Args:
            system_type: Type of chemical system
            active_space_size: (n_electrons, n_orbitals) tuple
            
        Returns:
            Dict of recommended parameters
        """
        n_electrons, n_orbitals = active_space_size
        
        if system_type == "transition_metal":
            return {
                "ipea_shift": 0.25,
                "imaginary_shift": 0.1 if n_orbitals > 8 else 0.0,
                "convergence_threshold": 1e-7,
                "max_iterations": 100,
                "memory_mb": 4000 if n_orbitals > 10 else 2000,
            }
        elif system_type == "organic":
            return {
                "ipea_shift": 0.0,
                "imaginary_shift": 0.0,
                "convergence_threshold": 1e-8,
                "max_iterations": 50,
                "memory_mb": 2000,
            }
        elif system_type == "biradical":
            return {
                "ipea_shift": 0.0,
                "imaginary_shift": 0.05,
                "convergence_threshold": 1e-7,
                "max_iterations": 75,
                "memory_mb": 3000,
            }
        else:
            # General defaults
            return {
                "ipea_shift": 0.0,
                "imaginary_shift": 0.0,
                "convergence_threshold": 1e-8,
                "max_iterations": 50,
                "memory_mb": 2000,
            }


class OpenMolcasCASPT2Interface(ExternalMethodInterface):
    """Specialized OpenMolcas interface for CASPT2 calculations."""
    
    def __init__(self, caspt2_method: CASPT2Method, **kwargs):
        """
        Initialize OpenMolcas CASPT2 interface.
        
        Args:
            caspt2_method: CASPT2 method instance
            **kwargs: Additional interface parameters
        """
        super().__init__(**kwargs)
        self.caspt2_method = caspt2_method
    
    def _get_software_name(self) -> str:
        """Return the name of the external software."""
        return "OpenMolcas"
    
    def _get_default_executable(self) -> str:
        """Return the default executable name."""
        return "pymolcas"
    
    def _prepare_input(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        work_path: Path,
        **kwargs
    ) -> Dict[str, str]:
        """
        Prepare comprehensive input files for OpenMolcas CASPT2.
        
        Args:
            scf_obj: SCF object
            active_space: Active space result
            work_path: Working directory
            **kwargs: Additional parameters
            
        Returns:
            Dict of input file contents
        """
        # Prepare auxiliary files
        xyz_file = self._prepare_xyz_file(scf_obj, work_path)
        
        # Generate main input file
        calculation_type = "ms_caspt2" if self.caspt2_method.multistate else "caspt2"
        input_content = self.caspt2_method.input_generator.generate_input(
            scf_obj, active_space, calculation_type, "molecule.xyz", **kwargs
        )
        
        # Prepare FCIDUMP file if needed (for debugging/validation)
        fcidump_file = work_path / "fcidump"
        try:
            symmetry_labels = detect_orbital_symmetries(scf_obj, active_space.orbital_coefficients)
            to_openmolcas_fcidump(
                active_space, scf_obj, str(fcidump_file), 
                symmetry_labels=symmetry_labels
            )
            fcidump_content = fcidump_file.read_text()
        except Exception as e:
            # FCIDUMP generation is optional for validation
            fcidump_content = f"! FCIDUMP generation failed: {e}"
        
        return {
            "caspt2.input": input_content,
            "molecule.xyz": Path(xyz_file).read_text(),
            "fcidump": fcidump_content,
        }
    
    def _prepare_xyz_file(self, scf_obj: Union[scf.hf.SCF, scf.uhf.UHF], work_path: Path) -> str:
        """Prepare XYZ coordinate file."""
        xyz_file = work_path / "molecule.xyz"
        mol = scf_obj.mol
        
        with open(xyz_file, 'w') as f:
            f.write(f"{mol.natm}\n")
            f.write("Generated from PySCF for OpenMolcas CASPT2\n")
            
            for i in range(mol.natm):
                symbol = mol.atom_symbol(i)
                # Convert from bohr to angstrom
                coord = mol.atom_coord(i) * 0.529177249
                f.write(f"{symbol:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        
        return str(xyz_file)
    
    def _run_calculation(
        self,
        work_path: Path,
        input_files: Dict[str, str],
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Execute OpenMolcas CASPT2 calculation with proper environment setup.
        
        Args:
            work_path: Working directory
            input_files: Input file contents
            **kwargs: Additional parameters
            
        Returns:
            Completed subprocess result
        """
        # Write all input files
        for filename, content in input_files.items():
            with open(work_path / filename, 'w') as f:
                f.write(content)
        
        # Set up environment variables
        env = os.environ.copy()
        
        if self.use_container:
            # Docker-based execution
            cmd = self._build_docker_command(work_path, **kwargs)
        else:
            # Native execution
            cmd = [self.software_path, "caspt2.input"]
            
            # Set OpenMolcas environment variables
            if 'MOLCAS' not in env:
                molcas_home = Path(self.software_path).parent.parent
                env['MOLCAS'] = str(molcas_home)
            
            # Memory allocation
            memory_mb = kwargs.get("memory_mb", 2000)
            env['MOLCAS_MEM'] = str(memory_mb)
            
            # Temporary directory
            env['MOLCAS_WORKDIR'] = str(work_path)
        
        # Execute calculation
        try:
            result = subprocess.run(
                cmd,
                cwd=work_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 7200)  # 2 hour default timeout
            )
            return result
            
        except subprocess.TimeoutExpired as e:
            raise ExternalSoftwareError(f"OpenMolcas CASPT2 calculation timed out: {e}")
        except subprocess.CalledProcessError as e:
            raise ExternalSoftwareError(f"OpenMolcas execution failed: {e}")
    
    def _build_docker_command(self, work_path: Path, **kwargs) -> List[str]:
        """Build Docker command for containerized execution."""
        memory_mb = kwargs.get("memory_mb", 2000)
        
        cmd = [
            "docker", "run", "--rm",
            f"--memory={memory_mb}m",
            f"-v", f"{work_path}:/opt/workdir",
            "-w", "/opt/workdir",
            "quantum-chemistry/openmolcas:latest",
            "pymolcas", "caspt2.input"
        ]
        
        return cmd
    
    def _parse_output(
        self,
        work_path: Path,
        process_result: subprocess.CompletedProcess,
        **kwargs
    ) -> ExternalMethodResult:
        """
        Parse OpenMolcas output and build external method result.
        
        Args:
            work_path: Working directory
            process_result: Subprocess result
            **kwargs: Additional parameters
            
        Returns:
            Parsed external method result
        """
        if process_result.returncode != 0:
            error_msg = (
                f"OpenMolcas failed with return code {process_result.returncode}\n"
                f"STDOUT: {process_result.stdout[:1000]}...\n"
                f"STDERR: {process_result.stderr[:1000]}..."
            )
            raise ExternalSoftwareError(error_msg)
        
        # Parse output using the output parser
        calculation_type = "ms_caspt2" if self.caspt2_method.multistate else "caspt2"
        parsed_results = self.caspt2_method.output_parser.parse_output(
            process_result.stdout, calculation_type
        )
        
        # Convert to external method result
        method_name = "MS-CASPT2" if self.caspt2_method.multistate else "CASPT2"
        external_result = self.caspt2_method.output_parser.to_external_result(
            parsed_results,
            method=method_name,
            input_files={k: v[:500] + "..." if len(v) > 500 else v 
                        for k, v in kwargs.get('input_files', {}).items()},
            output_files={"stdout": process_result.stdout[:2000] + "..." 
                         if len(process_result.stdout) > 2000 else process_result.stdout},
            work_directory=str(work_path)
        )
        
        return external_result