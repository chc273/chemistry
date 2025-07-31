"""
Base classes and interfaces for external multireference method integrations.

This module provides the abstract base class and common utilities for
integrating external quantum chemistry software packages with the
unified multireference interface.
"""

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from ..base import MultireferenceResult


class ExternalSoftwareError(Exception):
    """Base exception for external software integration errors."""
    pass


class SoftwareNotFoundError(ExternalSoftwareError):
    """Exception raised when required external software is not found."""
    pass


class ExternalMethodResult(BaseModel):
    """Results from external multireference method calculations."""
    
    method: str = Field(description="External method name")
    software: str = Field(description="External software used")
    energy: float = Field(description="Total energy")
    correlation_energy: Optional[float] = Field(
        default=None, description="Correlation energy"
    )
    
    # External method specific data
    external_data: Dict[str, Any] = Field(
        default_factory=dict, description="Method-specific external data"
    )
    
    # File management
    input_files: Dict[str, str] = Field(
        default_factory=dict, description="Input file contents"
    )
    output_files: Dict[str, str] = Field(
        default_factory=dict, description="Output file contents"
    )
    work_directory: Optional[str] = Field(
        default=None, description="Working directory path"
    )
    
    # Performance metrics
    wall_time: Optional[float] = Field(default=None, description="Wall clock time (seconds)")
    cpu_time: Optional[float] = Field(default=None, description="CPU time (seconds)")
    memory_mb: Optional[float] = Field(default=None, description="Peak memory usage (MB)")
    
    # Convergence and quality metrics
    converged: bool = Field(default=False, description="Calculation converged")
    convergence_info: Dict[str, Any] = Field(
        default_factory=dict, description="Convergence details"
    )
    error_bars: Optional[Dict[str, float]] = Field(
        default=None, description="Statistical error bars (for QMC methods)"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class ExternalMethodInterface(ABC):
    """
    Abstract base class for external multireference method interfaces.
    
    This class provides the common framework for integrating external
    quantum chemistry software packages with standardized input/output
    handling, error management, and result processing.
    """
    
    def __init__(self, 
                 software_path: Optional[str] = None,
                 work_dir: Optional[str] = None,
                 keep_files: bool = False,
                 use_container: Optional[bool] = None,
                 skip_validation: bool = False,
                 **kwargs):
        """
        Initialize external method interface.
        
        Args:
            software_path: Path to external software executable
            work_dir: Working directory for calculations (None for temp dir)
            keep_files: Whether to keep temporary files after calculation
            use_container: Force container usage (None for auto-detect)
            skip_validation: Skip software validation (for testing)
            **kwargs: Additional method-specific parameters
        """
        self.software_path = software_path
        self.work_dir = work_dir
        self.keep_files = keep_files
        self.force_container = use_container
        self.use_container = False  # Will be set during validation
        self.method_params = kwargs
        self.skip_validation = skip_validation
        
        # Validate software availability unless skipped
        if not skip_validation:
            self._validate_software()
    
    @abstractmethod
    def _get_software_name(self) -> str:
        """Return the name of the external software."""
        pass
    
    @abstractmethod
    def _get_default_executable(self) -> str:
        """Return the default executable name for the software."""
        pass
    
    def _validate_software(self):
        """Validate that the external software is available."""
        if self.software_path is None:
            # First try to find software in PATH
            executable = self._get_default_executable()
            self.software_path = shutil.which(executable)
            
            # If not found, check for containerized installation
            if self.software_path is None:
                containerized_path = self._check_containerized_installation()
                if containerized_path:
                    self.software_path = containerized_path
                    self.use_container = True
                else:
                    raise SoftwareNotFoundError(
                        f"{self._get_software_name()} executable '{executable}' "
                        f"not found in PATH or Docker containers. Please install "
                        f"{self._get_software_name()} natively or build the Docker container."
                    )
            else:
                self.use_container = False
        
        elif not os.path.exists(self.software_path):
            raise SoftwareNotFoundError(
                f"{self._get_software_name()} executable not found at: "
                f"{self.software_path}"
            )
        else:
            self.use_container = False
    
    def _check_containerized_installation(self) -> Optional[str]:
        """
        Check if the software is available as a Docker container.
        
        Returns:
            Docker run command if container is available, None otherwise
        """
        if self.force_container is False:
            return None
        
        # Check if Docker is available
        if not shutil.which("docker"):
            return None
        
        # Map software names to container images
        container_map = {
            "Block2": "quantum-chemistry/block2:latest",
            "OpenMolcas": "quantum-chemistry/openmolcas:latest", 
            "Dice (SHCI)": "quantum-chemistry/dice:latest",
            "Quantum Package (CIPSI)": "quantum-chemistry/quantum-package:latest"
        }
        
        software_name = self._get_software_name()
        container_image = container_map.get(software_name)
        
        if not container_image:
            return None
        
        # Check if container image exists
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", container_image],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Return docker run command
                return f"docker run --rm -v $(pwd):/opt/workdir -w /opt/workdir {container_image}"
            else:
                return None
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _setup_work_directory(self) -> Path:
        """Set up working directory for calculation."""
        if self.work_dir is not None:
            work_path = Path(self.work_dir)
            work_path.mkdir(parents=True, exist_ok=True)
        else:
            work_path = Path(tempfile.mkdtemp(
                prefix=f"{self._get_software_name().lower()}_"
            ))
        
        return work_path
    
    def _cleanup_work_directory(self, work_path: Path):
        """Clean up working directory after calculation."""
        if not self.keep_files and self.work_dir is None:
            # Only remove temp directories, not user-specified ones
            try:
                shutil.rmtree(work_path)
            except OSError as e:
                # Log warning but don't fail
                print(f"Warning: Could not remove work directory {work_path}: {e}")
    
    @abstractmethod
    def _prepare_input(self,
                      scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                      active_space: ActiveSpaceResult,
                      work_path: Path,
                      **kwargs) -> Dict[str, str]:
        """
        Prepare input files for external calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            work_path: Working directory path
            **kwargs: Additional calculation parameters
            
        Returns:
            Dict mapping filenames to file contents
        """
        pass
    
    @abstractmethod
    def _run_calculation(self, 
                        work_path: Path,
                        input_files: Dict[str, str],
                        **kwargs) -> subprocess.CompletedProcess:
        """
        Execute the external calculation.
        
        Args:
            work_path: Working directory path
            input_files: Input file contents
            **kwargs: Additional run parameters
            
        Returns:
            Completed subprocess result
        """
        pass
    
    @abstractmethod
    def _parse_output(self,
                     work_path: Path,
                     process_result: subprocess.CompletedProcess,
                     **kwargs) -> ExternalMethodResult:
        """
        Parse calculation output and extract results.
        
        Args:
            work_path: Working directory path
            process_result: Completed subprocess result
            **kwargs: Additional parsing parameters
            
        Returns:
            Parsed external method result
        """
        pass
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform external multireference calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            StandardizedMultireferenceResult with external method results
        """
        work_path = self._setup_work_directory()
        
        try:
            # Prepare input files
            input_files = self._prepare_input(scf_obj, active_space, work_path, **kwargs)
            
            # Run external calculation
            process_result = self._run_calculation(work_path, input_files, **kwargs)
            
            # Parse results
            external_result = self._parse_output(work_path, process_result, **kwargs)
            
            # Convert to standard multireference result
            mr_result = self._convert_to_mr_result(
                external_result, scf_obj, active_space
            )
            
            return mr_result
            
        except subprocess.TimeoutExpired as e:
            raise ExternalSoftwareError(f"Calculation timed out: {e}")
        except subprocess.CalledProcessError as e:
            raise ExternalSoftwareError(
                f"Calculation failed with return code {e.returncode}: {e.stderr}"
            )
        except Exception as e:
            raise ExternalSoftwareError(f"Unexpected error during calculation: {e}")
        finally:
            # Always clean up
            self._cleanup_work_directory(work_path)
    
    def _convert_to_mr_result(self,
                             external_result: ExternalMethodResult,
                             scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                             active_space: ActiveSpaceResult) -> MultireferenceResult:
        """
        Convert external method result to standard MultireferenceResult.
        
        Args:
            external_result: Result from external method
            scf_obj: SCF object
            active_space: Active space result
            
        Returns:
            Standardized MultireferenceResult
        """
        # Extract computational cost information
        computational_cost = {
            'wall_time': external_result.wall_time or 0.0,
            'cpu_time': external_result.cpu_time or 0.0,
            'memory_mb': external_result.memory_mb or 0.0
        }
        
        # Build active space info with external method details
        active_space_info = {
            'n_electrons': active_space.n_active_electrons,
            'n_orbitals': active_space.n_active_orbitals,
            'selection_method': active_space.method,
            'external_software': external_result.software,
            'external_data': external_result.external_data
        }
        
        # Add statistical error information for QMC methods
        if external_result.error_bars:
            active_space_info['error_bars'] = external_result.error_bars
        
        return MultireferenceResult(
            method=external_result.method,
            energy=external_result.energy,
            correlation_energy=external_result.correlation_energy,
            active_space_info=active_space_info,
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            convergence_info=external_result.convergence_info,
            computational_cost=computational_cost,
            natural_orbitals=None,  # May be populated by specific implementations
            occupation_numbers=None,  # May be populated by specific implementations
            basis_set=scf_obj.mol.basis,
            software_version=f"{external_result.software} (external)"
        )
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for external method calculation.
        
        This base implementation provides generic scaling estimates.
        Subclasses should override with method-specific estimates.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates
        """
        # Generic estimates - subclasses should provide better estimates
        memory_mb = (n_orbitals ** 4) * 8e-6 + (basis_size ** 2) * 8e-6
        time_seconds = (n_orbitals ** 6) * 1e-6
        disk_mb = memory_mb * 3  # Rough estimate for intermediates
        
        return {
            'memory_mb': memory_mb,
            'time_seconds': time_seconds,
            'disk_mb': disk_mb,
            'method_note': 'Generic estimate - actual costs may vary significantly'
        }
    
    def get_version_info(self) -> Dict[str, str]:
        """
        Get version information for the external software.
        
        Returns:
            Dict with version information
        """
        try:
            result = subprocess.run(
                [self.software_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                'software': self._get_software_name(),
                'executable': self.software_path,
                'version_output': result.stdout.strip(),
                'available': True
            }
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return {
                'software': self._get_software_name(),
                'executable': self.software_path,
                'version_output': 'Could not determine version',
                'available': False
            }