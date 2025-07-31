"""
OpenMolcas output parsing utilities.

This module provides robust parsing of OpenMolcas output files with proper
error handling, energy extraction, and convergence analysis.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from ..base import ExternalMethodResult


class OpenMolcasResults(BaseModel):
    """
    Structured container for OpenMolcas calculation results.
    
    This class provides validated storage for all extracted data from
    OpenMolcas output files with proper type checking and units.
    """
    
    # Energy results
    scf_energy: Optional[float] = Field(None, description="SCF energy in Hartree")
    casscf_energy: Optional[float] = Field(None, description="CASSCF energy in Hartree")
    caspt2_energy: Optional[float] = Field(None, description="CASPT2 energy in Hartree")
    correlation_energy: Optional[float] = Field(None, description="Correlation energy in Hartree")
    
    # Multi-state results
    state_energies: Optional[List[float]] = Field(
        None, description="All state energies for MS-CASPT2"
    )
    state_weights: Optional[List[float]] = Field(
        None, description="State weights for MS-CASPT2"
    )
    
    # Convergence information
    scf_converged: bool = Field(False, description="SCF convergence status")
    casscf_converged: bool = Field(False, description="CASSCF convergence status")
    caspt2_converged: bool = Field(False, description="CASPT2 convergence status")
    
    scf_iterations: Optional[int] = Field(None, description="Number of SCF iterations")
    casscf_iterations: Optional[int] = Field(None, description="Number of CASSCF iterations")
    caspt2_iterations: Optional[int] = Field(None, description="Number of CASPT2 iterations")
    
    # Orbital and electronic structure information
    active_space_size: Optional[Tuple[int, int]] = Field(
        None, description="(n_electrons, n_orbitals) in active space"
    )
    natural_occupations: Optional[List[float]] = Field(
        None, description="Natural orbital occupation numbers"
    )
    
    # Performance metrics
    wall_time: Optional[float] = Field(None, description="Total wall time in seconds")
    cpu_time: Optional[float] = Field(None, description="Total CPU time in seconds")
    memory_usage: Optional[float] = Field(None, description="Peak memory usage in MB")
    
    # Error and warning information
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    # Raw output sections for debugging
    raw_output: Optional[str] = Field(None, description="Raw output content (truncated)")


class OpenMolcasOutputParser:
    """
    Comprehensive OpenMolcas output file parser.
    
    This class provides robust parsing of OpenMolcas output files with support
    for CASSCF, CASPT2, and MS-CASPT2 calculations. It includes proper error
    handling and extraction of convergence information.
    """
    
    def __init__(self):
        """Initialize output parser with regex patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for output parsing."""
        # Energy patterns
        self.patterns = {
            # SCF energies
            "scf_energy": re.compile(
                r"Total SCF energy:\s*([-+]?\d+\.\d+)", re.IGNORECASE
            ),
            
            # CASSCF energies
            "casscf_energy": re.compile(
                r"RASSCF root number\s+1\s+Total energy:\s*([-+]?\d+\.\d+)"
            ),
            "casscf_final": re.compile(
                r"Final state energy\(ies\):\s*\n.*?Total energy:\s*([-+]?\d+\.\d+)",
                re.MULTILINE | re.DOTALL
            ),
            
            # CASPT2 energies
            "caspt2_energy": re.compile(
                r"CASPT2 Total Energy:\s*([-+]?\d+\.\d+)"
            ),
            "caspt2_reference": re.compile(
                r"Reference energy:\s*([-+]?\d+\.\d+)"
            ),
            
            # MS-CASPT2 energies
            "ms_caspt2_energies": re.compile(
                r"MS-CASPT2 Root\s+(\d+)\s+Total energy:\s*([-+]?\d+\.\d+)"
            ),
            "ms_caspt2_weights": re.compile(
                r"MS-CASPT2 Root\s+(\d+)\s+Weight:\s*([-+]?\d+\.\d+)"
            ),
            
            # Convergence patterns
            "scf_convergence": re.compile(
                r"SCF has converged", re.IGNORECASE
            ),
            "casscf_convergence": re.compile(
                r"RASSCF has converged|CASSCF.*converged", re.IGNORECASE
            ),
            "caspt2_convergence": re.compile(
                r"CASPT2.*converged|PT2 has converged", re.IGNORECASE
            ),
            
            # Iteration counts
            "scf_iterations": re.compile(
                r"SCF converged in\s+(\d+)\s+iterations"
            ),
            "casscf_iterations": re.compile(
                r"RASSCF converged in\s+(\d+)\s+iterations"
            ),
            "caspt2_iterations": re.compile(
                r"CASPT2 converged in\s+(\d+)\s+iterations"
            ),
            
            # Active space information
            "active_space": re.compile(
                r"Number of active electrons:\s+(\d+).*?"
                r"Number of active orbitals:\s+(\d+)",
                re.MULTILINE | re.DOTALL
            ),
            
            # Natural occupations
            "natural_occupations": re.compile(
                r"Natural orbitals and occupation numbers.*?\n((?:\s*\d+\s+[-+]?\d+\.\d+.*?\n)+)",
                re.MULTILINE | re.DOTALL
            ),
            
            # Timing information
            "wall_time": re.compile(
                r"Total wall time:\s*([\d.]+)", re.IGNORECASE
            ),
            "cpu_time": re.compile(
                r"Total CPU time:\s*([\d.]+)", re.IGNORECASE
            ),
            
            # Memory usage
            "memory_usage": re.compile(
                r"Maximum memory used:\s*([\d.]+)\s*MB", re.IGNORECASE
            ),
            
            # Error and warning patterns
            "warnings": re.compile(
                r"WARNING:?\s*(.*?)(?:\n|$)", re.IGNORECASE | re.MULTILINE
            ),
            "errors": re.compile(
                r"ERROR:?\s*(.*?)(?:\n|$)", re.IGNORECASE | re.MULTILINE
            ),
            "fatal_errors": re.compile(
                r"FATAL ERROR|Fatal error|Segmentation fault", re.IGNORECASE
            ),
        }
    
    def parse_output(
        self, 
        output_content: str,
        calculation_type: str = "caspt2"
    ) -> OpenMolcasResults:
        """
        Parse OpenMolcas output and extract results.
        
        Args:
            output_content: Full output file content
            calculation_type: Type of calculation ("casscf", "caspt2", "ms_caspt2")
            
        Returns:
            Parsed results in structured format
        """
        results = OpenMolcasResults()
        
        # Extract basic energy information
        self._extract_energies(output_content, results, calculation_type)
        
        # Extract convergence information
        self._extract_convergence(output_content, results)
        
        # Extract orbital and electronic structure information
        self._extract_orbital_info(output_content, results)
        
        # Extract performance metrics
        self._extract_performance(output_content, results)
        
        # Extract warnings and errors
        self._extract_diagnostics(output_content, results)
        
        # Store truncated raw output for debugging
        if len(output_content) > 5000:
            results.raw_output = output_content[:2500] + "\n...\n" + output_content[-2500:]
        else:
            results.raw_output = output_content
        
        # Validate and post-process results
        self._validate_results(results, calculation_type)
        
        return results
    
    def _extract_energies(
        self, 
        output_content: str, 
        results: OpenMolcasResults,
        calculation_type: str
    ):
        """Extract energy values from output."""
        # SCF energy
        scf_match = self.patterns["scf_energy"].search(output_content)
        if scf_match:
            results.scf_energy = float(scf_match.group(1))
        
        # CASSCF energy
        casscf_match = (
            self.patterns["casscf_final"].search(output_content) or
            self.patterns["casscf_energy"].search(output_content)
        )
        if casscf_match:
            results.casscf_energy = float(casscf_match.group(1))
        
        # CASPT2 energies
        if calculation_type in ["caspt2", "ms_caspt2"]:
            if calculation_type == "ms_caspt2":
                # Multi-state CASPT2
                energy_matches = self.patterns["ms_caspt2_energies"].findall(output_content)
                if energy_matches:
                    # Sort by root number and extract energies
                    sorted_energies = sorted(energy_matches, key=lambda x: int(x[0]))
                    results.state_energies = [float(energy) for _, energy in sorted_energies]
                    results.caspt2_energy = results.state_energies[0]  # Ground state
                
                # Extract state weights if available
                weight_matches = self.patterns["ms_caspt2_weights"].findall(output_content)
                if weight_matches:
                    sorted_weights = sorted(weight_matches, key=lambda x: int(x[0]))
                    results.state_weights = [float(weight) for _, weight in sorted_weights]
            else:
                # Single-state CASPT2
                caspt2_match = self.patterns["caspt2_energy"].search(output_content)
                if caspt2_match:
                    results.caspt2_energy = float(caspt2_match.group(1))
        
        # Calculate correlation energy
        if results.caspt2_energy and results.casscf_energy:
            results.correlation_energy = results.caspt2_energy - results.casscf_energy
    
    def _extract_convergence(self, output_content: str, results: OpenMolcasResults):
        """Extract convergence information."""
        # Check convergence status
        results.scf_converged = bool(self.patterns["scf_convergence"].search(output_content))
        results.casscf_converged = bool(self.patterns["casscf_convergence"].search(output_content))
        results.caspt2_converged = bool(self.patterns["caspt2_convergence"].search(output_content))
        
        # Extract iteration counts
        for method in ["scf", "casscf", "caspt2"]:
            pattern_name = f"{method}_iterations"
            if pattern_name in self.patterns:
                match = self.patterns[pattern_name].search(output_content)
                if match:
                    setattr(results, pattern_name, int(match.group(1)))
    
    def _extract_orbital_info(self, output_content: str, results: OpenMolcasResults):
        """Extract orbital and electronic structure information."""
        # Active space size
        active_match = self.patterns["active_space"].search(output_content)
        if active_match:
            n_electrons = int(active_match.group(1))
            n_orbitals = int(active_match.group(2))
            results.active_space_size = (n_electrons, n_orbitals)
        
        # Natural orbital occupations
        occ_match = self.patterns["natural_occupations"].search(output_content)
        if occ_match:
            occ_text = occ_match.group(1)
            # Parse occupation numbers from text
            occ_lines = [line.strip() for line in occ_text.split('\n') if line.strip()]
            occupations = []
            for line in occ_lines:
                # Extract occupation number (typically second column)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        occ = float(parts[1])
                        occupations.append(occ)
                    except (ValueError, IndexError):
                        continue
            if occupations:
                results.natural_occupations = occupations
    
    def _extract_performance(self, output_content: str, results: OpenMolcasResults):
        """Extract performance and timing information."""
        # Wall time
        wall_match = self.patterns["wall_time"].search(output_content)
        if wall_match:
            results.wall_time = float(wall_match.group(1))
        
        # CPU time
        cpu_match = self.patterns["cpu_time"].search(output_content)
        if cpu_match:
            results.cpu_time = float(cpu_match.group(1))
        
        # Memory usage
        mem_match = self.patterns["memory_usage"].search(output_content)
        if mem_match:
            results.memory_usage = float(mem_match.group(1))
    
    def _extract_diagnostics(self, output_content: str, results: OpenMolcasResults):
        """Extract warning and error messages."""
        # Warnings
        warning_matches = self.patterns["warnings"].findall(output_content)
        results.warnings = [w.strip() for w in warning_matches if w.strip()]
        
        # Errors
        error_matches = self.patterns["errors"].findall(output_content)
        results.errors = [e.strip() for e in error_matches if e.strip()]
        
        # Check for fatal errors
        if self.patterns["fatal_errors"].search(output_content):
            results.errors.append("Fatal error detected in calculation")
    
    def _validate_results(self, results: OpenMolcasResults, calculation_type: str):
        """Validate parsed results and add derived information."""
        # Check for required energies based on calculation type
        if calculation_type == "casscf":
            if results.casscf_energy is None:
                results.errors.append("CASSCF energy not found in output")
        elif calculation_type in ["caspt2", "ms_caspt2"]:
            if results.caspt2_energy is None:
                results.errors.append("CASPT2 energy not found in output")
        
        # Check convergence consistency
        if calculation_type == "caspt2" and results.caspt2_energy is not None:
            if not results.caspt2_converged:
                results.warnings.append("CASPT2 energy found but convergence not confirmed")
        
        # Validate multi-state results
        if calculation_type == "ms_caspt2":
            if results.state_energies and len(results.state_energies) == 1:
                results.warnings.append("MS-CASPT2 requested but only one state found")
    
    def parse_from_file(self, output_file: Union[str, Path], **kwargs) -> OpenMolcasResults:
        """
        Parse OpenMolcas output from file.
        
        Args:
            output_file: Path to output file
            **kwargs: Additional arguments for parse_output
            
        Returns:
            Parsed results
        """
        output_path = Path(output_file)
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not found: {output_path}")
        
        with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return self.parse_output(content, **kwargs)
    
    def to_external_result(
        self, 
        results: OpenMolcasResults,
        method: str = "CASPT2",
        **kwargs
    ) -> ExternalMethodResult:
        """
        Convert OpenMolcas results to standard external method result format.
        
        Args:
            results: Parsed OpenMolcas results
            method: Method name for result
            **kwargs: Additional parameters for external result
            
        Returns:
            Standardized external method result
        """
        # Determine final energy and convergence
        if results.caspt2_energy is not None:
            final_energy = results.caspt2_energy
            converged = results.caspt2_converged
        elif results.casscf_energy is not None:
            final_energy = results.casscf_energy
            converged = results.casscf_converged
        else:
            raise ValueError("No valid energy found in OpenMolcas results")
        
        # Build external data dictionary
        external_data = {
            "scf_energy": results.scf_energy,
            "casscf_energy": results.casscf_energy,
            "caspt2_energy": results.caspt2_energy,
            "active_space_size": results.active_space_size,
            "natural_occupations": results.natural_occupations,
            "warnings": results.warnings,
            "errors": results.errors,
        }
        
        # Add multi-state data if available
        if results.state_energies:
            external_data["state_energies"] = results.state_energies
            external_data["state_weights"] = results.state_weights
        
        # Build convergence info
        convergence_info = {
            "method": method,
            "converged": converged,
            "scf_converged": results.scf_converged,
            "casscf_converged": results.casscf_converged,
            "caspt2_converged": results.caspt2_converged,
        }
        
        # Add iteration counts if available
        for attr in ["scf_iterations", "casscf_iterations", "caspt2_iterations"]:
            value = getattr(results, attr)
            if value is not None:
                convergence_info[attr] = value
        
        return ExternalMethodResult(
            method=method,
            software="OpenMolcas",
            energy=final_energy,
            correlation_energy=results.correlation_energy,
            external_data=external_data,
            wall_time=results.wall_time,
            cpu_time=results.cpu_time,
            memory_mb=results.memory_usage,
            converged=converged,
            convergence_info=convergence_info,
            **kwargs
        )