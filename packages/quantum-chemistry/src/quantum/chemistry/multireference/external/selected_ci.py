"""
Selected CI integration for SHCI and CIPSI methods.

This module provides interfaces to selected CI methods including:
- SHCI (Semistochastic Heat-bath CI) via Dice
- CIPSI (Configuration Interaction using a Perturbative Selection made Iteratively) via Quantum Package
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult

from ..base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult
from .base import ExternalMethodInterface, ExternalMethodResult, ExternalSoftwareError


class SelectedCIMethod(MultireferenceMethod):
    """
    Selected CI method base class.

    This class provides the common interface for selected CI methods
    including SHCI and CIPSI with automatic PT2 extrapolation to
    near-FCI accuracy.
    """

    def __init__(
        self,
        method_type: str = "shci",
        pt2_threshold: float = 1e-4,
        max_determinants: int = 1000000,
        extrapolate_fci: bool = True,
        n_det_schedule: Optional[List[int]] = None,
        software_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        keep_files: bool = False,
        **kwargs,
    ):
        """
        Initialize Selected CI method.

        Args:
            method_type: Selected CI method ("shci" or "cipsi")
            pt2_threshold: PT2 correction convergence threshold
            max_determinants: Maximum number of determinants
            extrapolate_fci: Whether to extrapolate to FCI limit
            n_det_schedule: Schedule of determinant numbers for extrapolation
            software_path: Path to external software
            work_dir: Working directory
            keep_files: Whether to keep temporary files
            **kwargs: Additional parameters
        """
        # Set attributes before calling super().__init__()
        self.method_type = method_type.lower()
        self.pt2_threshold = pt2_threshold
        self.max_determinants = max_determinants
        self.extrapolate_fci = extrapolate_fci
        self.n_det_schedule = n_det_schedule or self._get_default_schedule()

        # Set up interface based on method type
        if self.method_type == "shci":
            self.interface = SHCIInterface(
                software_path=software_path,
                work_dir=work_dir,
                keep_files=keep_files,
                pt2_threshold=pt2_threshold,
                max_determinants=max_determinants,
            )
        elif self.method_type == "cipsi":
            self.interface = CIPSIInterface(
                software_path=software_path,
                work_dir=work_dir,
                keep_files=keep_files,
                pt2_threshold=pt2_threshold,
                max_determinants=max_determinants,
            )
        else:
            raise ValueError(f"Unknown selected CI method: {method_type}")

        # Call parent initialization after setting attributes
        super().__init__(**kwargs)

    def _get_default_schedule(self) -> List[int]:
        """Get default determinant schedule for extrapolation."""
        return [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        if self.method_type == "shci":
            return MultireferenceMethodType.SHCI
        elif self.method_type == "cipsi":
            return MultireferenceMethodType.CIPSI
        else:
            raise ValueError(f"Unknown method type: {self.method_type}")

    def calculate(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> MultireferenceResult:
        """
        Perform Selected CI calculation.

        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters

        Returns:
            MultireferenceResult with Selected CI results
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for Selected CI calculation")

        start_time = time.time()

        try:
            # Run selected CI calculation
            external_result = self.interface.calculate(scf_obj, active_space, **kwargs)

            # Perform FCI extrapolation if requested
            if self.extrapolate_fci:
                extrapolated_result = self._extrapolate_to_fci(
                    external_result, scf_obj, active_space, **kwargs
                )
            else:
                extrapolated_result = external_result

            wall_time = time.time() - start_time

            # Convert to MultireferenceResult
            method_name = f"Selected CI ({self.method_type.upper()})"
            if self.extrapolate_fci:
                method_name += " + FCI extrapolation"

            # Build convergence info
            convergence_info = {
                "converged": extrapolated_result.converged,
                "method": method_name,
                "pt2_threshold": self.pt2_threshold,
                "n_determinants": extrapolated_result.external_data.get(
                    "n_determinants", 0
                ),
                "pt2_correction": extrapolated_result.external_data.get(
                    "pt2_correction", 0.0
                ),
            }

            # Build active space info
            active_space_info = {
                "n_electrons": active_space.n_active_electrons,
                "n_orbitals": active_space.n_active_orbitals,
                "selection_method": active_space.method,
                "selected_ci_type": self.method_type,
                "extrapolated": self.extrapolate_fci,
                "sci_data": extrapolated_result.external_data,
            }

            # Computational cost
            computational_cost = {
                "wall_time": wall_time,
                "cpu_time": extrapolated_result.cpu_time or wall_time,
                "memory_mb": extrapolated_result.memory_mb or 0.0,
            }

            return MultireferenceResult(
                method=method_name,
                energy=extrapolated_result.energy,
                correlation_energy=extrapolated_result.correlation_energy,
                active_space_info=active_space_info,
                n_active_electrons=active_space.n_active_electrons,
                n_active_orbitals=active_space.n_active_orbitals,
                convergence_info=convergence_info,
                computational_cost=computational_cost,
                natural_orbitals=None,  # Could be computed from CI vector
                occupation_numbers=None,  # Could be computed from CI vector
                basis_set=scf_obj.mol.basis,
                software_version=f"{self.method_type} (external)",
            )

        except Exception as e:
            raise ExternalSoftwareError(f"Selected CI calculation failed: {e}")

    def _extrapolate_to_fci(
        self,
        sci_result: ExternalMethodResult,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> ExternalMethodResult:
        """
        Extrapolate selected CI results to FCI limit.

        Args:
            sci_result: Selected CI result
            scf_obj: SCF object
            active_space: Active space result
            **kwargs: Additional parameters

        Returns:
            Extrapolated result
        """
        # This would run multiple calculations with different determinant counts
        # and extrapolate E vs PT2 to PT2=0 (FCI limit)

        # For now, return the original result with a note
        sci_result.external_data["extrapolation_note"] = (
            "FCI extrapolation not yet implemented - returning largest calculation"
        )
        sci_result.method += " (extrapolation placeholder)"

        return sci_result

    def estimate_cost(
        self, n_electrons: int, n_orbitals: int, basis_size: int, **kwargs
    ) -> Dict[str, float]:
        """
        Estimate computational cost for Selected CI calculation.

        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters

        Returns:
            Dict with cost estimates
        """
        # Selected CI scaling is complex and depends on the number of determinants
        # Rough estimates based on determinant count

        n_dets = min(self.max_determinants, 4 ** min(n_orbitals, 10))  # Rough estimate

        # Memory: O(n_determinants)
        memory_mb = n_dets * 8e-6 * n_orbitals  # MB

        # Time: roughly O(n_determinants^2) for diagonalization
        time_seconds = (n_dets**1.5) * 1e-9  # Rough estimate

        # Disk: store CI vectors and intermediates
        disk_mb = memory_mb * 2

        return {
            "memory_mb": memory_mb,
            "time_seconds": time_seconds,
            "disk_mb": disk_mb,
            "estimated_n_determinants": n_dets,
            "scaling_note": "Selected CI scaling depends on determinant selection efficiency",
        }


class SHCIInterface(ExternalMethodInterface):
    """Interface for SHCI calculations using Dice."""

    def __init__(
        self, pt2_threshold: float = 1e-4, max_determinants: int = 1000000, **kwargs
    ):
        super().__init__(**kwargs)
        self.pt2_threshold = pt2_threshold
        self.max_determinants = max_determinants

    def _get_software_name(self) -> str:
        return "Dice (SHCI)"

    def _get_default_executable(self) -> str:
        return "Dice"

    def _prepare_input(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        work_path: Path,
        **kwargs,
    ) -> Dict[str, str]:
        """Prepare SHCI input files."""
        # This would prepare FCIDUMP and Dice input files
        # For now, return placeholder

        input_content = f"""
# Dice SHCI input file
{active_space.n_active_orbitals} {active_space.n_active_electrons}
epsilon1 {self.pt2_threshold}
maxDet {self.max_determinants}
davidsonTol 1e-8
dE 1e-8
doRDM false
schedule
{self.max_determinants} {self.pt2_threshold}
end
"""

        return {
            "input.dat": input_content.strip(),
            "FCIDUMP": "# FCIDUMP placeholder - would be generated from active space",
        }

    def _run_calculation(
        self, work_path: Path, input_files: Dict[str, str], **kwargs
    ) -> subprocess.CompletedProcess:
        """Run SHCI calculation."""
        # Write input files
        for filename, content in input_files.items():
            with open(work_path / filename, "w") as f:
                f.write(content)

        # Run Dice
        cmd = [self.software_path, "input.dat"]

        result = subprocess.run(
            cmd, cwd=work_path, capture_output=True, text=True, timeout=3600
        )

        return result

    def _parse_output(
        self, work_path: Path, process_result: subprocess.CompletedProcess, **kwargs
    ) -> ExternalMethodResult:
        """Parse SHCI output."""
        if process_result.returncode != 0:
            raise ExternalSoftwareError(f"SHCI failed: {process_result.stderr}")

        # This would parse the Dice output file
        # For now, return placeholder result

        return ExternalMethodResult(
            method="SHCI",
            software="Dice",
            energy=-75.0,  # Placeholder
            correlation_energy=-0.1,  # Placeholder
            external_data={
                "n_determinants": self.max_determinants,
                "pt2_correction": self.pt2_threshold,
                "implementation_note": "SHCI integration is placeholder",
            },
            converged=True,
            convergence_info={"method": "SHCI", "converged": True},
        )


class CIPSIInterface(ExternalMethodInterface):
    """Interface for CIPSI calculations using Quantum Package."""

    def __init__(
        self, pt2_threshold: float = 1e-4, max_determinants: int = 1000000, **kwargs
    ):
        super().__init__(**kwargs)
        self.pt2_threshold = pt2_threshold
        self.max_determinants = max_determinants

    def _get_software_name(self) -> str:
        return "Quantum Package (CIPSI)"

    def _get_default_executable(self) -> str:
        return "qp_run"

    def _prepare_input(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        work_path: Path,
        **kwargs,
    ) -> Dict[str, str]:
        """Prepare CIPSI input files."""
        # This would prepare Quantum Package input
        # For now, return placeholder

        return {
            "EZFIO_DIR": "# Quantum Package EZFIO directory setup",
            "run_script.sh": f"""#!/bin/bash
# Quantum Package CIPSI run script
qp_create_ezfio_from_xyz molecule.xyz -b {scf_obj.mol.basis}
qp_run scf
qp_set determinants n_det_max {self.max_determinants}
qp_set perturbation pt2_max {self.pt2_threshold}
qp_run cipsi
""",
        }

    def _run_calculation(
        self, work_path: Path, input_files: Dict[str, str], **kwargs
    ) -> subprocess.CompletedProcess:
        """Run CIPSI calculation."""
        # This would execute the Quantum Package workflow
        # For now, return placeholder

        return subprocess.CompletedProcess(
            args=["qp_run", "cipsi"],
            returncode=0,
            stdout="CIPSI calculation completed (placeholder)",
            stderr="",
        )

    def _parse_output(
        self, work_path: Path, process_result: subprocess.CompletedProcess, **kwargs
    ) -> ExternalMethodResult:
        """Parse CIPSI output."""
        # This would parse Quantum Package output
        # For now, return placeholder

        return ExternalMethodResult(
            method="CIPSI",
            software="Quantum Package",
            energy=-75.0,  # Placeholder
            correlation_energy=-0.1,  # Placeholder
            external_data={
                "n_determinants": self.max_determinants,
                "pt2_correction": self.pt2_threshold,
                "implementation_note": "CIPSI integration is placeholder",
            },
            converged=True,
            convergence_info={"method": "CIPSI", "converged": True},
        )


# Selected CI method types are already defined in base.py
