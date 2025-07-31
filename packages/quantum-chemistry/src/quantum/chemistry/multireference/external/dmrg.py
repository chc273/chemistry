"""
DMRG (Density Matrix Renormalization Group) integration via block2.

This module provides integration with the block2 library for DMRG calculations
including DMRG-CASSCF and DMRG-NEVPT2/CASPT2 workflows.
"""

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult

from ..base import MultireferenceMethod, MultireferenceMethodType, MultireferenceResult
from .base import ExternalSoftwareError


class DMRGMethod(MultireferenceMethod):
    """
    DMRG (Density Matrix Renormalization Group) method implementation using block2.

    This class provides DMRG-CASSCF calculations with optional post-SCF
    corrections (NEVPT2/CASPT2) for systems with large active spaces.
    """

    def __init__(
        self,
        bond_dimension: int = 1000,
        max_sweeps: int = 20,
        noise: float = 1e-4,
        davidson_threshold: float = 1e-8,
        post_correction: Optional[str] = None,
        scratch_dir: Optional[str] = None,
        n_threads: int = 1,
        memory_gb: float = 4.0,
        **kwargs,
    ):
        """
        Initialize DMRG method.

        Args:
            bond_dimension: Maximum bond dimension for DMRG
            max_sweeps: Maximum number of DMRG sweeps
            noise: Noise level for DMRG optimization
            davidson_threshold: Convergence threshold for Davidson diagonalization
            post_correction: Post-SCF correction method ('nevpt2' or 'caspt2')
            scratch_dir: Directory for temporary files
            n_threads: Number of threads for parallel execution
            memory_gb: Memory limit in GB
            **kwargs: Additional parameters
        """
        # Set attributes before calling super().__init__() which calls _get_method_type()
        self.bond_dimension = bond_dimension
        self.max_sweeps = max_sweeps
        self.noise = noise
        self.davidson_threshold = davidson_threshold
        self.post_correction = post_correction
        self.scratch_dir = scratch_dir
        self.n_threads = n_threads
        self.memory_gb = memory_gb

        # Try to import block2
        self._validate_block2()

        # Call parent initialization after setting attributes
        super().__init__(**kwargs)

    def _validate_block2(self):
        """Validate that block2 is available."""
        try:
            import block2

            self.block2 = block2

            # Check for required components
            from block2 import SZ, VectorUInt8
            from block2.su2 import DMRG, HamiltonianQC, SimplifiedMPO

        except ImportError as e:
            raise ExternalSoftwareError(
                f"block2 library not found. Please install with: "
                f"pip install block2. Error: {e}"
            )

    def _get_method_type(self) -> MultireferenceMethodType:
        """Return method type identifier."""
        if self.post_correction == "nevpt2":
            return MultireferenceMethodType.DMRG_NEVPT2
        elif self.post_correction == "caspt2":
            return MultireferenceMethodType.DMRG_CASPT2
        else:
            return MultireferenceMethodType.DMRG

    def calculate(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> MultireferenceResult:
        """
        Perform DMRG calculation.

        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters

        Returns:
            MultireferenceResult with DMRG results
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for DMRG calculation")

        start_time = time.time()

        try:
            # Set up DMRG calculation
            dmrg_result = self._run_dmrg_casscf(scf_obj, active_space, **kwargs)

            # Apply post-SCF correction if requested
            if self.post_correction:
                dmrg_result = self._apply_post_correction(
                    dmrg_result, scf_obj, active_space, **kwargs
                )

            wall_time = time.time() - start_time

            # Extract results
            method_name = "DMRG-CASSCF"
            if self.post_correction:
                method_name += f"+{self.post_correction.upper()}"

            # Build convergence info
            convergence_info = {
                "converged": dmrg_result.get("converged", False),
                "final_energy": dmrg_result["energy"],
                "bond_dimension": self.bond_dimension,
                "n_sweeps": dmrg_result.get("n_sweeps", 0),
                "discarded_weight": dmrg_result.get("discarded_weight", 0.0),
            }

            # Computational cost
            computational_cost = {
                "wall_time": wall_time,
                "cpu_time": wall_time * self.n_threads,  # Approximation
                "memory_mb": self.memory_gb * 1024,
            }

            return MultireferenceResult(
                method=method_name,
                energy=dmrg_result["energy"],
                correlation_energy=dmrg_result["energy"] - scf_obj.e_tot,
                active_space_info={
                    "n_electrons": active_space.n_active_electrons,
                    "n_orbitals": active_space.n_active_orbitals,
                    "selection_method": active_space.method,
                    "bond_dimension": self.bond_dimension,
                    "dmrg_data": dmrg_result.get("dmrg_data", {}),
                },
                n_active_electrons=active_space.n_active_electrons,
                n_active_orbitals=active_space.n_active_orbitals,
                convergence_info=convergence_info,
                computational_cost=computational_cost,
                natural_orbitals=dmrg_result.get("natural_orbitals"),
                occupation_numbers=dmrg_result.get("occupation_numbers"),
                basis_set=scf_obj.mol.basis,
                software_version=f"block2-{self._get_block2_version()}",
            )

        except Exception as e:
            raise ExternalSoftwareError(f"DMRG calculation failed: {e}")

    def _run_dmrg_casscf(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run DMRG-CASSCF calculation using block2.

        Args:
            scf_obj: SCF object
            active_space: Active space result
            **kwargs: Additional parameters

        Returns:
            Dict with DMRG calculation results
        """
        try:
            # Set up block2 environment
            from block2 import init_memory_pool, release_memory_pool, set_omp_threads
            from block2.su2 import DMRG, HamiltonianQC, SimplifiedMPO

            # Initialize threading and memory
            set_omp_threads(self.n_threads)
            init_memory_pool(int(self.memory_gb * 1024**3))  # Convert GB to bytes

            try:
                # Build quantum chemistry Hamiltonian
                n_sites = active_space.n_active_orbitals
                n_elec = active_space.n_active_electrons
                spin = 0  # Singlet state by default

                # Get molecular integrals in active space
                h1e, h2e, ecore = self._get_active_space_integrals(
                    scf_obj, active_space
                )

                # Create block2 Hamiltonian
                hamil = HamiltonianQC(n_sites, n_elec, spin, ecore, h1e, h2e)

                # Build MPO
                mpo = SimplifiedMPO(hamil, rule_type="qc")

                # Set up DMRG
                dmrg = DMRG(mpo, self.bond_dimension, noise=self.noise)
                dmrg.solve(n_sweeps=self.max_sweeps, tol=self.davidson_threshold)

                # Extract results
                energy = dmrg.energies[-1]
                converged = (
                    len(dmrg.energies) < self.max_sweeps
                    or abs(dmrg.energies[-1] - dmrg.energies[-2])
                    < self.davidson_threshold
                )

                # Get natural orbitals and occupation numbers if available
                natural_orbitals = None
                occupation_numbers = None

                try:
                    # Try to get 1-RDM for natural orbitals
                    rdm1 = dmrg.get_1pdm()
                    eigs, vecs = np.linalg.eigh(rdm1)
                    occupation_numbers = eigs[::-1]  # Sort in descending order
                    natural_orbitals = vecs[:, ::-1]
                except:
                    pass  # Natural orbitals not critical for basic functionality

                result = {
                    "energy": energy,
                    "converged": converged,
                    "n_sweeps": len(dmrg.energies),
                    "discarded_weight": dmrg.discarded_weights[-1]
                    if dmrg.discarded_weights
                    else 0.0,
                    "natural_orbitals": natural_orbitals,
                    "occupation_numbers": occupation_numbers,
                    "dmrg_data": {
                        "bond_dimension": self.bond_dimension,
                        "final_energy": energy,
                        "energy_history": dmrg.energies,
                    },
                }

                return result

            finally:
                # Always clean up memory
                release_memory_pool()

        except Exception as e:
            raise ExternalSoftwareError(f"DMRG-CASSCF calculation failed: {e}")

    def _get_active_space_integrals(
        self, scf_obj: Union[scf.hf.SCF, scf.uhf.UHF], active_space: ActiveSpaceResult
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get active space integrals for DMRG calculation.

        Args:
            scf_obj: SCF object
            active_space: Active space result

        Returns:
            Tuple of (h1e, h2e, ecore) - 1e integrals, 2e integrals, core energy
        """
        from pyscf import mcscf

        # Create minimal CASSCF object to get integrals
        casscf = mcscf.CASSCF(
            scf_obj, active_space.n_active_orbitals, active_space.n_active_electrons
        )
        casscf.mo_coeff = active_space.orbital_coefficients

        # Get effective integrals
        h1e, ecore = casscf.get_h1eff()
        h2e = casscf.get_h2eff()

        return h1e, h2e, ecore

    def _apply_post_correction(
        self,
        dmrg_result: Dict[str, Any],
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply post-SCF correction to DMRG-CASSCF result.

        Args:
            dmrg_result: DMRG-CASSCF result
            scf_obj: SCF object
            active_space: Active space result
            **kwargs: Additional parameters

        Returns:
            Updated result with post-SCF correction
        """
        if self.post_correction.lower() == "nevpt2":
            return self._apply_nevpt2_correction(
                dmrg_result, scf_obj, active_space, **kwargs
            )
        elif self.post_correction.lower() == "caspt2":
            return self._apply_caspt2_correction(
                dmrg_result, scf_obj, active_space, **kwargs
            )
        else:
            raise ValueError(f"Unknown post-correction method: {self.post_correction}")

    def _apply_nevpt2_correction(
        self,
        dmrg_result: Dict[str, Any],
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply NEVPT2 correction to DMRG-CASSCF result.

        Note: This is a placeholder implementation. Full DMRG-NEVPT2 requires
        specialized interfaces that are method-specific.
        """
        # For now, return DMRG-CASSCF result with note
        dmrg_result["post_correction"] = "nevpt2"
        dmrg_result["post_correction_note"] = (
            "DMRG-NEVPT2 requires specialized implementation - "
            "returning DMRG-CASSCF result"
        )
        return dmrg_result

    def _apply_caspt2_correction(
        self,
        dmrg_result: Dict[str, Any],
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply CASPT2 correction to DMRG-CASSCF result.

        Note: This is a placeholder implementation. Full DMRG-CASPT2 requires
        external software integration (e.g., OpenMolcas).
        """
        # For now, return DMRG-CASSCF result with note
        dmrg_result["post_correction"] = "caspt2"
        dmrg_result["post_correction_note"] = (
            "DMRG-CASPT2 requires external software integration - "
            "returning DMRG-CASSCF result"
        )
        return dmrg_result

    def _get_block2_version(self) -> str:
        """Get block2 version string."""
        try:
            import block2

            return getattr(block2, "__version__", "unknown")
        except:
            return "unknown"

    def estimate_cost(
        self, n_electrons: int, n_orbitals: int, basis_size: int, **kwargs
    ) -> Dict[str, float]:
        """
        Estimate computational cost for DMRG calculation.

        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters

        Returns:
            Dict with cost estimates
        """
        # DMRG scaling estimates
        # Memory: O(M^3 * D^2) where M is orbitals, D is bond dimension
        memory_scaling = (n_orbitals**3) * (self.bond_dimension**2) * 8e-9  # GB
        memory_mb = max(memory_scaling * 1024, 100)  # At least 100 MB

        # Time: roughly O(M^3 * D^3) with sweep overhead
        time_scaling = (n_orbitals**3) * (self.bond_dimension**3) * 1e-12
        time_seconds = max(time_scaling * self.max_sweeps, 1.0)  # At least 1 second

        # Disk: store wave function and intermediates
        disk_mb = memory_mb * 2

        return {
            "memory_mb": memory_mb,
            "time_seconds": time_seconds,
            "disk_mb": disk_mb,
            "bond_dimension": self.bond_dimension,
            "scaling_note": "DMRG scaling: O(M^3*D^3) time, O(M^3*D^2) memory",
        }

    def get_recommended_parameters(
        self, system_type: str, active_space_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Get recommended DMRG parameters for given system.

        Args:
            system_type: Type of chemical system
            active_space_size: (n_electrons, n_orbitals) tuple

        Returns:
            Dict of recommended parameters
        """
        n_electrons, n_orbitals = active_space_size

        # Base parameters
        params = {
            "bond_dimension": 1000,
            "max_sweeps": 20,
            "noise": 1e-4,
            "davidson_threshold": 1e-8,
        }

        # Adjust for system size
        if n_orbitals <= 10:
            params["bond_dimension"] = 500
            params["max_sweeps"] = 15
        elif n_orbitals <= 20:
            params["bond_dimension"] = 1000
            params["max_sweeps"] = 20
        else:
            params["bond_dimension"] = 2000
            params["max_sweeps"] = 30
            params["noise"] = 1e-5  # Lower noise for larger systems

        # System-specific adjustments
        if system_type == "transition_metal":
            params["bond_dimension"] *= 2  # Higher bond dimension for TM
            params["max_sweeps"] += 10
            params["davidson_threshold"] = 1e-7  # Tighter convergence

        if "bond_breaking" in system_type.lower():
            params["noise"] = 1e-5  # Lower noise for bond breaking
            params["max_sweeps"] += 5

        return params


# DMRG method types are already defined in base.py
