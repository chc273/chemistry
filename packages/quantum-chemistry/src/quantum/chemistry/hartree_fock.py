"""Hartree-Fock self-consistent field calculations using PySCF."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pyscf import gto, scf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from quantum.core import BaseCalculator, BaseSystem, Molecule


class HartreeFockCalculator(BaseCalculator):
    """
    Hartree-Fock self-consistent field calculator using PySCF backend.

    This class provides a high-level interface to PySCF's Hartree-Fock
    implementation, supporting both restricted (RHF) and unrestricted (UHF)
    calculations.
    """

    def __init__(
        self,
        method: str = "rhf",
        basis_set: str = "sto-3g",
        convergence_threshold: float = 1e-8,
        max_iterations: int = 100,
        diis: bool = True,
        diis_space: int = 8,
        level_shift: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initialize Hartree-Fock calculator.

        Args:
            method: HF method ('rhf', 'uhf', 'rohf')
            basis_set: Basis set name
            convergence_threshold: SCF convergence threshold
            max_iterations: Maximum SCF iterations
            diis: Enable DIIS acceleration
            diis_space: DIIS space size
            level_shift: Level shift parameter for convergence
            **kwargs: Additional PySCF parameters
        """
        if not PYSCF_AVAILABLE:
            raise ImportError(
                "PySCF is required for Hartree-Fock calculations. Install with: pip install pyscf"
            )

        super().__init__(method, basis_set, convergence_threshold, max_iterations)
        self.diis = diis
        self.diis_space = diis_space
        self.level_shift = level_shift
        self.pyscf_kwargs = kwargs

        # PySCF objects
        self._mol: gto.Mole | None = None
        self._mf: scf.hf.SCF | None = None
        self._results: dict[str, Any] = {}

    def run_calculation(self, system: BaseSystem) -> dict[str, Any]:
        """
        Run Hartree-Fock SCF calculation using PySCF.

        Args:
            system: Molecular system for calculation

        Returns:
            Dictionary containing calculation results
        """
        # Convert system to PySCF Mole object
        self._mol = self._create_pyscf_mol(system)

        # Create SCF object based on method
        method_lower = self.method.lower()
        if method_lower == "rhf":
            self._mf = scf.RHF(self._mol)
        elif method_lower == "uhf":
            self._mf = scf.UHF(self._mol)
        elif method_lower == "rohf":
            self._mf = scf.ROHF(self._mol)
        else:
            raise ValueError(f"Unsupported HF method: {self.method}")

        # Set SCF parameters
        self._mf.conv_tol = self.convergence_threshold
        self._mf.max_cycle = self.max_iterations
        self._mf.diis = self.diis
        self._mf.diis_space = self.diis_space
        self._mf.level_shift = self.level_shift

        # Apply additional PySCF parameters
        for key, value in self.pyscf_kwargs.items():
            setattr(self._mf, key, value)

        # Run SCF calculation
        energy = self._mf.kernel()

        # Extract results
        self._results = self._extract_results(energy)

        return self._results

    def get_energy(self) -> float:
        """Get total energy from last calculation."""
        if "energy" not in self._results:
            raise RuntimeError("No calculation results available")
        return self._results["energy"]

    def get_homo_lumo_gap(self) -> float | None:
        """Get HOMO-LUMO gap in Hartree."""
        if "homo_energy" not in self._results or "lumo_energy" not in self._results:
            return None

        lumo = self._results["lumo_energy"]
        if lumo is None:
            return None

        return lumo - self._results["homo_energy"]

    def get_mulliken_charges(self, system: BaseSystem) -> np.ndarray:
        """Calculate Mulliken population charges."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")

        pop = self._mf.mulliken_pop(verbose=0)
        return pop[1]  # Atomic charges

    def get_molecular_orbitals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get molecular orbital coefficients and energies.

        Returns:
            Tuple of (orbital_coefficients, orbital_energies)
        """
        if "molecular_orbitals" not in self._results:
            raise RuntimeError("No calculation results available")

        return self._results["molecular_orbitals"], self._results["orbital_energies"]

    def get_density_matrix(self) -> np.ndarray:
        """Get density matrix."""
        if "density_matrix" not in self._results:
            raise RuntimeError("No calculation results available")

        return self._results["density_matrix"]

    def get_fock_matrix(self) -> np.ndarray:
        """Get Fock matrix."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")

        return self._mf.get_fock()

    def get_overlap_matrix(self) -> np.ndarray:
        """Get overlap matrix."""
        if self._mol is None:
            raise RuntimeError("No calculation results available")

        return self._mol.intor("int1e_ovlp")

    def get_dipole_moment(self) -> np.ndarray:
        """Calculate dipole moment in Debye."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")


        # Calculate dipole moment
        dipole_ao = self._mol.intor("int1e_r")
        dm = self._mf.make_rdm1()

        dipole = np.einsum("xij,ji->x", dipole_ao, dm)

        # Add nuclear contribution
        charges = self._mol.atom_charges()
        coords = self._mol.atom_coords()
        nuclear_dipole = np.einsum("i,ix->x", charges, coords)

        total_dipole = nuclear_dipole - dipole  # Electronic contribution is negative

        # Convert to Debye (1 a.u. = 2.54174 Debye)
        return total_dipole * 2.54174

    def _create_pyscf_mol(self, system: BaseSystem) -> gto.Mole:
        """Convert system to PySCF Mole object."""
        if isinstance(system, Molecule):
            # Use ASE backend for consistency
            ase_atoms = system.to_ase_atoms()
            symbols = ase_atoms.get_chemical_symbols()
            positions = ase_atoms.get_positions()
        else:
            # Assume system has the required attributes
            symbols = system.atoms
            positions = system.coordinates

        # Build atom string for PySCF
        atom_string = []
        for symbol, pos in zip(symbols, positions):
            atom_string.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        mol = gto.Mole()
        mol.atom = ";".join(atom_string)
        mol.basis = self.basis_set
        mol.charge = system.charge
        mol.spin = system.multiplicity - 1  # PySCF uses 2S, not 2S+1
        mol.unit = "Angstrom"
        mol.build(verbose=0)

        return mol

    def _extract_results(self, energy: float) -> dict[str, Any]:
        """Extract results from PySCF calculation."""
        if self._mf is None or self._mol is None:
            raise RuntimeError("SCF calculation not completed")

        # Get orbital information
        mo_coeff = self._mf.mo_coeff
        mo_energy = self._mf.mo_energy

        # Get density matrix
        dm = self._mf.make_rdm1()

        # Determine HOMO/LUMO
        occ = self._mf.mo_occ
        homo_idx = np.where(occ > 0)[0]
        if len(homo_idx) > 0:
            homo_energy = mo_energy[homo_idx[-1]]
            lumo_idx = np.where(occ == 0)[0]
            lumo_energy = mo_energy[lumo_idx[0]] if len(lumo_idx) > 0 else None
        else:
            homo_energy = None
            lumo_energy = None

        results = {
            "energy": energy,
            "converged": self._mf.converged,
            "iterations": getattr(self._mf, "niter", None),
            "orbital_energies": mo_energy,
            "molecular_orbitals": mo_coeff,
            "density_matrix": dm,
            "homo_energy": homo_energy,
            "lumo_energy": lumo_energy,
            "nuclear_repulsion": self._mol.energy_nuc(),
            "electronic_energy": energy - self._mol.energy_nuc(),
        }

        return results


class RestrictedHF(HartreeFockCalculator):
    """Convenience class for Restricted Hartree-Fock calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="rhf", **kwargs)


class UnrestrictedHF(HartreeFockCalculator):
    """Convenience class for Unrestricted Hartree-Fock calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="uhf", **kwargs)


class RestrictedOpenHF(HartreeFockCalculator):
    """Convenience class for Restricted Open-shell Hartree-Fock calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="rohf", **kwargs)
