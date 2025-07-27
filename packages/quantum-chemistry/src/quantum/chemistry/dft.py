"""Density Functional Theory calculations using PySCF."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pyscf import dft, gto

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from quantum.core import BaseCalculator, BaseSystem, Molecule


class DFTCalculator(BaseCalculator):
    """
    Density Functional Theory calculator using PySCF backend.

    This class provides a high-level interface to PySCF's DFT implementation,
    supporting various exchange-correlation functionals and basis sets.
    """

    def __init__(
        self,
        method: str = "b3lyp",
        basis_set: str = "6-31g*",
        convergence_threshold: float = 1e-8,
        max_iterations: int = 100,
        grid_level: int = 3,
        unrestricted: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize DFT calculator.

        Args:
            method: DFT functional (e.g., 'b3lyp', 'pbe', 'wb97x', 'm06-2x')
            basis_set: Basis set name
            convergence_threshold: SCF convergence threshold
            max_iterations: Maximum SCF iterations
            grid_level: DFT integration grid quality (0-9, higher = better)
            unrestricted: Use unrestricted DFT (UKS)
            **kwargs: Additional PySCF parameters
        """
        if not PYSCF_AVAILABLE:
            raise ImportError(
                "PySCF is required for DFT calculations. Install with: pip install pyscf"
            )

        super().__init__(method, basis_set, convergence_threshold, max_iterations)
        self.grid_level = grid_level
        self.unrestricted = unrestricted
        self.pyscf_kwargs = kwargs

        # PySCF objects
        self._mol: gto.Mole | None = None
        self._mf: dft.rks.RKS | dft.uks.UKS | None = None
        self._results: dict[str, Any] = {}

    def run_calculation(self, system: BaseSystem) -> dict[str, Any]:
        """
        Run DFT SCF calculation using PySCF.

        Args:
            system: Molecular system for calculation

        Returns:
            Dictionary containing calculation results
        """
        # Convert system to PySCF Mole object
        self._mol = self._create_pyscf_mol(system)

        # Create DFT object
        if self.unrestricted or system.multiplicity > 1:
            self._mf = dft.UKS(self._mol)
        else:
            self._mf = dft.RKS(self._mol)

        # Set functional
        self._mf.xc = self.method

        # Set SCF parameters
        self._mf.conv_tol = self.convergence_threshold
        self._mf.max_cycle = self.max_iterations

        # Set grid parameters
        self._mf.grids.level = self.grid_level

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

    def get_kohn_sham_matrix(self) -> np.ndarray:
        """Get Kohn-Sham matrix."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")

        return self._mf.get_fock()

    def get_xc_energy(self) -> float:
        """Get exchange-correlation energy."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")

        dm = self._mf.make_rdm1()
        return self._mf.get_veff(dm=dm).exc

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

    def get_polarizability(self) -> np.ndarray:
        """Calculate static polarizability tensor."""
        if self._mf is None:
            raise RuntimeError("No calculation results available")

        try:
            from pyscf.prop import polarizability

            polar = polarizability.polarizability(self._mf)
            return polar.polarizability()
        except ImportError:
            raise ImportError("Polarizability calculation requires pyscf.prop module")

    def calculate_tddft(self, nstates: int = 10) -> dict[str, Any]:
        """
        Calculate excited states using time-dependent DFT.

        Args:
            nstates: Number of excited states to calculate

        Returns:
            Dictionary containing TDDFT results
        """
        if self._mf is None:
            raise RuntimeError("No ground state calculation available")

        try:
            from pyscf import tddft

            if isinstance(self._mf, dft.uks.UKS):
                td = tddft.TDDFT(self._mf)
            else:
                td = tddft.TDHF(self._mf)  # Use TDHF for RKS

            td.nstates = nstates
            td.kernel()

            # Extract excitation energies and oscillator strengths
            excitation_energies = td.e  # in Hartree
            oscillator_strengths = td.oscillator_strength()

            return {
                "excitation_energies_hartree": excitation_energies,
                "excitation_energies_ev": excitation_energies
                * 27.211386,  # Convert to eV
                "oscillator_strengths": oscillator_strengths,
                "converged": td.converged,
            }
        except ImportError:
            raise ImportError("TDDFT calculation requires pyscf.tddft module")

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
        """Extract results from PySCF DFT calculation."""
        if self._mf is None or self._mol is None:
            raise RuntimeError("SCF calculation not completed")

        # Get orbital information
        mo_coeff = self._mf.mo_coeff
        mo_energy = self._mf.mo_energy

        # Get density matrix
        dm = self._mf.make_rdm1()

        # Determine HOMO/LUMO
        occ = self._mf.mo_occ
        if isinstance(occ, tuple):  # UKS case
            occ = occ[0] + occ[1]  # Total occupation
            mo_energy = mo_energy[0]  # Alpha energies

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
            "functional": self.method,
        }

        return results


class B3LYPCalculator(DFTCalculator):
    """Convenience class for B3LYP calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="b3lyp", **kwargs)


class PBECalculator(DFTCalculator):
    """Convenience class for PBE calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="pbe", **kwargs)


class M06Calculator(DFTCalculator):
    """Convenience class for M06 calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="m06", **kwargs)


class wB97XDCalculator(DFTCalculator):
    """Convenience class for ωB97X-D calculations."""

    def __init__(self, **kwargs: Any):
        super().__init__(method="wb97x", **kwargs)
