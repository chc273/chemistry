"""Unit tests for quantum chemistry calculations."""

import numpy as np
import pytest
from ase import Atoms

from quantum.chemistry import DFTCalculator, HartreeFockCalculator
from quantum.core import Molecule

# Skip tests if PySCF is not available
pytest_plugins = []

try:
    import pyscf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


class TestHartreeFockCalculator:
    """Test cases for HartreeFock calculator with PySCF backend."""

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_hf_calculator_creation(self):
        """Test Hartree-Fock calculator initialization."""
        calc = HartreeFockCalculator(
            method="rhf", basis_set="sto-3g", convergence_threshold=1e-6
        )

        assert calc.method == "rhf"
        assert calc.basis_set == "sto-3g"
        assert calc.convergence_threshold == 1e-6

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_h2_calculation(self):
        """Test Hartree-Fock calculation on H2 molecule."""
        # Create H2 molecule
        atoms = Atoms(
            symbols=["H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],  # 1.4 Bohr
        )
        h2 = Molecule(atoms=atoms, name="H2")

        # Run HF calculation
        calc = HartreeFockCalculator(basis_set="sto-3g")
        results = calc.run_calculation(h2)

        # Check basic results
        assert "energy" in results
        assert "converged" in results
        assert results["converged"] is True
        assert results["energy"] < 0  # Should be negative
        assert results["energy"] > -5  # Reasonable range for H2

        # Check orbital information
        assert "homo_energy" in results
        assert "orbital_energies" in results
        assert len(results["orbital_energies"]) == 2  # STO-3G for H2

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_water_calculation(self):
        """Test Hartree-Fock calculation on water molecule."""
        # Create water molecule
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [0.0000, 0.0000, 0.1173],
                [0.0000, 0.7572, -0.4692],
                [0.0000, -0.7572, -0.4692],
            ],
        )
        water = Molecule(atoms=atoms, name="water")

        # Run HF calculation
        calc = HartreeFockCalculator(basis_set="sto-3g")
        results = calc.run_calculation(water)

        # Check basic results
        assert results["converged"] is True
        assert results["energy"] < -70  # Water should be around -76 Hartree
        assert results["energy"] > -85

        # Check HOMO-LUMO gap
        gap = calc.get_homo_lumo_gap()
        assert gap is not None
        assert gap > 0  # HOMO-LUMO gap should be positive

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_mulliken_charges(self):
        """Test Mulliken population analysis."""
        # Create H2O molecule
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        water = Molecule(atoms=atoms, name="water")

        calc = HartreeFockCalculator(basis_set="sto-3g")
        calc.run_calculation(water)

        charges = calc.get_mulliken_charges(water)

        assert len(charges) == 3  # Three atoms
        assert charges[0] < 0  # Oxygen should be negative
        assert charges[1] > 0  # Hydrogens should be positive
        assert charges[2] > 0

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_dipole_moment(self):
        """Test dipole moment calculation."""
        # Create water molecule
        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [0.0000, 0.0000, 0.1173],
                [0.0000, 0.7572, -0.4692],
                [0.0000, -0.7572, -0.4692],
            ],
        )
        water = Molecule(atoms=atoms, name="water")

        calc = HartreeFockCalculator(basis_set="sto-3g")
        calc.run_calculation(water)

        dipole = calc.get_dipole_moment()

        assert len(dipole) == 3  # x, y, z components
        assert np.linalg.norm(dipole) > 0  # Water has non-zero dipole


class TestDFTCalculator:
    """Test cases for DFT calculator with PySCF backend."""

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_dft_calculator_creation(self):
        """Test DFT calculator initialization."""
        calc = DFTCalculator(method="b3lyp", basis_set="6-31g", grid_level=2)

        assert calc.method == "b3lyp"
        assert calc.basis_set == "6-31g"
        assert calc.grid_level == 2

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_b3lyp_h2_calculation(self):
        """Test B3LYP calculation on H2 molecule."""
        # Create H2 molecule
        atoms = Atoms(symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        h2 = Molecule(atoms=atoms, name="H2")

        # Run B3LYP calculation
        calc = DFTCalculator(method="b3lyp", basis_set="sto-3g")
        results = calc.run_calculation(h2)

        # Check basic results
        assert results["converged"] is True
        assert results["energy"] < 0
        assert results["functional"] == "b3lyp"

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_unrestricted_calculation(self):
        """Test unrestricted DFT calculation."""
        # Create H atom (doublet)
        atoms = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])
        h_atom = Molecule(atoms=atoms, name="H", multiplicity=2)

        calc = DFTCalculator(method="pbe", basis_set="sto-3g", unrestricted=True)
        results = calc.run_calculation(h_atom)

        assert results["converged"] is True
        assert results["energy"] < 0

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_convenience_calculators(self):
        """Test convenience calculator classes."""
        from quantum.chemistry import B3LYPCalculator, PBECalculator

        b3lyp_calc = B3LYPCalculator(basis_set="sto-3g")
        assert b3lyp_calc.method == "b3lyp"

        pbe_calc = PBECalculator(basis_set="sto-3g")
        assert pbe_calc.method == "pbe"


class TestCalculatorIntegration:
    """Test integration between calculators and molecule objects."""

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_multiple_format_compatibility(self):
        """Test that calculators work with different molecule formats."""
        # Create molecule from XYZ string
        xyz_string = """2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 1.4
"""
        mol = Molecule.from_xyz_string(xyz_string, name="H2")

        # Test HF calculation
        hf_calc = HartreeFockCalculator(basis_set="sto-3g")
        hf_results = hf_calc.run_calculation(mol)

        # Test DFT calculation on same molecule
        dft_calc = DFTCalculator(method="pbe", basis_set="sto-3g")
        dft_results = dft_calc.run_calculation(mol)

        assert hf_results["converged"] is True
        assert dft_results["converged"] is True

        # DFT and HF energies should be different
        assert abs(hf_results["energy"] - dft_results["energy"]) > 1e-6

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_calculator_error_handling(self):
        """Test error handling in calculators."""
        # Test with invalid method
        with pytest.raises(ValueError):
            calc = HartreeFockCalculator(method="invalid_method")
            atoms = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])
            mol = Molecule(atoms=atoms)
            calc.run_calculation(mol)

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_results_consistency(self):
        """Test that results are consistent across runs."""
        atoms = Atoms(symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        mol = Molecule(atoms=atoms, name="H2")

        calc = HartreeFockCalculator(basis_set="sto-3g")

        # Run calculation twice
        results1 = calc.run_calculation(mol)
        results2 = calc.run_calculation(mol)

        # Results should be identical
        assert abs(results1["energy"] - results2["energy"]) < 1e-10
        assert np.allclose(results1["orbital_energies"], results2["orbital_energies"])
