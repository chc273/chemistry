"""Unit tests for molecular format conversion utilities."""

import numpy as np
import pytest
from ase import Atoms

from quantum.core import Molecule
from quantum.core.converters import (
    from_file,
    get_supported_formats,
    to_ase_atoms,
    to_pymatgen_molecule,
    to_qcschema,
    to_quantum_molecule,
)

# Skip tests requiring external libraries if not available
try:
    import pymatgen

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import qcelemental

    QCELEMENTAL_AVAILABLE = True
except ImportError:
    QCELEMENTAL_AVAILABLE = False


class TestFormatConversions:
    """Test format conversion utilities."""

    def test_to_ase_atoms_from_molecule(self):
        """Test conversion from Molecule to ASE Atoms."""
        # Create molecule from ASE atoms
        ase_atoms = Atoms(
            symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
        )
        mol = Molecule(atoms=ase_atoms, name="H2")

        # Convert back to ASE
        converted_ase = to_ase_atoms(mol)

        assert len(converted_ase) == 2
        assert converted_ase.get_chemical_symbols() == ["H", "H"]
        assert np.allclose(converted_ase.get_positions(), ase_atoms.get_positions())

    def test_to_ase_atoms_from_ase(self):
        """Test conversion from ASE Atoms to ASE Atoms (identity)."""
        ase_atoms = Atoms(
            symbols=["C", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        converted = to_ase_atoms(ase_atoms)

        assert len(converted) == 2
        assert converted.get_chemical_symbols() == ["C", "H"]
        assert np.allclose(converted.get_positions(), ase_atoms.get_positions())

    @pytest.mark.skipif(not PYMATGEN_AVAILABLE, reason="PyMatGen not available")
    def test_to_pymatgen_molecule(self):
        """Test conversion to PyMatGen Molecule."""
        ase_atoms = Atoms(
            symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
        )

        pymatgen_mol = to_pymatgen_molecule(ase_atoms)

        assert len(pymatgen_mol) == 2
        assert [str(site.specie) for site in pymatgen_mol] == ["H", "H"]

    @pytest.mark.skipif(not QCELEMENTAL_AVAILABLE, reason="QCElemental not available")
    def test_to_qcschema(self):
        """Test conversion to QCSchema format."""
        ase_atoms = Atoms(
            symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
        )

        qc_mol = to_qcschema(ase_atoms, charge=0, multiplicity=1)

        assert len(qc_mol.symbols) == 2
        assert qc_mol.symbols == ["H", "H"]
        assert qc_mol.molecular_charge == 0
        assert qc_mol.molecular_multiplicity == 1

    def test_to_quantum_molecule(self):
        """Test conversion to quantum.core.Molecule."""
        ase_atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        mol = to_quantum_molecule(ase_atoms, name="water", charge=0)

        assert isinstance(mol, Molecule)
        assert mol.name == "water"
        assert len(mol.atoms) == 3
        assert mol.atoms == ["O", "H", "H"]

    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        with pytest.raises(TypeError):
            to_ase_atoms("invalid_format")

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = get_supported_formats()

        assert isinstance(formats, dict)
        assert "read" in formats
        assert "write" in formats
        assert "common" in formats

        # Check that common formats are included
        common_formats = formats["common"]
        assert "xyz" in common_formats
        assert "pdb" in common_formats

    @pytest.mark.skipif(
        not (PYMATGEN_AVAILABLE and QCELEMENTAL_AVAILABLE),
        reason="PyMatGen and QCElemental required",
    )
    def test_round_trip_conversions(self):
        """Test round-trip conversions between formats."""
        # Start with ASE Atoms
        original_ase = Atoms(
            symbols=["C", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        # ASE -> Molecule -> PyMatGen -> ASE
        mol = to_quantum_molecule(original_ase)
        pymatgen_mol = to_pymatgen_molecule(mol)
        final_ase = to_ase_atoms(pymatgen_mol)

        assert len(final_ase) == len(original_ase)
        assert final_ase.get_chemical_symbols() == original_ase.get_chemical_symbols()
        assert np.allclose(
            final_ase.get_positions(), original_ase.get_positions(), atol=1e-6
        )

    def test_xyz_string_conversion(self):
        """Test XYZ string conversion utilities."""
        from quantum.core.converters import qcschema_to_xyz, xyz_to_qcschema

        xyz_string = """3
water molecule
O 0.000000 0.000000 0.000000
H 0.757200 0.000000 0.469200
H -0.757200 0.000000 0.469200
"""

        # Test XYZ to QCSchema (skip if qcelemental not available)
        if QCELEMENTAL_AVAILABLE:
            qc_mol = xyz_to_qcschema(xyz_string, charge=0, multiplicity=1)
            assert len(qc_mol.symbols) == 3
            assert qc_mol.symbols == ["O", "H", "H"]

            # Test QCSchema back to XYZ
            xyz_back = qcschema_to_xyz(qc_mol)
            lines = xyz_back.strip().split("\\n")
            assert lines[0] == "3"
            assert "O" in lines[2]
            assert "H" in lines[3]


class TestFileIO:
    """Test file input/output operations."""

    def test_from_file_output_formats(self):
        """Test different output formats from file reading."""
        # Create a temporary XYZ file content
        xyz_content = """2
H2 molecule
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 1.400000
"""

        # Write to temporary file
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_filename = f.name

        try:
            # Test different output formats
            mol = from_file(temp_filename, output_format="quantum")
            assert isinstance(mol, Molecule)
            assert len(mol.atoms) == 2

            ase_atoms = from_file(temp_filename, output_format="ase")
            assert isinstance(ase_atoms, Atoms)
            assert len(ase_atoms) == 2

            if PYMATGEN_AVAILABLE:
                pymatgen_mol = from_file(temp_filename, output_format="pymatgen")
                assert len(pymatgen_mol) == 2

            if QCELEMENTAL_AVAILABLE:
                qc_mol = from_file(temp_filename, output_format="qcschema")
                assert len(qc_mol.symbols) == 2

        finally:
            # Clean up temporary file
            os.unlink(temp_filename)

    def test_invalid_output_format(self):
        """Test error handling for invalid output format."""
        import os
        import tempfile

        xyz_content = """1
H atom
H 0.000000 0.000000 0.000000
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_filename = f.name

        try:
            with pytest.raises(ValueError):
                from_file(temp_filename, output_format="invalid_format")
        finally:
            os.unlink(temp_filename)

    def test_to_file_functionality(self):
        """Test writing molecules to files."""
        import os
        import tempfile

        from quantum.core.converters import to_file

        # Create a molecule
        ase_atoms = Atoms(
            symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
        )

        # Test writing to XYZ file
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_filename = f.name

        try:
            to_file(ase_atoms, temp_filename, format="xyz")

            # Verify file was written correctly
            assert os.path.exists(temp_filename)

            # Read it back
            mol_back = from_file(temp_filename, output_format="quantum")
            assert len(mol_back.atoms) == 2
            assert mol_back.atoms == ["H", "H"]

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestUnitConversions:
    """Test unit conversion utilities."""

    def test_convert_units(self):
        """Test unit conversion between Angstrom and Bohr."""
        from quantum.core.converters import convert_units

        # Create atoms in Angstrom
        ase_atoms = Atoms(
            symbols=["H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # 1 Angstrom apart
        )

        # Convert to Bohr
        atoms_bohr = convert_units(ase_atoms, "angstrom", "bohr")

        # Check conversion factor (1 Angstrom ≈ 1.889726 Bohr)
        expected_distance_bohr = 1.0 / 0.52917721067
        actual_distance_bohr = atoms_bohr.get_positions()[1, 2]

        assert np.isclose(actual_distance_bohr, expected_distance_bohr, rtol=1e-6)

        # Convert back to Angstrom
        atoms_ang_back = convert_units(atoms_bohr, "bohr", "angstrom")

        # Should match original
        assert np.allclose(
            atoms_ang_back.get_positions(), ase_atoms.get_positions(), rtol=1e-10
        )

    def test_invalid_unit_conversion(self):
        """Test error handling for invalid unit conversions."""
        from quantum.core.converters import convert_units

        ase_atoms = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError):
            convert_units(ase_atoms, "invalid", "angstrom")

        with pytest.raises(ValueError):
            convert_units(ase_atoms, "angstrom", "invalid")
