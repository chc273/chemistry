"""Unit tests for Molecule class with ASE/PyMatGen integration."""

import numpy as np
from ase import Atoms

from quantum.core import Molecule


class TestMolecule:
    """Test cases for integrated Molecule class."""

    def test_molecule_creation_from_ase_atoms(self):
        """Test molecule creation from ASE Atoms."""
        atoms = Atoms(
            symbols=["C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        )

        mol = Molecule(
            atoms=atoms,
            name="methane",
            charge=0,
            multiplicity=1,
        )

        assert mol.name == "methane"
        assert len(mol.atoms) == 5
        assert mol.charge == 0
        assert mol.multiplicity == 1
        assert mol.coordinates.shape == (5, 3)

    def test_molecule_from_xyz_string(self):
        """Test molecule creation from XYZ string."""
        xyz_string = """5
methane molecule
C    0.000000    0.000000    0.000000
H    1.000000    0.000000    0.000000
H   -1.000000    0.000000    0.000000
H    0.000000    1.000000    0.000000
H    0.000000   -1.000000    0.000000
"""

        mol = Molecule.from_xyz_string(xyz_string, name="test_methane")

        assert mol.name == "test_methane"
        assert len(mol.atoms) == 5
        assert mol.atoms[0] == "C"
        assert mol.atoms[1] == "H"
        assert np.allclose(mol.coordinates[0], [0.0, 0.0, 0.0])

    def test_get_atomic_numbers(self):
        """Test atomic number retrieval."""
        atoms = Atoms(symbols=["H", "C", "N", "O"])
        mol = Molecule(atoms=atoms, name="test")
        atomic_nums = mol.get_atomic_numbers()

        assert atomic_nums == [1, 6, 7, 8]

    def test_get_num_electrons(self):
        """Test electron count calculation."""
        atoms = Atoms(symbols=["C", "H", "H", "H", "H"])

        # Neutral molecule
        mol = Molecule(atoms=atoms, name="methane", charge=0)
        assert mol.get_num_electrons() == 10  # 6 + 4*1

        # Charged molecule
        mol_charged = Molecule(atoms=atoms, name="methane_ion", charge=1)
        assert mol_charged.get_num_electrons() == 9  # 10 - 1

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        atoms = Atoms(
            symbols=["C", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
        )

        mol = Molecule(atoms=atoms, name="test")
        com = mol.get_center_of_mass()

        # Should be close to origin due to symmetry
        assert np.allclose(com, [0.0, 0.0, 0.0], atol=0.1)

    def test_translation(self):
        """Test molecule translation."""
        atoms = Atoms(symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        mol = Molecule(atoms=atoms, name="H2")

        # Translate by [1, 1, 1]
        translated_mol = mol.translate(np.array([1.0, 1.0, 1.0]))

        expected_coords = mol.coordinates + np.array([1.0, 1.0, 1.0])
        assert np.allclose(translated_mol.coordinates, expected_coords)

        # Original molecule should be unchanged
        assert np.allclose(mol.coordinates, atoms.get_positions())

    def test_bond_lengths(self):
        """Test bond length calculation."""
        atoms = Atoms(symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        mol = Molecule(atoms=atoms, name="H2")
        bonds = mol.get_bond_lengths()

        assert len(bonds) == 1
        assert bonds[0][0] == 0  # First atom index
        assert bonds[0][1] == 1  # Second atom index
        assert np.isclose(bonds[0][2], 1.0)  # Distance

    def test_to_xyz_string(self):
        """Test XYZ string generation."""
        atoms = Atoms(
            symbols=["C", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        mol = Molecule(atoms=atoms, name="test_molecule")
        xyz_string = mol.to_xyz_string()

        lines = xyz_string.strip().split("\\n")
        assert lines[0] == "3"  # Number of atoms
        assert lines[1] == "test_molecule"  # Name
        assert "C " in lines[2]  # First atom line
        assert "H " in lines[3]  # Second atom line

    def test_format_conversions(self):
        """Test conversions between different formats."""
        # Create molecule from ASE
        ase_atoms = Atoms(
            symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        )
        mol = Molecule.from_ase_atoms(ase_atoms, name="H2")

        # Test ASE conversion
        converted_ase = mol.to_ase_atoms()
        assert len(converted_ase) == 2
        assert converted_ase.get_chemical_symbols() == ["H", "H"]

        # Test PyMatGen conversion
        pymatgen_mol = mol.to_pymatgen_molecule()
        assert len(pymatgen_mol) == 2

        # Test QCSchema conversion
        qc_mol = mol.to_qcschema()
        assert len(qc_mol.symbols) == 2
        assert qc_mol.symbols == ["H", "H"]

    def test_compute_nuclear_repulsion(self):
        """Test nuclear repulsion energy calculation."""
        atoms = Atoms(
            symbols=["H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],  # 1.4 Bohr ≈ 0.74 Å
        )
        mol = Molecule(atoms=atoms, name="H2")

        repulsion = mol.compute_nuclear_repulsion()
        # Should be approximately 1 * 1 / 1.4 ≈ 0.714 Hartree
        assert repulsion > 0.5
        assert repulsion < 1.0

    def test_rotation(self):
        """Test molecule rotation."""
        atoms = Atoms(symbols=["H", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mol = Molecule(atoms=atoms, name="H2")

        # Rotate 90 degrees around z-axis
        rotated_mol = mol.rotate(np.pi / 2, "z")

        # Second H atom should now be at approximately [0, 1, 0]
        expected_pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(rotated_mol.coordinates, expected_pos, atol=1e-10)

    def test_distance_matrix(self):
        """Test distance matrix calculation."""
        atoms = Atoms(
            symbols=["H", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        mol = Molecule(atoms=atoms, name="H3")

        dist_matrix = mol.get_distance_matrix()

        assert dist_matrix.shape == (3, 3)
        assert np.isclose(dist_matrix[0, 1], 1.0)  # Distance H1-H2
        assert np.isclose(dist_matrix[0, 2], 1.0)  # Distance H1-H3
        assert np.isclose(dist_matrix[1, 2], np.sqrt(2))  # Distance H2-H3
