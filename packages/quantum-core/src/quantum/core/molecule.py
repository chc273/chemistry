"""Molecular system representation with ASE/PyMatGen/QCSchema integration."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.core import Molecule as PyMatGenMolecule
from pymatgen.io.ase import AseAtomsAdaptor
from qcelemental.models import Molecule as QCMolecule

from .base import BaseSystem


class Molecule(BaseSystem):
    """
    Unified molecular system representation integrating ASE, PyMatGen, and QCSchema.

    This class acts as a wrapper around different molecular representations,
    providing seamless conversion between ASE Atoms, PyMatGen Molecule,
    and QCElemental Molecule objects.
    """

    def __init__(
        self,
        atoms: Atoms | PyMatGenMolecule | QCMolecule | None = None,
        name: str | None = None,
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs: Any,
    ):
        """
        Initialize Molecule from various input formats.

        Args:
            atoms: ASE Atoms, PyMatGen Molecule, or QCElemental Molecule object
            name: Molecule name
            charge: Total charge
            multiplicity: Spin multiplicity
            **kwargs: Additional parameters for BaseSystem
        """
        super().__init__(name=name, charge=charge, multiplicity=multiplicity, **kwargs)

        if atoms is None:
            self._ase_atoms = Atoms()
        elif isinstance(atoms, Atoms):
            self._ase_atoms = atoms.copy()
        elif isinstance(atoms, PyMatGenMolecule):
            self._ase_atoms = AseAtomsAdaptor.get_atoms(atoms)
        elif isinstance(atoms, QCMolecule):
            # Convert QCMolecule to ASE Atoms
            symbols = atoms.symbols
            positions = (
                np.array(atoms.geometry).reshape(-1, 3) * 0.52917721067
            )  # Bohr to Angstrom
            self._ase_atoms = Atoms(symbols=symbols, positions=positions)
        else:
            raise TypeError(f"Unsupported input type: {type(atoms)}")

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, **kwargs: Any) -> Molecule:
        """Create Molecule from ASE Atoms object."""
        return cls(atoms=atoms, **kwargs)

    @classmethod
    def from_pymatgen_molecule(
        cls, molecule: PyMatGenMolecule, **kwargs: Any
    ) -> Molecule:
        """Create Molecule from PyMatGen Molecule object."""
        return cls(atoms=molecule, **kwargs)

    @classmethod
    def from_qcschema(cls, qc_molecule: QCMolecule, **kwargs: Any) -> Molecule:
        """Create Molecule from QCElemental Molecule object."""
        return cls(atoms=qc_molecule, **kwargs)

    @classmethod
    def from_file(
        cls, filename: str, format: str | None = None, **kwargs: Any
    ) -> Molecule:
        """
        Create molecule from file using ASE's extensive format support.

        Supports 30+ file formats including XYZ, CIF, POSCAR, PDB, etc.
        """
        atoms = ase_read(filename, format=format)
        return cls.from_ase_atoms(atoms, **kwargs)

    @classmethod
    def from_xyz_string(cls, xyz_string: str, **kwargs: Any) -> Molecule:
        """Create molecule from XYZ format string."""
        lines = xyz_string.strip().split("\n")
        num_atoms = int(lines[0])

        symbols = []
        positions = []

        for line in lines[2 : 2 + num_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])

        atoms = Atoms(symbols=symbols, positions=positions)
        return cls.from_ase_atoms(atoms, **kwargs)

    @classmethod
    def from_xyz_file(cls, filename: str, **kwargs: Any) -> Molecule:
        """Create molecule from XYZ file."""
        return cls.from_file(filename, format="xyz", **kwargs)

    def to_ase_atoms(self) -> Atoms:
        """Convert to ASE Atoms object."""
        return self._ase_atoms.copy()

    def to_pymatgen_molecule(self) -> PyMatGenMolecule:
        """Convert to PyMatGen Molecule object."""
        return AseAtomsAdaptor.get_molecule(self._ase_atoms)

    def to_qcschema(self) -> QCMolecule:
        """Convert to QCElemental Molecule object (QCSchema format)."""
        symbols = self._ase_atoms.get_chemical_symbols()
        # Convert Angstrom to Bohr for QCSchema
        geometry = self._ase_atoms.get_positions().flatten() / 0.52917721067

        return QCMolecule(
            symbols=symbols,
            geometry=geometry.tolist(),
            molecular_charge=self.charge,
            molecular_multiplicity=self.multiplicity,
        )

    def to_file(self, filename: str, format: str | None = None) -> None:
        """Write molecule to file using ASE's format support."""
        ase_write(filename, self._ase_atoms, format=format)

    def to_xyz_string(self) -> str:
        """Convert molecule to XYZ format string."""
        symbols = self._ase_atoms.get_chemical_symbols()
        positions = self._ase_atoms.get_positions()

        lines = [str(len(symbols)), self.name or ""]

        for symbol, pos in zip(symbols, positions, strict=False):
            line = f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
            lines.append(line)

        return "\\n".join(lines)

    def to_xyz_file(self, filename: str) -> None:
        """Write molecule to XYZ file."""
        self.to_file(filename, format="xyz")

    # Properties and methods compatible with original interface
    @property
    def atoms(self) -> list[str]:
        """Get list of atomic symbols."""
        return self._ase_atoms.get_chemical_symbols()

    @property
    def coordinates(self) -> np.ndarray:
        """Get atomic coordinates in Angstroms."""
        return self._ase_atoms.get_positions()

    def get_geometry(self) -> np.ndarray:
        """Return molecular geometry."""
        return self.coordinates.copy()

    def get_atomic_numbers(self) -> list[int]:
        """Return list of atomic numbers."""
        return self._ase_atoms.get_atomic_numbers().tolist()

    def get_atomic_masses(self) -> list[float]:
        """Get atomic masses in atomic mass units."""
        return self._ase_atoms.get_masses().tolist()

    def compute_nuclear_repulsion(self) -> float:
        """Compute nuclear repulsion energy in Hartree."""
        positions = self.coordinates
        atomic_numbers = self.get_atomic_numbers()

        energy = 0.0
        for i in range(len(atomic_numbers)):
            for j in range(i + 1, len(atomic_numbers)):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                # Convert Angstrom to Bohr for atomic units
                r_ij_bohr = r_ij / 0.52917721067
                energy += atomic_numbers[i] * atomic_numbers[j] / r_ij_bohr

        return energy

    def get_bond_lengths(self) -> list[tuple[int, int, float]]:
        """Calculate all bond lengths in Angstroms."""
        positions = self.coordinates
        bonds = []

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                bonds.append((i, j, distance))

        return bonds

    def get_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        return self._ase_atoms.get_center_of_mass()

    def translate(self, vector: np.ndarray) -> Molecule:
        """Translate molecule by given vector."""
        new_atoms = self._ase_atoms.copy()
        new_atoms.translate(vector)
        return Molecule.from_ase_atoms(
            new_atoms,
            name=self.name,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )

    def center_at_origin(self) -> Molecule:
        """Center molecule at origin (center of mass)."""
        com = self.get_center_of_mass()
        return self.translate(-com)

    def rotate(self, angle: float, axis: str | np.ndarray) -> Molecule:
        """Rotate molecule around given axis."""
        new_atoms = self._ase_atoms.copy()
        if isinstance(axis, str):
            axis_vector = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis.lower()]
        else:
            axis_vector = axis

        new_atoms.rotate(angle, axis_vector)
        return Molecule.from_ase_atoms(
            new_atoms,
            name=self.name,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )

    def get_distance_matrix(self) -> np.ndarray:
        """Get distance matrix between all atoms."""
        return self._ase_atoms.get_all_distances()

    def get_num_electrons(self) -> int:
        """Get total number of electrons."""
        return sum(self.get_atomic_numbers()) - self.charge

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._ase_atoms)

    def __repr__(self) -> str:
        """String representation."""
        formula = self._ase_atoms.get_chemical_formula()
        return f"Molecule(formula={formula}, charge={self.charge}, multiplicity={self.multiplicity})"


# Compatibility aliases for common usage patterns
MolecularSystem = Molecule  # Alias for compatibility
