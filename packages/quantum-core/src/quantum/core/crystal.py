"""Crystal structure representation for materials science calculations."""

import numpy as np
from pydantic import Field, validator

from .base import BaseSystem


class Crystal(BaseSystem):
    """Represents a crystal structure for materials science calculations."""

    lattice_vectors: np.ndarray = Field(
        description="3x3 matrix of lattice vectors in Angstroms"
    )
    fractional_coordinates: np.ndarray = Field(
        description="Fractional coordinates of atoms in unit cell"
    )
    atoms: list[str] = Field(description="List of atomic symbols")
    space_group: str | None = Field(default=None, description="Space group")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @validator("lattice_vectors")
    def validate_lattice(cls, v):
        """Validate lattice vectors."""
        if v.shape != (3, 3):
            raise ValueError("Lattice vectors must be 3x3 matrix")
        return v

    @validator("fractional_coordinates")
    def validate_frac_coords(cls, v, values):
        """Validate fractional coordinates."""
        if "atoms" in values and len(values["atoms"]) != len(v):
            raise ValueError("Number of atoms must match coordinates")
        if v.shape[1] != 3:
            raise ValueError("Fractional coordinates must be Nx3 array")
        return v

    @classmethod
    def from_cif_file(cls, filename: str, **kwargs) -> "Crystal":
        """Create crystal from CIF file (simplified implementation)."""
        # This would typically use a library like pymatgen
        raise NotImplementedError("CIF parsing requires pymatgen integration")

    @classmethod
    def cubic_cell(
        cls, atoms: list[str], lattice_parameter: float, **kwargs
    ) -> "Crystal":
        """Create a cubic unit cell."""
        lattice = np.eye(3) * lattice_parameter

        # Simple cubic structure
        if len(atoms) == 1:
            frac_coords = np.array([[0.0, 0.0, 0.0]])
        else:
            # Distribute atoms equally in the unit cell
            n_atoms = len(atoms)
            frac_coords = np.random.random((n_atoms, 3))

        return cls(
            lattice_vectors=lattice,
            fractional_coordinates=frac_coords,
            atoms=atoms,
            **kwargs,
        )

    def get_geometry(self) -> np.ndarray:
        """Return Cartesian coordinates of atoms."""
        return self.fractional_coordinates @ self.lattice_vectors

    def get_atomic_numbers(self) -> list[int]:
        """Return list of atomic numbers."""
        atomic_numbers = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            "Ti": 22,
            "Fe": 26,
        }
        return [atomic_numbers[atom] for atom in self.atoms]

    def compute_nuclear_repulsion(self) -> float:
        """Compute nuclear repulsion energy (simplified for unit cell)."""
        cartesian_coords = self.get_geometry()
        energy = 0.0
        atomic_nums = self.get_atomic_numbers()

        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                r_ij = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])
                if r_ij > 0:  # Avoid division by zero
                    energy += atomic_nums[i] * atomic_nums[j] / r_ij

        return energy

    def get_volume(self) -> float:
        """Calculate unit cell volume."""
        return abs(np.linalg.det(self.lattice_vectors))

    def get_density(self) -> float:
        """Calculate density in g/cm³."""
        masses = self._get_atomic_masses()
        total_mass = sum(masses) * 1.66054e-24  # Convert amu to grams
        volume = self.get_volume() * 1e-24  # Convert Ų to cm³
        return total_mass / volume

    def get_lattice_parameters(self) -> tuple[float, float, float, float, float, float]:
        """Get lattice parameters (a, b, c, alpha, beta, gamma)."""
        a_vec, b_vec, c_vec = self.lattice_vectors

        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)

        alpha = np.arccos(np.dot(b_vec, c_vec) / (b * c)) * 180 / np.pi
        beta = np.arccos(np.dot(a_vec, c_vec) / (a * c)) * 180 / np.pi
        gamma = np.arccos(np.dot(a_vec, b_vec) / (a * b)) * 180 / np.pi

        return a, b, c, alpha, beta, gamma

    def supercell(self, nx: int, ny: int, nz: int) -> "Crystal":
        """Create supercell with given dimensions."""
        new_lattice = self.lattice_vectors * np.array([nx, ny, nz])

        new_atoms = []
        new_frac_coords = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for atom, coord in zip(
                        self.atoms, self.fractional_coordinates, strict=False
                    ):
                        new_atoms.append(atom)
                        new_coord = (coord + np.array([i, j, k])) / np.array(
                            [nx, ny, nz]
                        )
                        new_frac_coords.append(new_coord)

        return Crystal(
            name=f"{self.name}_supercell_{nx}x{ny}x{nz}",
            lattice_vectors=new_lattice,
            fractional_coordinates=np.array(new_frac_coords),
            atoms=new_atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            space_group=self.space_group,
        )

    def _get_atomic_masses(self) -> list[float]:
        """Get atomic masses in atomic mass units."""
        masses = {
            "H": 1.008,
            "He": 4.003,
            "Li": 6.941,
            "Be": 9.012,
            "B": 10.811,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Ne": 20.180,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.086,
            "P": 30.974,
            "S": 32.065,
            "Cl": 35.453,
            "Ar": 39.948,
            "K": 39.098,
            "Ca": 40.078,
            "Ti": 47.867,
            "Fe": 55.845,
        }
        return [masses[atom] for atom in self.atoms]
