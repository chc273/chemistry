"""Universal molecular format conversion utilities."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.core import Molecule as PyMatGenMolecule
from pymatgen.core import Structure as PyMatGenStructure
from pymatgen.io.ase import AseAtomsAdaptor
from qcelemental.models import Molecule as QCMolecule

from .molecule import Molecule

# Type aliases for supported molecular formats
MolecularFormat = Union[
    Atoms, PyMatGenMolecule, PyMatGenStructure, QCMolecule, Molecule
]


def to_ase_atoms(obj: MolecularFormat) -> Atoms:
    """
    Convert any supported molecular format to ASE Atoms.

    Args:
        obj: Molecular object in any supported format

    Returns:
        ASE Atoms object

    Raises:
        TypeError: If input format is not supported
    """
    if isinstance(obj, Atoms):
        return obj.copy()
    elif isinstance(obj, Molecule):
        return obj.to_ase_atoms()
    elif isinstance(obj, (PyMatGenMolecule, PyMatGenStructure)):
        return AseAtomsAdaptor.get_atoms(obj)
    elif isinstance(obj, QCMolecule):
        symbols = obj.symbols
        # Convert Bohr to Angstrom
        positions = np.array(obj.geometry).reshape(-1, 3) * 0.52917721067
        return Atoms(symbols=symbols, positions=positions)
    else:
        raise TypeError(f"Unsupported molecular format: {type(obj)}")


def to_pymatgen_molecule(obj: MolecularFormat) -> PyMatGenMolecule:
    """
    Convert any supported molecular format to PyMatGen Molecule.

    Args:
        obj: Molecular object in any supported format

    Returns:
        PyMatGen Molecule object
    """
    if isinstance(obj, PyMatGenMolecule):
        return obj
    elif isinstance(obj, Molecule):
        return obj.to_pymatgen_molecule()
    else:
        # Convert to ASE first, then to PyMatGen
        ase_atoms = to_ase_atoms(obj)
        return AseAtomsAdaptor.get_molecule(ase_atoms)


def to_pymatgen_structure(obj: MolecularFormat) -> PyMatGenStructure:
    """
    Convert any supported molecular format to PyMatGen Structure.

    Args:
        obj: Molecular object in any supported format

    Returns:
        PyMatGen Structure object
    """
    if isinstance(obj, PyMatGenStructure):
        return obj
    else:
        # Convert to ASE first, then to PyMatGen Structure
        ase_atoms = to_ase_atoms(obj)
        return AseAtomsAdaptor.get_structure(ase_atoms)


def to_qcschema(
    obj: MolecularFormat, charge: int = 0, multiplicity: int = 1
) -> QCMolecule:
    """
    Convert any supported molecular format to QCElemental Molecule (QCSchema).

    Args:
        obj: Molecular object in any supported format
        charge: Molecular charge
        multiplicity: Spin multiplicity

    Returns:
        QCElemental Molecule object
    """
    if isinstance(obj, QCMolecule):
        return obj
    elif isinstance(obj, Molecule):
        return obj.to_qcschema()
    else:
        # Convert to ASE first
        ase_atoms = to_ase_atoms(obj)
        symbols = ase_atoms.get_chemical_symbols()
        # Convert Angstrom to Bohr for QCSchema
        geometry = ase_atoms.get_positions().flatten() / 0.52917721067

        return QCMolecule(
            symbols=symbols,
            geometry=geometry.tolist(),
            molecular_charge=charge,
            molecular_multiplicity=multiplicity,
        )


def to_quantum_molecule(obj: MolecularFormat, **kwargs: Any) -> Molecule:
    """
    Convert any supported molecular format to quantum.core.Molecule.

    Args:
        obj: Molecular object in any supported format
        **kwargs: Additional parameters for Molecule constructor

    Returns:
        quantum.core.Molecule object
    """
    if isinstance(obj, Molecule):
        return obj
    elif isinstance(obj, Atoms):
        return Molecule.from_ase_atoms(obj, **kwargs)
    elif isinstance(obj, PyMatGenMolecule):
        return Molecule.from_pymatgen_molecule(obj, **kwargs)
    elif isinstance(obj, QCMolecule):
        return Molecule.from_qcschema(obj, **kwargs)
    else:
        raise TypeError(f"Unsupported molecular format: {type(obj)}")


def from_file(
    filename: str, format: str | None = None, output_format: str = "quantum"
) -> Molecule | Atoms | PyMatGenMolecule | QCMolecule:
    """
    Read molecular structure from file and convert to desired format.

    Args:
        filename: Path to molecular structure file
        format: Input file format (auto-detected if None)
        output_format: Desired output format ('quantum', 'ase', 'pymatgen', 'qcschema')

    Returns:
        Molecular object in specified format

    Raises:
        ValueError: If output format is not supported
    """
    # Read using ASE (supports 30+ formats)
    ase_atoms = ase_read(filename, format=format)

    if output_format.lower() == "quantum":
        return Molecule.from_ase_atoms(ase_atoms)
    elif output_format.lower() == "ase":
        return ase_atoms
    elif output_format.lower() == "pymatgen":
        return AseAtomsAdaptor.get_molecule(ase_atoms)
    elif output_format.lower() == "qcschema":
        return to_qcschema(ase_atoms)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def to_file(obj: MolecularFormat, filename: str, format: str | None = None) -> None:
    """
    Write molecular structure to file.

    Args:
        obj: Molecular object in any supported format
        filename: Output filename
        format: Output file format (auto-detected from extension if None)
    """
    ase_atoms = to_ase_atoms(obj)
    ase_write(filename, ase_atoms, format=format)


def get_supported_formats() -> dict[str, list[str]]:
    """
    Get dictionary of supported file formats for reading and writing.

    Returns:
        Dictionary with 'read' and 'write' keys containing lists of formats
    """
    from ase.io.formats import ioformats

    read_formats = []
    write_formats = []

    for fmt_name, fmt_info in ioformats.items():
        if fmt_info.get("read", False):
            read_formats.append(fmt_name)
        if fmt_info.get("write", False):
            write_formats.append(fmt_name)

    return {
        "read": sorted(read_formats),
        "write": sorted(write_formats),
        "common": sorted(set(read_formats) & set(write_formats)),
    }


def convert_units(
    obj: MolecularFormat, from_unit: str, to_unit: str
) -> Atoms | np.ndarray:
    """
    Convert between different unit systems.

    Args:
        obj: Molecular object
        from_unit: Source unit ('angstrom', 'bohr')
        to_unit: Target unit ('angstrom', 'bohr')

    Returns:
        Converted object or positions array
    """
    conversion_factors = {
        ("angstrom", "bohr"): 1.0 / 0.52917721067,
        ("bohr", "angstrom"): 0.52917721067,
        ("angstrom", "angstrom"): 1.0,
        ("bohr", "bohr"): 1.0,
    }

    factor = conversion_factors.get((from_unit.lower(), to_unit.lower()))
    if factor is None:
        raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit}")

    if factor == 1.0:
        return obj

    ase_atoms = to_ase_atoms(obj)
    new_positions = ase_atoms.get_positions() * factor

    new_atoms = ase_atoms.copy()
    new_atoms.set_positions(new_positions)

    return new_atoms


# Convenience functions for common conversions
def xyz_to_qcschema(
    xyz_string: str, charge: int = 0, multiplicity: int = 1
) -> QCMolecule:
    """Convert XYZ string directly to QCSchema format."""
    mol = Molecule.from_xyz_string(xyz_string, charge=charge, multiplicity=multiplicity)
    return mol.to_qcschema()


def qcschema_to_xyz(qc_molecule: QCMolecule) -> str:
    """Convert QCSchema molecule directly to XYZ string."""
    mol = Molecule.from_qcschema(qc_molecule)
    return mol.to_xyz_string()
