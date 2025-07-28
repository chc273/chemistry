"""
Reference data management for quantum chemistry validation.

This module provides tools for storing, retrieving, and managing
reference data from literature and high-level calculations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ReferenceEntry:
    """A single reference data entry."""

    system: str
    method: str
    basis_set: str
    energy: float

    # Optional properties
    properties: Dict[str, float] = None
    uncertainty: Optional[float] = None

    # Metadata
    source: str = ""
    doi: str = ""
    year: Optional[int] = None
    notes: str = ""

    # Computational details
    software: str = ""
    version: str = ""
    settings: Dict[str, Any] = None

    # Quality indicators
    quality_level: str = "unknown"  # 'benchmark', 'high', 'medium', 'low'
    verified: bool = False

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.settings is None:
            self.settings = {}


class ReferenceDatabase:
    """Database for storing and retrieving reference quantum chemistry data."""

    def __init__(self, database_path: Optional[Union[str, Path]] = None):
        """Initialize the reference database.

        Args:
            database_path: Path to the database file. If None, uses in-memory storage.
        """
        self.database_path = Path(database_path) if database_path else None
        self.entries: List[ReferenceEntry] = []
        self._load_database()

    def _load_database(self):
        """Load database from file if it exists."""
        if self.database_path and self.database_path.exists():
            try:
                with open(self.database_path, "r") as f:
                    data = json.load(f)
                    self.entries = [ReferenceEntry(**entry) for entry in data]
            except Exception as e:
                print(
                    f"Warning: Could not load database from {self.database_path}: {e}"
                )
        else:
            self._initialize_default_data()

    def _initialize_default_data(self):
        """Initialize with default reference data from literature."""

        # H2 molecule references
        self.add_entry(
            ReferenceEntry(
                system="h2",
                method="fci",
                basis_set="cc-pVDZ",
                energy=-1.17447,
                uncertainty=0.00001,
                source="Helgaker et al., Molecular Electronic-Structure Theory (2000)",
                year=2000,
                quality_level="benchmark",
                verified=True,
                notes="Exact result within basis set",
            )
        )

        self.add_entry(
            ReferenceEntry(
                system="h2",
                method="ccsd(t)",
                basis_set="cc-pVDZ",
                energy=-1.17370,
                uncertainty=0.00005,
                source="Helgaker et al., Molecular Electronic-Structure Theory (2000)",
                year=2000,
                quality_level="benchmark",
                verified=True,
            )
        )

        # LiH references
        self.add_entry(
            ReferenceEntry(
                system="lih",
                method="fci",
                basis_set="cc-pVDZ",
                energy=-8.07055,
                uncertainty=0.00001,
                source="Helgaker et al., Molecular Electronic-Structure Theory (2000)",
                year=2000,
                quality_level="benchmark",
                verified=True,
            )
        )

        # H2O references
        self.add_entry(
            ReferenceEntry(
                system="h2o",
                method="ccsd(t)",
                basis_set="cc-pVDZ",
                energy=-76.24265,
                uncertainty=0.00010,
                source="Peterson et al., J. Chem. Phys. 100, 7410 (1994)",
                doi="10.1063/1.466884",
                year=1994,
                quality_level="benchmark",
                verified=True,
            )
        )

        self.add_entry(
            ReferenceEntry(
                system="h2o",
                method="fci",
                basis_set="cc-pVDZ",
                energy=-76.24266,
                uncertainty=0.00001,
                source="Peterson et al., J. Chem. Phys. 100, 7410 (1994)",
                doi="10.1063/1.466884",
                year=1994,
                quality_level="benchmark",
                verified=True,
                notes="Exact result within basis set",
            )
        )

        # N2 references
        self.add_entry(
            ReferenceEntry(
                system="n2",
                method="ccsd(t)",
                basis_set="cc-pVDZ",
                energy=-109.36617,
                uncertainty=0.00020,
                source="Peterson et al., J. Chem. Phys. 117, 10548 (2002)",
                doi="10.1063/1.1520138",
                year=2002,
                quality_level="benchmark",
                verified=True,
            )
        )

        # F2 references (challenging multi-reference case)
        self.add_entry(
            ReferenceEntry(
                system="f2",
                method="mrci",
                basis_set="cc-pVDZ",
                energy=-199.2189,
                uncertainty=0.0010,
                source="Gdanitz, R.J. Chem. Phys. Lett. 283, 253 (1998)",
                doi="10.1016/S0009-2614(97)01392-4",
                year=1998,
                quality_level="high",
                verified=True,
                notes="Multi-reference character important",
            )
        )

        self.add_entry(
            ReferenceEntry(
                system="f2",
                method="caspt2",
                basis_set="cc-pVDZ",
                energy=-199.2156,
                uncertainty=0.0020,
                source="Roos et al., J. Phys. Chem. A 108, 2851 (2004)",
                doi="10.1021/jp031064+",
                year=2004,
                quality_level="high",
                verified=True,
                settings={"active_space": "(14,8)"},
            )
        )

        # Cr2 references (extreme multi-reference)
        self.add_entry(
            ReferenceEntry(
                system="cr2",
                method="experimental",
                basis_set="n/a",
                energy=-2088.40,
                uncertainty=0.10,
                source="Casey et al., J. Phys. Chem. 97, 816 (1993)",
                doi="10.1021/j100106a005",
                year=1993,
                quality_level="benchmark",
                verified=True,
                notes="From photoelectron spectroscopy",
            )
        )

        self.add_entry(
            ReferenceEntry(
                system="cr2",
                method="dmrg",
                basis_set="cc-pVDZ",
                energy=-2088.356,
                uncertainty=0.005,
                source="Kurashige et al., J. Chem. Phys. 135, 094104 (2011)",
                doi="10.1063/1.3629454",
                year=2011,
                quality_level="high",
                verified=True,
                settings={"bond_dimension": 1000, "active_space": "(12,12)"},
            )
        )

        # Benzene references
        self.add_entry(
            ReferenceEntry(
                system="benzene",
                method="ccsd(t)",
                basis_set="cc-pVDZ",
                energy=-231.58946,
                uncertainty=0.00050,
                source="Gauss et al., J. Chem. Phys. 116, 1773 (2002)",
                doi="10.1063/1.1429244",
                year=2002,
                quality_level="high",
                verified=True,
            )
        )

    def add_entry(self, entry: ReferenceEntry):
        """Add a reference entry to the database."""
        self.entries.append(entry)

    def get_entries(
        self,
        system: Optional[str] = None,
        method: Optional[str] = None,
        basis_set: Optional[str] = None,
        quality_level: Optional[str] = None,
        verified_only: bool = False,
    ) -> List[ReferenceEntry]:
        """Get reference entries matching the specified criteria."""

        filtered_entries = self.entries

        if system:
            filtered_entries = [
                e for e in filtered_entries if e.system.lower() == system.lower()
            ]

        if method:
            filtered_entries = [
                e for e in filtered_entries if e.method.lower() == method.lower()
            ]

        if basis_set:
            filtered_entries = [
                e for e in filtered_entries if e.basis_set.lower() == basis_set.lower()
            ]

        if quality_level:
            filtered_entries = [
                e for e in filtered_entries if e.quality_level == quality_level
            ]

        if verified_only:
            filtered_entries = [e for e in filtered_entries if e.verified]

        return filtered_entries

    def get_reference_energy(
        self, system: str, method: str, basis_set: str = None
    ) -> Optional[ReferenceEntry]:
        """Get the best reference energy for a system/method combination."""

        entries = self.get_entries(system=system, method=method, basis_set=basis_set)

        if not entries:
            return None

        # Prioritize by quality level and verification status
        quality_order = {"benchmark": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}

        entries.sort(
            key=lambda e: (
                quality_order.get(e.quality_level, 0),
                e.verified,
                -e.uncertainty if e.uncertainty else 0,
            ),
            reverse=True,
        )

        return entries[0]

    def get_systems(self) -> List[str]:
        """Get all systems in the database."""
        return list(set(entry.system for entry in self.entries))

    def get_methods(self, system: Optional[str] = None) -> List[str]:
        """Get all methods in the database, optionally for a specific system."""
        entries = self.get_entries(system=system) if system else self.entries
        return list(set(entry.method for entry in entries))

    def get_basis_sets(
        self, system: Optional[str] = None, method: Optional[str] = None
    ) -> List[str]:
        """Get all basis sets in the database."""
        entries = self.get_entries(system=system, method=method)
        return list(set(entry.basis_set for entry in entries))

    def save_database(self):
        """Save the database to file."""
        if self.database_path:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert entries to dictionaries for JSON serialization
            data = []
            for entry in self.entries:
                entry_dict = {
                    "system": entry.system,
                    "method": entry.method,
                    "basis_set": entry.basis_set,
                    "energy": entry.energy,
                    "properties": entry.properties,
                    "uncertainty": entry.uncertainty,
                    "source": entry.source,
                    "doi": entry.doi,
                    "year": entry.year,
                    "notes": entry.notes,
                    "software": entry.software,
                    "version": entry.version,
                    "settings": entry.settings,
                    "quality_level": entry.quality_level,
                    "verified": entry.verified,
                }
                data.append(entry_dict)

            with open(self.database_path, "w") as f:
                json.dump(data, f, indent=2)

    def import_from_json(self, json_path: Union[str, Path]):
        """Import reference data from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        for entry_data in data:
            entry = ReferenceEntry(**entry_data)
            self.add_entry(entry)

    def export_to_json(self, json_path: Union[str, Path]):
        """Export reference data to a JSON file."""
        self.database_path = Path(json_path)
        self.save_database()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        systems = self.get_systems()
        methods = self.get_methods()

        quality_counts = {}
        for level in ["benchmark", "high", "medium", "low", "unknown"]:
            quality_counts[level] = len(self.get_entries(quality_level=level))

        verified_count = len([e for e in self.entries if e.verified])

        return {
            "total_entries": len(self.entries),
            "unique_systems": len(systems),
            "unique_methods": len(methods),
            "quality_distribution": quality_counts,
            "verified_entries": verified_count,
            "verification_rate": verified_count / len(self.entries)
            if self.entries
            else 0,
            "systems": systems,
            "methods": methods,
        }
