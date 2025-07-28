"""
Benchmark dataset management for multireference methods validation.

This module provides tools for managing and curating benchmark datasets
for systematic validation of multireference quantum chemistry methods.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator
from pyscf import gto

from ..base import MultireferenceResult


class SystemType(str, Enum):
    """Types of chemical systems for benchmarking."""
    
    ORGANIC = "organic"
    TRANSITION_METAL = "transition_metal"
    DIATOMIC = "diatomic"
    EXCITATION = "excitation"
    SPIN_STATE = "spin_state"
    BOND_DISSOCIATION = "bond_dissociation"


class BenchmarkMolecule(BaseModel):
    """
    Molecular system specification for benchmarking.
    
    This class defines a molecular system with all necessary metadata
    for systematic benchmarking studies.
    """
    
    name: str = Field(..., description="Systematic name for the molecule")
    atoms: List[Tuple[str, float, float, float]] = Field(..., description="Atomic coordinates")
    charge: int = Field(0, description="Total molecular charge")
    multiplicity: int = Field(1, description="Spin multiplicity (2S+1)")
    basis_set: str = Field("sto-3g", description="Basis set specification")
    system_type: SystemType = Field(..., description="Chemical system classification")
    
    # Metadata
    source: str = Field(..., description="Literature source or database")
    doi: Optional[str] = Field(None, description="DOI of source paper")
    experimental_data: Optional[Dict[str, float]] = Field(None, description="Experimental reference values")
    theoretical_references: Optional[Dict[str, float]] = Field(None, description="High-level theoretical references")
    
    # Computational details
    keywords: List[str] = Field(default_factory=list, description="Associated keywords/tags")
    difficulty_level: str = Field("standard", description="Expected computational difficulty")
    
    @validator('atoms')
    def validate_atoms(cls, v):
        """Validate atomic coordinate format."""
        if not v:
            raise ValueError("Atoms list cannot be empty")
        
        for atom in v:
            if len(atom) != 4:
                raise ValueError("Each atom must have format (symbol, x, y, z)")
            if not isinstance(atom[0], str):
                raise ValueError("Atomic symbol must be string")
            if not all(isinstance(coord, (int, float)) for coord in atom[1:]):
                raise ValueError("Coordinates must be numeric")
        
        return v
    
    def to_pyscf_molecule(self) -> gto.Mole:
        """Convert to PySCF Mole object."""
        mol = gto.Mole()
        
        # Build atom string
        atom_str = []
        for symbol, x, y, z in self.atoms:
            atom_str.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
        
        mol.atom = "; ".join(atom_str)
        mol.charge = self.charge
        mol.spin = self.multiplicity - 1
        mol.basis = self.basis_set
        
        try:
            mol.build()
        except Exception as e:
            raise ValueError(f"Failed to build PySCF molecule: {e}")
        
        return mol
    
    def get_system_hash(self) -> str:
        """Generate unique hash for this molecular system."""
        # Create deterministic representation
        system_data = {
            'atoms': sorted(self.atoms),
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'basis_set': self.basis_set
        }
        
        system_str = json.dumps(system_data, sort_keys=True)
        return hashlib.md5(system_str.encode()).hexdigest()[:8]


class BenchmarkEntry(BaseModel):
    """
    Single benchmark calculation entry with results and metadata.
    """
    
    system: BenchmarkMolecule = Field(..., description="Molecular system")
    method: str = Field(..., description="Quantum chemistry method used")
    
    # Active space information
    active_space_method: str = Field(..., description="Active space selection method")
    n_active_electrons: int = Field(..., description="Number of active electrons")
    n_active_orbitals: int = Field(..., description="Number of active orbitals")
    
    # Results
    energy: float = Field(..., description="Total energy in Hartree")
    correlation_energy: Optional[float] = Field(None, description="Correlation energy")
    properties: Optional[Dict[str, float]] = Field(None, description="Additional properties")
    
    # Validation and comparison
    reference_energy: Optional[float] = Field(None, description="Reference energy for comparison")
    absolute_error: Optional[float] = Field(None, description="Absolute error vs reference")
    relative_error: Optional[float] = Field(None, description="Relative error percentage")
    
    # Computational metadata
    computational_cost: Dict[str, float] = Field(default_factory=dict, description="Resource usage")
    convergence_info: Dict[str, Any] = Field(default_factory=dict, description="Convergence details")
    software_version: str = Field("PySCF", description="Software package version")
    
    # Execution metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Calculation timestamp")
    execution_time: Optional[float] = Field(None, description="Wall clock time in seconds")
    
    def calculate_errors(self) -> None:
        """Calculate error metrics if reference is available."""
        if self.reference_energy is not None:
            self.absolute_error = abs(self.energy - self.reference_energy)
            if abs(self.reference_energy) > 1e-10:
                self.relative_error = (self.absolute_error / abs(self.reference_energy)) * 100
    
    def to_multireference_result(self) -> MultireferenceResult:
        """Convert to MultireferenceResult for analysis."""
        return MultireferenceResult(
            method=self.method,
            energy=self.energy,
            correlation_energy=self.correlation_energy,
            active_space_info={
                'n_electrons': self.n_active_electrons,
                'n_orbitals': self.n_active_orbitals,
                'selection_method': self.active_space_method
            },
            n_active_electrons=self.n_active_electrons,
            n_active_orbitals=self.n_active_orbitals,
            properties=self.properties,
            convergence_info=self.convergence_info,
            computational_cost=self.computational_cost,
            timestamp=self.timestamp,
            software_version=self.software_version,
            basis_set=self.system.basis_set
        )


class BenchmarkDataset(BaseModel):
    """
    Collection of benchmark calculations for systematic analysis.
    """
    
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    version: str = Field("1.0", description="Dataset version")
    
    entries: List[BenchmarkEntry] = Field(default_factory=list, description="Benchmark entries")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    created_date: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    def add_entry(self, entry: BenchmarkEntry) -> None:
        """Add benchmark entry to dataset."""
        entry.calculate_errors()
        self.entries.append(entry)
    
    def filter_by_system_type(self, system_type: SystemType) -> BenchmarkDataset:
        """Filter entries by chemical system type."""
        filtered_entries = [
            entry for entry in self.entries 
            if entry.system.system_type == system_type
        ]
        
        return BenchmarkDataset(
            name=f"{self.name}_{system_type.value}",
            description=f"Filtered subset: {system_type.value} systems",
            entries=filtered_entries,
            metadata=self.metadata.copy()
        )
    
    def filter_by_method(self, method: str) -> BenchmarkDataset:
        """Filter entries by quantum chemistry method."""
        filtered_entries = [
            entry for entry in self.entries 
            if entry.method.lower() == method.lower()
        ]
        
        return BenchmarkDataset(
            name=f"{self.name}_{method}",
            description=f"Filtered subset: {method} calculations",
            entries=filtered_entries,
            metadata=self.metadata.copy()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        if not self.entries:
            return {"error": "No entries in dataset"}
        
        # Error statistics (only for entries with reference values)
        entries_with_ref = [e for e in self.entries if e.reference_energy is not None]
        
        stats = {
            "total_entries": len(self.entries),
            "entries_with_reference": len(entries_with_ref),
            "system_types": {},
            "methods": {},
            "basis_sets": {}
        }
        
        # Count by categories
        for entry in self.entries:
            # System types
            system_type = entry.system.system_type.value
            stats["system_types"][system_type] = stats["system_types"].get(system_type, 0) + 1
            
            # Methods
            stats["methods"][entry.method] = stats["methods"].get(entry.method, 0) + 1
            
            # Basis sets
            basis = entry.system.basis_set
            stats["basis_sets"][basis] = stats["basis_sets"].get(basis, 0) + 1
        
        # Error statistics
        if entries_with_ref:
            errors = [e.absolute_error for e in entries_with_ref if e.absolute_error is not None]
            if errors:
                stats["error_statistics"] = {
                    "mean_absolute_error": np.mean(errors),
                    "rmse": np.sqrt(np.mean([e**2 for e in errors])),
                    "max_error": np.max(errors),
                    "min_error": np.min(errors),
                    "std_error": np.std(errors)
                }
        
        return stats
    
    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            # Convert to JSON-serializable format
            data = self.dict()
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> BenchmarkDataset:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)


class BenchmarkDatasetBuilder:
    """
    Builder class for creating standardized benchmark datasets.
    """
    
    def __init__(self):
        """Initialize dataset builder."""
        self.dataset = BenchmarkDataset(
            name="custom_dataset",
            description="Custom benchmark dataset"
        )
    
    def set_metadata(self, name: str, description: str, **kwargs) -> BenchmarkDatasetBuilder:
        """Set dataset metadata."""
        self.dataset.name = name
        self.dataset.description = description
        self.dataset.metadata.update(kwargs)
        return self
    
    def add_questdb_subset(self) -> BenchmarkDatasetBuilder:
        """Add QUESTDB benchmark subset for vertical excitations."""
        # Add representative QUESTDB molecules
        questdb_molecules = [
            BenchmarkMolecule(
                name="water_s1",
                atoms=[("O", 0.0, 0.0, 0.0), ("H", 0.757, 0.0, 0.586), ("H", -0.757, 0.0, 0.586)],
                charge=0,
                multiplicity=1,
                basis_set="aug-cc-pVTZ",
                system_type=SystemType.EXCITATION,
                source="QUESTDB",
                theoretical_references={"vertical_excitation": 7.61}  # eV
            ),
            BenchmarkMolecule(
                name="formaldehyde_t1",
                atoms=[("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.208), ("H", 0.0, 0.943, -0.587), ("H", 0.0, -0.943, -0.587)],
                charge=0,
                multiplicity=1,
                basis_set="aug-cc-pVTZ",
                system_type=SystemType.EXCITATION,
                source="QUESTDB",
                theoretical_references={"vertical_excitation": 3.88}  # eV
            )
        ]
        
        for mol in questdb_molecules:
            entry = BenchmarkEntry(
                system=mol,
                method="reference",
                active_space_method="manual",
                n_active_electrons=8,
                n_active_orbitals=8,
                energy=0.0,  # Placeholder
                reference_energy=mol.theoretical_references.get("vertical_excitation")
            )
            self.dataset.add_entry(entry)
        
        return self
    
    def add_transition_metal_benchmarks(self) -> BenchmarkDatasetBuilder:
        """Add transition metal benchmark systems."""
        tm_molecules = [
            BenchmarkMolecule(
                name="fe_h2o_6_hs",
                atoms=[
                    ("Fe", 0.0, 0.0, 0.0),
                    ("O", 2.1, 0.0, 0.0), ("H", 2.5, 0.8, 0.0), ("H", 2.5, -0.8, 0.0),
                    ("O", -2.1, 0.0, 0.0), ("H", -2.5, 0.8, 0.0), ("H", -2.5, -0.8, 0.0),
                    ("O", 0.0, 2.1, 0.0), ("H", 0.8, 2.5, 0.0), ("H", -0.8, 2.5, 0.0),
                    ("O", 0.0, -2.1, 0.0), ("H", 0.8, -2.5, 0.0), ("H", -0.8, -2.5, 0.0),
                    ("O", 0.0, 0.0, 2.1), ("H", 0.0, 0.8, 2.5), ("H", 0.0, -0.8, 2.5),
                    ("O", 0.0, 0.0, -2.1), ("H", 0.0, 0.8, -2.5), ("H", 0.0, -0.8, -2.5)
                ],
                charge=2,
                multiplicity=6,  # High-spin Fe(II)
                basis_set="def2-SVP",
                system_type=SystemType.TRANSITION_METAL,
                source="Benchmark study",
                theoretical_references={"spin_state_energy": 0.0}  # Reference high-spin state
            )
        ]
        
        for mol in tm_molecules:
            entry = BenchmarkEntry(
                system=mol,
                method="reference",
                active_space_method="avas",
                n_active_electrons=6,
                n_active_orbitals=5,
                energy=0.0,  # Placeholder
                reference_energy=mol.theoretical_references.get("spin_state_energy")
            )
            self.dataset.add_entry(entry)
        
        return self
    
    def add_bond_dissociation_curves(self) -> BenchmarkDatasetBuilder:
        """Add bond dissociation benchmark systems."""
        bond_molecules = [
            BenchmarkMolecule(
                name="n2_equilibrium",
                atoms=[("N", 0.0, 0.0, -0.55), ("N", 0.0, 0.0, 0.55)],
                charge=0,
                multiplicity=1,
                basis_set="cc-pVTZ",
                system_type=SystemType.BOND_DISSOCIATION,
                source="Literature",
                theoretical_references={"bond_energy": -109.5}  # Total energy in Hartree
            ),
            BenchmarkMolecule(
                name="n2_stretched",
                atoms=[("N", 0.0, 0.0, -1.5), ("N", 0.0, 0.0, 1.5)],
                charge=0,
                multiplicity=1,
                basis_set="cc-pVTZ",
                system_type=SystemType.BOND_DISSOCIATION,
                source="Literature",
                theoretical_references={"bond_energy": -108.8}  # Total energy in Hartree
            )
        ]
        
        for mol in bond_molecules:
            entry = BenchmarkEntry(
                system=mol,
                method="reference",
                active_space_method="apc",
                n_active_electrons=10,
                n_active_orbitals=8,
                energy=0.0,  # Placeholder
                reference_energy=mol.theoretical_references.get("bond_energy")
            )
            self.dataset.add_entry(entry)
        
        return self
    
    def build(self) -> BenchmarkDataset:
        """Build and return the dataset."""
        return self.dataset


def create_standard_benchmark_datasets() -> Dict[str, BenchmarkDataset]:
    """Create standard benchmark datasets for validation."""
    datasets = {}
    
    # QUESTDB subset for vertical excitations  
    questdb = (BenchmarkDatasetBuilder()
               .set_metadata("questdb_subset", "QUESTDB subset for vertical excitations")
               .add_questdb_subset()
               .build())
    datasets["questdb"] = questdb
    
    # Transition metal complexes
    transition_metals = (BenchmarkDatasetBuilder()
                        .set_metadata("transition_metals", "Transition metal spin-state benchmarks")
                        .add_transition_metal_benchmarks() 
                        .build())
    datasets["transition_metals"] = transition_metals
    
    # Bond dissociation curves
    bond_dissociation = (BenchmarkDatasetBuilder()
                        .set_metadata("bond_dissociation", "Bond dissociation benchmark curves")
                        .add_bond_dissociation_curves()
                        .build())
    datasets["bond_dissociation"] = bond_dissociation
    
    return datasets