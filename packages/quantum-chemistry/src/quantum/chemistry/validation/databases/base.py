"""
Base classes and interfaces for quantum chemistry database integration.

Provides the foundation for accessing and managing standard benchmarking
databases with consistent data structures and validation procedures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import logging

import numpy as np
from pydantic import BaseModel, Field, validator
import pyscf.gto as gto

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of molecular properties available in databases."""
    ATOMIZATION_ENERGY = "atomization_energy"
    FORMATION_ENERGY = "formation_energy"
    REACTION_ENERGY = "reaction_energy"
    BOND_DISSOCIATION_ENERGY = "bond_dissociation_energy"
    IONIZATION_POTENTIAL = "ionization_potential"
    ELECTRON_AFFINITY = "electron_affinity"
    EXCITATION_ENERGY = "excitation_energy"
    BOND_LENGTH = "bond_length"
    BOND_ANGLE = "bond_angle"
    DIPOLE_MOMENT = "dipole_moment"
    POLARIZABILITY = "polarizability"


class BasisSetType(Enum):
    """Standard basis set types for benchmarking."""
    MINIMAL = "sto-3g"
    DOUBLE_ZETA = "cc-pvdz"
    TRIPLE_ZETA = "cc-pvtz"
    QUADRUPLE_ZETA = "cc-pvqz"
    QUINTUPLE_ZETA = "cc-pv5z"
    SEXTUPLE_ZETA = "cc-pv6z"
    AUG_DOUBLE_ZETA = "aug-cc-pvdz"
    AUG_TRIPLE_ZETA = "aug-cc-pvtz"
    AUG_QUADRUPLE_ZETA = "aug-cc-pvqz"
    CBS_LIMIT = "cbs"


@dataclass
class UncertaintyInfo:
    """Uncertainty information for reference values."""
    value: float
    error_bar: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    systematic_error: Optional[float] = None
    statistical_error: Optional[float] = None
    method_uncertainty: Optional[str] = None


class ReferenceDataEntry(BaseModel):
    """Reference data entry with uncertainty quantification."""
    
    property_type: PropertyType
    value: float
    unit: str
    uncertainty: Optional[UncertaintyInfo] = None
    method: str
    basis_set: Optional[str] = None
    level_of_theory: str
    source: str
    notes: Optional[str] = None
    
    class Config:
        use_enum_values = True


class MolecularEntry(BaseModel):
    """Molecular system entry with standardized geometry and properties."""
    
    # Basic molecular information
    name: str
    formula: str
    geometry: str  # XYZ format
    charge: int = 0
    multiplicity: int = 1
    
    # Database metadata
    database_id: str
    cas_number: Optional[str] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    
    # Reference properties
    reference_data: List[ReferenceDataEntry] = Field(default_factory=list)
    
    # Computational metadata
    point_group: Optional[str] = None
    electronic_state: Optional[str] = None
    bond_order: Optional[float] = None
    
    # Difficulty assessment
    multireference_character: Optional[float] = None  # 0-1 scale
    recommended_active_space: Optional[Tuple[int, int]] = None
    computational_difficulty: Optional[str] = None  # easy/medium/hard/expert
    
    # Literature references
    references: List[str] = Field(default_factory=list)
    
    @validator("geometry")
    def validate_geometry(cls, v):
        """Validate and standardize molecular geometry."""
        lines = v.strip().split('\n')
        if len(lines) < 1:
            raise ValueError("Geometry must contain at least one atom")
        
        # Basic format validation
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Invalid geometry line: {line}")
            
            # Check atomic symbol
            symbol = parts[0]
            if not symbol.isalpha() or len(symbol) > 2:
                raise ValueError(f"Invalid atomic symbol: {symbol}")
                
        return v
    
    def create_molecule(self, basis_set: str = "sto-3g") -> gto.Mole:
        """Create PySCF Mole object from the molecular entry."""
        mol = gto.Mole()
        mol.atom = self.geometry
        mol.basis = basis_set
        mol.charge = self.charge
        mol.spin = self.multiplicity - 1
        
        try:
            mol.build()
        except Exception as e:
            logger.error(f"Failed to build molecule {self.name}: {e}")
            raise
            
        return mol
    
    def get_reference_value(
        self, 
        property_type: PropertyType, 
        method: Optional[str] = None,
        prefer_experimental: bool = True
    ) -> Optional[ReferenceDataEntry]:
        """Get reference value for a specific property."""
        matching_entries = [
            entry for entry in self.reference_data 
            if entry.property_type == property_type
        ]
        
        if not matching_entries:
            return None
            
        # Filter by method if specified
        if method:
            method_matches = [
                entry for entry in matching_entries 
                if method.lower() in entry.method.lower()
            ]
            if method_matches:
                matching_entries = method_matches
        
        # Prefer experimental data if available
        if prefer_experimental:
            experimental = [
                entry for entry in matching_entries 
                if "experimental" in entry.method.lower()
            ]
            if experimental:
                return experimental[0]
        
        # Return highest level of theory
        if matching_entries:
            return max(matching_entries, key=lambda x: self._assess_theory_level(x.level_of_theory))
        
        return None
    
    def _assess_theory_level(self, level_of_theory: str) -> int:
        """Assess relative quality of theory level (higher is better)."""
        level = level_of_theory.lower()
        
        # Hierarchy of method quality
        if "fci" in level or "exact" in level:
            return 100
        elif "ccsd(t)" in level:
            return 90
        elif "ccsdt" in level:
            return 85
        elif "ccsd" in level:
            return 80
        elif "caspt2" in level or "nevpt2" in level:
            return 75
        elif "casscf" in level:
            return 60
        elif "mp4" in level:
            return 50
        elif "mp3" in level:
            return 40
        elif "mp2" in level:
            return 30
        elif "hf" in level:
            return 10
        else:
            return 0


class DatabaseInterface(ABC):
    """Abstract interface for quantum chemistry databases."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".quantum_chemistry" / "databases"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._molecules: Dict[str, MolecularEntry] = {}
        self._loaded = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Database name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Database description."""
        pass
    
    @property
    @abstractmethod
    def url(self) -> Optional[str]:
        """Database URL or DOI."""
        pass
    
    @property
    @abstractmethod
    def reference(self) -> str:
        """Primary literature reference."""
        pass
    
    @abstractmethod
    def _download_data(self) -> None:
        """Download and cache database files."""
        pass
    
    @abstractmethod
    def _parse_data(self) -> Dict[str, MolecularEntry]:
        """Parse downloaded data into MolecularEntry objects."""
        pass
    
    def load(self, force_reload: bool = False) -> None:
        """Load database data."""
        if self._loaded and not force_reload:
            return
            
        try:
            self._download_data()
            self._molecules = self._parse_data()
            self._loaded = True
            logger.info(f"Loaded {len(self._molecules)} molecules from {self.name}")
        except Exception as e:
            logger.error(f"Failed to load database {self.name}: {e}")
            raise
    
    def get_molecule(self, identifier: str) -> Optional[MolecularEntry]:
        """Get molecule by name or database ID."""
        if not self._loaded:
            self.load()
            
        # Try exact match first
        if identifier in self._molecules:
            return self._molecules[identifier]
            
        # Try case-insensitive search
        lower_id = identifier.lower()
        for key, molecule in self._molecules.items():
            if (key.lower() == lower_id or 
                molecule.name.lower() == lower_id or
                molecule.formula.lower() == lower_id):
                return molecule
                
        return None
    
    def get_molecules_by_formula(self, formula: str) -> List[MolecularEntry]:
        """Get all molecules with the given molecular formula."""
        if not self._loaded:
            self.load()
            
        return [
            mol for mol in self._molecules.values() 
            if mol.formula.lower() == formula.lower()
        ]
    
    def get_molecules_by_property(
        self, 
        property_type: PropertyType,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[MolecularEntry]:
        """Get molecules that have a specific property within a range."""
        if not self._loaded:
            self.load()
            
        matching_molecules = []
        for molecule in self._molecules.values():
            ref_entry = molecule.get_reference_value(property_type)
            if ref_entry is None:
                continue
                
            value = ref_entry.value
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
                
            matching_molecules.append(molecule)
            
        return matching_molecules
    
    def get_all_molecules(self) -> List[MolecularEntry]:
        """Get all molecules in the database."""
        if not self._loaded:
            self.load()
        return list(self._molecules.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._loaded:
            self.load()
            
        molecules = list(self._molecules.values())
        if not molecules:
            return {}
        
        # Count properties
        property_counts = {}
        for prop_type in PropertyType:
            count = sum(
                1 for mol in molecules 
                if mol.get_reference_value(prop_type) is not None
            )
            if count > 0:
                property_counts[prop_type.value] = count
        
        # Element distribution
        element_counts = {}
        for molecule in molecules:
            lines = molecule.geometry.strip().split('\n')
            for line in lines:
                element = line.strip().split()[0]
                element_counts[element] = element_counts.get(element, 0) + 1
        
        return {
            "total_molecules": len(molecules),
            "property_distribution": property_counts,
            "element_distribution": element_counts,
            "charge_distribution": {
                str(charge): sum(1 for mol in molecules if mol.charge == charge)
                for charge in set(mol.charge for mol in molecules)
            },
            "multiplicity_distribution": {
                str(mult): sum(1 for mol in molecules if mol.multiplicity == mult)
                for mult in set(mol.multiplicity for mol in molecules)
            }
        }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate database integrity and report issues."""
        if not self._loaded:
            self.load()
            
        issues = []
        warnings = []
        
        for mol_id, molecule in self._molecules.items():
            # Check geometry validity
            try:
                mol = molecule.create_molecule()
                if mol.natm == 0:
                    issues.append(f"{mol_id}: No atoms in geometry")
            except Exception as e:
                issues.append(f"{mol_id}: Invalid geometry - {e}")
            
            # Check reference data consistency
            if not molecule.reference_data:
                warnings.append(f"{mol_id}: No reference data available")
            
            # Check for required fields
            if not molecule.formula:
                issues.append(f"{mol_id}: Missing molecular formula")
        
        return {
            "total_molecules": len(self._molecules),
            "issues": issues,
            "warnings": warnings,
            "integrity_score": 1.0 - len(issues) / max(1, len(self._molecules))
        }