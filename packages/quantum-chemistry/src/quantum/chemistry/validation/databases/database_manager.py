"""
Database Manager for Unified Access to Quantum Chemistry Databases

Provides a centralized interface for accessing multiple databases with
caching, cross-database searches, and automated data curation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm

from .base import DatabaseInterface, MolecularEntry, PropertyType, ReferenceDataEntry
from .w4_11 import W4_11Database
from .g2_97 import G2_97Database
from .tmc_151 import TMC151Database
from .questdb import QuestDBDatabase

logger = logging.getLogger(__name__)


@dataclass
class DatabaseStats:
    """Statistics for a database."""
    name: str
    total_molecules: int
    unique_formulas: int
    property_coverage: Dict[str, int]
    element_coverage: Set[str]
    difficulty_distribution: Dict[str, int]
    multireference_systems: int


class DatabaseManager:
    """Unified manager for quantum chemistry databases."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".quantum_chemistry" / "databases"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.databases: Dict[str, DatabaseInterface] = {
            "w4_11": W4_11Database(self.cache_dir),
            "g2_97": G2_97Database(self.cache_dir),
            "tmc_151": TMC151Database(self.cache_dir),
            "questdb": QuestDBDatabase(self.cache_dir)
        }
        
        self._loaded_databases: Set[str] = set()
        self._unified_index: Optional[Dict[str, Dict]] = None
    
    def load_database(self, database_name: str, force_reload: bool = False) -> None:
        """Load a specific database."""
        if database_name not in self.databases:
            raise ValueError(f"Unknown database: {database_name}")
        
        if database_name in self._loaded_databases and not force_reload:
            logger.info(f"Database {database_name} already loaded")
            return
        
        logger.info(f"Loading database: {database_name}")
        self.databases[database_name].load(force_reload=force_reload)
        self._loaded_databases.add(database_name)
        
        # Invalidate unified index
        self._unified_index = None
    
    def load_all_databases(self, force_reload: bool = False) -> None:
        """Load all available databases."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.load_database, db_name, force_reload): db_name
                for db_name in self.databases.keys()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading databases"):
                db_name = futures[future]
                try:
                    future.result()
                    logger.info(f"Successfully loaded {db_name}")
                except Exception as e:
                    logger.error(f"Failed to load {db_name}: {e}")
    
    def get_database_stats(self, database_name: str) -> DatabaseStats:
        """Get statistics for a specific database."""
        if database_name not in self._loaded_databases:
            self.load_database(database_name)
        
        database = self.databases[database_name]
        molecules = database.get_all_molecules()
        
        # Calculate statistics
        formulas = set(mol.formula for mol in molecules)
        
        property_coverage = {}
        for prop_type in PropertyType:
            count = sum(
                1 for mol in molecules 
                if mol.get_reference_value(prop_type) is not None
            )
            if count > 0:
                property_coverage[prop_type.value] = count
        
        elements = set()
        for mol in molecules:
            for line in mol.geometry.strip().split('\n'):
                element = line.strip().split()[0]
                elements.add(element)
        
        difficulty_dist = {}
        mr_count = 0
        for mol in molecules:
            difficulty = mol.computational_difficulty or "unknown"
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            
            if mol.multireference_character and mol.multireference_character > 0.1:
                mr_count += 1
        
        return DatabaseStats(
            name=database_name,
            total_molecules=len(molecules),
            unique_formulas=len(formulas),
            property_coverage=property_coverage,
            element_coverage=elements,
            difficulty_distribution=difficulty_dist,
            multireference_systems=mr_count
        )
    
    def get_all_database_stats(self) -> Dict[str, DatabaseStats]:
        """Get statistics for all loaded databases."""
        stats = {}
        for db_name in self.databases.keys():
            try:
                stats[db_name] = self.get_database_stats(db_name)
            except Exception as e:
                logger.warning(f"Failed to get stats for {db_name}: {e}")
        return stats
    
    def search_by_formula(self, formula: str, databases: List[str] = None) -> Dict[str, List[MolecularEntry]]:
        """Search for molecules by formula across databases."""
        if databases is None:
            databases = list(self.databases.keys())
        
        results = {}
        for db_name in databases:
            if db_name not in self._loaded_databases:
                self.load_database(db_name)
            
            molecules = self.databases[db_name].get_molecules_by_formula(formula)
            if molecules:
                results[db_name] = molecules
        
        return results
    
    def search_by_property(
        self, 
        property_type: PropertyType,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        databases: List[str] = None
    ) -> Dict[str, List[MolecularEntry]]:
        """Search for molecules by property values across databases."""
        if databases is None:
            databases = list(self.databases.keys())
        
        results = {}
        for db_name in databases:
            if db_name not in self._loaded_databases:
                self.load_database(db_name)
            
            molecules = self.databases[db_name].get_molecules_by_property(
                property_type, min_value, max_value
            )
            if molecules:
                results[db_name] = molecules
        
        return results
    
    def find_common_molecules(self, databases: List[str] = None) -> Dict[str, Dict[str, MolecularEntry]]:
        """Find molecules that appear in multiple databases."""
        if databases is None:
            databases = list(self.databases.keys())
        
        # Load all specified databases
        for db_name in databases:
            if db_name not in self._loaded_databases:
                self.load_database(db_name)
        
        # Collect all molecules by formula
        formula_to_molecules = {}
        for db_name in databases:
            molecules = self.databases[db_name].get_all_molecules()
            for molecule in molecules:
                formula = molecule.formula
                if formula not in formula_to_molecules:
                    formula_to_molecules[formula] = {}
                formula_to_molecules[formula][db_name] = molecule
        
        # Filter to common molecules (appearing in multiple databases)
        common_molecules = {}
        for formula, db_molecules in formula_to_molecules.items():
            if len(db_molecules) > 1:
                common_molecules[formula] = db_molecules
        
        return common_molecules
    
    def create_unified_benchmark_set(
        self,
        target_size: int = 100,
        difficulty_levels: List[str] = None,
        required_properties: List[PropertyType] = None,
        max_multireference_character: float = 0.5
    ) -> List[MolecularEntry]:
        """Create a unified benchmark set from all databases."""
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium"]
        
        if required_properties is None:
            required_properties = [PropertyType.ATOMIZATION_ENERGY, PropertyType.FORMATION_ENERGY]
        
        # Load all databases
        self.load_all_databases()
        
        # Collect candidate molecules
        candidates = []
        seen_formulas = set()
        
        for db_name, database in self.databases.items():
            molecules = database.get_all_molecules()
            
            for molecule in molecules:
                # Skip if formula already seen (avoid duplicates)
                if molecule.formula in seen_formulas:
                    continue
                
                # Check difficulty level
                if molecule.computational_difficulty not in difficulty_levels:
                    continue
                
                # Check multireference character
                mr_char = molecule.multireference_character or 0.0
                if mr_char > max_multireference_character:
                    continue
                
                # Check for required properties
                has_required_props = any(
                    molecule.get_reference_value(prop_type) is not None
                    for prop_type in required_properties
                )
                if not has_required_props:
                    continue
                
                candidates.append(molecule)
                seen_formulas.add(molecule.formula)
        
        # Sort by quality/reliability criteria
        def quality_score(mol: MolecularEntry) -> float:
            score = 0.0
            
            # Prefer molecules with experimental data
            for ref_data in mol.reference_data:
                if "experimental" in ref_data.method.lower():
                    score += 10.0
                elif "w4" in ref_data.method.lower():
                    score += 8.0
                elif "ccsd(t)" in ref_data.level_of_theory.lower():
                    score += 6.0
                else:
                    score += 2.0
            
            # Prefer smaller systems (easier to compute)
            n_atoms = len(mol.geometry.strip().split('\n'))
            score += max(0, 20 - n_atoms)  # Bonus for smaller systems
            
            # Penalty for high multireference character
            mr_char = mol.multireference_character or 0.0
            score -= mr_char * 10
            
            return score
        
        candidates.sort(key=quality_score, reverse=True)
        
        # Select top candidates up to target size
        benchmark_set = candidates[:target_size]
        
        logger.info(f"Created unified benchmark set with {len(benchmark_set)} molecules")
        return benchmark_set
    
    def compare_database_coverage(self) -> Dict[str, Any]:
        """Compare coverage across databases."""
        self.load_all_databases()
        
        # Collect all unique formulas
        all_formulas = set()
        database_formulas = {}
        
        for db_name, database in self.databases.items():
            molecules = database.get_all_molecules()
            formulas = set(mol.formula for mol in molecules)
            database_formulas[db_name] = formulas
            all_formulas.update(formulas)
        
        # Calculate overlaps
        coverage_matrix = {}
        for db1 in self.databases.keys():
            coverage_matrix[db1] = {}
            for db2 in self.databases.keys():
                if db1 == db2:
                    coverage_matrix[db1][db2] = len(database_formulas[db1])
                else:
                    overlap = len(database_formulas[db1].intersection(database_formulas[db2]))
                    coverage_matrix[db1][db2] = overlap
        
        # Calculate unique contributions
        unique_contributions = {}
        for db_name in self.databases.keys():
            other_formulas = set()
            for other_db in self.databases.keys():
                if other_db != db_name:
                    other_formulas.update(database_formulas[other_db])
            
            unique = database_formulas[db_name] - other_formulas
            unique_contributions[db_name] = len(unique)
        
        return {
            "total_unique_formulas": len(all_formulas),
            "database_sizes": {db: len(formulas) for db, formulas in database_formulas.items()},
            "overlap_matrix": coverage_matrix,
            "unique_contributions": unique_contributions,
            "database_formulas": {db: list(formulas) for db, formulas in database_formulas.items()}
        }
    
    def export_unified_dataset(
        self, 
        output_path: Path,
        format: str = "json",
        include_geometry: bool = True,
        include_properties: List[PropertyType] = None
    ) -> None:
        """Export unified dataset in various formats."""
        self.load_all_databases()
        
        if include_properties is None:
            include_properties = list(PropertyType)
        
        # Collect all molecules
        all_molecules = []
        for db_name, database in self.databases.items():
            molecules = database.get_all_molecules()
            for molecule in molecules:
                mol_data = {
                    "database": db_name,
                    "name": molecule.name,
                    "formula": molecule.formula,
                    "charge": molecule.charge,
                    "multiplicity": molecule.multiplicity
                }
                
                if include_geometry:
                    mol_data["geometry"] = molecule.geometry
                
                # Add properties
                properties = {}
                for prop_type in include_properties:
                    ref_entry = molecule.get_reference_value(prop_type)
                    if ref_entry:
                        properties[prop_type.value] = {
                            "value": ref_entry.value,
                            "unit": ref_entry.unit,
                            "method": ref_entry.method,
                            "uncertainty": ref_entry.uncertainty.error_bar if ref_entry.uncertainty else None
                        }
                
                if properties:
                    mol_data["properties"] = properties
                
                # Add metadata
                mol_data["metadata"] = {
                    "computational_difficulty": molecule.computational_difficulty,
                    "multireference_character": molecule.multireference_character,
                    "point_group": molecule.point_group,
                    "electronic_state": molecule.electronic_state
                }
                
                all_molecules.append(mol_data)
        
        # Export in requested format
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump({
                    "metadata": {
                        "total_molecules": len(all_molecules),
                        "databases": list(self.databases.keys()),
                        "export_timestamp": pd.Timestamp.now().isoformat()
                    },
                    "molecules": all_molecules
                }, f, indent=2)
        
        elif format.lower() == "csv":
            # Flatten for CSV export
            flattened_data = []
            for mol_data in all_molecules:
                base_data = {
                    "database": mol_data["database"],
                    "name": mol_data["name"],
                    "formula": mol_data["formula"],
                    "charge": mol_data["charge"],
                    "multiplicity": mol_data["multiplicity"]
                }
                
                if include_geometry:
                    base_data["geometry"] = mol_data["geometry"]
                
                # Add metadata
                base_data.update(mol_data["metadata"])
                
                # Add properties as separate columns
                if "properties" in mol_data:
                    for prop_name, prop_data in mol_data["properties"].items():
                        base_data[f"{prop_name}_value"] = prop_data["value"]
                        base_data[f"{prop_name}_unit"] = prop_data["unit"]
                        base_data[f"{prop_name}_method"] = prop_data["method"]
                        base_data[f"{prop_name}_uncertainty"] = prop_data.get("uncertainty")
                
                flattened_data.append(base_data)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        
        elif format.lower() == "hdf5":
            # Export to HDF5 for efficient access
            df = pd.DataFrame([
                {
                    "database": mol_data["database"],
                    "name": mol_data["name"],
                    "formula": mol_data["formula"],
                    "charge": mol_data["charge"],
                    "multiplicity": mol_data["multiplicity"],
                    "geometry": mol_data.get("geometry", ""),
                    **mol_data["metadata"]
                }
                for mol_data in all_molecules
            ])
            
            with pd.HDFStore(output_path, 'w') as store:
                store.put('molecules', df, format='table')
                
                # Store properties separately
                for prop_type in include_properties:
                    prop_data = []
                    for mol_data in all_molecules:
                        if "properties" in mol_data and prop_type.value in mol_data["properties"]:
                            prop_info = mol_data["properties"][prop_type.value]
                            prop_data.append({
                                "name": mol_data["name"],
                                "database": mol_data["database"],
                                "value": prop_info["value"],
                                "unit": prop_info["unit"],
                                "method": prop_info["method"],
                                "uncertainty": prop_info.get("uncertainty")
                            })
                    
                    if prop_data:
                        prop_df = pd.DataFrame(prop_data)
                        store.put(f'properties/{prop_type.value}', prop_df, format='table')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported unified dataset to {output_path} ({len(all_molecules)} molecules)")
    
    def validate_all_databases(self) -> Dict[str, Dict[str, Any]]:
        """Validate integrity of all databases."""
        validation_results = {}
        
        for db_name in self.databases.keys():
            try:
                if db_name not in self._loaded_databases:
                    self.load_database(db_name)
                
                database = self.databases[db_name]
                validation_results[db_name] = database.validate_data_integrity()
                
            except Exception as e:
                validation_results[db_name] = {
                    "error": str(e),
                    "integrity_score": 0.0
                }
        
        return validation_results
    
    def get_cross_database_molecule(self, identifier: str) -> Dict[str, Optional[MolecularEntry]]:
        """Find a molecule across all databases."""
        results = {}
        
        for db_name, database in self.databases.items():
            if db_name not in self._loaded_databases:
                self.load_database(db_name)
            
            molecule = database.get_molecule(identifier)
            results[db_name] = molecule
        
        return results