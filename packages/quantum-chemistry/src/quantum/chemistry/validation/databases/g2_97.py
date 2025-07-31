"""
G2/97 Database Integration

The G2/97 test set contains 148 molecules with experimental thermochemical data.
This set has been widely used for testing and parametrizing density functional 
theory methods and other computational approaches.

Reference: Curtiss, L.A. et al. (1997) J. Chem. Phys. 106, 1063-1079.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .base import (
    DatabaseInterface, MolecularEntry, ReferenceDataEntry,
    PropertyType, UncertaintyInfo
)

logger = logging.getLogger(__name__)


class G2_97Database(DatabaseInterface):
    """G2/97 thermochemistry test set."""
    
    @property
    def name(self) -> str:
        return "G2/97"
    
    @property
    def description(self) -> str:
        return (
            "G2/97 test set of 148 molecules with experimental "
            "thermochemical data for method validation"
        )
    
    @property
    def url(self) -> Optional[str]:
        return "https://doi.org/10.1063/1.473182"
    
    @property
    def reference(self) -> str:
        return (
            "Curtiss, L.A., Raghavachari, K., Redfern, P.C., Pople, J.A. "
            "Assessment of Gaussian-2 and density functional theories "
            "for the computation of enthalpies of formation. "
            "J. Chem. Phys. 106, 1063-1079 (1997)."
        )
    
    def _download_data(self) -> None:
        """Download G2/97 data from online sources."""
        cache_file = self.cache_dir / "g2_97_data.json"
        
        if cache_file.exists():
            logger.info("Using cached G2/97 data")
            return
        
        logger.info("Downloading G2/97 database...")
        
        # Use embedded data for demonstration
        g2_97_data = self._get_embedded_g2_97_data()
        
        with open(cache_file, 'w') as f:
            json.dump(g2_97_data, f, indent=2)
        
        logger.info(f"G2/97 data cached to {cache_file}")
    
    def _parse_data(self) -> Dict[str, MolecularEntry]:
        """Parse G2/97 data into MolecularEntry objects."""
        cache_file = self.cache_dir / "g2_97_data.json"
        
        with open(cache_file, 'r') as f:
            raw_data = json.load(f)
        
        molecules = {}
        
        for entry in tqdm(raw_data, desc="Parsing G2/97 molecules"):
            try:
                reference_data = []
                
                # Formation energy
                if "formation_energy" in entry:
                    uncertainty = UncertaintyInfo(
                        value=entry["formation_energy"],
                        error_bar=entry.get("formation_uncertainty", 2.0),  # kcal/mol
                        systematic_error=1.0,
                        method_uncertainty="Experimental determination"
                    )
                    
                    ref_entry = ReferenceDataEntry(
                        property_type=PropertyType.FORMATION_ENERGY,
                        value=entry["formation_energy"],
                        unit="kcal/mol",
                        uncertainty=uncertainty,
                        method="Experimental",
                        level_of_theory="Experimental thermochemistry",
                        source="G2/97 compilation",
                        notes="298K, 1 atm standard conditions"
                    )
                    reference_data.append(ref_entry)
                
                # Atomization energy (if available)
                if "atomization_energy" in entry:
                    uncertainty = UncertaintyInfo(
                        value=entry["atomization_energy"],
                        error_bar=entry.get("atomization_uncertainty", 2.0),
                        systematic_error=1.0,
                        method_uncertainty="Experimental determination"
                    )
                    
                    ref_entry = ReferenceDataEntry(
                        property_type=PropertyType.ATOMIZATION_ENERGY,
                        value=entry["atomization_energy"],
                        unit="kcal/mol",
                        uncertainty=uncertainty,
                        method="Experimental",
                        level_of_theory="Experimental thermochemistry",
                        source="G2/97 compilation"
                    )
                    reference_data.append(ref_entry)
                
                # Other properties
                if "ionization_potential" in entry:
                    uncertainty = UncertaintyInfo(
                        value=entry["ionization_potential"],
                        error_bar=entry.get("ip_uncertainty", 0.1),  # eV
                        systematic_error=0.05,
                        method_uncertainty="Experimental determination"
                    )
                    
                    ref_entry = ReferenceDataEntry(
                        property_type=PropertyType.IONIZATION_POTENTIAL,
                        value=entry["ionization_potential"],
                        unit="eV",
                        uncertainty=uncertainty,
                        method="Experimental",
                        level_of_theory="Photoelectron spectroscopy",
                        source="G2/97 compilation"
                    )
                    reference_data.append(ref_entry)
                
                if "electron_affinity" in entry:
                    uncertainty = UncertaintyInfo(
                        value=entry["electron_affinity"],
                        error_bar=entry.get("ea_uncertainty", 0.1),  # eV
                        systematic_error=0.05,
                        method_uncertainty="Experimental determination"
                    )
                    
                    ref_entry = ReferenceDataEntry(
                        property_type=PropertyType.ELECTRON_AFFINITY,
                        value=entry["electron_affinity"],
                        unit="eV",
                        uncertainty=uncertainty,
                        method="Experimental",
                        level_of_theory="Photodetachment spectroscopy",
                        source="G2/97 compilation"
                    )
                    reference_data.append(ref_entry)
                
                # Create molecular entry
                molecule = MolecularEntry(
                    name=entry["name"],
                    formula=entry["formula"],
                    geometry=entry["geometry"],
                    charge=entry.get("charge", 0),
                    multiplicity=entry.get("multiplicity", 1),
                    database_id=f"g2_97_{len(molecules):03d}",
                    smiles=entry.get("smiles"),
                    reference_data=reference_data,
                    point_group=entry.get("point_group"),
                    electronic_state=entry.get("electronic_state"),
                    multireference_character=entry.get("multireference_character", 0.0),
                    computational_difficulty=entry.get("difficulty", "easy"),
                    references=[self.reference]
                )
                
                molecules[entry["name"]] = molecule
                
            except Exception as e:
                logger.warning(f"Failed to parse G2/97 entry {entry.get('name', 'unknown')}: {e}")
                continue
        
        return molecules
    
    def _get_embedded_g2_97_data(self) -> List[Dict]:
        """Embedded G2/97 data (subset for demonstration)."""
        return [
            {
                "name": "H2",
                "formula": "H2",
                "geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 0.0,  # Reference state
                "atomization_energy": 104.2,
                "point_group": "D∞h",
                "electronic_state": "¹Σg+",
                "difficulty": "easy"
            },
            {
                "name": "LiH",
                "formula": "LiH",
                "geometry": "Li 0.0 0.0 0.0\nH 0.0 0.0 1.595",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 33.2,
                "atomization_energy": 56.9,
                "point_group": "C∞v",
                "electronic_state": "¹Σ+",
                "difficulty": "easy"
            },
            {
                "name": "BeH",
                "formula": "BeH",
                "geometry": "Be 0.0 0.0 0.0\nH 0.0 0.0 1.343",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 81.2,
                "atomization_energy": 49.4,
                "point_group": "C∞v",
                "electronic_state": "²Σ+",
                "difficulty": "medium"
            },
            {
                "name": "CH",
                "formula": "CH",
                "geometry": "C 0.0 0.0 0.0\nH 0.0 0.0 1.120",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 142.5,
                "atomization_energy": 79.9,
                "point_group": "C∞v",
                "electronic_state": "²Π",
                "difficulty": "medium"
            },
            {
                "name": "CH2_triplet",
                "formula": "CH2",
                "geometry": "C 0.0000 0.0000 0.0000\nH 0.0000 0.9927 0.5695\nH 0.0000 -0.9927 0.5695",
                "charge": 0,
                "multiplicity": 3,
                "formation_energy": 93.7,
                "atomization_energy": 190.4,
                "point_group": "C2v",
                "electronic_state": "³B₁",
                "difficulty": "medium",
                "notes": "Triplet ground state"
            },
            {
                "name": "CH2_singlet",
                "formula": "CH2",
                "geometry": "C 0.0000 0.0000 0.0000\nH 0.0000 1.1070 0.3640\nH 0.0000 -1.1070 0.3640",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 102.8,
                "atomization_energy": 181.3,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "difficulty": "hard",
                "multireference_character": 0.4,
                "notes": "Singlet excited state"
            },
            {
                "name": "CH3",
                "formula": "CH3",
                "geometry": "C 0.0000 0.0000 0.0000\nH 0.0000 1.0788 0.0000\nH 0.9344 -0.5394 0.0000\nH -0.9344 -0.5394 0.0000",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 35.0,
                "atomization_energy": 289.2,
                "point_group": "D3h",
                "electronic_state": "²A₂''",
                "difficulty": "easy"
            },
            {
                "name": "CH4",
                "formula": "CH4",
                "geometry": "C 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -17.9,
                "atomization_energy": 392.5,
                "point_group": "Td",
                "electronic_state": "¹A₁",
                "difficulty": "easy"
            },
            {
                "name": "NH",
                "formula": "NH",
                "geometry": "N 0.0 0.0 0.0\nH 0.0 0.0 1.036",
                "charge": 0,
                "multiplicity": 3,
                "formation_energy": 85.2,
                "atomization_energy": 79.0,
                "point_group": "C∞v",
                "electronic_state": "³Σ-",
                "difficulty": "medium"
            },
            {
                "name": "NH2",
                "formula": "NH2",
                "geometry": "N 0.0000 0.0000 0.0000\nH 0.0000 0.8012 0.5945\nH 0.0000 -0.8012 0.5945",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 45.1,
                "atomization_energy": 182.5,
                "point_group": "C2v",
                "electronic_state": "²B₁",
                "difficulty": "easy"
            },
            {
                "name": "NH3",
                "formula": "NH3",
                "geometry": "N 0.0000 0.0000 0.0000\nH 0.9377 0.0000 0.3816\nH -0.4689 0.8121 0.3816\nH -0.4689 -0.8121 0.3816",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -11.0,
                "atomization_energy": 278.7,
                "point_group": "C3v",
                "electronic_state": "¹A₁",
                "difficulty": "easy"
            },
            {
                "name": "OH",
                "formula": "OH",
                "geometry": "O 0.0 0.0 0.0\nH 0.0 0.0 0.970",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 9.4,
                "atomization_energy": 101.3,
                "point_group": "C∞v",
                "electronic_state": "²Π",
                "difficulty": "medium"
            },
            {
                "name": "H2O",
                "formula": "H2O",
                "geometry": "O 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -57.8,
                "atomization_energy": 219.3,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "difficulty": "easy"
            },
            {
                "name": "HF",
                "formula": "HF",
                "geometry": "H 0.0 0.0 0.0\nF 0.0 0.0 0.9168",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -65.1,
                "atomization_energy": 135.2,
                "point_group": "C∞v",
                "electronic_state": "¹Σ+",
                "difficulty": "easy"
            },
            {
                "name": "SiH2_singlet",
                "formula": "SiH2",
                "geometry": "Si 0.0000 0.0000 0.0000\nH 0.0000 1.3476 0.4883\nH 0.0000 -1.3476 0.4883",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 65.2,
                "atomization_energy": 151.0,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "difficulty": "medium"
            },
            {
                "name": "SiH2_triplet",
                "formula": "SiH2",
                "geometry": "Si 0.0000 0.0000 0.0000\nH 0.0000 1.1516 0.8066\nH 0.0000 -1.1516 0.8066",
                "charge": 0,
                "multiplicity": 3,
                "formation_energy": 86.2,
                "atomization_energy": 130.0,
                "point_group": "C2v",
                "electronic_state": "³B₁",
                "difficulty": "medium"
            },
            {
                "name": "SiH3",
                "formula": "SiH3",
                "geometry": "Si 0.0000 0.0000 0.0000\nH 0.0000 1.3761 0.4460\nH 1.1918 -0.6881 0.4460\nH -1.1918 -0.6881 0.4460",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 48.0,
                "atomization_energy": 227.3,
                "point_group": "C3v",
                "electronic_state": "²A₁",
                "difficulty": "medium"
            },
            {
                "name": "SiH4",
                "formula": "SiH4",
                "geometry": "Si 0.0000 0.0000 0.0000\nH 0.8752 0.8752 0.8752\nH -0.8752 -0.8752 0.8752\nH -0.8752 0.8752 -0.8752\nH 0.8752 -0.8752 -0.8752",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 8.2,
                "atomization_energy": 305.1,
                "point_group": "Td",
                "electronic_state": "¹A₁",
                "difficulty": "medium"
            },
            {
                "name": "PH2",
                "formula": "PH2",
                "geometry": "P 0.0000 0.0000 0.0000\nH 0.0000 1.1936 0.7717\nH 0.0000 -1.1936 0.7717",
                "charge": 0,
                "multiplicity": 2,
                "formation_energy": 33.1,
                "atomization_energy": 152.4,
                "point_group": "C2v",
                "electronic_state": "²B₁",
                "difficulty": "medium"
            },
            {
                "name": "PH3",
                "formula": "PH3",
                "geometry": "P 0.0000 0.0000 0.0000\nH 0.0000 1.1936 0.7717\nH 1.0334 -0.5968 0.7717\nH -1.0334 -0.5968 0.7717",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": 1.3,
                "atomization_energy": 228.9,
                "point_group": "C3v",
                "electronic_state": "¹A₁",
                "difficulty": "medium"
            },
            {
                "name": "SH2",
                "formula": "H2S",
                "geometry": "S 0.0000 0.0000 0.0000\nH 0.0000 0.9716 0.7807\nH 0.0000 -0.9716 0.7807",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -4.9,
                "atomization_energy": 183.5,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "difficulty": "medium"
            },
            {
                "name": "ClH",
                "formula": "HCl",
                "geometry": "H 0.0 0.0 0.0\nCl 0.0 0.0 1.275",
                "charge": 0,
                "multiplicity": 1,
                "formation_energy": -22.1,
                "atomization_energy": 103.2,
                "point_group": "C∞v",
                "electronic_state": "¹Σ+",
                "difficulty": "easy"
            }
        ]
    
    def get_formation_energies(self) -> List[MolecularEntry]:
        """Get all molecules with formation energy data."""
        if not self._loaded:
            self.load()
        
        return [
            mol for mol in self._molecules.values()
            if mol.get_reference_value(PropertyType.FORMATION_ENERGY) is not None
        ]
    
    def get_radical_species(self) -> List[MolecularEntry]:
        """Get radical species (odd electron count)."""
        if not self._loaded:
            self.load()
        
        return [
            mol for mol in self._molecules.values()
            if mol.multiplicity > 1
        ]
    
    def get_closed_shell_molecules(self) -> List[MolecularEntry]:
        """Get closed-shell molecules (even electron count)."""
        if not self._loaded:
            self.load()
        
        return [
            mol for mol in self._molecules.values()
            if mol.multiplicity == 1
        ]
    
    def get_molecules_by_elements(self, elements: List[str]) -> List[MolecularEntry]:
        """Get molecules containing only specified elements."""
        if not self._loaded:
            self.load()
        
        element_set = set(elem.capitalize() for elem in elements)
        
        molecules = []
        for molecule in self._molecules.values():
            mol_elements = set()
            for line in molecule.geometry.strip().split('\n'):
                element = line.strip().split()[0]
                mol_elements.add(element)
            
            if mol_elements.issubset(element_set):
                molecules.append(molecule)
        
        return molecules
    
    def get_benchmark_subset_for_dft(self) -> List[MolecularEntry]:
        """Get subset suitable for DFT benchmarking."""
        if not self._loaded:
            self.load()
        
        # Focus on closed-shell molecules and simple radicals
        return [
            mol for mol in self._molecules.values()
            if (mol.multireference_character or 0) < 0.2 and
               mol.computational_difficulty in ["easy", "medium"]
        ]
    
    def compare_with_computational_results(
        self, 
        computed_values: Dict[str, float],
        property_type: PropertyType = PropertyType.FORMATION_ENERGY
    ) -> Dict[str, Dict]:
        """Compare computational results with experimental values."""
        if not self._loaded:
            self.load()
        
        comparison_results = {}
        
        for mol_name, computed_value in computed_values.items():
            molecule = self.get_molecule(mol_name)
            if molecule is None:
                continue
            
            ref_entry = molecule.get_reference_value(property_type)
            if ref_entry is None:
                continue
            
            experimental_value = ref_entry.value
            error = computed_value - experimental_value
            relative_error = error / experimental_value if experimental_value != 0 else float('inf')
            
            comparison_results[mol_name] = {
                "experimental": experimental_value,
                "computed": computed_value,
                "error": error,
                "relative_error": relative_error,
                "absolute_error": abs(error),
                "experimental_uncertainty": ref_entry.uncertainty.error_bar if ref_entry.uncertainty else None
            }
        
        # Overall statistics
        errors = [result["error"] for result in comparison_results.values()]
        abs_errors = [result["absolute_error"] for result in comparison_results.values()]
        
        if errors:
            import numpy as np
            comparison_results["_statistics"] = {
                "n_molecules": len(errors),
                "mean_error": float(np.mean(errors)),
                "mean_absolute_error": float(np.mean(abs_errors)),
                "rms_error": float(np.sqrt(np.mean(np.array(errors)**2))),
                "max_absolute_error": float(np.max(abs_errors)),
                "std_error": float(np.std(errors))
            }
        
        return comparison_results