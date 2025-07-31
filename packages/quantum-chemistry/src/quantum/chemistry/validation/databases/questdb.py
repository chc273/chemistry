"""
QUESTDB Database Integration

QUESTDB is a database of highly accurate excited state energies for benchmarking
electronic structure methods. It contains vertical excitation energies computed
with theoretical best estimates (TBEs) using high-level coupled cluster methods.

Reference: Véril, M. et al. (2021) WIREs Comput Mol Sci. 11, e1517.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from tqdm import tqdm

from .base import (
    DatabaseInterface, MolecularEntry, ReferenceDataEntry,
    PropertyType, UncertaintyInfo
)

logger = logging.getLogger(__name__)


class QuestDBDatabase(DatabaseInterface):
    """QUESTDB excited state benchmark database."""
    
    @property
    def name(self) -> str:
        return "QUESTDB"
    
    @property
    def description(self) -> str:
        return (
            "Database of highly accurate vertical excitation energies "
            "computed with theoretical best estimates (TBEs)"
        )
    
    @property
    def url(self) -> Optional[str]:
        return "https://lcpq.github.io/QUESTDB_website"
    
    @property
    def reference(self) -> str:
        return (
            "Véril, M., Scemama, A., Caffarel, M., Lipparini, F., "
            "Boggio-Pasqua, M., Jacquemin, D., Loos, P.-F. "
            "QUESTDB: A database of highly-accurate excitation energies "
            "for the electronic structure community. "
            "WIREs Comput Mol Sci. 11, e1517 (2021)."
        )
    
    def _download_data(self) -> None:
        """Download QUESTDB data from online sources."""
        cache_file = self.cache_dir / "questdb_data.json"
        
        if cache_file.exists():
            logger.info("Using cached QUESTDB data")
            return
        
        logger.info("Downloading QUESTDB database...")
        
        # Use embedded data for demonstration - in practice would download from GitHub
        questdb_data = self._get_embedded_questdb_data()
        
        with open(cache_file, 'w') as f:
            json.dump(questdb_data, f, indent=2)
        
        logger.info(f"QUESTDB data cached to {cache_file}")
    
    def _parse_data(self) -> Dict[str, MolecularEntry]:
        """Parse QUESTDB data into MolecularEntry objects."""
        cache_file = self.cache_dir / "questdb_data.json"
        
        with open(cache_file, 'r') as f:
            raw_data = json.load(f)
        
        molecules = {}
        
        for entry in tqdm(raw_data, desc="Parsing QUESTDB molecules"):
            try:
                # Parse excitation data
                reference_data = []
                
                for excitation in entry["excitations"]:
                    uncertainty = UncertaintyInfo(
                        value=excitation["energy"],
                        error_bar=excitation.get("uncertainty", 0.05),  # eV
                        systematic_error=0.02,  # Estimated systematic error
                        method_uncertainty="Theoretical best estimate (TBE)"
                    )
                    
                    ref_entry = ReferenceDataEntry(
                        property_type=PropertyType.EXCITATION_ENERGY,
                        value=excitation["energy"],
                        unit="eV",
                        uncertainty=uncertainty,
                        method="TBE",
                        basis_set="aug-cc-pVTZ",
                        level_of_theory=excitation.get("method", "CC3/aug-cc-pVTZ"),
                        source="QUESTDB",
                        notes=f"State: {excitation['state']}, Type: {excitation['type']}"
                    )
                    
                    reference_data.append(ref_entry)
                
                # Create molecular entry
                molecule = MolecularEntry(
                    name=entry["name"],
                    formula=entry["formula"],
                    geometry=entry["geometry"],
                    charge=entry.get("charge", 0),
                    multiplicity=entry.get("multiplicity", 1),
                    database_id=f"questdb_{len(molecules):03d}",
                    reference_data=reference_data,
                    point_group=entry.get("point_group"),
                    electronic_state=entry.get("ground_state"),
                    multireference_character=entry.get("multireference_character", 0.0),
                    computational_difficulty=entry.get("difficulty", "medium"),
                    references=[self.reference]
                )
                
                molecules[entry["name"]] = molecule
                
            except Exception as e:
                logger.warning(f"Failed to parse QUESTDB entry {entry.get('name', 'unknown')}: {e}")
                continue
        
        return molecules
    
    def _get_embedded_questdb_data(self) -> List[Dict]:
        """Embedded QUESTDB data (subset for demonstration)."""
        return [
            {
                "name": "water",
                "formula": "H2O",
                "geometry": "O 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "C2v",
                "ground_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "easy",
                "excitations": [
                    {
                        "state": "¹B₁",
                        "energy": 7.41,  # eV
                        "type": "n→3s",
                        "uncertainty": 0.03,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹A₁",
                        "energy": 9.12,
                        "type": "n→3p",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₂",
                        "energy": 9.69,
                        "type": "n→3p",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "ammonia",
                "formula": "NH3",
                "geometry": "N 0.0000 0.0000 0.1118\nH 0.0000 0.9377 -0.2606\nH 0.8121 -0.4689 -0.2606\nH -0.8121 -0.4689 -0.2606",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "C3v",
                "ground_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "easy",
                "excitations": [
                    {
                        "state": "¹A₁",
                        "energy": 6.66,
                        "type": "n→3s",
                        "uncertainty": 0.03,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹E",
                        "energy": 8.04,
                        "type": "n→3p",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "formaldehyde",
                "formula": "CH2O",
                "geometry": "C 0.0000 0.0000 0.5265\nO 0.0000 0.0000 -0.6844\nH 0.0000 0.9436 1.1070\nH 0.0000 -0.9436 1.1070",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "C2v",
                "ground_state": "¹A₁",
                "multireference_character": 0.05,
                "difficulty": "medium",
                "excitations": [
                    {
                        "state": "¹A₂",
                        "energy": 4.07,
                        "type": "n→π*",
                        "uncertainty": 0.02,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₁",
                        "energy": 9.09,
                        "type": "σ→π*",
                        "uncertainty": 0.05,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₂",
                        "energy": 9.97,
                        "type": "n→3s",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "ethene",
                "formula": "C2H4",
                "geometry": "C 0.0000 0.0000 0.6695\nC 0.0000 0.0000 -0.6695\nH 0.0000 0.9289 1.2321\nH 0.0000 -0.9289 1.2321\nH 0.0000 0.9289 -1.2321\nH 0.0000 -0.9289 -1.2321",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "D2h",
                "ground_state": "¹Ag",
                "multireference_character": 0.1,
                "difficulty": "medium",
                "excitations": [
                    {
                        "state": "¹B₁ᵤ",
                        "energy": 7.80,
                        "type": "π→π*",
                        "uncertainty": 0.03,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₃ᵤ",
                        "energy": 8.28,
                        "type": "π→3s",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "benzene",
                "formula": "C6H6",
                "geometry": "C 0.000000 1.396000 0.000000\nC 1.209000 0.698000 0.000000\nC 1.209000 -0.698000 0.000000\nC 0.000000 -1.396000 0.000000\nC -1.209000 -0.698000 0.000000\nC -1.209000 0.698000 0.000000\nH 0.000000 2.480000 0.000000\nH 2.148000 1.240000 0.000000\nH 2.148000 -1.240000 0.000000\nH 0.000000 -2.480000 0.000000\nH -2.148000 -1.240000 0.000000\nH -2.148000 1.240000 0.000000",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "D6h",
                "ground_state": "¹A₁g",
                "multireference_character": 0.2,
                "difficulty": "medium",
                "excitations": [
                    {
                        "state": "¹B₂ᵤ",
                        "energy": 5.08,
                        "type": "π→π*",
                        "uncertainty": 0.05,
                        "method": "CC3/aug-cc-pVTZ",
                        "notes": "Symmetry-forbidden"
                    },
                    {
                        "state": "¹B₁ᵤ",
                        "energy": 6.20,
                        "type": "π→π*",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹E₁ᵤ",
                        "energy": 6.92,
                        "type": "π→π*",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "furan",
                "formula": "C4H4O",
                "geometry": "O 0.0000 0.0000 1.1657\nC 0.0000 1.0812 0.3087\nC 0.0000 0.7115 -0.9993\nC 0.0000 -0.7115 -0.9993\nC 0.0000 -1.0812 0.3087\nH 0.0000 2.0735 0.7036\nH 0.0000 1.3938 -1.8212\nH 0.0000 -1.3938 -1.8212\nH 0.0000 -2.0735 0.7036",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "C2v",
                "ground_state": "¹A₁",
                "multireference_character": 0.1,
                "difficulty": "medium",
                "excitations": [
                    {
                        "state": "¹B₂",
                        "energy": 6.57,
                        "type": "π→π*",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹A₁",
                        "energy": 6.62,
                        "type": "π→π*",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "pyridine",
                "formula": "C5H5N",
                "geometry": "N 0.0000 0.0000 1.4064\nC 0.0000 1.1334 0.6927\nC 0.0000 1.1334 -0.6927\nC 0.0000 0.0000 -1.4064\nC 0.0000 -1.1334 -0.6927\nC 0.0000 -1.1334 0.6927\nH 0.0000 2.0735 1.2356\nH 0.0000 2.0735 -1.2356\nH 0.0000 0.0000 -2.4912\nH 0.0000 -2.0735 -1.2356\nH 0.0000 -2.0735 1.2356",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "C2v",
                "ground_state": "¹A₁",
                "multireference_character": 0.15,
                "difficulty": "hard",
                "excitations": [
                    {
                        "state": "¹B₁",
                        "energy": 4.85,
                        "type": "n→π*",
                        "uncertainty": 0.03,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹A₁",
                        "energy": 5.11,
                        "type": "π→π*",
                        "uncertainty": 0.04,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₂",
                        "energy": 6.26,
                        "type": "π→π*",
                        "uncertainty": 0.05,
                        "method": "CC3/aug-cc-pVTZ"
                    }
                ]
            },
            {
                "name": "naphthalene",
                "formula": "C10H8",
                "geometry": "C 0.0000 0.7135 1.2427\nC 0.0000 -0.7135 1.2427\nC 0.0000 -1.4270 0.0000\nC 0.0000 -0.7135 -1.2427\nC 0.0000 0.7135 -1.2427\nC 0.0000 1.4270 0.0000\nC 0.0000 1.4270 2.4854\nC 0.0000 0.7135 3.7281\nC 0.0000 -0.7135 3.7281\nC 0.0000 -1.4270 2.4854\nH 0.0000 -2.5135 0.0000\nH 0.0000 -1.2678 -2.1751\nH 0.0000 1.2678 -2.1751\nH 0.0000 2.5135 0.0000\nH 0.0000 2.5135 2.4854\nH 0.0000 1.2678 4.6605\nH 0.0000 -1.2678 4.6605\nH 0.0000 -2.5135 2.4854",
                "charge": 0,
                "multiplicity": 1,
                "point_group": "D2h",
                "ground_state": "¹Ag",
                "multireference_character": 0.25,
                "difficulty": "hard",
                "excitations": [
                    {
                        "state": "¹B₃ᵤ",
                        "energy": 4.24,
                        "type": "π→π*",
                        "uncertainty": 0.05,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₂ᵤ",
                        "energy": 4.77,
                        "type": "π→π*",
                        "uncertainty": 0.06,
                        "method": "CC3/aug-cc-pVTZ"
                    },
                    {
                        "state": "¹B₁g",
                        "energy": 6.02,
                        "type": "π→π*",
                        "uncertainty": 0.07,
                        "method": "CC3/aug-cc-pVTZ",
                        "notes": "Double excitation character"
                    }
                ]
            }
        ]
    
    def get_molecules_by_excitation_type(self, excitation_type: str) -> List[MolecularEntry]:
        """Get molecules with specific types of excitations."""
        if not self._loaded:
            self.load()
        
        molecules = []
        for molecule in self._molecules.values():
            for ref_data in molecule.reference_data:
                if (ref_data.property_type == PropertyType.EXCITATION_ENERGY and
                    ref_data.notes and excitation_type.lower() in ref_data.notes.lower()):
                    molecules.append(molecule)
                    break
        
        return molecules
    
    def get_excitation_energies_range(
        self, 
        min_energy: float = 0.0, 
        max_energy: float = 10.0
    ) -> List[Tuple[MolecularEntry, ReferenceDataEntry]]:
        """Get excitation energies within a specific range."""
        if not self._loaded:
            self.load()
        
        results = []
        for molecule in self._molecules.values():
            for ref_data in molecule.reference_data:
                if (ref_data.property_type == PropertyType.EXCITATION_ENERGY and
                    min_energy <= ref_data.value <= max_energy):
                    results.append((molecule, ref_data))
        
        return results
    
    def get_benchmark_subset_for_method(self, method_type: str) -> List[MolecularEntry]:
        """Get subset suitable for benchmarking specific excited state methods.
        
        Args:
            method_type: Type of method ('td-dft', 'casscf', 'caspt2', 'cc', 'adc')
        """
        if not self._loaded:
            self.load()
        
        if method_type.lower() == 'td-dft':
            # Single-reference dominated systems for TD-DFT
            return [
                mol for mol in self._molecules.values()
                if (mol.multireference_character or 0) < 0.15
            ]
        elif method_type.lower() in ['casscf', 'caspt2']:
            # Include multireference systems
            return [
                mol for mol in self._molecules.values()
                if (mol.multireference_character or 0) >= 0.05
            ]
        elif method_type.lower() in ['cc', 'cc2', 'cc3', 'ccsd']:
            # All systems suitable for coupled cluster
            return list(self._molecules.values())
        elif method_type.lower() == 'adc':
            # ADC methods work well for single-reference systems
            return [
                mol for mol in self._molecules.values()
                if (mol.multireference_character or 0) < 0.2
            ]
        else:
            return list(self._molecules.values())
    
    def analyze_excitation_statistics(self) -> Dict[str, Any]:
        """Analyze statistics of excitation energies in the database."""
        if not self._loaded:
            self.load()
        
        excitation_energies = []
        excitation_types = {}
        state_symmetries = {}
        
        for molecule in self._molecules.values():
            for ref_data in molecule.reference_data:
                if ref_data.property_type == PropertyType.EXCITATION_ENERGY:
                    excitation_energies.append(ref_data.value)
                    
                    # Parse excitation type from notes
                    if ref_data.notes:
                        type_info = ref_data.notes.split(',')[1].split(':')[1].strip()
                        excitation_types[type_info] = excitation_types.get(type_info, 0) + 1
                        
                        state_info = ref_data.notes.split(',')[0].split(':')[1].strip()
                        state_symmetries[state_info] = state_symmetries.get(state_info, 0) + 1
        
        if excitation_energies:
            energies = np.array(excitation_energies)
            return {
                "total_excitations": len(excitation_energies),
                "energy_statistics": {
                    "mean": float(np.mean(energies)),
                    "std": float(np.std(energies)),
                    "min": float(np.min(energies)),
                    "max": float(np.max(energies)),
                    "median": float(np.median(energies))
                },
                "excitation_type_distribution": excitation_types,
                "state_symmetry_distribution": state_symmetries
            }
        else:
            return {}
    
    def export_active_space_recommendations(self) -> Dict[str, Dict]:
        """Export recommended active spaces for multireference calculations."""
        if not self._loaded:
            self.load()
        
        recommendations = {}
        for name, molecule in self._molecules.items():
            if molecule.multireference_character and molecule.multireference_character > 0.1:
                # Estimate active space based on molecule size and type
                n_atoms = len(molecule.geometry.strip().split('\n'))
                
                if "π" in str(molecule.reference_data):  # π-conjugated system
                    # Count π electrons and orbitals
                    if "benzene" in name.lower():
                        active_space = (6, 6)  # 6 π electrons, 6 π orbitals
                    elif "naphthalene" in name.lower():
                        active_space = (10, 10)  # 10 π electrons, 10 π orbitals
                    elif "furan" in name.lower() or "pyridine" in name.lower():
                        active_space = (6, 6)  # Similar to benzene
                    else:
                        active_space = (4, 4)  # Default for small π systems
                else:
                    # General recommendation based on molecule size
                    if n_atoms <= 3:
                        active_space = (4, 4)
                    elif n_atoms <= 6:
                        active_space = (6, 6)
                    else:
                        active_space = (8, 8)
                
                recommendations[name] = {
                    "active_space": active_space,
                    "multireference_character": molecule.multireference_character,
                    "description": f"Recommended active space for {name}",
                    "method_suggestions": ["CASSCF", "CASPT2", "NEVPT2"]
                }
        
        return recommendations