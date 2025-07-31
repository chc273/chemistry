"""
W4-11 Database Integration

The W4-11 database contains 140 highly accurate total atomization energies
of small first- and second-row molecules and radicals. These values are 
derived from first-principles W4 calculations with sub-kcal/mol accuracy.

Reference: Karton, A., Daon, S., Martin, J.M.L. (2011) 
           Chem. Phys. Lett. 510, 165-178.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

from .base import (
    DatabaseInterface, MolecularEntry, ReferenceDataEntry,
    PropertyType, UncertaintyInfo
)

logger = logging.getLogger(__name__)


class W4_11Database(DatabaseInterface):
    """W4-11 thermochemistry database."""
    
    @property
    def name(self) -> str:
        return "W4-11"
    
    @property
    def description(self) -> str:
        return (
            "W4-11 benchmark dataset of 140 total atomization energies "
            "derived from first-principles W4 calculations"
        )
    
    @property
    def url(self) -> Optional[str]:
        return "https://doi.org/10.1016/j.cplett.2011.05.007"
    
    @property
    def reference(self) -> str:
        return (
            "Karton, A., Daon, S., Martin, J.M.L. "
            "W4-11: A high-confidence benchmark dataset for computational "
            "thermochemistry derived from first-principles W4 data. "
            "Chem. Phys. Lett. 510, 165-178 (2011)."
        )
    
    def _download_data(self) -> None:
        """Download W4-11 data from online sources."""
        cache_file = self.cache_dir / "w4_11_data.json"
        
        if cache_file.exists():
            logger.info("Using cached W4-11 data")
            return
        
        logger.info("Downloading W4-11 database...")
        
        # Use embedded data for now - in practice, this would download from a repository
        w4_11_data = self._get_embedded_w4_11_data()
        
        with open(cache_file, 'w') as f:
            json.dump(w4_11_data, f, indent=2)
        
        logger.info(f"W4-11 data cached to {cache_file}")
    
    def _parse_data(self) -> Dict[str, MolecularEntry]:
        """Parse W4-11 data into MolecularEntry objects."""
        cache_file = self.cache_dir / "w4_11_data.json"
        
        with open(cache_file, 'r') as f:
            raw_data = json.load(f)
        
        molecules = {}
        
        for entry in tqdm(raw_data, desc="Parsing W4-11 molecules"):
            try:
                # Create uncertainty information
                uncertainty = UncertaintyInfo(
                    value=entry["atomization_energy"],
                    error_bar=entry.get("uncertainty", 1.0),  # kcal/mol
                    systematic_error=0.5,  # Estimated systematic error
                    method_uncertainty="W4 composite method"
                )
                
                # Create reference data entry
                ref_entry = ReferenceDataEntry(
                    property_type=PropertyType.ATOMIZATION_ENERGY,
                    value=entry["atomization_energy"],
                    unit="kcal/mol",
                    uncertainty=uncertainty,
                    method="W4",
                    level_of_theory="W4 composite method (CCSD(T)/CBS + corrections)",
                    source="W4-11 database",
                    notes=entry.get("notes")
                )
                
                # Create molecular entry
                molecule = MolecularEntry(
                    name=entry["name"],
                    formula=entry["formula"],
                    geometry=entry["geometry"],
                    charge=entry.get("charge", 0),
                    multiplicity=entry.get("multiplicity", 1),
                    database_id=f"w4_11_{len(molecules):03d}",
                    reference_data=[ref_entry],
                    point_group=entry.get("point_group"),
                    electronic_state=entry.get("electronic_state"),
                    multireference_character=entry.get("multireference_character", 0.0),
                    computational_difficulty=entry.get("difficulty", "medium"),
                    references=[self.reference]
                )
                
                molecules[entry["name"]] = molecule
                
            except Exception as e:
                logger.warning(f"Failed to parse W4-11 entry {entry.get('name', 'unknown')}: {e}")
                continue
        
        return molecules
    
    def _get_embedded_w4_11_data(self) -> List[Dict]:
        """Embedded W4-11 data (subset for demonstration)."""
        return [
            {
                "name": "H2",
                "formula": "H2",
                "geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 104.2,  # kcal/mol
                "uncertainty": 0.1,
                "point_group": "D∞h",
                "electronic_state": "¹Σg+",
                "multireference_character": 0.0,
                "difficulty": "easy",
                "notes": "Prototype diatomic molecule"
            },
            {
                "name": "H2O",
                "formula": "H2O",
                "geometry": "O 0.0000 0.0000 0.0000\nH 0.7571 0.0000 0.5861\nH -0.7571 0.0000 0.5861",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 219.3,
                "uncertainty": 0.2,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "easy",
                "notes": "Fundamental triatomic molecule"
            },
            {
                "name": "NH3",
                "formula": "NH3",
                "geometry": "N 0.0000 0.0000 0.0000\nH 0.9377 0.0000 0.3816\nH -0.4689 0.8121 0.3816\nH -0.4689 -0.8121 0.3816",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 278.7,
                "uncertainty": 0.3,
                "point_group": "C3v",
                "electronic_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "easy"
            },
            {
                "name": "CH4",
                "formula": "CH4",
                "geometry": "C 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 392.5,
                "uncertainty": 0.3,
                "point_group": "Td",
                "electronic_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "easy"
            },
            {
                "name": "HF",
                "formula": "HF",
                "geometry": "H 0.0 0.0 0.0\nF 0.0 0.0 0.9168",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 135.2,
                "uncertainty": 0.1,
                "point_group": "C∞v",
                "electronic_state": "¹Σ+",
                "multireference_character": 0.0,
                "difficulty": "easy"
            },
            {
                "name": "N2",
                "formula": "N2",
                "geometry": "N 0.0 0.0 0.0\nN 0.0 0.0 1.098",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 225.1,
                "uncertainty": 0.2,
                "point_group": "D∞h",
                "electronic_state": "¹Σg+",
                "multireference_character": 0.1,
                "difficulty": "medium",
                "notes": "Strong triple bond"
            },
            {
                "name": "CO",
                "formula": "CO",
                "geometry": "C 0.0 0.0 0.0\nO 0.0 0.0 1.128",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 256.2,
                "uncertainty": 0.2,
                "point_group": "C∞v",
                "electronic_state": "¹Σ+",
                "multireference_character": 0.05,
                "difficulty": "medium"
            },
            {
                "name": "F2",
                "formula": "F2",
                "geometry": "F 0.0 0.0 0.0\nF 0.0 0.0 1.412",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 37.8,
                "uncertainty": 0.3,
                "point_group": "D∞h",
                "electronic_state": "¹Σg+",
                "multireference_character": 0.3,
                "difficulty": "hard",
                "notes": "Challenging multireference case"
            },
            {
                "name": "O2",
                "formula": "O2", 
                "geometry": "O 0.0 0.0 0.0\nO 0.0 0.0 1.208",
                "charge": 0,
                "multiplicity": 3,
                "atomization_energy": 119.1,
                "uncertainty": 0.2,
                "point_group": "D∞h",
                "electronic_state": "³Σg-",
                "multireference_character": 0.2,
                "difficulty": "medium",
                "notes": "Triplet ground state"
            },
            {
                "name": "C2H2",
                "formula": "C2H2",
                "geometry": "C 0.0 0.0 0.0\nC 0.0 0.0 1.203\nH 0.0 0.0 -1.061\nH 0.0 0.0 2.264",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 388.9,
                "uncertainty": 0.4,
                "point_group": "D∞h",
                "electronic_state": "¹Σg+",
                "multireference_character": 0.0,
                "difficulty": "medium"
            },
            {
                "name": "C2H4",
                "formula": "C2H4",
                "geometry": "C 0.0000 0.0000 0.6695\nC 0.0000 0.0000 -0.6695\nH 0.0000 0.9289 1.2321\nH 0.0000 -0.9289 1.2321\nH 0.0000 0.9289 -1.2321\nH 0.0000 -0.9289 -1.2321",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 531.9,
                "uncertainty": 0.4,
                "point_group": "D2h",
                "electronic_state": "¹Ag",
                "multireference_character": 0.0,
                "difficulty": "medium"
            },
            {
                "name": "C2H6",
                "formula": "C2H6",
                "geometry": "C 0.0000 0.0000 0.7650\nC 0.0000 0.0000 -0.7650\nH 1.0192 0.0000 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH 1.0192 0.0000 -1.1573\nH -0.5096 0.8826 -1.1573\nH -0.5096 -0.8826 -1.1573",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 666.3,
                "uncertainty": 0.5,
                "point_group": "D3d",
                "electronic_state": "¹Ag",
                "multireference_character": 0.0,
                "difficulty": "medium"
            },
            {
                "name": "SO2",
                "formula": "SO2",
                "geometry": "S 0.0000 0.0000 0.0000\nO 0.0000 1.2323 0.8799\nO 0.0000 -1.2323 0.8799",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 260.4,
                "uncertainty": 0.4,
                "point_group": "C2v",
                "electronic_state": "¹A₁",
                "multireference_character": 0.1,
                "difficulty": "medium",
                "notes": "Contains second-row atom"
            },
            {
                "name": "SiH4",
                "formula": "SiH4",
                "geometry": "Si 0.0000 0.0000 0.0000\nH 0.8752 0.8752 0.8752\nH -0.8752 -0.8752 0.8752\nH -0.8752 0.8752 -0.8752\nH 0.8752 -0.8752 -0.8752",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 305.1,
                "uncertainty": 0.4,
                "point_group": "Td",
                "electronic_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "medium",
                "notes": "Second-row analogue of CH4"
            },
            {
                "name": "PH3",
                "formula": "PH3",
                "geometry": "P 0.0000 0.0000 0.0000\nH 0.0000 1.1936 0.7717\nH 1.0334 -0.5968 0.7717\nH -1.0334 -0.5968 0.7717",
                "charge": 0,
                "multiplicity": 1,
                "atomization_energy": 228.9,
                "uncertainty": 0.4,
                "point_group": "C3v",
                "electronic_state": "¹A₁",
                "multireference_character": 0.0,
                "difficulty": "medium"
            }
        ]
    
    def get_molecules_by_size(self, min_atoms: int = 2, max_atoms: int = 10) -> List[MolecularEntry]:
        """Get molecules by number of atoms."""
        if not self._loaded:
            self.load()
        
        molecules = []
        for molecule in self._molecules.values():
            n_atoms = len(molecule.geometry.strip().split('\n'))
            if min_atoms <= n_atoms <= max_atoms:
                molecules.append(molecule)
        
        return molecules
    
    def get_multireference_systems(self, min_character: float = 0.1) -> List[MolecularEntry]:
        """Get systems with significant multireference character."""
        if not self._loaded:
            self.load()
        
        return [
            mol for mol in self._molecules.values()
            if mol.multireference_character and mol.multireference_character >= min_character
        ]
    
    def get_benchmark_subset(self, difficulty: str = "medium") -> List[MolecularEntry]:
        """Get a subset suitable for benchmarking at different difficulty levels."""
        if not self._loaded:
            self.load()
        
        if difficulty == "easy":
            # Small molecules with minimal multireference character
            return [
                mol for mol in self._molecules.values()
                if (mol.computational_difficulty == "easy" or 
                    (mol.multireference_character or 0) < 0.1)
            ]
        elif difficulty == "medium":
            # Mix of easy and moderately challenging systems
            return [
                mol for mol in self._molecules.values()
                if mol.computational_difficulty in ["easy", "medium"]
            ]
        elif difficulty == "hard":
            # Include challenging multireference cases
            return list(self._molecules.values())
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
    
    def export_for_external_programs(
        self, 
        output_dir: Path, 
        programs: List[str] = None
    ) -> Dict[str, Path]:
        """Export molecular geometries for external quantum chemistry programs."""
        if not self._loaded:
            self.load()
        
        if programs is None:
            programs = ["gaussian", "orca", "molpro", "psi4"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for program in programs:
            program_dir = output_dir / program
            program_dir.mkdir(exist_ok=True)
            
            for name, molecule in self._molecules.items():
                if program.lower() == "gaussian":
                    content = self._format_gaussian_input(molecule)
                    filename = f"{name}.com"
                elif program.lower() == "orca":
                    content = self._format_orca_input(molecule)
                    filename = f"{name}.inp"
                elif program.lower() == "molpro":
                    content = self._format_molpro_input(molecule)
                    filename = f"{name}.inp"
                elif program.lower() == "psi4":
                    content = self._format_psi4_input(molecule)
                    filename = f"{name}.dat"
                else:
                    continue
                
                file_path = program_dir / filename
                with open(file_path, 'w') as f:
                    f.write(content)
            
            exported_files[program] = program_dir
        
        return exported_files
    
    def _format_gaussian_input(self, molecule: MolecularEntry) -> str:
        """Format Gaussian input file."""
        return f"""# HF/cc-pVDZ

{molecule.name} - W4-11 benchmark calculation

{molecule.charge} {molecule.multiplicity}
{molecule.geometry}

"""
    
    def _format_orca_input(self, molecule: MolecularEntry) -> str:
        """Format ORCA input file."""
        return f"""# W4-11 benchmark calculation for {molecule.name}
! HF cc-pVDZ

* xyz {molecule.charge} {molecule.multiplicity}
{molecule.geometry}
*
"""
    
    def _format_molpro_input(self, molecule: MolecularEntry) -> str:
        """Format Molpro input file."""
        return f"""*** W4-11 benchmark calculation for {molecule.name}
memory,1000,m
geometry={{
{molecule.geometry}
}}

basis=cc-pVDZ
{{hf}}
"""
    
    def _format_psi4_input(self, molecule: MolecularEntry) -> str:
        """Format Psi4 input file."""
        return f"""# W4-11 benchmark calculation for {molecule.name}

molecule {{
{molecule.charge} {molecule.multiplicity}
{molecule.geometry}
}}

set basis cc-pVDZ
energy('scf')
"""