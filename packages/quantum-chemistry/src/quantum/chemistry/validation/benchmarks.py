"""
Benchmark molecules and test systems for quantum chemistry validation.

This module provides a collection of well-studied molecular systems
with known reference energies for validating quantum chemistry methods.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pyscf import gto, scf

from ..active_space import ActiveSpaceResult
from ..multireference.external import ExternalMethodRunner


@dataclass
class BenchmarkSystem:
    """A benchmark molecular system with reference data."""

    name: str
    description: str
    geometry: str
    basis_set: str
    charge: int
    spin: int

    # Reference energies from high-level calculations or experiments
    reference_energies: Dict[str, float]  # method -> energy
    reference_properties: Dict[str, Any]  # property -> value

    # Active space information
    recommended_active_space: Optional[Tuple[int, int]]  # (electrons, orbitals)
    active_space_description: Optional[str]

    # Computational details
    difficulty_level: str  # 'easy', 'medium', 'hard', 'expert'
    computational_cost: str  # 'low', 'medium', 'high', 'very_high'

    # Literature references
    references: List[str]

    def create_molecule(self) -> gto.Mole:
        """Create a PySCF molecule object."""
        mol = gto.Mole()
        mol.atom = self.geometry
        mol.basis = self.basis_set
        mol.charge = self.charge
        mol.spin = self.spin
        mol.build()
        return mol

    def run_scf(self) -> scf.hf.SCF:
        """Run SCF calculation and return converged object."""
        mol = self.create_molecule()

        if self.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.kernel()

        if not mf.converged:
            raise RuntimeError(f"SCF failed to converge for {self.name}")

        return mf

    def get_reference_energy(self, method: str) -> Optional[float]:
        """Get reference energy for a specific method."""
        return self.reference_energies.get(method.lower())

    def get_active_space(self) -> Optional[ActiveSpaceResult]:
        """Get recommended active space for this system."""
        if self.recommended_active_space is None:
            return None

        n_electrons, n_orbitals = self.recommended_active_space

        # Create a simple active space result
        # In practice, this would come from an active space selection method
        return ActiveSpaceResult(
            method="recommended",
            n_active_electrons=n_electrons,
            n_active_orbitals=n_orbitals,
            active_orbitals=list(range(n_orbitals)),
            orbital_energies=None,
            occupation_numbers=None,
            selection_info={
                "description": self.active_space_description,
                "system": self.name,
            },
        )

    def run_external_method(self, method: str, **kwargs) -> Dict[str, Any]:
        """Run external quantum chemistry method using Docker integration.

        Args:
            method: External method name (e.g., 'molpro', 'orca', 'gaussian')
            **kwargs: Method-specific calculation parameters

        Returns:
            Dictionary containing calculation results
        """
        runner = ExternalMethodRunner()

        # Prepare input for external method
        input_data = {
            "geometry": self.geometry,
            "basis_set": self.basis_set,
            "charge": self.charge,
            "spin": self.spin,
            "method": method,
            **kwargs,
        }

        # Add active space information if available
        if self.recommended_active_space:
            input_data["active_space"] = self.recommended_active_space

        return runner.run_calculation(method, input_data)


class BenchmarkSuite:
    """Collection of benchmark systems for systematic validation."""

    def __init__(self):
        self.systems = {}
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize the collection of benchmark systems."""

        # Small molecules - easy benchmarks
        self.systems["h2"] = BenchmarkSystem(
            name="H2",
            description="Hydrogen molecule - fundamental bond dissociation",
            geometry="H 0 0 0; H 0 0 0.74",
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -1.12870,
                "fci": -1.17447,
                "ccsd": -1.16820,
                "ccsd(t)": -1.17370,
            },
            reference_properties={
                "bond_length": 0.74,
                "dissociation_energy": 4.52,  # eV
            },
            recommended_active_space=(2, 2),
            active_space_description="σ and σ* orbitals",
            difficulty_level="easy",
            computational_cost="low",
            references=[
                "Helgaker, T. et al. Molecular Electronic-Structure Theory (2000)",
                "NIST Chemistry WebBook",
            ],
        )

        self.systems["lih"] = BenchmarkSystem(
            name="LiH",
            description="Lithium hydride - ionic-covalent character",
            geometry="Li 0 0 0; H 0 0 1.595",
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -7.98726,
                "fci": -8.07055,
                "ccsd": -8.05558,
                "ccsd(t)": -8.06985,
            },
            reference_properties={
                "bond_length": 1.595,
                "dipole_moment": 5.88,  # Debye
            },
            recommended_active_space=(4, 6),
            active_space_description="Li 2s, H 1s, and correlating orbitals",
            difficulty_level="easy",
            computational_cost="low",
            references=[
                "Helgaker, T. et al. Molecular Electronic-Structure Theory (2000)"
            ],
        )

        self.systems["h2o"] = BenchmarkSystem(
            name="H2O",
            description="Water molecule - fundamental importance",
            geometry="""
                O  0.0000  0.0000  0.0000
                H  0.7571  0.0000  0.5861
                H -0.7571  0.0000  0.5861
            """,
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -76.02676,
                "mp2": -76.23081,
                "ccsd": -76.24125,
                "ccsd(t)": -76.24265,
                "fci": -76.24266,
            },
            reference_properties={
                "bond_length": 0.9572,
                "bond_angle": 104.52,
                "dipole_moment": 1.85,
            },
            recommended_active_space=(8, 6),
            active_space_description="Oxygen valence orbitals",
            difficulty_level="medium",
            computational_cost="medium",
            references=[
                "Peterson, K.A. et al. J. Chem. Phys. 100, 7410 (1994)",
                "Dunning, T.H. J. Chem. Phys. 90, 1007 (1989)",
            ],
        )

        self.systems["n2"] = BenchmarkSystem(
            name="N2",
            description="Nitrogen molecule - strong triple bond",
            geometry="N 0 0 0; N 0 0 1.098",
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -108.95412,
                "mp2": -109.28289,
                "ccsd": -109.34338,
                "ccsd(t)": -109.36617,
                "fci": -109.36617,
            },
            reference_properties={
                "bond_length": 1.098,
                "dissociation_energy": 9.79,
                "vibrational_frequency": 2358,
            },
            recommended_active_space=(10, 8),
            active_space_description="Nitrogen valence orbitals (2s, 2p)",
            difficulty_level="medium",
            computational_cost="medium",
            references=["Peterson, K.A. et al. J. Chem. Phys. 117, 10548 (2002)"],
        )

        # More challenging systems
        self.systems["f2"] = BenchmarkSystem(
            name="F2",
            description="Fluorine molecule - challenging multi-reference case",
            geometry="F 0 0 0; F 0 0 1.412",
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -198.76864,
                "casscf_6_6": -199.0345,
                "caspt2_6_6": -199.2156,
                "mrci": -199.2189,
                "fci": -199.2189,
            },
            reference_properties={"bond_length": 1.412, "dissociation_energy": 1.60},
            recommended_active_space=(14, 8),
            active_space_description="F 2p orbitals and correlating π* orbitals",
            difficulty_level="hard",
            computational_cost="high",
            references=[
                "Gdanitz, R.J. Chem. Phys. Lett. 283, 253 (1998)",
                "Roos, B.O. et al. J. Phys. Chem. A 108, 2851 (2004)",
            ],
        )

        self.systems["cr2"] = BenchmarkSystem(
            name="Cr2",
            description="Chromium dimer - extreme multi-reference character",
            geometry="Cr 0 0 0; Cr 0 0 1.679",
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -2087.8234,  # Very poor reference
                "casscf_12_12": -2088.156,
                "caspt2_12_12": -2088.345,
                "dmrg": -2088.356,
                "experimental": -2088.40,  # From spectroscopy
            },
            reference_properties={
                "bond_length": 1.679,
                "dissociation_energy": 1.47,
                "ground_state": "1Σg+",
            },
            recommended_active_space=(12, 12),
            active_space_description="Cr 3d and 4s orbitals",
            difficulty_level="expert",
            computational_cost="very_high",
            references=[
                "Casey, S.M. et al. J. Phys. Chem. 97, 816 (1993)",
                "Kurashige, Y. et al. J. Chem. Phys. 135, 094104 (2011)",
            ],
        )

        # Organic molecules
        self.systems["benzene"] = BenchmarkSystem(
            name="benzene",
            description="Benzene - aromatic π system",
            geometry="""
                C  0.000000  1.396000  0.000000
                C  1.209000  0.698000  0.000000
                C  1.209000 -0.698000  0.000000
                C  0.000000 -1.396000  0.000000
                C -1.209000 -0.698000  0.000000
                C -1.209000  0.698000  0.000000
                H  0.000000  2.480000  0.000000
                H  2.148000  1.240000  0.000000
                H  2.148000 -1.240000  0.000000
                H  0.000000 -2.480000  0.000000
                H -2.148000 -1.240000  0.000000
                H -2.148000  1.240000  0.000000
            """,
            basis_set="cc-pVDZ",
            charge=0,
            spin=0,
            reference_energies={
                "hf": -230.71868,
                "mp2": -231.46523,
                "ccsd": -231.53456,
                "ccsd(t)": -231.58946,
            },
            reference_properties={
                "cc_bond_length": 1.396,
                "ch_bond_length": 1.084,
                "aromaticity_index": 0.95,
            },
            recommended_active_space=(6, 6),
            active_space_description="π orbitals (delocalized)",
            difficulty_level="medium",
            computational_cost="high",
            references=["Gauss, J. et al. J. Chem. Phys. 116, 1773 (2002)"],
        )

    def get_system(self, name: str) -> Optional[BenchmarkSystem]:
        """Get a benchmark system by name."""
        return self.systems.get(name.lower())

    def list_systems(self) -> List[str]:
        """List all available benchmark systems."""
        return list(self.systems.keys())

    def get_systems_by_difficulty(self, difficulty: str) -> List[BenchmarkSystem]:
        """Get systems by difficulty level."""
        return [
            sys for sys in self.systems.values() if sys.difficulty_level == difficulty
        ]

    def get_systems_by_cost(self, cost: str) -> List[BenchmarkSystem]:
        """Get systems by computational cost."""
        return [sys for sys in self.systems.values() if sys.computational_cost == cost]

    def get_validation_suite(
        self, difficulty_levels: List[str] = None
    ) -> List[BenchmarkSystem]:
        """Get a validation suite with systems of specified difficulty levels."""
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium"]

        systems = []
        for level in difficulty_levels:
            systems.extend(self.get_systems_by_difficulty(level))

        return systems

    def get_external_method_suite(
        self, external_methods: List[str], difficulty_levels: List[str] = None
    ) -> Dict[str, List[BenchmarkSystem]]:
        """Get benchmark systems optimized for specific external methods.

        Args:
            external_methods: List of external method names
            difficulty_levels: Difficulty levels to include

        Returns:
            Dictionary mapping method names to suitable benchmark systems
        """
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium"]

        # Method-specific system recommendations
        method_preferences = {
            "molpro": ["h2", "h2o", "n2", "f2"],  # Excellent for CASSCF/CASPT2
            "orca": ["h2", "lih", "h2o", "benzene"],  # Good all-around package
            "gaussian": ["h2o", "benzene"],  # Popular for organic chemistry
            "psi4": ["h2", "lih", "h2o", "n2"],  # Strong correlation methods
            "mrcc": ["h2", "lih", "f2"],  # Specialized for high-level correlation
            "cfour": ["h2", "lih", "n2"],  # High accuracy coupled cluster
            "nwchem": ["h2o", "benzene"],  # Good for large systems
            "qchem": ["h2", "h2o", "benzene"],  # Versatile package
        }

        result = {}
        for method in external_methods:
            # Get method-specific preferred systems
            preferred_systems = method_preferences.get(
                method.lower(), list(self.systems.keys())[:4]
            )

            # Filter by difficulty and availability
            method_systems = []
            for system_name in preferred_systems:
                system = self.get_system(system_name)
                if system and system.difficulty_level in difficulty_levels:
                    method_systems.append(system)

            result[method] = method_systems

        return result

    def add_external_reference_data(
        self,
        system_name: str,
        method: str,
        energy: float,
        source: str = "External calculation",
    ):
        """Add reference data from external method calculations.

        Args:
            system_name: Name of the benchmark system
            method: External method name
            energy: Reference energy value
            source: Source of the reference data
        """
        system = self.get_system(system_name)
        if system:
            system.reference_energies[method.lower()] = energy
            if source not in system.references:
                system.references.append(source)

    def validate_external_method_compatibility(self, method: str) -> Dict[str, Any]:
        """Check which systems are compatible with an external method.

        Args:
            method: External method name

        Returns:
            Dictionary with compatibility information
        """
        runner = ExternalMethodRunner()

        compatibility = {
            "method": method,
            "available": runner.is_method_available(method),
            "compatible_systems": [],
            "incompatible_systems": [],
            "recommended_systems": [],
        }

        if not compatibility["available"]:
            return compatibility

        # Check each system
        for system_name, system in self.systems.items():
            try:
                # Test if system can be run with this method
                test_input = {
                    "geometry": system.geometry,
                    "basis_set": system.basis_set,
                    "charge": system.charge,
                    "spin": system.spin,
                    "method": method,
                    "test_run": True,  # Flag for validation check only
                }

                runner.validate_input(method, test_input)
                compatibility["compatible_systems"].append(system_name)

                # Recommend based on difficulty and cost
                if system.difficulty_level in ["easy", "medium"]:
                    compatibility["recommended_systems"].append(system_name)

            except Exception:
                compatibility["incompatible_systems"].append(system_name)

        return compatibility
