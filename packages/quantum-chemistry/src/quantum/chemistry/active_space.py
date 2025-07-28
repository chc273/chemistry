"""
Unified Active Space Finding Module

This module provides a unified interface for various active space selection methods
including:

- AVAS (Atomic Valence Active Space)
- APC (Atomic Population Coefficient)
- DMET-CAS (Density Matrix Embedding Theory for Complete Active Space)
- Natural Orbitals from MP2
- Boys and Pipek-Mezey orbital localization
- IAO (Intrinsic Atomic Orbitals)
- IBO (Intrinsic Bond Orbitals)
- Energy window-based selection
- Manual orbital selection

All methods are unified under a common interface for easy comparison and automatic
selection of the best active space for a given system.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from pydantic import BaseModel, Field
from pyscf import lo, scf
from pyscf.mcscf import apc, avas
from pyscf.tools import molden


def _load_avas_defaults() -> Dict[str, List[str]]:
    """Load AVAS default orbital selections from YAML file."""
    config_path = Path(__file__).parent / "avas_defaults.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ActiveSpaceMethod(str, Enum):
    """Available active space selection methods."""

    AVAS = "avas"  # Atomic Valence Active Space
    APC = "apc"  # Atomic Population Coefficient
    BOYS_LOCALIZATION = "boys"  # Boys localization
    PIPEK_MEZEY = "pipek_mezey"  # Pipek-Mezey localization
    NATURAL_ORBITALS = "natural_orbitals"  # Natural orbital selection from MP2
    ENERGY_WINDOW = "energy_window"  # Energy-based selection
    DMET_CAS = "dmet_cas"  # DMET-CAS method
    IAO_LOCALIZATION = "iao"  # Intrinsic Atomic Orbitals
    IBO_LOCALIZATION = "ibo"  # Intrinsic Bond Orbitals
    MANUAL = "manual"  # Manual orbital selection


class ActiveSpaceResult(BaseModel):
    """Results from active space selection."""

    method: ActiveSpaceMethod = Field(description="Method used for selection")
    n_active_electrons: int = Field(description="Number of active electrons")
    n_active_orbitals: int = Field(description="Number of active orbitals")
    active_orbital_indices: List[int] = Field(description="Indices of active orbitals")
    orbital_energies: Optional[np.ndarray] = Field(
        default=None, description="Energies of selected orbitals"
    )
    orbital_coefficients: Optional[np.ndarray] = Field(
        default=None, description="Coefficients of active orbitals"
    )
    selection_scores: Optional[np.ndarray] = Field(
        default=None, description="Selection scores for each orbital"
    )
    avas_data: Optional[Dict[str, Any]] = Field(
        default=None, description="AVAS-specific data"
    )
    localization_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Localization-specific data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class BaseActiveSpaceSelector(ABC):
    """Abstract base class for active space selectors."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space orbitals."""
        pass


class AVASSelector(BaseActiveSpaceSelector):
    """
    Atomic Valence Active Space (AVAS) selector.

    AVAS automatically selects active space orbitals based on their overlap with
    atomic valence orbitals. If no atoms are specified, it uses default valence
    orbital selections from the YAML configuration.
    """

    def __init__(
        self,
        avas_atoms: Optional[List[Union[str, int]]] = None,
        ao_labels: Optional[List[str]] = None,
        minao: str = "minao",
        threshold: float = 0.2,
        canonicalize: bool = True,
        openshell_option: int = 2,
        **kwargs,
    ):
        """
        Initialize AVAS selector.

        Args:
            avas_atoms: Atoms to include in active space (list of symbols or indices).
                       If None, uses all heavy atoms with default valence orbitals.
            ao_labels: Explicit atomic orbital labels for AVAS. If provided, overrides
                      automatic generation from avas_atoms.
            minao: Minimal atomic orbital basis for AVAS
            threshold: Threshold for orbital selection
            canonicalize: Whether to canonicalize orbitals
            openshell_option: Option for open-shell systems
        """
        super().__init__(**kwargs)
        self.avas_atoms = avas_atoms
        self.ao_labels = ao_labels
        self.minao = minao
        self.threshold = threshold
        self.canonicalize = canonicalize
        self.openshell_option = openshell_option

        # Load default AVAS orbital configurations
        self._avas_defaults = _load_avas_defaults()

    def _get_ao_labels(self, mol) -> List[str]:
        """Get AO labels for AVAS based on molecular composition."""
        if self.ao_labels is not None:
            return self.ao_labels

        ao_labels = []

        if self.avas_atoms is not None:
            # Use specified atoms
            for atom_spec in self.avas_atoms:
                if isinstance(atom_spec, int):
                    # Atom index
                    element = mol.atom[atom_spec][0]
                else:
                    # Element symbol
                    element = atom_spec

                if element in self._avas_defaults:
                    ao_labels.extend(self._avas_defaults[element])
                else:
                    # Fallback for unknown elements
                    ao_labels.append(f"{element} 3d")
        else:
            # Default: use all heavy atoms with their valence orbitals
            for atom in mol.atom:
                element = atom[0]
                if element not in ["H", "h"]:
                    if element in self._avas_defaults:
                        ao_labels.extend(self._avas_defaults[element])
                    else:
                        # Fallback for unknown elements
                        ao_labels.append(f"{element} 3d")

        return ao_labels

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using AVAS method."""
        mol = mf.mol

        # Get AO labels for AVAS
        ao_labels = self._get_ao_labels(mol)

        # Run AVAS
        norb_act, ne_act, orbs_act = avas.avas(
            mf,
            ao_labels,
            minao=self.minao,
            threshold=self.threshold,
            canonicalize=self.canonicalize,
            openshell_option=self.openshell_option,
        )

        # Get orbital indices
        active_indices = list(range(norb_act))

        # Get orbital energies if available
        orbital_energies = None
        if hasattr(mf, "mo_energy") and mf.mo_energy is not None:
            if isinstance(mf.mo_energy, tuple):  # UHF case
                orbital_energies = mf.mo_energy[0][:norb_act] if norb_act > 0 else None
            else:
                orbital_energies = mf.mo_energy[:norb_act] if norb_act > 0 else None

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.AVAS,
            n_active_electrons=ne_act,
            n_active_orbitals=norb_act,
            active_orbital_indices=active_indices,
            orbital_energies=orbital_energies,
            orbital_coefficients=orbs_act,
            avas_data={
                "ao_labels": ao_labels,
                "avas_atoms": self.avas_atoms,
                "threshold": self.threshold,
                "minao": self.minao,
                "canonicalize": self.canonicalize,
                "openshell_option": self.openshell_option,
            },
        )


class APCSelector(BaseActiveSpaceSelector):
    """
    Approximate Pair Coefficient (APC) selector using PySCF's APC implementation.

    The APC method is a ranked-orbital approach for automated active space selection
    that estimates the importance of orbitals through pair-interaction framework from
    orbital energies and features of the Hartree-Fock exchange matrix.

    Based on the research by Daniel S. King and Laura Gagliardi:
    - https://doi.org/10.1021/acs.jctc.1c00037 (Ranked-Orbital Approach)
    - https://doi.org/10.1021/acs.jctc.2c00630 (Large-Scale Benchmarking)
    """

    def __init__(
        self,
        max_size: Union[int, Tuple[int, int]] = (8, 8),
        n: int = 2,
        eps: float = 1e-3,
        verbose: int = 4,
        **kwargs,
    ):
        """
        Initialize APC selector.

        Args:
            max_size: Maximum active space size constraint.
                     If tuple, interpreted as (nelecas, ncas)
                     If int, interpreted as max number of orbitals
            n: Number of APC iterations (APC-N approach)
            eps: Small offset for orbital ranking
            verbose: Verbosity level for PySCF output
        """
        super().__init__(**kwargs)
        self.max_size = max_size
        self.n = n
        self.eps = eps
        self.verbose = verbose

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using PySCF's APC method."""

        # Initialize PySCF APC object
        apc_obj = apc.APC(
            mf, max_size=self.max_size, n=self.n, eps=self.eps, verbose=self.verbose
        )

        # Run APC calculation
        ncas, nelecas, mo_coeff = apc_obj.kernel()

        # Get the active orbital indices
        active_indices = (
            apc_obj.active_idx if hasattr(apc_obj, "active_idx") else list(range(ncas))
        )

        # Get orbital energies if available
        orbital_energies = None
        if hasattr(mf, "mo_energy") and mf.mo_energy is not None:
            if isinstance(mf.mo_energy, tuple):  # UHF case
                orbital_energies = (
                    mf.mo_energy[0][active_indices] if len(active_indices) > 0 else None
                )
            else:
                orbital_energies = (
                    mf.mo_energy[active_indices] if len(active_indices) > 0 else None
                )

        # Get APC entropies as selection scores
        selection_scores = None
        if hasattr(apc_obj, "entropies"):
            selection_scores = (
                apc_obj.entropies[active_indices] if len(active_indices) > 0 else None
            )

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.APC,
            n_active_electrons=nelecas,
            n_active_orbitals=ncas,
            active_orbital_indices=active_indices,
            orbital_energies=orbital_energies,
            orbital_coefficients=mo_coeff,
            selection_scores=selection_scores,
            metadata={
                "max_size": self.max_size,
                "n_iterations": self.n,
                "eps": self.eps,
                "apc_algorithm": "ranked_orbital_approach",
                "reference": "https://doi.org/10.1021/acs.jctc.1c00037",
            },
        )


class LocalizationSelector(BaseActiveSpaceSelector):
    """Orbital localization-based active space selector."""

    def __init__(
        self,
        localization_method: str = "boys",
        energy_window: Tuple[float, float] = (2.0, 2.0),
        target_atoms: Optional[List[Union[str, int]]] = None,
        localization_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize localization-based selector.

        Args:
            localization_method: 'boys' or 'pipek_mezey'
            energy_window: Energy window around HOMO-LUMO (HOMO-x, LUMO+x)
            target_atoms: Target atoms for localization analysis
            localization_threshold: Threshold for localized orbital selection
        """
        super().__init__(**kwargs)
        self.localization_method = localization_method
        self.energy_window = energy_window
        self.target_atoms = target_atoms
        self.localization_threshold = localization_threshold

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using orbital localization."""

        mol = mf.mol

        # Get molecular orbital data
        if isinstance(mf.mo_coeff, tuple):  # UHF case
            mo_coeff = mf.mo_coeff[0]
            mo_energy = mf.mo_energy[0]
        else:
            mo_coeff = mf.mo_coeff
            mo_energy = mf.mo_energy

        # Define energy window
        homo_idx = mf.mol.nelectron // 2 - 1
        lumo_idx = homo_idx + 1

        start_idx = max(0, homo_idx - int(self.energy_window[0]))
        end_idx = min(len(mo_energy), lumo_idx + int(self.energy_window[1]))

        # Extract orbitals in energy window
        window_orbitals = mo_coeff[:, start_idx:end_idx]

        # Perform localization
        if self.localization_method.lower() == "boys":
            localizer = lo.Boys(mol, window_orbitals)
        elif self.localization_method.lower() == "pipek_mezey":
            localizer = lo.PipekMezey(mol, window_orbitals)
        else:
            raise ValueError(f"Unknown localization method: {self.localization_method}")

        localized_orbitals = localizer.kernel()

        # Analyze localization on target atoms
        target_atoms = self.target_atoms
        if target_atoms is None:
            # Default: use all heavy atoms
            target_atoms = []
            for i, atom in enumerate(mol.atom):
                if atom[0] not in ["H", "h"]:
                    target_atoms.append(i)

        # Calculate localization scores
        localization_scores = []
        selected_orbitals = []

        for i, orbital in enumerate(localized_orbitals.T):
            atom_contributions = 0.0
            for atom_idx in target_atoms:
                atom_basis_start = mol.aoslice_by_atom()[atom_idx, 2]
                atom_basis_end = mol.aoslice_by_atom()[atom_idx, 3]
                atom_contributions += np.sum(
                    orbital[atom_basis_start:atom_basis_end] ** 2
                )

            localization_scores.append(atom_contributions)

            if atom_contributions >= self.localization_threshold:
                selected_orbitals.append(start_idx + i)

        # Calculate number of active electrons
        n_active_electrons = 0
        for orbital_idx in selected_orbitals:
            if orbital_idx <= homo_idx:
                n_active_electrons += 2

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.BOYS_LOCALIZATION
            if self.localization_method.lower() == "boys"
            else ActiveSpaceMethod.PIPEK_MEZEY,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(selected_orbitals),
            active_orbital_indices=selected_orbitals,
            orbital_energies=mo_energy[selected_orbitals],
            orbital_coefficients=localized_orbitals,
            selection_scores=np.array(localization_scores),
            localization_data={
                "localization_method": self.localization_method,
                "energy_window": self.energy_window,
                "target_atoms": target_atoms,
                "localization_threshold": self.localization_threshold,
            },
        )


class EnergyWindowSelector(BaseActiveSpaceSelector):
    """Energy window-based active space selector."""

    def __init__(
        self,
        energy_window: Tuple[float, float] = (2.0, 2.0),
        max_orbitals: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize energy window selector.

        Args:
            energy_window: Energy window (HOMO-x, LUMO+x) in eV
            max_orbitals: Maximum number of orbitals to select
        """
        super().__init__(**kwargs)
        self.energy_window = energy_window
        self.max_orbitals = max_orbitals

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space based on energy window."""

        # Get molecular orbital data
        if isinstance(mf.mo_coeff, tuple):  # UHF case
            mo_coeff = mf.mo_coeff[0]
            mo_energy = mf.mo_energy[0]
        else:
            mo_coeff = mf.mo_coeff
            mo_energy = mf.mo_energy

        # Define energy window
        homo_idx = mf.mol.nelectron // 2 - 1
        lumo_idx = homo_idx + 1

        if homo_idx < len(mo_energy):
            homo_energy = mo_energy[homo_idx]
        else:
            homo_energy = mo_energy[-1]

        if lumo_idx < len(mo_energy):
            lumo_energy = mo_energy[lumo_idx]
        else:
            lumo_energy = homo_energy + 0.5  # Estimate

        # Convert eV to Hartree (1 eV = 0.036749 Hartree)
        ev_to_hartree = 0.036749
        energy_low = homo_energy - self.energy_window[0] * ev_to_hartree
        energy_high = lumo_energy + self.energy_window[1] * ev_to_hartree

        # Select orbitals in energy window
        selected_orbitals = []
        for i, energy in enumerate(mo_energy):
            if energy_low <= energy <= energy_high:
                selected_orbitals.append(i)

        # Apply maximum orbital limit
        if self.max_orbitals is not None and len(selected_orbitals) > self.max_orbitals:
            # Sort by distance from HOMO-LUMO gap and select closest
            gap_center = (homo_energy + lumo_energy) / 2
            distances = [abs(mo_energy[i] - gap_center) for i in selected_orbitals]
            sorted_indices = sorted(range(len(distances)), key=lambda x: distances[x])
            selected_orbitals = [
                selected_orbitals[i] for i in sorted_indices[: self.max_orbitals]
            ]

        # Calculate number of active electrons
        n_active_electrons = 0
        for orbital_idx in selected_orbitals:
            if orbital_idx <= homo_idx:
                n_active_electrons += 2

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.ENERGY_WINDOW,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(selected_orbitals),
            active_orbital_indices=selected_orbitals,
            orbital_energies=mo_energy[selected_orbitals],
            orbital_coefficients=mo_coeff[:, selected_orbitals],
            metadata={
                "energy_window": self.energy_window,
                "energy_window_hartree": (energy_low, energy_high),
                "max_orbitals": self.max_orbitals,
                "homo_energy": homo_energy,
                "lumo_energy": lumo_energy,
            },
        )


class DMETCASSelector(BaseActiveSpaceSelector):
    """DMET-CAS active space selector."""

    def __init__(
        self,
        target_atoms: Optional[List[Union[str, int]]] = None,
        **kwargs,
    ):
        """
        Initialize DMET-CAS selector.

        Args:
            target_atoms: Target atoms for active space selection
        """
        super().__init__(**kwargs)
        self.target_atoms = target_atoms

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using DMET-CAS method."""

        try:
            from pyscf.mcscf import dmet_cas
        except ImportError:
            raise ImportError("DMET-CAS not available in this PySCF version")

        mol = mf.mol

        # Determine target atoms for DMET-CAS
        target_atoms = self.target_atoms
        if target_atoms is None:
            # Default: use all heavy atoms
            ao_labels = []
            for i, atom in enumerate(mol.atom):
                if atom[0] not in ["H", "h"]:
                    ao_labels.append(f"{atom[0]} 2p")  # Common valence orbitals
        else:
            # Convert atom indices/symbols to AO labels
            ao_labels = []
            for atom in target_atoms:
                if isinstance(atom, int):
                    atom_symbol = mol.atom[atom][0]
                else:
                    atom_symbol = atom
                ao_labels.append(f"{atom_symbol} 2p")

        # Run DMET-CAS
        rdm1 = mf.make_rdm1()
        ncas, nelecas, mo = dmet_cas.guess_cas(mf, rdm1, ao_labels)

        # Calculate active orbital indices
        active_indices = list(range(ncas))

        # Get orbital energies if available
        orbital_energies = None
        if hasattr(mf, "mo_energy") and mf.mo_energy is not None:
            if isinstance(mf.mo_energy, tuple):  # UHF case
                orbital_energies = mf.mo_energy[0][:ncas]
            else:
                orbital_energies = mf.mo_energy[:ncas]

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.DMET_CAS,
            n_active_electrons=nelecas,
            n_active_orbitals=ncas,
            active_orbital_indices=active_indices,
            orbital_energies=orbital_energies,
            orbital_coefficients=mo[:, :ncas],
            metadata={
                "target_atoms": target_atoms,
                "ao_labels": ao_labels,
            },
        )


class NaturalOrbitalSelector(BaseActiveSpaceSelector):
    """Natural orbital-based active space selector using MP2."""

    def __init__(
        self,
        occupation_threshold: float = 0.02,
        max_orbitals: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize natural orbital selector.

        Args:
            occupation_threshold: Minimum occupation for orbital selection
            max_orbitals: Maximum number of orbitals to select
        """
        super().__init__(**kwargs)
        self.occupation_threshold = occupation_threshold
        self.max_orbitals = max_orbitals

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using MP2 natural orbitals."""

        try:
            from pyscf import mp
            from pyscf.mcscf import addons
        except ImportError:
            raise ImportError("MP2 module not available in this PySCF version")

        # Run MP2 calculation
        if isinstance(mf, scf.uhf.UHF):
            mp2 = mp.UMP2(mf)
        else:
            mp2 = mp.MP2(mf)
        mp2.kernel()

        # Get natural orbitals and occupations
        noons, natorbs = addons.make_natural_orbitals(mp2)

        # Select orbitals based on occupation threshold
        selected_indices = []
        for i, occ in enumerate(noons):
            if occ >= self.occupation_threshold and occ <= (
                2.0 - self.occupation_threshold
            ):
                selected_indices.append(i)

        # Apply maximum orbital limit
        if self.max_orbitals is not None and len(selected_indices) > self.max_orbitals:
            # Sort by distance from 1.0 occupation (most active)
            distances = [abs(noons[i] - 1.0) for i in selected_indices]
            sorted_indices = sorted(range(len(distances)), key=lambda x: distances[x])
            selected_indices = [
                selected_indices[i] for i in sorted_indices[: self.max_orbitals]
            ]

        # Calculate number of active electrons
        n_active_electrons = int(sum(noons[i] for i in selected_indices))

        # Get orbital energies if available
        orbital_energies = None
        if hasattr(mf, "mo_energy") and mf.mo_energy is not None:
            if isinstance(mf.mo_energy, tuple):  # UHF case
                orbital_energies = mf.mo_energy[0][selected_indices]
            else:
                orbital_energies = mf.mo_energy[selected_indices]

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.NATURAL_ORBITALS,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(selected_indices),
            active_orbital_indices=selected_indices,
            orbital_energies=orbital_energies,
            orbital_coefficients=natorbs[:, selected_indices],
            selection_scores=noons[selected_indices],
            metadata={
                "occupation_threshold": self.occupation_threshold,
                "max_orbitals": self.max_orbitals,
                "natural_occupations": noons,
            },
        )


class IAOSelector(BaseActiveSpaceSelector):
    """Intrinsic Atomic Orbital (IAO) based active space selector."""

    def __init__(
        self,
        target_atoms: Optional[List[Union[str, int]]] = None,
        minao_basis: str = "minao",
        energy_window: Tuple[float, float] = (2.0, 2.0),
        **kwargs,
    ):
        """
        Initialize IAO selector.

        Args:
            target_atoms: Target atoms for IAO analysis
            minao_basis: Minimal basis for IAO construction
            energy_window: Energy window around HOMO-LUMO
        """
        super().__init__(**kwargs)
        self.target_atoms = target_atoms
        self.minao_basis = minao_basis
        self.energy_window = energy_window

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using IAO analysis."""

        try:
            from pyscf import lo
        except ImportError:
            raise ImportError("Localization module not available in this PySCF version")

        mol = mf.mol

        # Get molecular orbital data
        if isinstance(mf.mo_coeff, tuple):  # UHF case
            mo_coeff = mf.mo_coeff[0]
            mo_energy = mf.mo_energy[0]
        else:
            mo_coeff = mf.mo_coeff
            mo_energy = mf.mo_energy

        # Define energy window
        homo_idx = mf.mol.nelectron // 2 - 1
        lumo_idx = homo_idx + 1

        start_idx = max(0, homo_idx - int(self.energy_window[0]))
        end_idx = min(len(mo_energy), lumo_idx + int(self.energy_window[1]))

        # Extract orbitals in energy window
        window_orbitals = mo_coeff[:, start_idx:end_idx]

        # Compute IAOs for occupied orbitals
        occ_orbitals = mo_coeff[:, mo_energy <= mo_energy[homo_idx]]
        iao_coeff = lo.iao.iao(mol, occ_orbitals, minao=self.minao_basis)

        # Orthogonalize IAOs
        ovlp = mf.get_ovlp()
        iao_coeff = lo.vec_lowdin(iao_coeff, ovlp)

        # Project window orbitals onto IAO space
        iao_proj = np.dot(iao_coeff.T, np.dot(ovlp, window_orbitals))
        iao_norms = np.linalg.norm(iao_proj, axis=0)

        # Select orbitals with significant IAO character
        threshold = 0.3  # Adjustable threshold
        selected_indices = []
        for i, norm in enumerate(iao_norms):
            if norm >= threshold:
                selected_indices.append(start_idx + i)

        # Calculate number of active electrons
        n_active_electrons = 0
        for orbital_idx in selected_indices:
            if orbital_idx <= homo_idx:
                n_active_electrons += 2

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.IAO_LOCALIZATION,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(selected_indices),
            active_orbital_indices=selected_indices,
            orbital_energies=mo_energy[selected_indices],
            orbital_coefficients=window_orbitals[
                :, [i - start_idx for i in selected_indices]
            ],
            selection_scores=iao_norms[[i - start_idx for i in selected_indices]],
            localization_data={
                "minao_basis": self.minao_basis,
                "energy_window": self.energy_window,
                "iao_coefficients": iao_coeff,
            },
        )


class IBOSelector(BaseActiveSpaceSelector):
    """Intrinsic Bond Orbital (IBO) based active space selector."""

    def __init__(
        self,
        target_atoms: Optional[List[Union[str, int]]] = None,
        minao_basis: str = "minao",
        energy_window: Tuple[float, float] = (2.0, 2.0),
        **kwargs,
    ):
        """
        Initialize IBO selector.

        Args:
            target_atoms: Target atoms for IBO analysis
            minao_basis: Minimal basis for IAO construction
            energy_window: Energy window around HOMO-LUMO
        """
        super().__init__(**kwargs)
        self.target_atoms = target_atoms
        self.minao_basis = minao_basis
        self.energy_window = energy_window

    def select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """Select active space using IBO analysis."""

        try:
            from pyscf import lo
        except ImportError:
            raise ImportError("Localization module not available in this PySCF version")

        mol = mf.mol

        # Get molecular orbital data
        if isinstance(mf.mo_coeff, tuple):  # UHF case
            mo_coeff = mf.mo_coeff[0]
            mo_energy = mf.mo_energy[0]
        else:
            mo_coeff = mf.mo_coeff
            mo_energy = mf.mo_energy

        # Define energy window
        homo_idx = mf.mol.nelectron // 2 - 1
        lumo_idx = homo_idx + 1

        start_idx = max(0, homo_idx - int(self.energy_window[0]))
        end_idx = min(len(mo_energy), lumo_idx + int(self.energy_window[1]))

        # Extract occupied orbitals for IBO construction
        occ_orbitals = mo_coeff[:, mo_energy <= mo_energy[homo_idx]]

        # Compute IAOs first (required for IBOs)
        iao_coeff = lo.iao.iao(mol, occ_orbitals, minao=self.minao_basis)
        ovlp = mf.get_ovlp()
        iao_coeff = lo.vec_lowdin(iao_coeff, ovlp)

        # Compute IBOs
        ibo_coeff = lo.ibo.ibo(mol, occ_orbitals, iaos=iao_coeff)

        # Extract orbitals in energy window
        window_orbitals = mo_coeff[:, start_idx:end_idx]

        # Project window orbitals onto IBO space
        ibo_proj = np.dot(ibo_coeff.T, np.dot(ovlp, window_orbitals))
        ibo_norms = np.linalg.norm(ibo_proj, axis=0)

        # Select orbitals with significant IBO character
        threshold = 0.3  # Adjustable threshold
        selected_indices = []
        for i, norm in enumerate(ibo_norms):
            if norm >= threshold:
                selected_indices.append(start_idx + i)

        # Calculate number of active electrons
        n_active_electrons = 0
        for orbital_idx in selected_indices:
            if orbital_idx <= homo_idx:
                n_active_electrons += 2

        return ActiveSpaceResult(
            method=ActiveSpaceMethod.IBO_LOCALIZATION,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(selected_indices),
            active_orbital_indices=selected_indices,
            orbital_energies=mo_energy[selected_indices],
            orbital_coefficients=window_orbitals[
                :, [i - start_idx for i in selected_indices]
            ],
            selection_scores=ibo_norms[[i - start_idx for i in selected_indices]],
            localization_data={
                "minao_basis": self.minao_basis,
                "energy_window": self.energy_window,
                "iao_coefficients": iao_coeff,
                "ibo_coefficients": ibo_coeff,
            },
        )


class UnifiedActiveSpaceFinder:
    """
    Unified interface for active space selection methods.

    This class provides a single interface to access multiple active space
    selection methods including AVAS, APC, localization methods, and others.
    """

    def __init__(self):
        """Initialize the unified active space finder."""
        self._selectors = {
            ActiveSpaceMethod.AVAS: AVASSelector,
            ActiveSpaceMethod.APC: APCSelector,
            ActiveSpaceMethod.BOYS_LOCALIZATION: lambda **kwargs: LocalizationSelector(
                localization_method="boys", **kwargs
            ),
            ActiveSpaceMethod.PIPEK_MEZEY: lambda **kwargs: LocalizationSelector(
                localization_method="pipek_mezey", **kwargs
            ),
            ActiveSpaceMethod.ENERGY_WINDOW: EnergyWindowSelector,
            ActiveSpaceMethod.DMET_CAS: DMETCASSelector,
            ActiveSpaceMethod.NATURAL_ORBITALS: NaturalOrbitalSelector,
            ActiveSpaceMethod.IAO_LOCALIZATION: IAOSelector,
            ActiveSpaceMethod.IBO_LOCALIZATION: IBOSelector,
        }

    def find_active_space(
        self,
        method: Union[str, ActiveSpaceMethod],
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        **kwargs,
    ) -> ActiveSpaceResult:
        """
        Find active space using specified method.

        Args:
            method: Active space selection method
            mf: PySCF mean field object
            **kwargs: Method-specific parameters

        Returns:
            ActiveSpaceResult containing selected active space
        """
        if isinstance(method, str):
            method = ActiveSpaceMethod(method.lower())

        if method not in self._selectors:
            raise ValueError(f"Unknown active space method: {method}")

        selector = self._selectors[method](**kwargs)
        return selector.select_active_space(mf, **kwargs)

    def compare_methods(
        self,
        methods: List[Union[str, ActiveSpaceMethod]],
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, ActiveSpaceResult]:
        """
        Compare multiple active space selection methods.

        Args:
            methods: List of methods to compare
            mf: PySCF mean field object
            method_kwargs: Method-specific kwargs

        Returns:
            Dictionary mapping method names to results
        """
        results = {}
        method_kwargs = method_kwargs or {}

        for method in methods:
            method_name = str(method)
            kwargs = method_kwargs.get(method_name, {})

            try:
                result = self.find_active_space(method, mf, **kwargs)
                results[method_name] = result
            except Exception as e:
                warnings.warn(f"Failed to run method {method}: {e}")
                continue

        return results

    def auto_select_active_space(
        self,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        target_size: Optional[Tuple[int, int]] = None,
        priority_methods: Optional[List[ActiveSpaceMethod]] = None,
        **kwargs,
    ) -> ActiveSpaceResult:
        """
        Automatically select best active space based on multiple criteria.

        Args:
            mf: PySCF mean field object
            target_size: Target (n_electrons, n_orbitals) if specified
            priority_methods: Methods to try in order of preference
            **kwargs: Additional parameters

        Returns:
            Best active space result
        """
        if priority_methods is None:
            priority_methods = [
                ActiveSpaceMethod.AVAS,
                ActiveSpaceMethod.DMET_CAS,
                ActiveSpaceMethod.NATURAL_ORBITALS,
                ActiveSpaceMethod.APC,
                ActiveSpaceMethod.ENERGY_WINDOW,
            ]

        best_result = None
        best_score = -1

        for method in priority_methods:
            try:
                result = self.find_active_space(method, mf, **kwargs)

                # Score based on target size match and orbital energy gap
                score = self._score_active_space(result, target_size)

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                warnings.warn(f"Auto-selection failed for method {method}: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All active space selection methods failed")

        return best_result

    def _score_active_space(
        self,
        result: ActiveSpaceResult,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Score active space quality."""
        score = 0.0

        # Size matching score
        if target_size is not None:
            target_electrons, target_orbitals = target_size
            electron_diff = abs(result.n_active_electrons - target_electrons)
            orbital_diff = abs(result.n_active_orbitals - target_orbitals)
            size_score = 1.0 / (1.0 + electron_diff + orbital_diff)
            score += size_score * 0.5

        # Energy gap score (prefer well-separated HOMO-LUMO)
        if result.orbital_energies is not None and len(result.orbital_energies) > 1:
            energy_gap = np.max(result.orbital_energies) - np.min(
                result.orbital_energies
            )
            gap_score = min(1.0, energy_gap / 0.1)  # Normalize to ~0.1 Hartree
            score += gap_score * 0.3

        # Selection score (if available)
        if result.selection_scores is not None:
            avg_score = np.mean(result.selection_scores)
            score += min(1.0, avg_score) * 0.2

        return score

    def export_molden(
        self,
        result: ActiveSpaceResult,
        mf: Union[scf.hf.SCF, scf.uhf.UHF],
        filename: str,
    ) -> None:
        """
        Export active space orbitals to Molden format.

        Args:
            result: Active space result
            mf: PySCF mean field object
            filename: Output filename
        """

        if result.orbital_coefficients is None:
            raise ValueError("No orbital coefficients available for export")

        # Create Molden file with active orbitals
        with open(filename, "w") as f:
            molden.header(mf.mol, f)
            molden.orbital_coeff(
                mf.mol,
                f,
                result.orbital_coefficients,
                ene=result.orbital_energies,
                occ=np.ones(result.n_active_orbitals),
            )

    def get_available_methods(self) -> List[str]:
        """Get list of available active space selection methods."""
        return [method.value for method in self._selectors.keys()]


# Convenience functions
def find_active_space_avas(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    avas_atoms: Optional[List[Union[str, int]]] = None,
    ao_labels: Optional[List[str]] = None,
    threshold: float = 0.2,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for AVAS (Atomic Valence Active Space) active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.AVAS,
        mf,
        avas_atoms=avas_atoms,
        ao_labels=ao_labels,
        threshold=threshold,
        **kwargs,
    )


def find_active_space_apc(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    max_size: Union[int, Tuple[int, int]] = (8, 8),
    n: int = 2,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for APC (Approximate Pair Coefficient) active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.APC, mf, max_size=max_size, n=n, **kwargs
    )


def find_active_space_energy_window(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    energy_window: Tuple[float, float] = (2.0, 2.0),
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for energy window active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.ENERGY_WINDOW,
        mf,
        energy_window=energy_window,
        **kwargs,
    )


def find_active_space_dmet_cas(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    target_atoms: Optional[List[Union[str, int]]] = None,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for DMET-CAS active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.DMET_CAS, mf, target_atoms=target_atoms, **kwargs
    )


def find_active_space_natural_orbitals(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    occupation_threshold: float = 0.02,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for natural orbital active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.NATURAL_ORBITALS,
        mf,
        occupation_threshold=occupation_threshold,
        **kwargs,
    )


def find_active_space_iao(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    target_atoms: Optional[List[Union[str, int]]] = None,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for IAO active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.IAO_LOCALIZATION,
        mf,
        target_atoms=target_atoms,
        **kwargs,
    )


def find_active_space_ibo(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    target_atoms: Optional[List[Union[str, int]]] = None,
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for IBO active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.find_active_space(
        ActiveSpaceMethod.IBO_LOCALIZATION,
        mf,
        target_atoms=target_atoms,
        **kwargs,
    )


def auto_find_active_space(
    mf: Union[scf.hf.SCF, scf.uhf.UHF],
    **kwargs,
) -> ActiveSpaceResult:
    """Convenience function for automatic active space selection."""
    finder = UnifiedActiveSpaceFinder()
    return finder.auto_select_active_space(mf, **kwargs)
