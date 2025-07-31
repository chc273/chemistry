"""Create FCIDUMP files from active space selections with proper core energy treatment."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pyscf import ao2mo, gto, scf, symm
from pyscf.mcscf import CASSCF
from pyscf.tools.fcidump import from_integrals, from_mcscf

from quantum.chemistry.active_space import (
    ActiveSpaceMethod,
    ActiveSpaceResult,
    UnifiedActiveSpaceFinder,
)


def create_minimal_casscf_for_integrals(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    ncas: int,
    nelecas: Union[int, Tuple[int, int]],
    mo_coeff: np.ndarray,
) -> CASSCF:
    """
    Create a minimal CASSCF object to compute effective integrals without optimization.

    This allows us to get h1eff and ecore that include core electron contributions
    without running the expensive CASSCF optimization.

    Args:
        scf_obj: Converged SCF object
        ncas: Number of active orbitals
        nelecas: Number of active electrons (int or tuple)
        mo_coeff: MO coefficient matrix ordered as [core|active|virtual]

    Returns:
        CASSCF object with properly set MO coefficients
    """
    # Create CASSCF object
    casscf = CASSCF(scf_obj, ncas, nelecas)
    casscf.mo_coeff = mo_coeff

    # Set up the active space parameters
    ncore = casscf.ncore

    # For AVAS, mo_coeff is already the full reordered matrix
    # CASSCF will slice it internally using ncore and ncas

    return casscf


def from_active_space_result(
    active_space_result: ActiveSpaceResult,
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
) -> str:
    """
    Create FCIDUMP from ActiveSpaceResult with proper core energy treatment.

    This computes the effective 1-electron integrals that include core electron
    field effects, rather than just the bare active space integrals.

    Args:
        active_space_result: Result from active space selection
        scf_obj: Converged SCF object
        filename: Output FCIDUMP filename

    Returns:
        Path to created FCIDUMP file
    """
    ncas = active_space_result.n_active_orbitals
    nelecas = active_space_result.n_active_electrons
    mo_coeff = active_space_result.orbital_coefficients

    # Create minimal CASSCF object to get effective integrals
    casscf = create_minimal_casscf_for_integrals(scf_obj, ncas, nelecas, mo_coeff)

    # Get effective integrals (includes core electron effects)
    h1eff, ecore = casscf.get_h1eff()
    h2eff = casscf.get_h2eff()

    # Handle electron count
    if isinstance(nelecas, tuple):
        n_alpha, n_beta = nelecas
        ms = n_alpha - n_beta
        total_nelecas = n_alpha + n_beta
    else:
        ms = 0
        total_nelecas = nelecas

    # Write FCIDUMP with effective integrals
    from_integrals(
        filename,
        h1eff,  # Effective 1e integrals (includes core effects)
        h2eff,  # Active space 2e integrals
        ncas,  # Number of active orbitals
        total_nelecas,  # Total active electrons
        ecore,  # Core energy (includes all core contributions)
        ms,  # Spin multiplicity
    )

    return filename


def to_openmolcas_fcidump(
    active_space_result: ActiveSpaceResult,
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    orbital_reordering: Optional[List[int]] = None,
    symmetry_labels: Optional[List[str]] = None,
) -> str:
    """
    Create OpenMolcas-compatible FCIDUMP file with proper orbital ordering.
    
    OpenMolcas has specific requirements for FCIDUMP files:
    - Orbitals must be ordered by symmetry
    - Symmetry labels must be provided
    - Integral format must match OpenMolcas conventions
    
    Args:
        active_space_result: Result from active space selection
        scf_obj: Converged SCF object
        filename: Output FCIDUMP filename
        orbital_reordering: Manual orbital reordering (None for automatic)
        symmetry_labels: Symmetry labels for each orbital
        
    Returns:
        Path to created FCIDUMP file
    """
    ncas = active_space_result.n_active_orbitals
    nelecas = active_space_result.n_active_electrons
    mo_coeff = active_space_result.orbital_coefficients
    
    # Create minimal CASSCF object to get effective integrals
    casscf = create_minimal_casscf_for_integrals(scf_obj, ncas, nelecas, mo_coeff)
    h1eff, ecore = casscf.get_h1eff()
    h2eff_2d = casscf.get_h2eff()
    # Convert h2eff from 2D to 4D format for OpenMolcas
    h2eff = ao2mo.restore(1, h2eff_2d, ncas)
    
    # Apply orbital reordering if specified
    if orbital_reordering is not None:
        if len(orbital_reordering) != ncas:
            raise ValueError(
                f"Orbital reordering length ({len(orbital_reordering)}) "
                f"must match number of active orbitals ({ncas})"
            )
        # Reorder integrals according to specified ordering
        h1eff = h1eff[np.ix_(orbital_reordering, orbital_reordering)]
        h2eff = h2eff[np.ix_(orbital_reordering, orbital_reordering, 
                           orbital_reordering, orbital_reordering)]
    
    # Handle electron count for OpenMolcas format
    if isinstance(nelecas, tuple):
        n_alpha, n_beta = nelecas
        ms = n_alpha - n_beta
        total_nelecas = n_alpha + n_beta
    else:
        ms = 0
        total_nelecas = nelecas
    
    # Write FCIDUMP with OpenMolcas-compatible format
    _write_openmolcas_fcidump(
        filename, h1eff, h2eff, ncas, total_nelecas, ecore, ms, 
        symmetry_labels
    )
    
    return filename


def _write_openmolcas_fcidump(
    filename: str,
    h1eff: np.ndarray,
    h2eff: np.ndarray,
    ncas: int,
    nelecas: int,
    ecore: float,
    ms: int,
    symmetry_labels: Optional[List[str]] = None,
):
    """
    Write FCIDUMP file in OpenMolcas-compatible format.
    
    Args:
        filename: Output filename
        h1eff: Effective 1-electron integrals
        h2eff: 2-electron integrals in active space
        ncas: Number of active orbitals
        nelecas: Number of active electrons
        ecore: Core energy
        ms: Spin multiplicity (2S)
        symmetry_labels: Symmetry labels for orbitals
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("&FCI NORB={:d},NELEC={:d},MS2={:d},\n".format(
            ncas, nelecas, ms
        ))
        
        # Add symmetry information if available
        if symmetry_labels:
            orbsym_str = ",".join(str(_get_symmetry_index(label)) 
                                for label in symmetry_labels)
            f.write("ORBSYM={},\n".format(orbsym_str))
        
        f.write("ISYM=1,\n")
        f.write("&END\n")
        
        # Write two-electron integrals (ijkl) format
        for i in range(ncas):
            for j in range(ncas):
                for k in range(ncas):
                    for l in range(ncas):
                        if abs(h2eff[i,j,k,l]) > 1e-15:
                            f.write(f"{h2eff[i,j,k,l]:20.12E} "
                                  f"{i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}\n")
        
        # Write one-electron integrals
        for i in range(ncas):
            for j in range(ncas):
                if abs(h1eff[i,j]) > 1e-15:
                    f.write(f"{h1eff[i,j]:20.12E} "
                          f"{i+1:4d} {j+1:4d}    0    0\n")
        
        # Write core energy
        f.write(f"{ecore:20.12E}    0    0    0    0\n")


def _get_symmetry_index(symmetry_label: str) -> int:
    """
    Convert symmetry label to OpenMolcas symmetry index.
    
    Args:
        symmetry_label: Symmetry label (e.g., 'A1', 'B2', etc.)
        
    Returns:
        Symmetry index for OpenMolcas
    """
    # Basic mapping for common point groups
    # This is a simplified version - full implementation would need
    # proper point group analysis
    symmetry_map = {
        'A1': 1, 'A': 1,    # Totally symmetric
        'A2': 2,
        'B1': 3, 'B': 2,   # Antisymmetric
        'B2': 4,
        'E': 5,            # Doubly degenerate
        'T1': 6,           # Triply degenerate
        'T2': 7,
    }
    
    # Try exact match first
    if symmetry_label in symmetry_map:
        return symmetry_map[symmetry_label]
    
    # Try prefix match for compound labels
    for label, index in symmetry_map.items():
        if symmetry_label.startswith(label):
            return index
    
    # Default to totally symmetric if not found
    return 1


def detect_orbital_symmetries(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    active_orbitals: np.ndarray,
) -> List[str]:
    """
    Detect symmetry labels for active orbitals.
    
    Args:
        scf_obj: SCF object with molecule
        active_orbitals: Active orbital coefficients
        
    Returns:
        List of symmetry labels for each active orbital
    """
    mol = scf_obj.mol
    
    # If molecule has symmetry, use it
    if hasattr(mol, 'symmetry') and mol.symmetry:
        try:
            # Use PySCF's symmetry analysis
            irrep_ids = symm.label_orb_symm(
                mol, mol.irrep_id, mol.symm_orb, active_orbitals
            )
            return [mol.irrep_name[i] for i in irrep_ids]
        except (AttributeError, IndexError):
            pass
    
    # Fallback: assign all orbitals to totally symmetric representation
    n_orbitals = active_orbitals.shape[1]
    return ['A1'] * n_orbitals


def active_space_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    method: Union[str, ActiveSpaceMethod] = ActiveSpaceMethod.AVAS,
    approach: Literal["casscf", "effective"] = "effective",
    openmolcas_compatible: bool = False,
    **method_kwargs,
) -> str:
    """
    Convert active space selection to FCIDUMP format with different approaches.

    Args:
        scf_obj: Converged SCF object (RHF, UHF, or ROHF)
        filename: Output filename for the FCIDUMP file
        method: Active space selection method
        approach: How to treat core electrons:
            - "casscf": Run full CASSCF optimization (most accurate)
            - "effective": Use effective integrals without CASSCF optimization (recommended)
        openmolcas_compatible: Generate OpenMolcas-compatible FCIDUMP format
        **method_kwargs: Method-specific parameters

    Returns:
        Path to created FCIDUMP file

    Examples:
        >>> # Recommended approach - effective integrals without CASSCF cost
        >>> active_space_to_fcidump(rhf_obj, "active.fcidump", "avas", "effective", threshold=0.3)

        >>> # OpenMolcas-compatible format
        >>> active_space_to_fcidump(rhf_obj, "active.fcidump", "avas", "effective", 
        ...                        openmolcas_compatible=True, threshold=0.3)
    """
    # Select active space using unified finder
    finder = UnifiedActiveSpaceFinder()
    result = finder.find_active_space(method, scf_obj, **method_kwargs)

    if approach == "casscf":
        # Full CASSCF optimization
        casscf = CASSCF(scf_obj, result.n_active_orbitals, result.n_active_electrons)
        casscf.kernel(mo_coeff=result.orbital_coefficients)
        return from_mcscf(casscf, filename)

    elif approach == "effective":
        # Effective integrals without CASSCF optimization
        if openmolcas_compatible:
            # Use OpenMolcas-compatible format
            symmetry_labels = detect_orbital_symmetries(
                scf_obj, result.orbital_coefficients
            )
            return to_openmolcas_fcidump(
                result, scf_obj, filename, symmetry_labels=symmetry_labels
            )
        else:
            # Standard PySCF format
            return from_active_space_result(result, scf_obj, filename)

    else:
        raise ValueError(f"Unknown approach '{approach}'. Use 'casscf' or 'effective'")


# Convenience functions for specific methods
def avas_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    approach: Literal["casscf", "effective"] = "effective",
    threshold: float = 0.2,
    **kwargs,
) -> str:
    """AVAS-specific FCIDUMP creation."""
    return active_space_to_fcidump(
        scf_obj,
        filename,
        ActiveSpaceMethod.AVAS,
        approach,
        threshold=threshold,
        **kwargs,
    )


def apc_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    approach: Literal["casscf", "effective"] = "effective",
    max_size: Union[int, Tuple[int, int]] = (8, 8),
    **kwargs,
) -> str:
    """APC-specific FCIDUMP creation."""
    return active_space_to_fcidump(
        scf_obj,
        filename,
        ActiveSpaceMethod.APC,
        approach,
        max_size=max_size,
        **kwargs,
    )


def natural_orbitals_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    approach: Literal["casscf", "effective"] = "effective",
    occupation_threshold: float = 0.02,
    **kwargs,
) -> str:
    """Natural orbitals-specific FCIDUMP creation."""
    return active_space_to_fcidump(
        scf_obj,
        filename,
        ActiveSpaceMethod.NATURAL_ORBITALS,
        approach,
        occupation_threshold=occupation_threshold,
        **kwargs,
    )


def dmet_cas_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    approach: Literal["casscf", "effective"] = "effective",
    **kwargs,
) -> str:
    """DMET-CAS-specific FCIDUMP creation."""
    return active_space_to_fcidump(
        scf_obj, filename, ActiveSpaceMethod.DMET_CAS, approach, **kwargs
    )


if __name__ == "__main__":
    # Example usage
    from pyscf import gto, scf

    # Set up molecule
    mol = gto.Mole()
    mol.atom = "F 0 0 0; F 0 0 1.41"
    mol.basis = "sto-3g"
    mol.build()

    # Create quantum system (mock for this example)
    class MockSystem:
        def __init__(self, mol):
            self.mol = mol
            self.name = "F2"

    system = MockSystem(mol)

    # Run SCF
    rhf = scf.RHF(mol)
    rhf.kernel()

    # Create FCIDUMP files with different approaches
    print("Creating FCIDUMP files...")

    # Recommended: effective integrals (fast, accurate)
    avas_to_fcidump(rhf, "f2_avas_effective.fcidump", "effective")

    # Full CASSCF (slow, most accurate)
    avas_to_fcidump(rhf, "f2_avas_casscf.fcidump", "casscf")

    # Different methods
    try:
        apc_to_fcidump(rhf, "f2_apc_effective.fcidump", "effective")
    except ImportError:
        print("APC method not available in this PySCF version")

    natural_orbitals_to_fcidump(rhf, "f2_no_effective.fcidump", "effective")

    print("Done! The 'effective' and 'casscf' approaches should give similar energies.")
