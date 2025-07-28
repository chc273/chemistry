"""Create FCIDUMP files from active space selections with proper core energy treatment."""

from typing import Literal, Tuple, Union

import numpy as np
from pyscf import gto, scf
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


def active_space_to_fcidump(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    filename: str,
    method: Union[str, ActiveSpaceMethod] = ActiveSpaceMethod.AVAS,
    approach: Literal["casscf", "effective"] = "effective",
    **method_kwargs,
) -> str:
    """
    Convert active space selection to FCIDUMP format with different approaches.

    Args:
        scf_obj: Converged SCF object (RHF, UHF, or ROHF)
        system: Quantum system object
        filename: Output filename for the FCIDUMP file
        method: Active space selection method
        approach: How to treat core electrons:
            - "casscf": Run full CASSCF optimization (most accurate)
            - "effective": Use effective integrals without CASSCF optimization (recommended)
        **method_kwargs: Method-specific parameters

    Returns:
        Path to created FCIDUMP file

    Examples:
        >>> # Recommended approach - effective integrals without CASSCF cost
        >>> active_space_to_fcidump(rhf_obj, mol, "active.fcidump", "avas", "effective", threshold=0.3)

        >>> # Full CASSCF for highest accuracy
        >>> active_space_to_fcidump(rhf_obj, mol, "active.fcidump", "avas", "casscf", threshold=0.3)
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
