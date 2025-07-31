"""
Fast screening diagnostics for multireference character assessment.

This module implements computationally inexpensive diagnostic methods that can
be performed on DFT/SCF calculations within minutes. These methods provide
initial screening for multireference character before deciding whether to
run expensive coupled cluster diagnostics.

References:
- HOMO-LUMO gap: Roos, B. O. Adv. Chem. Phys. 1987, 69, 399.
- Spin contamination: Yamaguchi, K. et al. Theor. Chim. Acta 1988, 73, 337.
- NOONs: Groß, E. K. U. et al. Adv. Quantum Chem. 1982, 21, 255.
- FOD: Grimme, S. J. Chem. Phys. 2013, 138, 244104.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
from pyscf import scf, lo
from scipy.linalg import eigh

from .models.core_models import (
    DiagnosticResult,
    DiagnosticMethod,
    DiagnosticConfig,
    SystemClassification,
    MultireferenceCharacter,
)


def calculate_homo_lumo_gap(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate HOMO-LUMO gap as indicator of multireference character.
    
    Small HOMO-LUMO gaps indicate near-degeneracy and potential
    multireference character. This is a fast and reliable screening method.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration (uses defaults if None)
        
    Returns:
        DiagnosticResult with HOMO-LUMO gap analysis
        
    References:
        Roos, B. O. "The Complete Active Space Self-Consistent Field Method 
        and its Applications in Electronic Structure Calculations" 
        Adv. Chem. Phys. 1987, 69, 399-445.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    # Extract orbital energies
    if isinstance(scf_obj, scf.uhf.UHF):
        # For UHF, use alpha orbitals
        mo_energy = scf_obj.mo_energy[0]
        mo_occ = scf_obj.mo_occ[0]
    else:
        mo_energy = scf_obj.mo_energy
        mo_occ = scf_obj.mo_occ
    
    # Find HOMO and LUMO
    homo_idx = np.where(mo_occ > 0)[0][-1]
    lumo_idx = np.where(mo_occ == 0)[0][0]
    
    homo_energy = mo_energy[homo_idx]
    lumo_energy = mo_energy[lumo_idx]
    
    # Gap in eV
    gap_hartree = lumo_energy - homo_energy
    gap_ev = gap_hartree * 27.2114  # Hartree to eV conversion
    
    # Classify system based on molecular composition
    system_class = _classify_system(scf_obj)
    
    # Apply system-specific thresholds if needed
    thresholds = config.get_thresholds(DiagnosticMethod.HOMO_LUMO_GAP)
    
    # Adjust thresholds for transition metals (typically smaller gaps)
    if system_class == SystemClassification.TRANSITION_METAL:
        thresholds = {k: v * 0.7 for k, v in thresholds.items()}
    
    # Determine multireference character (inverted logic - smaller gaps = more MR)
    if gap_ev > thresholds.get("weak", 3.0):
        mr_character = MultireferenceCharacter.NONE
    elif gap_ev > thresholds.get("moderate", 2.0):
        mr_character = MultireferenceCharacter.WEAK
    elif gap_ev > thresholds.get("strong", 1.0):
        mr_character = MultireferenceCharacter.MODERATE
    elif gap_ev > thresholds.get("very_strong", 0.5):
        mr_character = MultireferenceCharacter.STRONG
    else:
        mr_character = MultireferenceCharacter.VERY_STRONG
    
    # Confidence based on gap size (very confident for extreme values)
    if gap_ev > 4.0:
        confidence = 0.95  # Very confident no MR character
    elif gap_ev < 0.5:
        confidence = 0.95  # Very confident strong MR character
    else:
        confidence = 0.8   # Moderate confidence
    
    calc_time = time.time() - start_time
    
    return DiagnosticResult(
        method=DiagnosticMethod.HOMO_LUMO_GAP,
        value=gap_ev,
        multireference_character=mr_character,
        confidence=confidence,
        threshold_weak=thresholds.get("weak"),
        threshold_moderate=thresholds.get("moderate"),
        threshold_strong=thresholds.get("strong"),
        threshold_very_strong=thresholds.get("very_strong"),
        metadata={
            "homo_energy_hartree": homo_energy,
            "lumo_energy_hartree": lumo_energy,
            "gap_hartree": gap_hartree,
            "homo_index": int(homo_idx),
            "lumo_index": int(lumo_idx),
        },
        calculation_time=calc_time,
        system_classification=system_class,
        reference="Roos, B. O. Adv. Chem. Phys. 1987, 69, 399",
    )


def calculate_spin_contamination(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate spin contamination as indicator of multireference character.
    
    For UHF calculations, spin contamination measured as <S²> - S(S+1)
    indicates broken-symmetry solutions and multireference character.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with spin contamination analysis
        
    References:
        Yamaguchi, K. et al. "Symmetry and broken symmetry in molecular 
        orbital descriptions of unstable molecules"
        Theor. Chim. Acta 1988, 73, 337-364.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    # Only meaningful for UHF calculations
    if not isinstance(scf_obj, scf.uhf.UHF):
        # For RHF, return zero contamination
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.SPIN_CONTAMINATION,
            value=0.0,
            multireference_character=MultireferenceCharacter.NONE,
            confidence=1.0,
            metadata={"note": "RHF calculation, no spin contamination"},
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            reference="Yamaguchi, K. et al. Theor. Chim. Acta 1988, 73, 337",
        )
    
    # Calculate <S²> expectation value
    s_squared = scf_obj.spin_square()[0]
    
    # Calculate theoretical S(S+1) for pure spin state
    n_unpaired = abs(scf_obj.nelec[0] - scf_obj.nelec[1])
    s_value = n_unpaired / 2.0
    s_squared_pure = s_value * (s_value + 1.0)
    
    # Spin contamination
    contamination = s_squared - s_squared_pure
    
    # Classify system
    system_class = _classify_system(scf_obj)
    
    # Determine multireference character
    mr_character = config.classify_value(
        DiagnosticMethod.SPIN_CONTAMINATION, contamination
    )
    
    # Confidence based on contamination magnitude
    if contamination < 0.01:
        confidence = 0.9   # Low contamination, confident assessment
    elif contamination > 0.3:
        confidence = 0.95  # High contamination, very confident
    else:
        confidence = 0.8   # Moderate contamination
    
    thresholds = config.get_thresholds(DiagnosticMethod.SPIN_CONTAMINATION)
    calc_time = time.time() - start_time
    
    return DiagnosticResult(
        method=DiagnosticMethod.SPIN_CONTAMINATION,
        value=contamination,
        multireference_character=mr_character,
        confidence=confidence,
        threshold_weak=thresholds.get("weak"),
        threshold_moderate=thresholds.get("moderate"),
        threshold_strong=thresholds.get("strong"),
        threshold_very_strong=thresholds.get("very_strong"),
        metadata={
            "s_squared_calculated": s_squared,
            "s_squared_pure": s_squared_pure,
            "s_value": s_value,
            "n_unpaired_electrons": n_unpaired,
            "nelec_alpha": scf_obj.nelec[0],
            "nelec_beta": scf_obj.nelec[1],
        },
        calculation_time=calc_time,
        system_classification=system_class,
        reference="Yamaguchi, K. et al. Theor. Chim. Acta 1988, 73, 337",
    )


def calculate_natural_orbital_occupations(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate natural orbital occupation numbers (NOONs) diagnostic.
    
    Natural orbital occupations deviating significantly from 0 or 2
    indicate multireference character. This method analyzes the
    one-particle density matrix from SCF calculations.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with NOON analysis
        
    References:
        Groß, E. K. U. et al. "Density-Functional Theory for Atoms and Molecules"
        Adv. Quantum Chem. 1982, 21, 255-291.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    # Calculate natural orbitals from density matrix
    if isinstance(scf_obj, scf.uhf.UHF):
        # For UHF, use total density matrix
        dm = scf_obj.make_rdm1()
        if isinstance(dm, tuple):
            dm = dm[0] + dm[1]  # Total density matrix
    else:
        dm = scf_obj.make_rdm1()
    
    # Get overlap matrix
    ovlp = scf_obj.get_ovlp()
    
    # Diagonalize density matrix in AO basis
    # P = C @ n @ C.T, where n are natural occupations
    try:
        # Use overlap-orthogonalized basis
        s_sqrt_inv = np.linalg.inv(np.linalg.cholesky(ovlp))
        dm_ortho = s_sqrt_inv @ dm @ s_sqrt_inv.T
        occupations, nat_orbs_ortho = eigh(dm_ortho)
        
        # Sort by occupation (descending)
        idx = np.argsort(occupations)[::-1]
        occupations = occupations[idx]
        
        # Clamp occupations to valid range [0, 2] to handle numerical precision issues
        occupations = np.clip(occupations, 0.0, 2.0)
        
        # Transform back to AO basis
        nat_orbs = s_sqrt_inv.T @ nat_orbs_ortho[:, idx]
        
    except np.linalg.LinAlgError:
        # Fallback: direct diagonalization (less numerically stable)
        eigenvals, eigenvecs = eigh(dm, ovlp)
        idx = np.argsort(eigenvals)[::-1]
        occupations = eigenvals[idx]
        
        # Clamp occupations to valid range [0, 2] to handle numerical precision issues
        occupations = np.clip(occupations, 0.0, 2.0)
        
        nat_orbs = eigenvecs[:, idx]
    
    # Calculate deviations from integer occupations
    # Look for occupations significantly different from 0 or 2
    deviations = []
    for occ in occupations:
        if occ > 1.9:
            deviation = 2.0 - occ  # Deviation from 2
        elif occ < 0.1:
            deviation = occ - 0.0  # Deviation from 0  
        else:
            # Fractional occupation - calculate distance to nearest integer
            deviation = min(abs(occ - 0.0), abs(occ - 2.0))
        deviations.append(deviation)
    
    # Calculate diagnostic value as maximum deviation
    max_deviation = max(deviations) if deviations else 0.0
    
    # Alternative: sum of all significant deviations
    significant_deviations = [d for d in deviations if d > 0.05]
    sum_deviations = sum(significant_deviations)
    
    # Use maximum deviation as primary diagnostic
    diagnostic_value = max_deviation
    
    # Classify system
    system_class = _classify_system(scf_obj)
    
    # Determine multireference character
    mr_character = config.classify_value(
        DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS, diagnostic_value
    )
    
    # Confidence based on magnitude and consistency
    if max_deviation < 0.05:
        confidence = 0.9   # Clear single-reference
    elif max_deviation > 0.4:
        confidence = 0.95  # Clear multireference
    else:
        confidence = 0.8   # Borderline case
    
    thresholds = config.get_thresholds(DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS)
    calc_time = time.time() - start_time
    
    return DiagnosticResult(
        method=DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS,
        value=diagnostic_value,
        multireference_character=mr_character,
        confidence=confidence,
        threshold_weak=thresholds.get("weak"),
        threshold_moderate=thresholds.get("moderate"),
        threshold_strong=thresholds.get("strong"),
        threshold_very_strong=thresholds.get("very_strong"),
        metadata={
            "natural_occupations": occupations.tolist(),
            "max_deviation": max_deviation,
            "sum_significant_deviations": sum_deviations,
            "n_significant_deviations": len(significant_deviations),
            "all_deviations": deviations,
        },
        calculation_time=calc_time,
        system_classification=system_class,
        reference="Groß, E. K. U. et al. Adv. Quantum Chem. 1982, 21, 255",
    )


def calculate_fractional_occupation_density(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate Fractional Occupation Density (FOD) diagnostic.
    
    FOD analysis identifies regions of fractional electron occupation
    that indicate multireference character. This method is particularly
    useful for transition metal complexes and biradical systems.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with FOD analysis
        
    References:
        Grimme, S. "A General Quantum Mechanically Derived Force Field (QMDFF) 
        for Molecules and Condensed Phase Simulations"
        J. Chem. Phys. 2013, 138, 244104.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    # FOD analysis requires natural orbitals
    # We'll use a simplified version based on Mulliken populations
    
    if isinstance(scf_obj, scf.uhf.UHF):
        # For UHF, analyze spin density
        dm_alpha = scf_obj.make_rdm1()[0]
        dm_beta = scf_obj.make_rdm1()[1]
        dm_spin = dm_alpha - dm_beta
        
        # Mulliken spin populations
        ovlp = scf_obj.get_ovlp()
        pop_spin = np.diag(dm_spin @ ovlp)
        
        # Sum atomic spin populations
        mol = scf_obj.mol
        atomic_spins = []
        
        for i in range(mol.natm):
            # Get basis functions on this atom
            atom_basis = []
            for j, (atom_id, *_) in enumerate(mol.ao_labels()):
                if atom_id == i:
                    atom_basis.append(j)
            
            if atom_basis:
                atomic_spin = sum(pop_spin[j] for j in atom_basis)
                atomic_spins.append(abs(atomic_spin))
        
        # FOD estimate: sum of fractional atomic spins
        fod_electrons = sum(s for s in atomic_spins if 0.1 < s < 1.9)
        
        metadata = {
            "atomic_spins": atomic_spins,
            "total_spin_population": sum(atomic_spins),
            "method": "UHF_mulliken_spin",
        }
        
    else:
        # For RHF, use natural orbital occupations as proxy
        try:
            # Get natural orbitals from previous calculation
            noon_result = calculate_natural_orbital_occupations(scf_obj, config)
            occupations = noon_result.metadata["natural_occupations"]
            
            # Count electrons in fractionally occupied orbitals
            fod_electrons = sum(occ for occ in occupations if 0.1 < occ < 1.9)
            
            metadata = {
                "natural_occupations": occupations,
                "method": "RHF_natural_orbitals",
            }
            
        except Exception:
            # Fallback: assume no fractional occupation for RHF
            fod_electrons = 0.0
            metadata = {"method": "RHF_fallback", "note": "Assumed no FOD for RHF"}
    
    # Classify system
    system_class = _classify_system(scf_obj)
    
    # Determine multireference character
    mr_character = config.classify_value(
        DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY, fod_electrons
    )
    
    # Confidence based on FOD magnitude
    if fod_electrons < 0.5:
        confidence = 0.85  # Low FOD
    elif fod_electrons > 3.0:
        confidence = 0.95  # High FOD, very confident
    else:
        confidence = 0.8   # Moderate FOD
    
    thresholds = config.get_thresholds(DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY)
    calc_time = time.time() - start_time
    
    return DiagnosticResult(
        method=DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY,
        value=fod_electrons,
        multireference_character=mr_character,
        confidence=confidence,
        threshold_weak=thresholds.get("weak"),
        threshold_moderate=thresholds.get("moderate"),
        threshold_strong=thresholds.get("strong"),
        threshold_very_strong=thresholds.get("very_strong"),
        metadata=metadata,
        calculation_time=calc_time,
        system_classification=system_class,
        reference="Grimme, S. J. Chem. Phys. 2013, 138, 244104",
    )


def calculate_bond_order_fluctuation(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate bond order fluctuation as diagnostic for multireference character.
    
    This method analyzes fluctuations in bond orders that can indicate
    resonance structures and multireference character. Uses Mayer bond
    order analysis.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with bond order fluctuation analysis
        
    References:
        Mayer, I. "Charge, bond order and valence in the AB initio SCF theory"
        Chem. Phys. Lett. 1983, 97, 270-274.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    try:
        # Calculate Mayer bond orders
        bond_orders = _calculate_mayer_bond_orders(scf_obj)
        
        if not bond_orders:
            # No bonds found or calculation failed
            calc_time = time.time() - start_time
            return DiagnosticResult(
                method=DiagnosticMethod.BOND_ORDER_FLUCTUATION,
                value=0.0,
                multireference_character=MultireferenceCharacter.NONE,
                confidence=0.5,
                metadata={"note": "No significant bonds found"},
                calculation_time=calc_time,
                system_classification=_classify_system(scf_obj),
                reference="Mayer, I. Chem. Phys. Lett. 1983, 97, 270",
            )
        
        # Analyze bond order fluctuations
        bond_values = list(bond_orders.values())
        
        # Calculate deviation from ideal integer bond orders
        deviations = []
        for bo in bond_values:
            # Find nearest integer bond order
            nearest_int = round(bo)
            if nearest_int == 0 and bo > 0.1:
                nearest_int = 1  # Minimum significant bond
            deviation = abs(bo - nearest_int)
            deviations.append(deviation)
        
        # Diagnostic value: maximum deviation or average deviation
        max_deviation = max(deviations) if deviations else 0.0
        avg_deviation = np.mean(deviations) if deviations else 0.0
        
        # Use maximum deviation as diagnostic
        diagnostic_value = max_deviation
        
        # Classify system
        system_class = _classify_system(scf_obj)
        
        # Determine multireference character
        mr_character = config.classify_value(
            DiagnosticMethod.BOND_ORDER_FLUCTUATION, diagnostic_value
        )
        
        # Confidence based on consistency
        if max_deviation < 0.05:
            confidence = 0.8   # Clear integer bond orders
        elif max_deviation > 0.3:
            confidence = 0.9   # Clear non-integer bond orders
        else:
            confidence = 0.75  # Borderline case
        
        thresholds = config.get_thresholds(DiagnosticMethod.BOND_ORDER_FLUCTUATION)
        calc_time = time.time() - start_time
        
        return DiagnosticResult(
            method=DiagnosticMethod.BOND_ORDER_FLUCTUATION,
            value=diagnostic_value,
            multireference_character=mr_character,
            confidence=confidence,
            threshold_weak=thresholds.get("weak"),
            threshold_moderate=thresholds.get("moderate"),
            threshold_strong=thresholds.get("strong"),
            threshold_very_strong=thresholds.get("very_strong"),
            metadata={
                "bond_orders": bond_orders,
                "max_deviation": max_deviation,
                "avg_deviation": avg_deviation,
                "all_deviations": deviations,
                "n_bonds_analyzed": len(bond_values),
            },
            calculation_time=calc_time,
            system_classification=system_class,
            reference="Mayer, I. Chem. Phys. Lett. 1983, 97, 270",
        )
        
    except Exception as e:
        # Fallback if bond order calculation fails
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.BOND_ORDER_FLUCTUATION,
            value=0.0,
            multireference_character=MultireferenceCharacter.NONE,
            confidence=0.3,
            metadata={"error": str(e), "note": "Bond order calculation failed"},
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            converged=False,
            reference="Mayer, I. Chem. Phys. Lett. 1983, 97, 270",
        )


def _classify_system(scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]) -> SystemClassification:
    """
    Classify chemical system type based on molecular composition.
    
    Args:
        scf_obj: SCF calculation object
        
    Returns:
        System classification
    """
    mol = scf_obj.mol
    
    # Get unique elements
    elements = {mol.atom_symbol(i) for i in range(mol.natm)}
    
    # Transition metals
    transition_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
    }
    
    if elements & transition_metals:
        # Check if it's a cluster (multiple transition metals)
        tm_count = sum(1 for i in range(mol.natm) 
                      if mol.atom_symbol(i) in transition_metals)
        if tm_count > 1:
            return SystemClassification.METAL_CLUSTER
        else:
            return SystemClassification.TRANSITION_METAL
    
    # Check for biradical indicators (simplified)
    if isinstance(scf_obj, scf.uhf.UHF):
        n_unpaired = abs(scf_obj.nelec[0] - scf_obj.nelec[1])
        if n_unpaired >= 2 and len(elements & {'C', 'N', 'O'}) >= 2:
            return SystemClassification.BIRADICAL
    
    # Organic molecules
    if len(elements & {'C', 'H', 'N', 'O', 'S', 'P'}) >= 2:
        return SystemClassification.ORGANIC
    
    return SystemClassification.GENERAL


def _calculate_mayer_bond_orders(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
) -> dict:
    """
    Calculate Mayer bond orders between atoms.
    
    Args:
        scf_obj: SCF calculation object
        
    Returns:
        Dictionary of bond orders keyed by (atom1, atom2) tuples
    """
    mol = scf_obj.mol
    
    # Get density matrix and overlap
    if isinstance(scf_obj, scf.uhf.UHF):
        dm = scf_obj.make_rdm1()
        if isinstance(dm, tuple):
            dm = dm[0] + dm[1]  # Total density matrix
    else:
        dm = scf_obj.make_rdm1()
    
    ovlp = scf_obj.get_ovlp()
    
    # Calculate PS matrix (density-overlap product)
    ps = dm @ ovlp
    
    # Group basis functions by atoms
    atom_basis = {}
    for i, (atom_id, *_) in enumerate(mol.ao_labels()):
        if atom_id not in atom_basis:
            atom_basis[atom_id] = []
        atom_basis[atom_id].append(i)
    
    # Calculate bond orders between atoms
    bond_orders = {}
    
    for atom_a in range(mol.natm):
        for atom_b in range(atom_a + 1, mol.natm):
            if atom_a in atom_basis and atom_b in atom_basis:
                # Mayer bond order: 2 * sum_{mu on A, nu on B} (PS)_mu,nu * (PS)_nu,mu
                bond_order = 0.0
                
                for mu in atom_basis[atom_a]:
                    for nu in atom_basis[atom_b]:
                        bond_order += ps[mu, nu] * ps[nu, mu]
                
                bond_order *= 2.0
                
                # Only store significant bonds
                if bond_order > 0.05:
                    bond_orders[(atom_a, atom_b)] = bond_order
    
    return bond_orders