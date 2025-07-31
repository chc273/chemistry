"""
Reference diagnostic methods for multireference character assessment.

This module implements expensive but highly accurate diagnostic methods based on
coupled cluster calculations. These methods provide definitive assessment of
multireference character but require significant computational resources.

References:
- T1 diagnostic: Lee, T. J. & Taylor, P. R. Int. J. Quantum Chem. 1989, 23, 199.
- D1 diagnostic: Janssen, C. L. & Nielsen, I. M. B. Chem. Phys. Lett. 1998, 290, 423.
- S diagnostic: Krylov, A. I. J. Chem. Phys. 2000, 113, 6052.
"""

from __future__ import annotations

import time
from typing import Optional, Union, Tuple

import numpy as np
from pyscf import scf, cc, mp

from .models.core_models import (
    DiagnosticResult,
    DiagnosticMethod,
    DiagnosticConfig,
    SystemClassification,
    MultireferenceCharacter,
)
from .fast_screening import _classify_system


def calculate_t1_diagnostic(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate T1 diagnostic from CCSD calculation.
    
    The T1 diagnostic measures the magnitude of single excitation amplitudes
    in coupled cluster theory. Large T1 values indicate substantial single
    excitation character and potential multireference behavior.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with T1 diagnostic analysis
        
    References:
        Lee, T. J. & Taylor, P. R. "A diagnostic for determining the quality of 
        single-reference electron correlation methods"
        Int. J. Quantum Chem. 1989, 36, 199-207.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    try:
        # Set up CCSD calculation
        if isinstance(scf_obj, scf.uhf.UHF):
            mycc = cc.UCCSD(scf_obj)
        else:
            mycc = cc.CCSD(scf_obj)
        
        # Run CCSD calculation
        mycc.verbose = 0  # Suppress output
        e_ccsd, t1, t2 = mycc.kernel()
        
        if not mycc.converged:
            raise ValueError("CCSD calculation did not converge")
        
        # Calculate T1 diagnostic
        if isinstance(mycc, cc.UCCSD):
            # For UCCSD, average T1 norms for alpha and beta
            t1_norm_a = np.linalg.norm(t1[0])
            t1_norm_b = np.linalg.norm(t1[1])
            t1_diagnostic = (t1_norm_a + t1_norm_b) / 2.0
            
            metadata = {
                "t1_norm_alpha": t1_norm_a,
                "t1_norm_beta": t1_norm_b,
                "ccsd_energy": e_ccsd,
                "ccsd_converged": mycc.converged,
                "method": "UCCSD",
            }
        else:
            # For RCCSD
            t1_diagnostic = np.linalg.norm(t1)
            
            metadata = {
                "t1_norm": t1_diagnostic,
                "ccsd_energy": e_ccsd,
                "ccsd_converged": mycc.converged,
                "method": "RCCSD",
            }
        
        # Normalize by number of electrons (Lee & Taylor recommendation)
        n_electrons = sum(scf_obj.nelec) if hasattr(scf_obj, 'nelec') else scf_obj.mol.nelectron
        t1_diagnostic_normalized = t1_diagnostic / np.sqrt(n_electrons)
        
        # Use normalized value as diagnostic
        diagnostic_value = t1_diagnostic_normalized
        
        # Classify system
        system_class = _classify_system(scf_obj)
        
        # Adjust thresholds for system type
        thresholds = config.get_thresholds(DiagnosticMethod.T1_DIAGNOSTIC).copy()
        if system_class == SystemClassification.TRANSITION_METAL:
            # Transition metals typically have higher T1 values
            thresholds = {k: v * 1.5 for k, v in thresholds.items()}
        
        # Determine multireference character using adjusted thresholds
        if diagnostic_value < thresholds["weak"]:
            mr_character = MultireferenceCharacter.NONE
        elif diagnostic_value < thresholds["moderate"]:
            mr_character = MultireferenceCharacter.WEAK
        elif diagnostic_value < thresholds["strong"]:
            mr_character = MultireferenceCharacter.MODERATE
        elif diagnostic_value < thresholds["very_strong"]:
            mr_character = MultireferenceCharacter.STRONG
        else:
            mr_character = MultireferenceCharacter.VERY_STRONG
        
        # High confidence due to accurate CCSD calculation
        confidence = 0.95
        
        metadata.update({
            "t1_diagnostic_raw": t1_diagnostic,
            "t1_diagnostic_normalized": t1_diagnostic_normalized,
            "n_electrons": n_electrons,
            "thresholds_adjusted": thresholds,
        })
        
        calc_time = time.time() - start_time
        
        return DiagnosticResult(
            method=DiagnosticMethod.T1_DIAGNOSTIC,
            value=diagnostic_value,
            multireference_character=mr_character,
            confidence=confidence,
            threshold_weak=thresholds["weak"],
            threshold_moderate=thresholds["moderate"],
            threshold_strong=thresholds["strong"],
            threshold_very_strong=thresholds["very_strong"],
            metadata=metadata,
            calculation_time=calc_time,
            system_classification=system_class,
            reference="Lee, T. J. & Taylor, P. R. Int. J. Quantum Chem. 1989, 36, 199",
        )
        
    except Exception as e:
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.T1_DIAGNOSTIC,
            value=0.0,
            multireference_character=MultireferenceCharacter.NONE,
            confidence=0.0,
            metadata={
                "error": str(e),
                "note": "CCSD calculation failed",
            },
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            converged=False,
            reference="Lee, T. J. & Taylor, P. R. Int. J. Quantum Chem. 1989, 36, 199",
        )


def calculate_d1_diagnostic(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate D1 diagnostic from CCSD calculation.
    
    The D1 diagnostic is based on the norm of the CCSD density matrix
    difference from Hartree-Fock. It provides complementary information
    to the T1 diagnostic for assessing multireference character.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with D1 diagnostic analysis
        
    References:
        Janssen, C. L. & Nielsen, I. M. B. "New diagnostics for coupled-cluster 
        and Møller-Plesset perturbation theory"
        Chem. Phys. Lett. 1998, 290, 423-430.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    try:
        # Set up CCSD calculation
        if isinstance(scf_obj, scf.uhf.UHF):
            mycc = cc.UCCSD(scf_obj)
        else:
            mycc = cc.CCSD(scf_obj)
        
        # Run CCSD calculation
        mycc.verbose = 0
        e_ccsd, t1, t2 = mycc.kernel()
        
        if not mycc.converged:
            raise ValueError("CCSD calculation did not converge")
        
        # Calculate CCSD density matrix
        dm_cc = mycc.make_rdm1()
        
        # Get HF density matrix
        dm_hf = scf_obj.make_rdm1()
        
        # Calculate D1 diagnostic
        if isinstance(mycc, cc.UCCSD):
            # For UCCSD, handle alpha and beta separately
            if isinstance(dm_cc, tuple) and isinstance(dm_hf, tuple):
                # Both are tuples (alpha, beta)
                dm_diff_a = dm_cc[0] - dm_hf[0]
                dm_diff_b = dm_cc[1] - dm_hf[1]
                d1_a = np.linalg.norm(dm_diff_a)
                d1_b = np.linalg.norm(dm_diff_b)
                d1_diagnostic = (d1_a + d1_b) / 2.0
                
                metadata = {
                    "d1_alpha": d1_a,
                    "d1_beta": d1_b,
                    "method": "UCCSD",
                }
            else:
                # Handle case where one is total density matrix
                if isinstance(dm_cc, tuple):
                    dm_cc_total = dm_cc[0] + dm_cc[1]
                else:
                    dm_cc_total = dm_cc
                
                if isinstance(dm_hf, tuple):
                    dm_hf_total = dm_hf[0] + dm_hf[1]
                else:
                    dm_hf_total = dm_hf
                
                dm_diff = dm_cc_total - dm_hf_total
                d1_diagnostic = np.linalg.norm(dm_diff)
                
                metadata = {
                    "d1_total": d1_diagnostic,
                    "method": "UCCSD_total",
                }
        else:
            # For RCCSD
            dm_diff = dm_cc - dm_hf
            d1_diagnostic = np.linalg.norm(dm_diff)
            
            metadata = {
                "d1_norm": d1_diagnostic,
                "method": "RCCSD",
            }
        
        # Normalize by number of occupied orbitals
        if isinstance(scf_obj, scf.uhf.UHF):
            n_occ = max(scf_obj.nelec)
        else:
            n_occ = scf_obj.nelec // 2
        
        d1_diagnostic_normalized = d1_diagnostic / np.sqrt(n_occ)
        
        # Use normalized value as diagnostic
        diagnostic_value = d1_diagnostic_normalized
        
        # Classify system
        system_class = _classify_system(scf_obj)
        
        # Determine multireference character
        mr_character = config.classify_value(DiagnosticMethod.D1_DIAGNOSTIC, diagnostic_value)
        
        # High confidence due to accurate CCSD calculation
        confidence = 0.95
        
        metadata.update({
            "d1_diagnostic_raw": d1_diagnostic,
            "d1_diagnostic_normalized": d1_diagnostic_normalized,
            "n_occupied_orbitals": n_occ,
            "ccsd_energy": e_ccsd,
            "ccsd_converged": mycc.converged,
        })
        
        thresholds = config.get_thresholds(DiagnosticMethod.D1_DIAGNOSTIC)
        calc_time = time.time() - start_time
        
        return DiagnosticResult(
            method=DiagnosticMethod.D1_DIAGNOSTIC,
            value=diagnostic_value,
            multireference_character=mr_character,
            confidence=confidence,
            threshold_weak=thresholds.get("weak"),
            threshold_moderate=thresholds.get("moderate"),
            threshold_strong=thresholds.get("strong"),
            threshold_very_strong=thresholds.get("very_strong"),
            metadata=metadata,
            calculation_time=calc_time,
            system_classification=system_class,
            reference="Janssen, C. L. & Nielsen, I. M. B. Chem. Phys. Lett. 1998, 290, 423",
        )
        
    except Exception as e:
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.D1_DIAGNOSTIC,
            value=0.0,
            multireference_character=MultireferenceCharacter.NONE,
            confidence=0.0,
            metadata={
                "error": str(e),
                "note": "CCSD calculation failed",
            },
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            converged=False,
            reference="Janssen, C. L. & Nielsen, I. M. B. Chem. Phys. Lett. 1998, 290, 423",
        )


def calculate_correlation_recovery(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate correlation energy recovery diagnostic.
    
    This diagnostic compares the correlation energy recovered by single-reference
    methods (MP2, CCSD) to estimate the percentage of correlation energy that
    can be captured without multireference treatment.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with correlation recovery analysis
        
    References:
        Ramos-Cordoba, E. et al. "A simple approach to counteract the aufbau 
        violation in molecular DFT calculations"
        J. Chem. Theory Comput. 2015, 11, 1501-1508.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    try:
        # Calculate MP2 correlation energy
        if isinstance(scf_obj, scf.uhf.UHF):
            mp2_obj = mp.UMP2(scf_obj)
        else:
            mp2_obj = mp.MP2(scf_obj)
        
        mp2_obj.verbose = 0
        e_mp2, _ = mp2_obj.kernel()
        e_corr_mp2 = e_mp2 - scf_obj.e_tot
        
        # Estimate full correlation energy using CCSD
        try:
            if isinstance(scf_obj, scf.uhf.UHF):
                ccsd_obj = cc.UCCSD(scf_obj)
            else:
                ccsd_obj = cc.CCSD(scf_obj)
            
            ccsd_obj.verbose = 0
            e_ccsd, _, _ = ccsd_obj.kernel()
            e_corr_ccsd = e_ccsd - scf_obj.e_tot
            
            # Use CCSD as estimate of full correlation energy
            e_corr_full = e_corr_ccsd
            method_used = "CCSD"
            ccsd_converged = ccsd_obj.converged
            
        except Exception:
            # Fallback: use MP2 with empirical scaling
            e_corr_full = e_corr_mp2 * 1.2  # Rough estimate
            method_used = "MP2_scaled"
            ccsd_converged = False
        
        # Calculate recovery percentage
        if abs(e_corr_full) > 1e-10:
            recovery_percent = (e_corr_mp2 / e_corr_full) * 100.0
        else:
            recovery_percent = 100.0  # No correlation energy to recover
        
        # Diagnostic value is recovery percentage
        diagnostic_value = recovery_percent
        
        # Classify system
        system_class = _classify_system(scf_obj)
        
        # Determine multireference character (inverted logic - lower recovery = more MR)
        mr_character = config.classify_value(
            DiagnosticMethod.CORRELATION_RECOVERY, diagnostic_value
        )
        
        # Confidence based on method used and convergence
        if method_used == "CCSD" and ccsd_converged:
            confidence = 0.9
        elif method_used == "CCSD":
            confidence = 0.7  # CCSD didn't converge
        else:
            confidence = 0.6  # Using MP2 scaling
        
        metadata = {
            "mp2_correlation_energy": e_corr_mp2,
            "full_correlation_energy": e_corr_full,
            "recovery_percentage": recovery_percent,
            "method_used": method_used,
            "ccsd_converged": ccsd_converged,
            "scf_energy": scf_obj.e_tot,
        }
        
        if method_used == "CCSD":
            metadata["ccsd_energy"] = e_ccsd
        
        thresholds = config.get_thresholds(DiagnosticMethod.CORRELATION_RECOVERY)
        calc_time = time.time() - start_time
        
        return DiagnosticResult(
            method=DiagnosticMethod.CORRELATION_RECOVERY,
            value=diagnostic_value,
            multireference_character=mr_character,
            confidence=confidence,
            threshold_weak=thresholds.get("weak"),
            threshold_moderate=thresholds.get("moderate"),
            threshold_strong=thresholds.get("strong"),
            threshold_very_strong=thresholds.get("very_strong"),
            metadata=metadata,
            calculation_time=calc_time,
            system_classification=system_class,
            reference="Ramos-Cordoba, E. et al. J. Chem. Theory Comput. 2015, 11, 1501",
        )
        
    except Exception as e:
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.CORRELATION_RECOVERY,
            value=100.0,  # Assume full recovery on error
            multireference_character=MultireferenceCharacter.NONE,
            confidence=0.0,
            metadata={
                "error": str(e),
                "note": "Correlation energy calculation failed",
            },
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            converged=False,
            reference="Ramos-Cordoba, E. et al. J. Chem. Theory Comput. 2015, 11, 1501",
        )


def calculate_s_diagnostic(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticResult:
    """
    Calculate S diagnostic for strong correlation assessment.
    
    The S diagnostic is a newer method (2023) that combines information from
    natural orbital occupations and orbital energies to provide a robust
    measure of strong correlation effects.
    
    Args:
        scf_obj: Converged SCF calculation object
        config: Diagnostic configuration
        
    Returns:
        DiagnosticResult with S diagnostic analysis
        
    References:
        Krylov, A. I. "Size-consistent wave functions for bond-breaking: 
        the equation-of-motion spin-flip model"
        J. Chem. Phys. 2000, 113, 6052-6062.
    """
    start_time = time.time()
    
    if config is None:
        config = DiagnosticConfig()
    
    try:
        # This is a simplified implementation of the S diagnostic concept
        # A full implementation would require more sophisticated analysis
        
        # Get natural orbital occupations (from previous calculation)
        from .fast_screening import calculate_natural_orbital_occupations
        noon_result = calculate_natural_orbital_occupations(scf_obj, config)
        
        if not noon_result.converged:
            raise ValueError("Natural orbital calculation failed")
        
        occupations = np.array(noon_result.metadata["natural_occupations"])
        
        # Calculate entropy-like measure of occupation distribution
        # S = -sum(n_i * ln(n_i) + (2-n_i) * ln(2-n_i)) for significantly occupied orbitals
        s_diagnostic = 0.0
        n_active_orbitals = 0
        
        for occ in occupations:
            if 0.01 < occ < 1.99:  # Significantly fractionally occupied
                n_active_orbitals += 1
                
                # Avoid log(0) issues
                if occ > 0.01:
                    s_diagnostic -= occ * np.log(occ)
                if (2.0 - occ) > 0.01:
                    s_diagnostic -= (2.0 - occ) * np.log(2.0 - occ)
        
        # Normalize by number of active orbitals
        if n_active_orbitals > 0:
            s_diagnostic = s_diagnostic / n_active_orbitals
        
        # Take absolute value and scale
        diagnostic_value = abs(s_diagnostic)
        
        # Classify system
        system_class = _classify_system(scf_obj)
        
        # Determine multireference character
        mr_character = config.classify_value(DiagnosticMethod.S_DIAGNOSTIC, diagnostic_value)
        
        # Confidence based on number of active orbitals
        if n_active_orbitals == 0:
            confidence = 0.9  # Clear single-reference
        elif n_active_orbitals > 4:
            confidence = 0.85  # Many active orbitals, less certain
        else:
            confidence = 0.8   # Moderate confidence
        
        metadata = {
            "s_diagnostic_raw": s_diagnostic,
            "n_active_orbitals": n_active_orbitals,
            "natural_occupations": occupations.tolist(),
            "noon_diagnostic_value": noon_result.value,
            "method": "simplified_entropy",
        }
        
        thresholds = config.get_thresholds(DiagnosticMethod.S_DIAGNOSTIC)
        calc_time = time.time() - start_time
        
        return DiagnosticResult(
            method=DiagnosticMethod.S_DIAGNOSTIC,
            value=diagnostic_value,
            multireference_character=mr_character,
            confidence=confidence,
            threshold_weak=thresholds.get("weak"),
            threshold_moderate=thresholds.get("moderate"),
            threshold_strong=thresholds.get("strong"),
            threshold_very_strong=thresholds.get("very_strong"),
            metadata=metadata,
            calculation_time=calc_time,
            system_classification=system_class,
            reference="Krylov, A. I. J. Chem. Phys. 2000, 113, 6052",
        )
        
    except Exception as e:
        calc_time = time.time() - start_time
        return DiagnosticResult(
            method=DiagnosticMethod.S_DIAGNOSTIC,
            value=0.0,
            multireference_character=MultireferenceCharacter.NONE,
            confidence=0.0,
            metadata={
                "error": str(e),
                "note": "S diagnostic calculation failed",
            },
            calculation_time=calc_time,
            system_classification=_classify_system(scf_obj),
            converged=False,
            reference="Krylov, A. I. J. Chem. Phys. 2000, 113, 6052",
        )


def _estimate_full_correlation_energy(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
) -> Tuple[float, str]:
    """
    Estimate full correlation energy using available methods.
    
    Args:
        scf_obj: SCF calculation object
        
    Returns:
        Tuple of (correlation_energy, method_used)
    """
    try:
        # Try CCSD first
        if isinstance(scf_obj, scf.uhf.UHF):
            ccsd_obj = cc.UCCSD(scf_obj)
        else:
            ccsd_obj = cc.CCSD(scf_obj)
        
        ccsd_obj.verbose = 0
        e_ccsd, _, _ = ccsd_obj.kernel()
        
        if ccsd_obj.converged:
            e_corr = e_ccsd - scf_obj.e_tot
            return e_corr, "CCSD"
    
    except Exception:
        pass
    
    try:
        # Fallback to MP2
        if isinstance(scf_obj, scf.uhf.UHF):
            mp2_obj = mp.UMP2(scf_obj)
        else:
            mp2_obj = mp.MP2(scf_obj)
        
        mp2_obj.verbose = 0
        e_mp2, _ = mp2_obj.kernel()
        e_corr = e_mp2 - scf_obj.e_tot
        
        # Scale MP2 to estimate full correlation (rough approximation)
        return e_corr * 1.2, "MP2_scaled"
    
    except Exception:
        # Last resort: assume no correlation energy
        return 0.0, "none"