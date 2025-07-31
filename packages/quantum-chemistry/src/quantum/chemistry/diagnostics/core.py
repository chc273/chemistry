"""
Main orchestrator class for multireference diagnostics.

This module provides the central MultireferenceDiagnostics class that coordinates
the execution of various diagnostic methods, implements hierarchical screening,
and provides automated recommendations for method selection.
"""

from __future__ import annotations

import time
from typing import List, Optional, Union, Dict, Any

import numpy as np
from pyscf import scf

from .models.core_models import (
    DiagnosticResult,
    DiagnosticMethod,
    DiagnosticConfig,
    SystemClassification,
    MultireferenceCharacter,
    ComprehensiveDiagnosticResult,
)
from .fast_screening import (
    calculate_homo_lumo_gap,
    calculate_spin_contamination,
    calculate_natural_orbital_occupations,
    calculate_fractional_occupation_density,
    calculate_bond_order_fluctuation,
    _classify_system,
)
from .reference_methods import (
    calculate_t1_diagnostic,
    calculate_d1_diagnostic,
    calculate_correlation_recovery,
    calculate_s_diagnostic,
)


class MultireferenceDiagnostics:
    """
    Main orchestrator for multireference character assessment.
    
    This class provides a unified interface for running diagnostic calculations,
    implementing hierarchical screening strategies, and generating comprehensive
    assessments of multireference character.
    
    Usage:
        diagnostics = MultireferenceDiagnostics()
        result = diagnostics.run_full_analysis(scf_obj)
        print(result.get_summary())
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize diagnostics orchestrator.
        
        Args:
            config: Diagnostic configuration (uses defaults if None)
        """
        self.config = config or DiagnosticConfig()
        
        # Registry of available diagnostic methods
        self._fast_methods = {
            DiagnosticMethod.HOMO_LUMO_GAP: calculate_homo_lumo_gap,
            DiagnosticMethod.SPIN_CONTAMINATION: calculate_spin_contamination,
            DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS: calculate_natural_orbital_occupations,
            DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY: calculate_fractional_occupation_density,
            DiagnosticMethod.BOND_ORDER_FLUCTUATION: calculate_bond_order_fluctuation,
        }
        
        self._ref_methods = {
            DiagnosticMethod.T1_DIAGNOSTIC: calculate_t1_diagnostic,
            DiagnosticMethod.D1_DIAGNOSTIC: calculate_d1_diagnostic,
            DiagnosticMethod.CORRELATION_RECOVERY: calculate_correlation_recovery,
            DiagnosticMethod.S_DIAGNOSTIC: calculate_s_diagnostic,
        }
    
    def run_fast_screening(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        methods: Optional[List[DiagnosticMethod]] = None
    ) -> List[DiagnosticResult]:
        """
        Run fast screening diagnostics.
        
        Args:
            scf_obj: Converged SCF calculation object
            methods: Specific methods to run (uses config default if None)
            
        Returns:
            List of diagnostic results
        """
        if methods is None:
            methods = self.config.fast_screening_methods
        
        results = []
        total_time = 0.0
        
        for method in methods:
            if method in self._fast_methods:
                try:
                    start_time = time.time()
                    result = self._fast_methods[method](scf_obj, self.config)
                    calc_time = time.time() - start_time
                    total_time += calc_time
                    
                    results.append(result)
                    
                    # Check time limit
                    if total_time > self.config.max_fast_screening_time:
                        break
                        
                except Exception as e:
                    # Create failed result
                    result = DiagnosticResult(
                        method=method,
                        value=0.0,
                        multireference_character=MultireferenceCharacter.NONE,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        calculation_time=0.0,
                        system_classification=_classify_system(scf_obj),
                        converged=False,
                    )
                    results.append(result)
        
        return results
    
    def run_reference_diagnostics(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        methods: Optional[List[DiagnosticMethod]] = None
    ) -> List[DiagnosticResult]:
        """
        Run expensive reference diagnostics.
        
        Args:
            scf_obj: Converged SCF calculation object
            methods: Specific methods to run (uses config default if None)
            
        Returns:
            List of diagnostic results
        """
        if methods is None:
            methods = self.config.reference_methods
        
        results = []
        total_time = 0.0
        
        for method in methods:
            if method in self._ref_methods:
                try:
                    start_time = time.time()
                    result = self._ref_methods[method](scf_obj, self.config)
                    calc_time = time.time() - start_time
                    total_time += calc_time
                    
                    results.append(result)
                    
                    # Check time limit
                    if total_time > self.config.max_reference_time:
                        break
                        
                except Exception as e:
                    # Create failed result
                    result = DiagnosticResult(
                        method=method,
                        value=0.0,
                        multireference_character=MultireferenceCharacter.NONE,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        calculation_time=0.0,
                        system_classification=_classify_system(scf_obj),
                        converged=False,
                    )
                    results.append(result)
        
        return results
    
    def run_hierarchical_screening(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
    ) -> ComprehensiveDiagnosticResult:
        """
        Run hierarchical diagnostic screening.
        
        This method implements intelligent screening where fast methods are
        run first, and expensive reference methods are only run if needed
        based on initial results.
        
        Args:
            scf_obj: Converged SCF calculation object
            
        Returns:
            Comprehensive diagnostic result
        """
        start_time = time.time()
        
        # Step 1: Run fast screening
        fast_results = self.run_fast_screening(scf_obj)
        
        # Step 2: Analyze fast screening results
        needs_reference = self._assess_need_for_reference_methods(fast_results)
        
        # Step 3: Run reference methods if needed
        ref_results = []
        if needs_reference:
            ref_results = self.run_reference_diagnostics(scf_obj)
        
        # Step 4: Combine results and generate consensus
        all_results = fast_results + ref_results
        consensus_result = self._generate_consensus(scf_obj, all_results)
        
        total_time = time.time() - start_time
        consensus_result.total_time = total_time
        
        return consensus_result
    
    def run_full_analysis(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        include_ml_prediction: bool = None
    ) -> ComprehensiveDiagnosticResult:
        """
        Run comprehensive multireference analysis.
        
        This method runs all available diagnostic methods and provides
        a complete assessment with method recommendations.
        
        Args:
            scf_obj: Converged SCF calculation object
            include_ml_prediction: Whether to include ML predictions
            
        Returns:
            Comprehensive diagnostic result
        """
        start_time = time.time()
        
        if include_ml_prediction is None:
            include_ml_prediction = self.config.use_ml_acceleration
        
        # Run all fast screening methods
        fast_results = self.run_fast_screening(
            scf_obj, list(self._fast_methods.keys())
        )
        
        # Run all reference methods
        ref_results = self.run_reference_diagnostics(
            scf_obj, list(self._ref_methods.keys())
        )
        
        # ML predictions (placeholder for now)
        ml_results = []
        if include_ml_prediction:
            try:
                ml_results = self._run_ml_predictions(scf_obj, fast_results)
            except Exception:
                pass  # ML predictions are optional
        
        # Combine all results
        all_results = fast_results + ref_results + ml_results
        
        # Generate comprehensive analysis
        consensus_result = self._generate_consensus(scf_obj, all_results)
        
        total_time = time.time() - start_time
        consensus_result.total_time = total_time
        
        return consensus_result
    
    def _assess_need_for_reference_methods(
        self, 
        fast_results: List[DiagnosticResult]
    ) -> bool:
        """
        Determine if expensive reference methods are needed based on fast screening.
        
        Args:
            fast_results: Results from fast screening methods
            
        Returns:
            True if reference methods should be run
        """
        if not fast_results:
            return True  # No fast results, need reference methods
        
        # Check for strong indicators of multireference character
        strong_indicators = 0
        weak_indicators = 0
        
        for result in fast_results:
            if not result.converged:
                continue
                
            if result.multireference_character in [
                MultireferenceCharacter.STRONG,
                MultireferenceCharacter.VERY_STRONG
            ]:
                strong_indicators += 1
            elif result.multireference_character in [
                MultireferenceCharacter.WEAK,
                MultireferenceCharacter.MODERATE
            ]:
                weak_indicators += 1
        
        # Decision logic
        if strong_indicators >= 2:
            return True  # Clear strong MR character, confirm with reference
        elif strong_indicators >= 1 and weak_indicators >= 1:
            return True  # Mixed signals, need clarification
        elif weak_indicators >= 3:
            return True  # Multiple weak indicators, investigate further
        else:
            return False  # Likely single-reference, skip expensive methods
    
    def _run_ml_predictions(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult]
    ) -> List[DiagnosticResult]:
        """
        Run machine learning predictions for expensive diagnostics.
        
        This is a placeholder for ML model integration.
        
        Args:
            scf_obj: SCF calculation object
            fast_results: Fast screening results to use as features
            
        Returns:
            List of ML-predicted diagnostic results
        """
        # Placeholder implementation
        # In a full implementation, this would:
        # 1. Extract molecular descriptors
        # 2. Prepare feature vectors from fast_results
        # 3. Load pre-trained ML models
        # 4. Make predictions with uncertainty quantification
        # 5. Return DiagnosticResult objects with ML predictions
        
        return []  # Return empty list for now
    
    def _generate_consensus(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        all_results: List[DiagnosticResult]
    ) -> ComprehensiveDiagnosticResult:
        """
        Generate consensus assessment from all diagnostic results.
        
        Args:
            scf_obj: SCF calculation object
            all_results: All diagnostic results
            
        Returns:
            Comprehensive consensus result
        """
        # Filter converged results
        valid_results = [r for r in all_results if r.converged]
        
        if not valid_results:
            # No valid results
            return ComprehensiveDiagnosticResult(
                individual_results=all_results,
                consensus_character=MultireferenceCharacter.NONE,
                consensus_confidence=0.0,
                system_classification=_classify_system(scf_obj),
                molecular_formula=self._get_molecular_formula(scf_obj),
                total_time=0.0,
                methods_run=[r.method for r in all_results],
                all_converged=False,
                method_agreement=0.0,
                config=self.config,
            )
        
        # Weighted consensus based on method reliability and confidence
        consensus_score = self._calculate_consensus_score(valid_results)
        consensus_character = self._score_to_character(consensus_score)
        
        # Calculate method agreement
        method_agreement = self._calculate_method_agreement(valid_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            scf_obj, consensus_character, valid_results
        )
        
        # System classification
        system_class = _classify_system(scf_obj)
        
        return ComprehensiveDiagnosticResult(
            individual_results=all_results,
            consensus_character=consensus_character,
            consensus_confidence=min(0.95, method_agreement * 0.9 + 0.1),
            system_classification=system_class,
            molecular_formula=self._get_molecular_formula(scf_obj),
            total_time=sum(r.calculation_time or 0.0 for r in all_results),
            methods_run=[r.method for r in all_results if r.converged],
            methods_predicted=[r.method for r in all_results 
                             if r.method.value.startswith("ml_")],
            recommended_mr_methods=recommendations["methods"],
            recommended_active_space_size=recommendations.get("active_space"),
            all_converged=all(r.converged for r in all_results),
            method_agreement=method_agreement,
            config=self.config,
        )
    
    def _calculate_consensus_score(self, results: List[DiagnosticResult]) -> float:
        """
        Calculate weighted consensus score from diagnostic results.
        
        Args:
            results: Valid diagnostic results
            
        Returns:
            Consensus score (0-4 scale matching MultireferenceCharacter)
        """
        # Character to numeric mapping
        char_scores = {
            MultireferenceCharacter.NONE: 0.0,
            MultireferenceCharacter.WEAK: 1.0,
            MultireferenceCharacter.MODERATE: 2.0,
            MultireferenceCharacter.STRONG: 3.0,
            MultireferenceCharacter.VERY_STRONG: 4.0,
        }
        
        # Method reliability weights (reference methods get higher weight)
        method_weights = {
            DiagnosticMethod.T1_DIAGNOSTIC: 1.0,
            DiagnosticMethod.D1_DIAGNOSTIC: 1.0,
            DiagnosticMethod.CORRELATION_RECOVERY: 0.9,
            DiagnosticMethod.S_DIAGNOSTIC: 0.8,
            DiagnosticMethod.HOMO_LUMO_GAP: 0.7,
            DiagnosticMethod.SPIN_CONTAMINATION: 0.7,
            DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS: 0.6,
            DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY: 0.5,
            DiagnosticMethod.BOND_ORDER_FLUCTUATION: 0.4,
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in results:
            char_score = char_scores[result.multireference_character]
            method_weight = method_weights.get(result.method, 0.3)
            confidence_weight = result.confidence
            
            total_weight += method_weight * confidence_weight
            weighted_score += char_score * method_weight * confidence_weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.0
    
    def _score_to_character(self, score: float) -> MultireferenceCharacter:
        """
        Convert numeric consensus score to MultireferenceCharacter.
        
        Args:
            score: Consensus score (0-4 scale)
            
        Returns:
            Multireference character classification
        """
        if score < 0.5:
            return MultireferenceCharacter.NONE
        elif score < 1.5:
            return MultireferenceCharacter.WEAK
        elif score < 2.5:
            return MultireferenceCharacter.MODERATE
        elif score < 3.5:
            return MultireferenceCharacter.STRONG
        else:
            return MultireferenceCharacter.VERY_STRONG
    
    def _calculate_method_agreement(self, results: List[DiagnosticResult]) -> float:
        """
        Calculate agreement between diagnostic methods.
        
        Args:
            results: Valid diagnostic results
            
        Returns:
            Agreement score (0-1)
        """
        if len(results) < 2:
            return 1.0
        
        # Convert characters to numeric scores
        char_scores = {
            MultireferenceCharacter.NONE: 0,
            MultireferenceCharacter.WEAK: 1,
            MultireferenceCharacter.MODERATE: 2,
            MultireferenceCharacter.STRONG: 3,
            MultireferenceCharacter.VERY_STRONG: 4,
        }
        
        scores = [char_scores[r.multireference_character] for r in results]
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                # Agreement based on score difference
                diff = abs(scores[i] - scores[j])
                agreement = max(0.0, 1.0 - diff / 4.0)
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 1.0
    
    def _generate_recommendations(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        consensus_character: MultireferenceCharacter,
        results: List[DiagnosticResult]
    ) -> Dict[str, Any]:
        """
        Generate method and active space recommendations.
        
        Args:
            scf_obj: SCF calculation object
            consensus_character: Consensus multireference character
            results: Valid diagnostic results
            
        Returns:
            Dictionary with recommendations
        """
        system_class = _classify_system(scf_obj)
        
        # Method recommendations based on consensus
        if consensus_character == MultireferenceCharacter.NONE:
            methods = ["MP2", "CCSD(T)"]
        elif consensus_character == MultireferenceCharacter.WEAK:
            methods = ["CCSD(T)", "CASSCF"]
        elif consensus_character == MultireferenceCharacter.MODERATE:
            methods = ["CASSCF", "NEVPT2"]
        elif consensus_character == MultireferenceCharacter.STRONG:
            methods = ["NEVPT2", "CASPT2"]
        else:  # VERY_STRONG
            methods = ["NEVPT2", "Selected CI", "DMRG"]
        
        # System-specific adjustments
        if system_class == SystemClassification.TRANSITION_METAL:
            if "NEVPT2" not in methods:
                methods.insert(0, "NEVPT2")
        elif system_class == SystemClassification.BIRADICAL:
            if "CASSCF" not in methods:
                methods.insert(0, "CASSCF")
        
        # Active space size recommendation
        active_space = self._recommend_active_space_size(
            scf_obj, consensus_character, results
        )
        
        return {
            "methods": methods,
            "active_space": active_space,
        }
    
    def _recommend_active_space_size(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        consensus_character: MultireferenceCharacter,
        results: List[DiagnosticResult]
    ) -> Optional[tuple]:
        """
        Recommend active space size based on diagnostic results.
        
        Args:
            scf_obj: SCF calculation object
            consensus_character: Consensus multireference character
            results: Valid diagnostic results
            
        Returns:
            Tuple of (n_electrons, n_orbitals) or None
        """
        # Base recommendation on multireference strength
        base_sizes = {
            MultireferenceCharacter.NONE: (2, 2),
            MultireferenceCharacter.WEAK: (4, 4),
            MultireferenceCharacter.MODERATE: (6, 6),
            MultireferenceCharacter.STRONG: (8, 8),
            MultireferenceCharacter.VERY_STRONG: (10, 10),
        }
        
        base_size = base_sizes[consensus_character]
        
        # Adjust based on system type
        system_class = _classify_system(scf_obj)
        
        if system_class == SystemClassification.TRANSITION_METAL:
            # Transition metals typically need larger active spaces
            n_e, n_o = base_size
            return (n_e + 2, n_o + 3)  # Add d orbitals
        elif system_class == SystemClassification.BIRADICAL:
            # Biradicals need at least 2 unpaired electrons
            n_e, n_o = base_size
            return (max(n_e, 2), max(n_o, 2))
        
        return base_size
    
    def _get_molecular_formula(
        self, 
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
    ) -> str:
        """
        Get molecular formula from SCF object.
        
        Args:
            scf_obj: SCF calculation object
            
        Returns:
            Molecular formula string
        """
        mol = scf_obj.mol
        
        # Count atoms by element
        element_counts = {}
        for i in range(mol.natm):
            element = mol.atom_symbol(i)
            element_counts[element] = element_counts.get(element, 0) + 1
        
        # Format as molecular formula
        formula_parts = []
        for element in sorted(element_counts.keys()):
            count = element_counts[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        
        return "".join(formula_parts)