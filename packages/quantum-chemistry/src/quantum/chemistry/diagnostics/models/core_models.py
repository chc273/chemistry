"""
Core data models for multireference diagnostics.

This module defines the fundamental data structures used throughout the
diagnostics system, including diagnostic results, configuration models,
and system classifications.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator


class MultireferenceCharacter(str, Enum):
    """
    Classification of multireference character strength.
    
    Based on literature thresholds for various diagnostic methods:
    - none: Single-reference methods are adequate
    - weak: Possible multireference character, single-reference may suffice
    - moderate: Clear multireference character, MR methods recommended
    - strong: Strong multireference character, MR methods required
    - very_strong: Very strong multireference, careful method selection needed
    """
    
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class DiagnosticMethod(str, Enum):
    """Available diagnostic method types."""
    
    # Fast screening methods (cost: <1 minute)
    HOMO_LUMO_GAP = "homo_lumo_gap"
    SPIN_CONTAMINATION = "spin_contamination"
    NATURAL_ORBITAL_OCCUPATIONS = "natural_orbital_occupations"
    FRACTIONAL_OCCUPATION_DENSITY = "fractional_occupation_density"
    BOND_ORDER_FLUCTUATION = "bond_order_fluctuation"
    
    # Reference methods (cost: 10-30 minutes)
    T1_DIAGNOSTIC = "t1_diagnostic"
    D1_DIAGNOSTIC = "d1_diagnostic"
    CORRELATION_RECOVERY = "correlation_recovery"
    S_DIAGNOSTIC = "s_diagnostic"
    
    # Machine learning predictions
    ML_FAST_PREDICTION = "ml_fast_prediction"
    ML_ORGANIC = "ml_organic"
    ML_TRANSITION_METAL = "ml_transition_metal"


class SystemClassification(str, Enum):
    """Chemical system classification for diagnostic interpretation."""
    
    ORGANIC = "organic"
    TRANSITION_METAL = "transition_metal"
    BIRADICAL = "biradical"
    METAL_CLUSTER = "metal_cluster"
    GENERAL = "general"


class DiagnosticResult(BaseModel):
    """
    Result container for individual diagnostic calculations.
    
    This class stores results from individual diagnostic methods along with
    confidence scores, thresholds, and interpretation information.
    """
    
    method: DiagnosticMethod = Field(..., description="Diagnostic method used")
    value: float = Field(..., description="Diagnostic value")
    multireference_character: MultireferenceCharacter = Field(
        ..., description="Assessed multireference character"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in assessment (0-1)"
    )
    
    # Threshold information
    threshold_weak: Optional[float] = Field(
        None, description="Threshold for weak multireference character"
    )
    threshold_moderate: Optional[float] = Field(
        None, description="Threshold for moderate multireference character"
    )
    threshold_strong: Optional[float] = Field(
        None, description="Threshold for strong multireference character"
    )
    threshold_very_strong: Optional[float] = Field(
        None, description="Threshold for very strong multireference character"
    )
    
    # Additional method-specific data
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Method-specific additional data"
    )
    
    # Computational details
    calculation_time: Optional[float] = Field(
        None, description="Wall time for calculation (seconds)"
    )
    system_classification: Optional[SystemClassification] = Field(
        None, description="System type used for threshold selection"
    )
    
    # Literature reference for thresholds
    reference: Optional[str] = Field(
        None, description="Literature reference for diagnostic method"
    )
    
    # Quality indicators
    converged: bool = Field(True, description="Whether calculation converged")
    uncertainty: Optional[float] = Field(
        None, description="Statistical uncertainty estimate"
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Calculation timestamp"
    )
    
    @validator('confidence')
    def confidence_bounds(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    def get_interpretation(self) -> str:
        """
        Get human-readable interpretation of diagnostic result.
        
        Returns:
            String interpretation of the diagnostic value
        """
        if self.multireference_character == MultireferenceCharacter.NONE:
            return (
                f"No significant multireference character detected "
                f"({self.method.value}={self.value:.3f}). "
                f"Single-reference methods are adequate."
            )
        elif self.multireference_character == MultireferenceCharacter.WEAK:
            return (
                f"Weak multireference character detected "
                f"({self.method.value}={self.value:.3f}). "
                f"Single-reference methods may suffice, but consider validation."
            )
        elif self.multireference_character == MultireferenceCharacter.MODERATE:
            return (
                f"Moderate multireference character detected "
                f"({self.method.value}={self.value:.3f}). "
                f"Multireference methods recommended."
            )
        elif self.multireference_character == MultireferenceCharacter.STRONG:
            return (
                f"Strong multireference character detected "
                f"({self.method.value}={self.value:.3f}). "
                f"Multireference methods required."
            )
        else:  # VERY_STRONG
            return (
                f"Very strong multireference character detected "
                f"({self.method.value}={self.value:.3f}). "
                f"Careful multireference method selection needed."
            )
    
    class Config:
        arbitrary_types_allowed = True


class DiagnosticConfig(BaseModel):
    """
    Configuration for multireference diagnostic calculations.
    
    This class defines thresholds, method selection criteria, and
    computational parameters for the diagnostic system.
    """
    
    # System-specific threshold overrides
    system_classification: Optional[SystemClassification] = Field(
        None, description="Force specific system classification"
    )
    
    # Fast screening thresholds
    homo_lumo_gap_thresholds: Dict[str, float] = Field(
        default={
            "weak": 2.0,      # eV - gaps above this are single-reference
            "moderate": 1.0,  # eV 
            "strong": 0.5,    # eV
            "very_strong": 0.4  # eV - small gaps indicate strong MR character
        },
        description="HOMO-LUMO gap thresholds (eV)"
    )
    
    spin_contamination_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.05,
            "moderate": 0.1,
            "strong": 0.2,
            "very_strong": 0.4
        },
        description="Spin contamination <S²> - S(S+1) thresholds"
    )
    
    noon_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.1,      # Deviation from 0/2
            "moderate": 0.2,
            "strong": 0.4,
            "very_strong": 0.6
        },
        description="Natural orbital occupation number deviation thresholds"
    )
    
    fod_thresholds: Dict[str, float] = Field(
        default={
            "weak": 1.0,      # Number of FOD electrons
            "moderate": 2.0,
            "strong": 4.0,
            "very_strong": 6.0
        },
        description="Fractional occupation density thresholds (electrons)"
    )
    
    bond_order_fluctuation_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.1,
            "moderate": 0.2,
            "strong": 0.4,
            "very_strong": 0.6
        },
        description="Bond order fluctuation thresholds"
    )
    
    # Reference method thresholds
    t1_diagnostic_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.02,
            "moderate": 0.04,
            "strong": 0.06,
            "very_strong": 0.08
        },
        description="T1 diagnostic thresholds"
    )
    
    d1_diagnostic_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.05,
            "moderate": 0.10,
            "strong": 0.15,
            "very_strong": 0.20
        },
        description="D1 diagnostic thresholds"
    )
    
    correlation_recovery_thresholds: Dict[str, float] = Field(
        default={
            "weak": 85.0,     # % recovery (inverted logic)
            "moderate": 80.0,
            "strong": 75.0,
            "very_strong": 70.0
        },
        description="Correlation energy recovery thresholds (%)"
    )
    
    s_diagnostic_thresholds: Dict[str, float] = Field(
        default={
            "weak": 0.5,
            "moderate": 1.0,
            "strong": 1.5,
            "very_strong": 2.0
        },
        description="S-diagnostic thresholds"
    )
    
    # Method selection preferences
    fast_screening_methods: List[DiagnosticMethod] = Field(
        default=[
            DiagnosticMethod.HOMO_LUMO_GAP,
            DiagnosticMethod.SPIN_CONTAMINATION,
            DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS,
        ],
        description="Default fast screening methods to run"
    )
    
    reference_methods: List[DiagnosticMethod] = Field(
        default=[
            DiagnosticMethod.T1_DIAGNOSTIC,
            DiagnosticMethod.D1_DIAGNOSTIC,
        ],
        description="Reference methods to run for accurate assessment"
    )
    
    # Computational parameters
    max_fast_screening_time: float = Field(
        60.0, description="Maximum wall time for fast screening (seconds)"
    )
    max_reference_time: float = Field(
        1800.0, description="Maximum wall time for reference methods (seconds)"
    )
    
    # Machine learning parameters
    use_ml_acceleration: bool = Field(
        True, description="Use ML models to predict expensive diagnostics"
    )
    ml_confidence_threshold: float = Field(
        0.8, description="Minimum ML confidence to skip expensive calculations"
    )
    
    # Integration parameters
    auto_active_space_selection: bool = Field(
        True, description="Automatically recommend active space size"
    )
    auto_method_selection: bool = Field(
        True, description="Automatically recommend MR method"
    )
    
    def get_thresholds(self, method: DiagnosticMethod) -> Dict[str, float]:
        """
        Get thresholds for a specific diagnostic method.
        
        Args:
            method: Diagnostic method
            
        Returns:
            Dictionary of thresholds for different MR character levels
        """
        threshold_map = {
            DiagnosticMethod.HOMO_LUMO_GAP: self.homo_lumo_gap_thresholds,
            DiagnosticMethod.SPIN_CONTAMINATION: self.spin_contamination_thresholds,
            DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS: self.noon_thresholds,
            DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY: self.fod_thresholds,
            DiagnosticMethod.BOND_ORDER_FLUCTUATION: self.bond_order_fluctuation_thresholds,
            DiagnosticMethod.T1_DIAGNOSTIC: self.t1_diagnostic_thresholds,
            DiagnosticMethod.D1_DIAGNOSTIC: self.d1_diagnostic_thresholds,
            DiagnosticMethod.CORRELATION_RECOVERY: self.correlation_recovery_thresholds,
            DiagnosticMethod.S_DIAGNOSTIC: self.s_diagnostic_thresholds,
        }
        
        return threshold_map.get(method, {})
    
    def classify_value(
        self, 
        method: DiagnosticMethod, 
        value: float
    ) -> MultireferenceCharacter:
        """
        Classify diagnostic value into multireference character category.
        
        Args:
            method: Diagnostic method used
            value: Diagnostic value
            
        Returns:
            Multireference character classification
        """
        thresholds = self.get_thresholds(method)
        
        if not thresholds:
            return MultireferenceCharacter.NONE
        
        # Handle inverted logic for methods where higher values indicate less MR character
        if method in [DiagnosticMethod.CORRELATION_RECOVERY, DiagnosticMethod.HOMO_LUMO_GAP]:
            if value >= thresholds["weak"]:
                return MultireferenceCharacter.NONE
            elif value >= thresholds["moderate"]:
                return MultireferenceCharacter.WEAK
            elif value >= thresholds["strong"]:
                return MultireferenceCharacter.MODERATE
            elif value >= thresholds["very_strong"]:
                return MultireferenceCharacter.STRONG
            else:
                return MultireferenceCharacter.VERY_STRONG
        
        # Standard logic for other methods (higher values = more MR character)
        if value < thresholds["weak"]:
            return MultireferenceCharacter.NONE
        elif value < thresholds["moderate"]:
            return MultireferenceCharacter.WEAK
        elif value < thresholds["strong"]:
            return MultireferenceCharacter.MODERATE
        elif value < thresholds["very_strong"]:
            return MultireferenceCharacter.STRONG
        else:
            return MultireferenceCharacter.VERY_STRONG
    
    class Config:
        arbitrary_types_allowed = True


class ComprehensiveDiagnosticResult(BaseModel):
    """
    Comprehensive diagnostic result containing all method results and consensus.
    
    This class aggregates results from multiple diagnostic methods and provides
    a consensus assessment of multireference character.
    """
    
    # Individual diagnostic results
    individual_results: List[DiagnosticResult] = Field(
        ..., description="Results from individual diagnostic methods"
    )
    
    # Consensus assessment
    consensus_character: MultireferenceCharacter = Field(
        ..., description="Consensus multireference character assessment"
    )
    consensus_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in consensus assessment"
    )
    
    # System information
    system_classification: SystemClassification = Field(
        ..., description="Classified system type"
    )
    molecular_formula: Optional[str] = Field(
        None, description="Molecular formula"
    )
    
    # Computational summary
    total_time: float = Field(..., description="Total diagnostic time (seconds)")
    methods_run: List[DiagnosticMethod] = Field(
        ..., description="Diagnostic methods actually executed"
    )
    methods_predicted: List[DiagnosticMethod] = Field(
        default_factory=list, description="Methods predicted by ML models"
    )
    
    # Recommendations
    recommended_mr_methods: List[str] = Field(
        default_factory=list, description="Recommended multireference methods"
    )
    recommended_active_space_size: Optional[tuple] = Field(
        None, description="Recommended (n_electrons, n_orbitals)"
    )
    
    # Quality indicators
    all_converged: bool = Field(
        True, description="Whether all calculations converged"
    )
    method_agreement: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement between methods (0-1)"
    )
    
    # Configuration used
    config: DiagnosticConfig = Field(..., description="Configuration used")
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of diagnostic analysis.
        
        Returns:
            Comprehensive summary string
        """
        summary_lines = [
            f"Multireference Character Assessment",
            f"=" * 40,
            f"System: {self.molecular_formula or 'Unknown'} ({self.system_classification.value})",
            f"Consensus: {self.consensus_character.value.upper()} "
            f"(confidence: {self.consensus_confidence:.2f})",
            f"Method agreement: {self.method_agreement:.2f}",
            f"Total analysis time: {self.total_time:.1f}s",
            f"",
            f"Individual Results:",
            f"-" * 20,
        ]
        
        for result in self.individual_results:
            summary_lines.append(
                f"{result.method.value}: {result.value:.3f} "
                f"({result.multireference_character.value}, "
                f"conf: {result.confidence:.2f})"
            )
        
        if self.recommended_mr_methods:
            summary_lines.extend([
                f"",
                f"Recommendations:",
                f"-" * 15,
                f"Methods: {', '.join(self.recommended_mr_methods)}",
            ])
            
            if self.recommended_active_space_size:
                n_e, n_o = self.recommended_active_space_size
                summary_lines.append(f"Active space: ({n_e}e, {n_o}o)")
        
        return "\n".join(summary_lines)
    
    class Config:
        arbitrary_types_allowed = True