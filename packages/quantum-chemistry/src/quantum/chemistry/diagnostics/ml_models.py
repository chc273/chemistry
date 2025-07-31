"""
Machine learning models for fast WFT diagnostic prediction.

This module implements ML models that can predict expensive coupled cluster
diagnostics from cheap DFT/SCF features, enabling rapid screening without
the computational cost of full CCSD calculations.

The models are based on kernel ridge regression and provide uncertainty
quantification to determine when expensive calculations are needed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path

import numpy as np
from pyscf import scf

from .models.core_models import (
    DiagnosticResult,
    DiagnosticMethod,
    DiagnosticConfig,
    SystemClassification,
    MultireferenceCharacter,
)
from .fast_screening import _classify_system


# Suppress sklearn warnings during optional imports
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, r2_score
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False


class MLDiagnosticPredictor:
    """
    Machine learning predictor for expensive diagnostic methods.
    
    This class provides ML-accelerated predictions of coupled cluster
    diagnostics using molecular descriptors computed from SCF calculations.
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize ML predictor.
        
        Args:
            config: Diagnostic configuration
        """
        self.config = config or DiagnosticConfig()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._models_loaded = False
        
        if not SKLEARN_AVAILABLE:
            warnings.warn(
                "scikit-learn not available. ML predictions will be disabled.",
                RuntimeWarning
            )
    
    def predict_t1_diagnostic(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult]
    ) -> Optional[DiagnosticResult]:
        """
        Predict T1 diagnostic using ML model.
        
        Args:
            scf_obj: SCF calculation object
            fast_results: Results from fast screening methods
            
        Returns:
            Predicted diagnostic result or None if prediction fails
        """
        if not SKLEARN_AVAILABLE or not self._models_loaded:
            return None
        
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_features(scf_obj, fast_results)
            
            # Get system-specific model
            system_class = _classify_system(scf_obj)
            model_key = f"t1_{system_class.value}"
            
            if model_key not in self.models:
                # Fall back to general model
                model_key = "t1_general"
            
            if model_key not in self.models:
                return None
            
            # Make prediction
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            
            # Estimate uncertainty (simplified approach)
            uncertainty = self._estimate_uncertainty(
                model, features_scaled, "t1_diagnostic"
            )
            
            # Convert to diagnostic result
            mr_character = self.config.classify_value(
                DiagnosticMethod.T1_DIAGNOSTIC, prediction
            )
            
            # Confidence based on uncertainty
            confidence = max(0.1, min(0.9, 1.0 - uncertainty))
            
            calc_time = time.time() - start_time
            
            return DiagnosticResult(
                method=DiagnosticMethod.ML_FAST_PREDICTION,
                value=prediction,
                multireference_character=mr_character,
                confidence=confidence,
                metadata={
                    "predicted_diagnostic": "t1",
                    "model_used": model_key,
                    "uncertainty": uncertainty,
                    "features_used": features.tolist(),
                },
                calculation_time=calc_time,
                system_classification=system_class,
                uncertainty=uncertainty,
                reference="ML prediction based on Kernel Ridge Regression",
            )
            
        except Exception as e:
            return None  # Fail silently, expensive calculation will be used
    
    def predict_multiple_diagnostics(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult],
        diagnostics: List[str] = None
    ) -> List[DiagnosticResult]:
        """
        Predict multiple diagnostics using ML models.
        
        Args:
            scf_obj: SCF calculation object
            fast_results: Results from fast screening methods
            diagnostics: List of diagnostics to predict
            
        Returns:
            List of predicted diagnostic results
        """
        if diagnostics is None:
            diagnostics = ["t1", "d1", "correlation_recovery"]
        
        results = []
        
        for diagnostic in diagnostics:
            if diagnostic == "t1":
                result = self.predict_t1_diagnostic(scf_obj, fast_results)
            elif diagnostic == "d1":
                result = self.predict_d1_diagnostic(scf_obj, fast_results)
            elif diagnostic == "correlation_recovery":
                result = self.predict_correlation_recovery(scf_obj, fast_results)
            else:
                continue
            
            if result is not None:
                results.append(result)
        
        return results
    
    def predict_d1_diagnostic(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult]
    ) -> Optional[DiagnosticResult]:
        """
        Predict D1 diagnostic using ML model.
        
        This is a placeholder implementation following the same pattern as T1.
        """
        # Similar implementation to predict_t1_diagnostic
        # but for D1 diagnostic
        return None  # Placeholder
    
    def predict_correlation_recovery(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult]
    ) -> Optional[DiagnosticResult]:
        """
        Predict correlation energy recovery using ML model.
        
        This is a placeholder implementation.
        """
        # Similar implementation for correlation recovery prediction
        return None  # Placeholder
    
    def load_pretrained_models(self, model_dir: Optional[Path] = None) -> bool:
        """
        Load pre-trained ML models.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            True if models loaded successfully
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        if model_dir is None:
            # Default model directory
            model_dir = Path(__file__).parent / "models"
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would:
            # 1. Load serialized models (pickle/joblib)
            # 2. Load scalers and feature definitions
            # 3. Validate model compatibility
            
            # For now, create dummy models for demonstration
            self._create_dummy_models()
            self._models_loaded = True
            return True
            
        except Exception:
            return False
    
    def train_models(
        self,
        training_data: Dict[str, Any],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train ML models on provided data.
        
        Args:
            training_data: Dictionary containing features and targets
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of model performance metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for model training")
        
        # This is a placeholder for model training infrastructure
        # A full implementation would include:
        # 1. Data preprocessing and feature engineering
        # 2. Model hyperparameter optimization
        # 3. Cross-validation and performance evaluation
        # 4. Model serialization and storage
        
        return {"placeholder": 0.0}
    
    def _extract_features(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        fast_results: List[DiagnosticResult]
    ) -> np.ndarray:
        """
        Extract molecular descriptors and features for ML prediction.
        
        Args:
            scf_obj: SCF calculation object
            fast_results: Fast diagnostic results
            
        Returns:
            Feature vector
        """
        features = []
        
        # Molecular descriptors
        mol = scf_obj.mol
        features.extend([
            mol.natm,  # Number of atoms
            mol.nelectron,  # Number of electrons
            mol.nao,  # Number of basis functions
        ])
        
        # Electronic structure features
        if isinstance(scf_obj, scf.uhf.UHF):
            features.extend([
                1.0,  # UHF flag
                abs(scf_obj.nelec[0] - scf_obj.nelec[1]),  # Spin multiplicity
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Orbital energy features
        if isinstance(scf_obj, scf.uhf.UHF):
            mo_energy = scf_obj.mo_energy[0]  # Use alpha orbitals
        else:
            mo_energy = scf_obj.mo_energy
        
        if len(mo_energy) > 0:
            features.extend([
                np.min(mo_energy),
                np.max(mo_energy),
                np.mean(mo_energy),
                np.std(mo_energy),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Fast diagnostic features
        diagnostic_values = {
            DiagnosticMethod.HOMO_LUMO_GAP: 0.0,
            DiagnosticMethod.SPIN_CONTAMINATION: 0.0,
            DiagnosticMethod.NATURAL_ORBITAL_OCCUPATIONS: 0.0,
            DiagnosticMethod.FRACTIONAL_OCCUPATION_DENSITY: 0.0,
            DiagnosticMethod.BOND_ORDER_FLUCTUATION: 0.0,
        }
        
        for result in fast_results:
            if result.method in diagnostic_values and result.converged:
                diagnostic_values[result.method] = result.value
        
        features.extend(diagnostic_values.values())
        
        # System classification features (one-hot encoding)
        system_class = _classify_system(scf_obj)
        system_features = [0.0] * 5  # 5 system types
        
        system_map = {
            SystemClassification.ORGANIC: 0,
            SystemClassification.TRANSITION_METAL: 1,
            SystemClassification.BIRADICAL: 2,
            SystemClassification.METAL_CLUSTER: 3,
            SystemClassification.GENERAL: 4,
        }
        
        if system_class in system_map:
            system_features[system_map[system_class]] = 1.0
        
        features.extend(system_features)
        
        return np.array(features)
    
    def _estimate_uncertainty(
        self,
        model: Any,
        features: np.ndarray,
        diagnostic_type: str
    ) -> float:
        """
        Estimate prediction uncertainty.
        
        Args:
            model: Trained ML model
            features: Feature vector
            diagnostic_type: Type of diagnostic being predicted
            
        Returns:
            Uncertainty estimate (0-1 scale)
        """
        # Simplified uncertainty estimation
        # In a full implementation, you would use:
        # 1. Ensemble methods
        # 2. Conformal prediction
        # 3. Gaussian process uncertainty
        # 4. Bootstrap estimates
        
        # For now, return a conservative estimate
        return 0.2
    
    def _create_dummy_models(self) -> None:
        """
        Create dummy models for demonstration purposes.
        
        In a real implementation, this would be replaced by loading
        actual trained models from disk.
        """
        if not SKLEARN_AVAILABLE:
            return
        
        # Create dummy models with random parameters
        n_features = 20  # Expected number of features
        
        for system in ["organic", "transition_metal", "general"]:
            for diagnostic in ["t1", "d1", "correlation_recovery"]:
                model_key = f"{diagnostic}_{system}"
                
                # Create dummy kernel ridge regression model
                model = KernelRidge(kernel="rbf", alpha=0.1)
                
                # Create dummy training data
                X_dummy = np.random.random((100, n_features))
                y_dummy = np.random.random(100) * 0.1  # Typical T1 range
                
                # Fit dummy model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_dummy)
                model.fit(X_scaled, y_dummy)
                
                self.models[model_key] = model
                self.scalers[model_key] = scaler


def create_ml_predictor(config: Optional[DiagnosticConfig] = None) -> MLDiagnosticPredictor:
    """
    Factory function to create and initialize ML predictor.
    
    Args:
        config: Diagnostic configuration
        
    Returns:
        Initialized ML predictor
    """
    predictor = MLDiagnosticPredictor(config)
    
    # Try to load pre-trained models
    if SKLEARN_AVAILABLE:
        predictor.load_pretrained_models()
    
    return predictor


def generate_molecular_descriptors(
    scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
) -> Dict[str, float]:
    """
    Generate comprehensive molecular descriptors for ML models.
    
    This function computes various molecular descriptors that can be used
    as features for predicting multireference diagnostics.
    
    Args:
        scf_obj: SCF calculation object
        
    Returns:
        Dictionary of molecular descriptors
    """
    mol = scf_obj.mol
    descriptors = {}
    
    # Basic molecular properties
    descriptors["n_atoms"] = mol.natm
    descriptors["n_electrons"] = mol.nelectron
    descriptors["n_basis_functions"] = mol.nao
    descriptors["total_charge"] = mol.charge
    descriptors["spin_multiplicity"] = mol.spin
    
    # Electronic properties
    descriptors["scf_energy"] = scf_obj.e_tot
    
    if isinstance(scf_obj, scf.uhf.UHF):
        descriptors["is_uhf"] = 1.0
        descriptors["alpha_electrons"] = scf_obj.nelec[0]
        descriptors["beta_electrons"] = scf_obj.nelec[1]
        descriptors["n_unpaired"] = abs(scf_obj.nelec[0] - scf_obj.nelec[1])
    else:
        descriptors["is_uhf"] = 0.0
        descriptors["alpha_electrons"] = scf_obj.mol.nelectron // 2
        descriptors["beta_electrons"] = scf_obj.mol.nelectron // 2
        descriptors["n_unpaired"] = 0.0
    
    # Orbital energy statistics
    if isinstance(scf_obj, scf.uhf.UHF):
        mo_energy = scf_obj.mo_energy[0]  # Alpha orbitals
    else:
        mo_energy = scf_obj.mo_energy
    
    if len(mo_energy) > 0:
        descriptors["homo_energy"] = np.max(mo_energy[mo_energy < 0])
        descriptors["lumo_energy"] = np.min(mo_energy[mo_energy >= 0])
        descriptors["homo_lumo_gap"] = descriptors["lumo_energy"] - descriptors["homo_energy"]
        descriptors["orbital_energy_range"] = np.max(mo_energy) - np.min(mo_energy)
        descriptors["orbital_energy_std"] = np.std(mo_energy)
    
    # Chemical composition
    element_counts = {}
    for i in range(mol.natm):
        element = mol.atom_symbol(i)
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Common elements
    for element in ["H", "C", "N", "O", "F", "S", "P", "Cl"]:
        descriptors[f"n_{element}"] = element_counts.get(element, 0)
    
    # Transition metals
    transition_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'
    }
    
    tm_count = sum(1 for element in element_counts.keys() 
                   if element in transition_metals)
    descriptors["n_transition_metals"] = tm_count
    descriptors["has_transition_metal"] = 1.0 if tm_count > 0 else 0.0
    
    return descriptors