"""
Tests for multireference quantum chemistry methods.

This module contains unit tests for the multireference methods
implementation, including CASSCF, NEVPT2, and workflow orchestration.
"""

import pytest
import numpy as np

from quantum.chemistry.active_space import find_active_space_avas
from quantum.chemistry.multireference import (
    CASSCFMethod,
    NEVPT2Method,
    MultireferenceWorkflow,
    MethodSelector
)


class TestCASSCFMethod:
    """Test CASSCF method implementation."""
    
    def test_casscf_h2o(self, h2o_scf):
        """Test CASSCF calculation on water molecule."""
        # Find active space using AVAS
        active_space = find_active_space_avas(h2o_scf, threshold=0.2)
        
        # Initialize CASSCF method
        casscf_method = CASSCFMethod()
        
        # Perform calculation
        result = casscf_method.calculate(h2o_scf, active_space)
        
        # Basic validation
        assert result.method == "CASSCF"
        assert result.energy <= h2o_scf.e_tot  # CASSCF energy should be at most equal to SCF
        assert result.n_active_electrons == active_space.n_active_electrons
        assert result.n_active_orbitals == active_space.n_active_orbitals
        assert result.correlation_energy is not None
        assert result.natural_orbitals is not None
        assert result.occupation_numbers is not None
    
    def test_casscf_cost_estimation(self):
        """Test computational cost estimation."""
        casscf_method = CASSCFMethod()
        
        cost = casscf_method.estimate_cost(
            n_electrons=6,
            n_orbitals=6,
            basis_size=50
        )
        
        assert 'memory_mb' in cost
        assert 'time_seconds' in cost
        assert 'disk_mb' in cost
        assert all(v >= 0 for v in cost.values())
    
    def test_casscf_parameter_recommendations(self):
        """Test parameter recommendation system."""
        casscf_method = CASSCFMethod()
        
        # Test organic system parameters
        params = casscf_method.get_recommended_parameters(
            system_type="organic",
            active_space_size=(6, 6)
        )
        
        assert 'max_cycle' in params
        assert 'conv_tol' in params
        assert params['max_cycle'] > 0
        
        # Test transition metal parameters
        tm_params = casscf_method.get_recommended_parameters(
            system_type="transition_metal", 
            active_space_size=(10, 10)
        )
        
        # Should have different (typically more relaxed) parameters
        assert tm_params['max_cycle'] >= params['max_cycle']


class TestNEVPT2Method:
    """Test NEVPT2 method implementation."""
    
    def test_nevpt2_h2o(self, h2o_scf):
        """Test NEVPT2 calculation on water molecule."""
        # Find active space
        active_space = find_active_space_avas(h2o_scf, threshold=0.2)
        
        # Initialize NEVPT2 method
        nevpt2_method = NEVPT2Method()
        
        # Perform calculation
        result = nevpt2_method.calculate(h2o_scf, active_space)
        
        # Validation
        assert result.method == "CASSCF+NEVPT2"
        assert result.energy < h2o_scf.e_tot  # Should recover more correlation
        assert result.correlation_energy is not None
        assert 'pt2_correction' in result.active_space_info
        assert 'nevpt2_type' in result.active_space_info
    
    def test_nevpt2_cost_higher_than_casscf(self):
        """Test that NEVPT2 cost estimates are higher than CASSCF."""
        casscf_method = CASSCFMethod()
        nevpt2_method = NEVPT2Method()
        
        casscf_cost = casscf_method.estimate_cost(6, 6, 50)
        nevpt2_cost = nevpt2_method.estimate_cost(6, 6, 50)
        
        # NEVPT2 should be more expensive
        assert nevpt2_cost['memory_mb'] >= casscf_cost['memory_mb']
        assert nevpt2_cost['time_seconds'] >= casscf_cost['time_seconds']


class TestMethodSelector:
    """Test automated method selection."""
    
    def test_method_selector_initialization(self):
        """Test method selector initialization."""
        selector = MethodSelector()
        
        # Should have basic registry
        assert hasattr(selector, '_method_registry')
        assert hasattr(selector, '_system_classifiers')
    
    def test_system_classification(self, h2o_scf, fe_scf):
        """Test system type classification."""
        selector = MethodSelector()
        
        # Create dummy active space for testing
        from quantum.chemistry.active_space import ActiveSpaceResult, ActiveSpaceMethod
        dummy_active_space = ActiveSpaceResult(
            method=ActiveSpaceMethod.AVAS,
            n_active_electrons=6,
            n_active_orbitals=6,
            active_orbital_indices=list(range(6)),
            orbital_coefficients=np.eye(6),
            selection_scores=np.ones(6)
        )
        
        # Test organic classification (H2O)
        organic_type = selector._classify_system(h2o_scf, dummy_active_space)
        assert organic_type == "organic"
        
        # Test transition metal classification (Fe complex)
        if fe_scf.converged:  # Only test if Fe calculation converged
            tm_type = selector._classify_system(fe_scf, dummy_active_space)
            assert tm_type == "transition_metal"
    
    def test_method_recommendation(self, h2o_scf):
        """Test method recommendation logic."""
        selector = MethodSelector()
        
        # Register methods for testing
        from quantum.chemistry.multireference import MultireferenceMethodType
        selector.register_method(
            MultireferenceMethodType.CASSCF,
            CASSCFMethod
        )
        selector.register_method(
            MultireferenceMethodType.NEVPT2,
            NEVPT2Method
        )
        
        # Create active space
        active_space = find_active_space_avas(h2o_scf, threshold=0.2)
        
        # Test recommendation
        recommended = selector.recommend_method(
            h2o_scf,
            active_space,
            system_type="organic"
        )
        
        assert recommended is not None
        assert isinstance(recommended, MultireferenceMethodType)


class TestMultireferenceWorkflow:
    """Test high-level workflow orchestration."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = MultireferenceWorkflow()
        
        assert workflow.method_selector is not None
        assert workflow.active_space_finder is not None
    
    def test_full_workflow_h2o(self, h2o_scf):
        """Test complete workflow on water molecule."""
        workflow = MultireferenceWorkflow()
        
        # Run calculation with automatic method selection
        results = workflow.run_calculation(
            h2o_scf,
            active_space_method="avas",
            mr_method="casscf",
            target_accuracy="standard"
        )
        
        # Validate results structure
        assert 'scf_energy' in results
        assert 'active_space' in results
        assert 'selected_method' in results
        assert 'multireference_result' in results
        assert 'analysis' in results
        
        # Validate content
        assert results['scf_energy'] == h2o_scf.e_tot
        assert results['selected_method'] == "casscf"
        assert results['multireference_result'].method == "CASSCF"
        
        # Validate analysis
        analysis = results['analysis']
        assert 'correlation_energy_recovery' in analysis
        assert 'active_space_analysis' in analysis
        assert 'convergence_quality' in analysis
    
    def test_method_comparison(self, h2o_scf):
        """Test method comparison functionality."""
        workflow = MultireferenceWorkflow()
        
        # Find active space
        active_space = find_active_space_avas(h2o_scf, threshold=0.2)
        
        # Compare methods
        comparison = workflow.compare_methods(
            h2o_scf,
            active_space,
            methods=["casscf", "nevpt2"]
        )
        
        # Should have results for both methods
        assert "casscf" in comparison
        assert "nevpt2" in comparison
        
        # NEVPT2 should have lower energy
        casscf_energy = comparison["casscf"].energy
        nevpt2_energy = comparison["nevpt2"].energy
        assert nevpt2_energy <= casscf_energy
    
    def test_cost_estimation(self, h2o_scf):
        """Test cost estimation functionality."""
        workflow = MultireferenceWorkflow()
        
        cost = workflow.estimate_calculation_cost(
            h2o_scf,
            active_space_size=(6, 6),
            method="casscf"
        )
        
        assert 'memory_mb' in cost
        assert 'time_seconds' in cost
        assert cost['memory_mb'] > 0
        assert cost['time_seconds'] > 0