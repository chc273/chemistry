"""
Tests for external multireference method integrations.

This module contains unit tests for the external method interfaces
including DMRG, AF-QMC, and Selected CI methods.
"""

import pytest

# Test if external methods are available
try:
    from quantum.chemistry.multireference.external import (
        AFQMCMethod,
        DMRGMethod,
        ExternalMethodInterface,
        ExternalSoftwareError,
        SelectedCIMethod,
    )

    EXTERNAL_METHODS_AVAILABLE = True
except ImportError:
    EXTERNAL_METHODS_AVAILABLE = False
    # Create a placeholder for missing ExternalSoftwareError
    class ExternalSoftwareError(Exception):
        pass


@pytest.mark.skipif(
    not EXTERNAL_METHODS_AVAILABLE, reason="External methods not available"
)
class TestExternalMethodInterface:
    """Test external method interface base class."""

    def test_interface_initialization(self):
        """Test that external method interfaces can be created."""
        # This would test the base interface
        # For now, we'll test that the classes can be imported
        assert ExternalMethodInterface is not None
        assert ExternalSoftwareError is not None


@pytest.mark.skipif(
    not EXTERNAL_METHODS_AVAILABLE, reason="External methods not available"
)
class TestDMRGMethod:
    """Test DMRG method implementation."""

    def test_dmrg_initialization(self):
        """Test DMRG method initialization."""
        # For DMRG, we catch the import error differently since it validates during __init__
        try:
            dmrg_method = DMRGMethod(
                bond_dimension=500, max_sweeps=10, post_correction=None
            )

            assert dmrg_method.bond_dimension == 500
            assert dmrg_method.max_sweeps == 10
            assert dmrg_method.post_correction is None

        except (ImportError, ExternalSoftwareError):
            pytest.skip("block2 not available for DMRG testing")

    def test_dmrg_parameter_recommendations(self):
        """Test DMRG parameter recommendation system."""
        try:
            dmrg_method = DMRGMethod()

            # Test organic system parameters
            params = dmrg_method.get_recommended_parameters(
                system_type="organic", active_space_size=(6, 6)
            )

            assert "bond_dimension" in params
            assert "max_sweeps" in params
            assert params["bond_dimension"] > 0

            # Test transition metal parameters
            tm_params = dmrg_method.get_recommended_parameters(
                system_type="transition_metal", active_space_size=(10, 10)
            )

            # Should have higher bond dimension for TM systems
            assert tm_params["bond_dimension"] >= params["bond_dimension"]

        except (ImportError, ExternalSoftwareError):
            pytest.skip("block2 not available for DMRG testing")

    def test_dmrg_cost_estimation(self):
        """Test DMRG computational cost estimation."""
        try:
            dmrg_method = DMRGMethod(bond_dimension=1000)

            cost = dmrg_method.estimate_cost(n_electrons=6, n_orbitals=6, basis_size=50)

            assert "memory_mb" in cost
            assert "time_seconds" in cost
            assert "bond_dimension" in cost
            assert cost["bond_dimension"] == 1000
            assert all(v >= 0 for v in cost.values() if isinstance(v, (int, float)))

        except (ImportError, ExternalSoftwareError):
            pytest.skip("block2 not available for DMRG testing")


@pytest.mark.skipif(
    not EXTERNAL_METHODS_AVAILABLE, reason="External methods not available"
)
class TestAFQMCMethod:
    """Test AF-QMC method implementation."""

    def test_afqmc_initialization(self):
        """Test AF-QMC method initialization."""
        try:
            afqmc_method = AFQMCMethod(
                backend="ipie", n_walkers=50, n_steps=500, timestep=0.01
            )

            assert afqmc_method.backend == "ipie"
            assert afqmc_method.n_walkers == 50
            assert afqmc_method.n_steps == 500
            assert afqmc_method.timestep == 0.01

        except (ImportError, ExternalSoftwareError):
            pytest.skip("ipie not available for AF-QMC testing")

    def test_afqmc_parameter_recommendations(self):
        """Test AF-QMC parameter recommendation system."""
        try:
            afqmc_method = AFQMCMethod()

            # Test organic system parameters
            params = afqmc_method.get_recommended_parameters(
                system_type="organic", active_space_size=(6, 6)
            )

            assert "n_walkers" in params
            assert "n_steps" in params
            assert "timestep" in params
            assert params["n_walkers"] > 0

            # Test transition metal parameters
            tm_params = afqmc_method.get_recommended_parameters(
                system_type="transition_metal", active_space_size=(10, 10)
            )

            # Should have more walkers for TM systems
            assert tm_params["n_walkers"] >= params["n_walkers"]

        except (ImportError, ExternalSoftwareError):
            pytest.skip("ipie not available for AF-QMC testing")

    def test_afqmc_cost_estimation(self):
        """Test AF-QMC computational cost estimation."""
        try:
            afqmc_method = AFQMCMethod(n_walkers=100, n_steps=1000)

            cost = afqmc_method.estimate_cost(
                n_electrons=6, n_orbitals=6, basis_size=50
            )

            assert "memory_mb" in cost
            assert "time_seconds" in cost
            assert "n_walkers" in cost
            assert "estimated_statistical_error" in cost
            assert cost["n_walkers"] == 100
            assert all(v >= 0 for v in cost.values() if isinstance(v, (int, float)))

        except (ImportError, ExternalSoftwareError):
            pytest.skip("ipie not available for AF-QMC testing")


@pytest.mark.skipif(
    not EXTERNAL_METHODS_AVAILABLE, reason="External methods not available"
)
class TestSelectedCIMethod:
    """Test Selected CI method implementations."""

    def test_shci_initialization(self):
        """Test SHCI method initialization."""
        try:
            shci_method = SelectedCIMethod(
                method_type="shci", 
                pt2_threshold=1e-4, 
                max_determinants=100000,
                skip_validation=True  # Skip software validation for testing
            )

            assert shci_method.method_type == "shci"
            assert shci_method.pt2_threshold == 1e-4
            assert shci_method.max_determinants == 100000

        except ImportError:
            pytest.skip("SHCI software not available for testing")

    def test_cipsi_initialization(self):
        """Test CIPSI method initialization."""
        try:
            cipsi_method = SelectedCIMethod(
                method_type="cipsi", 
                pt2_threshold=1e-4, 
                max_determinants=100000,
                skip_validation=True  # Skip software validation for testing
            )

            assert cipsi_method.method_type == "cipsi"
            assert cipsi_method.pt2_threshold == 1e-4
            assert cipsi_method.max_determinants == 100000

        except ImportError:
            pytest.skip("CIPSI software not available for testing")

    def test_selected_ci_cost_estimation(self):
        """Test Selected CI computational cost estimation."""
        try:
            sci_method = SelectedCIMethod(
                method_type="shci", 
                max_determinants=100000,
                skip_validation=True  # Skip software validation for testing
            )

            cost = sci_method.estimate_cost(n_electrons=6, n_orbitals=6, basis_size=50)

            assert "memory_mb" in cost
            assert "time_seconds" in cost
            assert "estimated_n_determinants" in cost
            assert all(v >= 0 for v in cost.values() if isinstance(v, (int, float)))

        except ImportError:
            pytest.skip("Selected CI software not available for testing")


@pytest.mark.skipif(
    not EXTERNAL_METHODS_AVAILABLE, reason="External methods not available"
)
class TestExternalMethodIntegration:
    """Test integration of external methods with the main framework."""

    def test_external_method_import(self):
        """Test that external methods can be imported through main interface."""
        from quantum.chemistry.multireference import _EXTERNAL_METHODS_AVAILABLE

        if _EXTERNAL_METHODS_AVAILABLE:
            from quantum.chemistry.multireference import (
                AFQMCMethod,
                DMRGMethod,
                SelectedCIMethod,
            )

            # Test that classes are available
            assert DMRGMethod is not None
            assert AFQMCMethod is not None
            assert SelectedCIMethod is not None

    def test_method_type_extensions(self):
        """Test that new method types are properly registered."""
        from quantum.chemistry.multireference import MultireferenceMethodType

        # Check that extended method types exist
        method_types = [mt.value for mt in MultireferenceMethodType]

        expected_types = [
            "casscf",
            "nevpt2",
            "caspt2",  # Core methods
            "dmrg",
            "dmrg_nevpt2",
            "dmrg_caspt2",  # DMRG methods
            "afqmc",  # AF-QMC
            "shci",
            "cipsi",  # Selected CI
        ]

        for expected in expected_types:
            assert expected in method_types, f"Method type {expected} not found"

    def test_external_method_workflow_integration(self):
        """Test that external methods can be integrated with workflows."""
        # This would test workflow integration
        # For now, just verify the framework is set up correctly

        from quantum.chemistry.multireference.workflows import MultireferenceWorkflow

        workflow = MultireferenceWorkflow()

        # Workflow should be able to handle external methods
        assert workflow.method_selector is not None
        assert hasattr(workflow.method_selector, "_method_registry")


@pytest.mark.skipif(
    EXTERNAL_METHODS_AVAILABLE,
    reason="Test only runs when external methods are not available",
)
class TestExternalMethodsUnavailable:
    """Test behavior when external methods are not available."""

    def test_graceful_import_failure(self):
        """Test that the system gracefully handles missing external dependencies."""
        from quantum.chemistry.multireference import (
            _EXTERNAL_IMPORT_ERROR,
            _EXTERNAL_METHODS_AVAILABLE,
        )

        assert not _EXTERNAL_METHODS_AVAILABLE
        assert isinstance(_EXTERNAL_IMPORT_ERROR, str)
        assert len(_EXTERNAL_IMPORT_ERROR) > 0

    def test_core_functionality_still_works(self):
        """Test that core multireference functionality works without external methods."""
        from quantum.chemistry.multireference import (
            CASSCFMethod,
            MultireferenceWorkflow,
            NEVPT2Method,
        )

        # Core methods should still be available
        assert CASSCFMethod is not None
        assert NEVPT2Method is not None
        assert MultireferenceWorkflow is not None
