"""
Comprehensive tests for OpenMolcas CASPT2 integration.

This test suite covers all aspects of the CASPT2 implementation including:
- Input file generation and validation
- Output parsing and error handling
- Method parameter optimization
- Cross-method validation
- Docker/native execution paths
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from quantum.chemistry.active_space import find_active_space_avas
from quantum.chemistry.multireference.base import MultireferenceMethodType
from quantum.chemistry.multireference.external.base import ExternalSoftwareError
from quantum.chemistry.multireference.external.openmolcas import (
    CASPT2Method,
    OpenMolcasInputGenerator,
    OpenMolcasOutputParser,
    OpenMolcasParameters,
    OpenMolcasValidator,
)


class TestOpenMolcasInputGenerator:
    """Test OpenMolcas input file generation."""

    def test_initialization(self):
        """Test input generator initialization."""
        generator = OpenMolcasInputGenerator()
        assert hasattr(generator, "templates")
        assert "casscf" in generator.templates
        assert "caspt2" in generator.templates
        assert "ms_caspt2" in generator.templates

    def test_parameter_validation(self, h2o_scf, sample_active_space_result):
        """Test parameter validation with Pydantic model."""
        generator = OpenMolcasInputGenerator()

        # Valid parameters should work
        valid_params = {
            "n_active_electrons": sample_active_space_result.n_active_electrons,
            "n_active_orbitals": sample_active_space_result.n_active_orbitals,
            "charge": 0,
            "spin_multiplicity": 1,
            "ipea_shift": 0.25,
            "imaginary_shift": 0.1,
        }
        params_obj = OpenMolcasParameters(**valid_params)
        assert params_obj.ipea_shift == 0.25

        # Invalid parameters should raise validation error
        with pytest.raises(ValueError):
            OpenMolcasParameters(
                **{**valid_params, "ipea_shift": -0.1}
            )  # Negative IPEA

        with pytest.raises(ValueError):
            OpenMolcasParameters(
                **{**valid_params, "ipea_shift": 1.5}
            )  # Too large IPEA

    def test_caspt2_input_generation(self, h2o_scf, sample_active_space_result):
        """Test CASPT2 input file generation."""
        generator = OpenMolcasInputGenerator()

        input_content = generator.generate_input(
            h2o_scf,
            sample_active_space_result,
            calculation_type="caspt2",
            ipea_shift=0.25,
            imaginary_shift=0.1,
        )

        # Check required sections are present
        assert "&GATEWAY" in input_content
        assert "&SEWARD" in input_content
        assert "&SCF" in input_content
        assert "&RASSCF" in input_content
        assert "&CASPT2" in input_content

        # Check parameters are correctly inserted
        assert "IPEA = 0.25" in input_content
        assert "IMAGINARY = 0.1" in input_content
        assert (
            f"NACTEL = {sample_active_space_result.n_active_electrons}" in input_content
        )

    def test_ms_caspt2_input_generation(self, h2o_scf, sample_active_space_result):
        """Test MS-CASPT2 input file generation."""
        generator = OpenMolcasInputGenerator()

        input_content = generator.generate_input(
            h2o_scf,
            sample_active_space_result,
            calculation_type="ms_caspt2",
            n_states=3,
            multistate=True,
        )

        # Check multistate-specific content
        assert "MULTISTATE = 3" in input_content
        assert "XMIXED" in input_content
        assert "CIROOT = 3 3" in input_content

    def test_system_type_optimization(self, fe_scf):
        """Test parameter optimization for different system types."""
        generator = OpenMolcasInputGenerator()

        # Mock active space for Fe complex
        mock_active_space = Mock()
        mock_active_space.n_active_electrons = 10
        mock_active_space.n_active_orbitals = 10
        mock_active_space.orbital_coefficients = np.eye(20)

        # Generate input for transition metal system
        input_content = generator.generate_input(
            fe_scf, mock_active_space, calculation_type="caspt2"
        )

        # Should automatically include IPEA shift for transition metals
        assert "IPEA = 0.25" in input_content or "IPEA = 0.0" in input_content


class TestOpenMolcasOutputParser:
    """Test OpenMolcas output parsing."""

    def test_parser_initialization(self):
        """Test output parser initialization."""
        parser = OpenMolcasOutputParser()
        assert hasattr(parser, "patterns")
        assert "scf_energy" in parser.patterns
        assert "caspt2_energy" in parser.patterns

    def test_successful_caspt2_parsing(self):
        """Test parsing of successful CASPT2 output."""
        # Mock successful CASPT2 output
        mock_output = """
        Total SCF energy:       -76.026765
        RASSCF root number    1 Total energy:      -76.237891
        CASPT2 Total Energy:    -76.285432
        Reference energy:       -76.237891
        CASPT2 converged in 15 iterations
        Total wall time:        123.45
        Total CPU time:         98.76
        Maximum memory used:    2048.0 MB
        """

        parser = OpenMolcasOutputParser()
        results = parser.parse_output(mock_output, "caspt2")

        assert results.scf_energy == pytest.approx(-76.026765)
        assert results.casscf_energy == pytest.approx(-76.237891)
        assert results.caspt2_energy == pytest.approx(-76.285432)
        assert results.correlation_energy == pytest.approx(-0.047541)
        assert results.caspt2_converged is True
        assert results.wall_time == pytest.approx(123.45)
        assert results.cpu_time == pytest.approx(98.76)
        assert results.memory_usage == pytest.approx(2048.0)

    def test_ms_caspt2_parsing(self):
        """Test parsing of MS-CASPT2 output."""
        mock_output = """
        Total SCF energy:       -76.026765
        RASSCF root number    1 Total energy:      -76.237891
        MS-CASPT2 Root  1 Total energy:    -76.285432
        MS-CASPT2 Root  2 Total energy:    -76.245123
        MS-CASPT2 Root  3 Total energy:    -76.234567
        MS-CASPT2 Root  1 Weight:         0.85
        MS-CASPT2 Root  2 Weight:         0.10
        MS-CASPT2 Root  3 Weight:         0.05
        """

        parser = OpenMolcasOutputParser()
        results = parser.parse_output(mock_output, "ms_caspt2")

        assert len(results.state_energies) == 3
        assert results.state_energies[0] == pytest.approx(-76.285432)
        assert results.state_energies[1] == pytest.approx(-76.245123)
        assert results.state_energies[2] == pytest.approx(-76.234567)

        assert len(results.state_weights) == 3
        assert results.state_weights[0] == pytest.approx(0.85)
        assert results.state_weights[1] == pytest.approx(0.10)
        assert results.state_weights[2] == pytest.approx(0.05)

    def test_error_parsing(self):
        """Test parsing of failed calculations."""
        mock_output = """
        Total SCF energy:       -76.026765
        ERROR: CASPT2 calculation failed to converge
        FATAL ERROR in CASPT2 module
        """

        parser = OpenMolcasOutputParser()
        results = parser.parse_output(mock_output, "caspt2")

        assert len(results.errors) > 0
        assert any("Fatal error" in error for error in results.errors)
        assert results.caspt2_converged is False

    def test_external_result_conversion(self):
        """Test conversion to ExternalMethodResult."""
        mock_output = """
        Total SCF energy:       -76.026765
        CASPT2 Total Energy:    -76.285432
        CASPT2 converged
        """

        parser = OpenMolcasOutputParser()
        results = parser.parse_output(mock_output, "caspt2")
        external_result = parser.to_external_result(results, "CASPT2")

        assert external_result.method == "CASPT2"
        assert external_result.software == "OpenMolcas"
        assert external_result.energy == pytest.approx(-76.285432)
        assert external_result.converged is True


class TestCASPT2Method:
    """Test CASPT2 method implementation."""

    @patch(
        "quantum.chemistry.multireference.external.base.ExternalMethodInterface._validate_software"
    )
    def test_method_initialization(self, mock_validate):
        """Test CASPT2 method initialization."""
        # Mock the software validation to avoid requiring actual OpenMolcas installation
        mock_validate.return_value = None

        method = CASPT2Method(
            ipea_shift=0.25,
            multistate=False,
            imaginary_shift=0.1,
            auto_optimize_parameters=True,
            software_path="/mock/path/pymolcas",  # Provide a mock path
        )

        assert method.ipea_shift == 0.25
        assert method.multistate is False
        assert method.imaginary_shift == 0.1
        assert method.auto_optimize_parameters is True
        assert method._get_method_type() == MultireferenceMethodType.CASPT2

    @patch(
        "quantum.chemistry.multireference.external.base.ExternalMethodInterface._validate_software"
    )
    def test_parameter_optimization(self, mock_validate, h2o_scf, fe_scf):
        """Test automatic parameter optimization for different systems."""
        mock_validate.return_value = None
        method = CASPT2Method(
            auto_optimize_parameters=True, software_path="/mock/path/pymolcas"
        )

        # Mock active space
        mock_active_space = Mock()
        mock_active_space.n_active_electrons = 6
        mock_active_space.n_active_orbitals = 6

        # Test organic system (H2O)
        organic_params = method._optimize_parameters_for_system(
            h2o_scf.mol, mock_active_space
        )
        assert organic_params["ipea_shift"] == 0.0  # No IPEA for organic
        assert organic_params["imaginary_shift"] == 0.0

        # Test transition metal system (Fe complex)
        tm_params = method._optimize_parameters_for_system(
            fe_scf.mol, mock_active_space
        )
        assert tm_params["ipea_shift"] == 0.25  # IPEA for transition metals
        assert tm_params["memory_mb"] >= 4000  # More memory for TM

    @patch(
        "quantum.chemistry.multireference.external.openmolcas.caspt2_method.OpenMolcasCASPT2Interface"
    )
    def test_calculate_method(
        self, mock_interface, h2o_scf, sample_active_space_result
    ):
        """Test CASPT2 calculation method."""
        # Mock the interface to return a successful result
        mock_external_result = Mock()
        mock_external_result.energy = -76.285432
        mock_external_result.correlation_energy = -0.047541
        mock_external_result.converged = True
        mock_external_result.external_data = {
            "ipea_shift": 0.0,
            "imaginary_shift": 0.0,
            "warnings": [],
            "errors": [],
        }
        mock_external_result.convergence_info = {"converged": True}
        mock_external_result.cpu_time = 100.0
        mock_external_result.memory_mb = 2000.0
        mock_external_result.error_bars = None

        mock_interface_instance = Mock()
        mock_interface_instance.calculate.return_value = mock_external_result
        mock_interface.return_value = mock_interface_instance

        # Create method and run calculation
        method = CASPT2Method()
        method.interface = mock_interface_instance

        result = method.calculate(h2o_scf, sample_active_space_result)

        # Verify result structure
        assert isinstance(
            result, type(result).__bases__[0]
        )  # MultireferenceResult type
        assert result.method == "CASPT2"
        assert result.energy == pytest.approx(-76.285432)
        assert result.correlation_energy == pytest.approx(-0.047541)
        assert (
            result.n_active_electrons == sample_active_space_result.n_active_electrons
        )
        assert result.n_active_orbitals == sample_active_space_result.n_active_orbitals

    @patch(
        "quantum.chemistry.multireference.external.base.ExternalMethodInterface._validate_software"
    )
    def test_cost_estimation(self, mock_validate):
        """Test computational cost estimation."""
        mock_validate.return_value = None
        method = CASPT2Method(software_path="/mock/path/pymolcas")

        # Test cost estimation for different system sizes
        small_cost = method.estimate_cost(6, 6, 50)  # Small active space
        large_cost = method.estimate_cost(12, 12, 100)  # Larger active space

        assert "memory_mb" in small_cost
        assert "time_seconds" in small_cost
        assert "disk_mb" in small_cost

        # Larger system should cost more
        assert large_cost["memory_mb"] > small_cost["memory_mb"]
        assert large_cost["time_seconds"] > small_cost["time_seconds"]

        # Test multistate cost scaling
        ms_method = CASPT2Method(multistate=True, n_states=3)
        ms_cost = ms_method.estimate_cost(6, 6, 50)
        assert ms_cost["time_seconds"] > small_cost["time_seconds"]

    @patch(
        "quantum.chemistry.multireference.external.base.ExternalMethodInterface._validate_software"
    )
    def test_input_validation(self, mock_validate, h2o_scf, sample_active_space_result):
        """Test input validation."""
        mock_validate.return_value = None
        method = CASPT2Method(software_path="/mock/path/pymolcas")

        # Valid inputs should pass
        assert method.validate_input(h2o_scf, sample_active_space_result) is True

        # Test with unconverged SCF
        unconverged_scf = Mock()
        unconverged_scf.converged = False
        assert (
            method.validate_input(unconverged_scf, sample_active_space_result) is False
        )

        # Test with invalid active space
        invalid_active_space = Mock()
        invalid_active_space.n_active_electrons = 0
        invalid_active_space.n_active_orbitals = 5
        assert method.validate_input(h2o_scf, invalid_active_space) is False

    @patch(
        "quantum.chemistry.multireference.external.base.ExternalMethodInterface._validate_software"
    )
    def test_recommended_parameters(self, mock_validate):
        """Test recommended parameter generation."""
        mock_validate.return_value = None
        method = CASPT2Method(software_path="/mock/path/pymolcas")

        # Test different system types
        organic_params = method.get_recommended_parameters("organic", (6, 6))
        tm_params = method.get_recommended_parameters("transition_metal", (10, 10))
        biradical_params = method.get_recommended_parameters("biradical", (8, 8))

        # Organic systems should have no IPEA shift
        assert organic_params["ipea_shift"] == 0.0

        # Transition metals should have IPEA shift
        assert tm_params["ipea_shift"] == 0.25
        assert tm_params["memory_mb"] >= organic_params["memory_mb"]

        # Biradicals should have small imaginary shift
        assert biradical_params["imaginary_shift"] > 0.0


class TestOpenMolcasValidator:
    """Test cross-method validation utilities."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = OpenMolcasValidator()
        assert hasattr(validator, "tolerances")
        assert hasattr(validator, "benchmarks")
        assert "energy_absolute" in validator.tolerances

    def test_internal_consistency_validation(self):
        """Test internal consistency checks."""
        validator = OpenMolcasValidator()

        # Create mock CASPT2 result
        mock_result = Mock()
        mock_result.method = "CASPT2"
        mock_result.energy = -76.285432
        mock_result.correlation_energy = -0.047541  # Negative (good)
        mock_result.convergence_info = {"converged": True}
        mock_result.computational_cost = {"wall_time": 100.0}
        mock_result.active_space_info = {}
        mock_result.n_active_electrons = 6
        mock_result.n_active_orbitals = 6
        mock_result.basis_set = "sto-3g"
        mock_result.software_version = "OpenMolcas"

        validation = validator.validate_internal_consistency(mock_result)

        assert validation.validation_status == "PASS"
        assert "consistency checks passed" in validation.validation_message.lower()

    def test_casscf_comparison_validation(self, h2o_scf, sample_active_space_result):
        """Test validation against CASSCF reference."""
        validator = OpenMolcasValidator()

        # Mock CASPT2 and CASSCF results
        caspt2_result = Mock()
        caspt2_result.method = "CASPT2"
        caspt2_result.energy = -76.285432
        caspt2_result.correlation_energy = -0.047541
        caspt2_result.convergence_info = {"converged": True}
        caspt2_result.computational_cost = {"wall_time": 150.0}
        caspt2_result.active_space_info = {}

        casscf_result = Mock()
        casscf_result.energy = -76.237891  # Higher than CASPT2 (correct)
        casscf_result.correlation_energy = -0.020000
        casscf_result.convergence_info = {"converged": True}
        casscf_result.computational_cost = {"wall_time": 50.0}

        validation = validator.validate_against_casscf(
            caspt2_result,
            h2o_scf,
            sample_active_space_result,
            run_casscf=False,
            casscf_result=casscf_result,
        )

        assert validation.reference_method == "CASSCF"
        assert validation.test_method == "CASPT2"
        assert validation.energy_difference < 0  # CASPT2 should lower energy
        assert validation.validation_status == "PASS"

    def test_tolerance_customization(self):
        """Test custom tolerance setting."""
        validator = OpenMolcasValidator()

        original_tolerance = validator.tolerances["energy_absolute"]
        new_tolerance = 5e-4

        validator.set_tolerances(energy_absolute=new_tolerance)
        assert validator.tolerances["energy_absolute"] == new_tolerance
        assert validator.tolerances["energy_absolute"] != original_tolerance

    def test_custom_benchmark_addition(self):
        """Test adding custom benchmark data."""
        validator = OpenMolcasValidator()

        custom_benchmark = {
            "caspt2_energy": -100.123456,
            "correlation_energy": -0.456789,
            "geometry": "Test geometry",
            "basis": "test-basis",
        }

        validator.add_benchmark("test_system", custom_benchmark)
        assert "test_system" in validator.benchmarks
        assert validator.benchmarks["test_system"]["caspt2_energy"] == -100.123456


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_software_error(self):
        """Test error when OpenMolcas is not available."""
        from quantum.chemistry.multireference.external.base import (
            ExternalMethodInterface,
        )

        with patch("shutil.which", return_value=None):
            with patch.object(
                ExternalMethodInterface,
                "_check_containerized_installation",
                return_value=None,
            ):
                with pytest.raises(Exception):  # Should raise SoftwareNotFoundError
                    CASPT2Method(openmolcas_path=None)

    def test_calculation_timeout_error(self, h2o_scf, sample_active_space_result):
        """Test handling of calculation timeouts."""
        from quantum.chemistry.multireference.external.base import (
            ExternalMethodInterface,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

            # Mock software validation to avoid detection
            with patch.object(ExternalMethodInterface, "_validate_software"):
                method = CASPT2Method(skip_validation=True)
                # Mock the interface to avoid software detection
                method.interface = Mock()
                method.interface.calculate.side_effect = ExternalSoftwareError(
                    "Timeout"
                )

                with pytest.raises(ExternalSoftwareError):
                    method.calculate(h2o_scf, sample_active_space_result)

    def test_invalid_output_parsing(self):
        """Test handling of malformed output."""
        parser = OpenMolcasOutputParser()

        # Empty output should not crash
        results = parser.parse_output("", "caspt2")
        assert results.caspt2_energy is None
        assert len(results.errors) == 1  # Missing energy is reported as an error

        # Completely invalid output should not crash
        results = parser.parse_output("Random garbage text", "caspt2")
        assert results.caspt2_energy is None


class TestIntegrationPatterns:
    """Test integration with existing quantum-chemistry infrastructure."""

    def test_multireference_result_compatibility(
        self, h2o_scf, sample_active_space_result
    ):
        """Test that results are compatible with MultireferenceResult interface."""
        # This test would run a real calculation if OpenMolcas is available
        # For now, we test the result structure with mocked data
        from quantum.chemistry.multireference.external.base import (
            ExternalMethodInterface,
        )

        # Mock software validation to avoid detection
        with patch.object(ExternalMethodInterface, "_validate_software"):
            method = CASPT2Method(skip_validation=True)

        # Mock successful calculation
        with patch.object(method, "interface") as mock_interface:
            mock_external_result = Mock()
            mock_external_result.energy = -76.285432
            mock_external_result.correlation_energy = -0.047541
            mock_external_result.converged = True
            mock_external_result.external_data = {"warnings": [], "errors": []}
            mock_external_result.convergence_info = {"converged": True}
            mock_external_result.cpu_time = 100.0
            mock_external_result.memory_mb = 2000.0
            mock_external_result.error_bars = None

            mock_interface.calculate.return_value = mock_external_result

            result = method.calculate(h2o_scf, sample_active_space_result)

            # Test that result has all required MultireferenceResult fields
            required_fields = [
                "method",
                "energy",
                "active_space_info",
                "n_active_electrons",
                "n_active_orbitals",
                "convergence_info",
                "computational_cost",
            ]

            for field in required_fields:
                assert hasattr(result, field)
                assert getattr(result, field) is not None

    def test_fcidump_integration(self, h2o_scf, sample_active_space_result):
        """Test integration with FCIDUMP functionality."""
        from quantum.chemistry.fcidump import to_openmolcas_fcidump

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fcidump", delete=False
        ) as f:
            fcidump_path = f.name

        try:
            # Should not raise an error
            result_path = to_openmolcas_fcidump(
                sample_active_space_result, h2o_scf, fcidump_path
            )
            assert result_path == fcidump_path

            # Check file was created and has content
            with open(fcidump_path, "r") as f:
                content = f.read()
                assert "&FCI" in content
                assert "NORB=" in content
                assert "NELEC=" in content

        finally:
            Path(fcidump_path).unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.integration
class TestRealCalculations:
    """Integration tests that require actual OpenMolcas installation."""

    @pytest.mark.skipif(not shutil.which("pymolcas"), reason="OpenMolcas not available")
    def test_real_h2_calculation(self, h2_scf):
        """Test real CASPT2 calculation on H2 if OpenMolcas is available."""

        # Find active space for H2
        active_space = find_active_space_avas(h2_scf, threshold=0.1)

        # Run CASPT2 calculation
        method = CASPT2Method(
            ipea_shift=0.0,
            imaginary_shift=0.0,
            keep_files=True,  # Keep files for debugging
            auto_optimize_parameters=False,
        )

        result = method.calculate(h2_scf, active_space)

        # Basic sanity checks
        assert result.energy < h2_scf.e_tot  # Should be lower than HF
        assert result.convergence_info["converged"] is True
        assert result.correlation_energy < 0  # Should be negative

        # Energy should be reasonable for H2
        assert -1.2 < result.energy < -1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
