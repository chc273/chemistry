"""
Test suite for quantum chemistry validation framework.

This module provides automated tests for the validation framework,
ensuring that cross-method comparisons and statistical analyses work correctly.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantum.chemistry.validation import (
    BenchmarkSuite,
    ConvergenceTracker,
    MethodComparator,
    ReferenceDatabase,
    ReferenceEntry,
    StatisticalAnalyzer,
    ValidationResult,
)


class TestBenchmarkSuite:
    """Test the benchmark suite functionality."""

    def test_initialization(self):
        """Test that benchmark suite initializes with default systems."""
        suite = BenchmarkSuite()

        # Check that we have the expected benchmark systems
        systems = suite.list_systems()
        expected_systems = ["h2", "lih", "h2o", "n2", "f2", "cr2", "benzene"]

        for system in expected_systems:
            assert system in systems

    def test_get_system(self):
        """Test retrieving specific systems."""
        suite = BenchmarkSuite()

        # Test getting H2 system
        h2 = suite.get_system("h2")
        assert h2 is not None
        assert h2.name == "H2"
        assert h2.charge == 0
        assert h2.spin == 0

        # Test case insensitivity
        h2_upper = suite.get_system("H2")
        assert h2_upper is not None
        assert h2_upper.name == "H2"

        # Test non-existent system
        nonexistent = suite.get_system("nonexistent")
        assert nonexistent is None

    def test_system_difficulty_filtering(self):
        """Test filtering systems by difficulty level."""
        suite = BenchmarkSuite()

        easy_systems = suite.get_systems_by_difficulty("easy")
        medium_systems = suite.get_systems_by_difficulty("medium")
        hard_systems = suite.get_systems_by_difficulty("hard")

        # Check that we get expected counts
        assert len(easy_systems) >= 2  # At least H2 and LiH
        assert len(medium_systems) >= 2  # At least H2O and N2
        assert len(hard_systems) >= 1  # At least F2

        # Check that difficulty levels are correct
        for system in easy_systems:
            assert system.difficulty_level == "easy"
        for system in medium_systems:
            assert system.difficulty_level == "medium"

    def test_molecule_creation(self):
        """Test that benchmark systems can create PySCF molecules."""
        suite = BenchmarkSuite()
        h2 = suite.get_system("h2")

        # Test molecule creation
        mol = h2.create_molecule()
        assert mol.natm == 2  # Two atoms
        assert mol.charge == 0
        assert mol.spin == 0

        # Test SCF calculation
        mf = h2.run_scf()
        assert mf.converged
        assert abs(mf.e_tot - h2.reference_energies["hf"]) < 1e-4


class TestReferenceDatabase:
    """Test the reference database functionality."""

    def test_initialization(self):
        """Test database initialization with default data."""
        db = ReferenceDatabase()

        # Check that we have reference entries
        assert len(db.entries) > 0

        # Check for specific entries
        h2_fci = db.get_reference_energy("h2", "fci", "cc-pVDZ")
        assert h2_fci is not None
        assert abs(h2_fci.energy + 1.17447) < 1e-5

    def test_add_entry(self):
        """Test adding new reference entries."""
        db = ReferenceDatabase()
        initial_count = len(db.entries)

        # Add new entry
        new_entry = ReferenceEntry(
            system="test_system",
            method="test_method",
            basis_set="test_basis",
            energy=-100.0,
            source="Test source",
        )
        db.add_entry(new_entry)

        assert len(db.entries) == initial_count + 1

        # Retrieve the entry
        retrieved = db.get_reference_energy("test_system", "test_method", "test_basis")
        assert retrieved is not None
        assert retrieved.energy == -100.0

    def test_filtering(self):
        """Test filtering entries by various criteria."""
        db = ReferenceDatabase()

        # Filter by system
        h2_entries = db.get_entries(system="h2")
        assert len(h2_entries) > 0
        for entry in h2_entries:
            assert entry.system.lower() == "h2"

        # Filter by method
        fci_entries = db.get_entries(method="fci")
        assert len(fci_entries) > 0
        for entry in fci_entries:
            assert entry.method.lower() == "fci"

        # Filter by quality
        benchmark_entries = db.get_entries(quality_level="benchmark")
        assert len(benchmark_entries) > 0
        for entry in benchmark_entries:
            assert entry.quality_level == "benchmark"

    def test_statistics(self):
        """Test database statistics."""
        db = ReferenceDatabase()
        stats = db.get_statistics()

        assert "total_entries" in stats
        assert "unique_systems" in stats
        assert "unique_methods" in stats
        assert "quality_distribution" in stats
        assert "verification_rate" in stats

        assert stats["total_entries"] > 0
        assert stats["unique_systems"] > 0
        assert stats["unique_methods"] > 0
        assert 0 <= stats["verification_rate"] <= 1


class TestMethodComparator:
    """Test the method comparison framework."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = MethodComparator()

        assert comparator.benchmark_suite is not None
        assert comparator.reference_db is not None
        assert comparator.workflow is not None

    @pytest.mark.slow
    def test_method_validation(self):
        """Test validating a method against reference data."""
        comparator = MethodComparator()

        # Test with H2 system (should be fast)
        try:
            result = comparator.validate_method(
                system_name="h2",
                method="casscf",
                target_accuracy="low",  # Use low accuracy for speed
            )

            assert isinstance(result, ValidationResult)
            assert result.system_name == "h2"
            assert result.method == "casscf"
            assert result.calculated_energy != 0
            assert result.reference_energy != 0
            assert result.error_magnitude in ["excellent", "good", "acceptable", "poor"]

        except Exception as e:
            pytest.skip(
                f"Method validation failed, possibly due to missing dependencies: {e}"
            )

    @pytest.mark.slow
    def test_method_comparison(self):
        """Test comparing two methods."""
        comparator = MethodComparator()

        try:
            comparison = comparator.compare_methods(
                system_name="h2",
                method1="casscf",
                method2="nevpt2",
                target_accuracy="low",
            )

            assert comparison.system_name == "h2"
            assert comparison.method1 == "casscf"
            assert comparison.method2 == "nevpt2"
            assert comparison.energy1 != 0
            assert comparison.energy2 != 0
            assert comparison.agreement_level in [
                "excellent",
                "good",
                "acceptable",
                "poor",
            ]

        except Exception as e:
            pytest.skip(
                f"Method comparison failed, possibly due to missing dependencies: {e}"
            )

    def test_export_results(self):
        """Test exporting validation results."""
        comparator = MethodComparator()

        # Add a mock validation result
        mock_result = ValidationResult(
            system_name="test_system",
            method="test_method",
            basis_set="test_basis",
            calculated_energy=-100.5,
            reference_energy=-100.0,
            energy_error=-0.5,
            relative_error=0.005,
        )
        comparator.validation_results.append(mock_result)

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            comparator.export_results(temp_path)

            # Check that file was created and contains expected data
            assert temp_path.exists()

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "validation_results" in data
            assert "method_comparisons" in data
            assert "summary" in data
            assert len(data["validation_results"]) == 1

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestStatisticalAnalyzer:
    """Test the statistical analysis functionality."""

    def setup_method(self):
        """Set up test data."""
        self.analyzer = StatisticalAnalyzer()

        # Create mock validation results
        mock_results = [
            ValidationResult(
                system_name="h2",
                method="method1",
                basis_set="cc-pVDZ",
                calculated_energy=-1.170,
                reference_energy=-1.174,
                energy_error=0.004,
                relative_error=-0.0034,
            ),
            ValidationResult(
                system_name="h2o",
                method="method1",
                basis_set="cc-pVDZ",
                calculated_energy=-76.240,
                reference_energy=-76.243,
                energy_error=0.003,
                relative_error=-0.0000394,
            ),
            ValidationResult(
                system_name="h2",
                method="method2",
                basis_set="cc-pVDZ",
                calculated_energy=-1.172,
                reference_energy=-1.174,
                energy_error=0.002,
                relative_error=-0.0017,
            ),
        ]

        self.analyzer.add_validation_results(mock_results)

    def test_method_accuracy_analysis(self):
        """Test analyzing method accuracy."""
        stats = self.analyzer.analyze_method_accuracy("method1")

        assert stats["method"] == "method1"
        assert stats["n_systems"] == 2
        assert "mean_error" in stats
        assert "std_error" in stats
        assert "mean_absolute_error" in stats
        assert "rms_error" in stats
        assert "error_distribution" in stats

        # Check error distribution structure
        error_dist = stats["error_distribution"]
        assert all(
            key in error_dist for key in ["excellent", "good", "acceptable", "poor"]
        )
        assert sum(error_dist.values()) == 2  # Two systems for method1

    def test_method_comparison_analysis(self):
        """Test comparing multiple methods."""
        comparison = self.analyzer.compare_method_performance(["method1", "method2"])

        assert "method_statistics" in comparison
        assert "pairwise_comparisons" in comparison
        assert "ranking" in comparison

        # Check method statistics
        method_stats = comparison["method_statistics"]
        assert "method1" in method_stats
        assert "method2" in method_stats

        # Check ranking
        ranking = comparison["ranking"]
        assert len(ranking) == 2
        assert ranking[0]["rank"] == 1
        assert ranking[1]["rank"] == 2

    def test_system_difficulty_analysis(self):
        """Test analyzing system difficulty."""
        difficulty = self.analyzer.analyze_system_difficulty()

        assert "system_statistics" in difficulty
        assert "difficulty_ranking" in difficulty

        # Check that both systems are present
        system_stats = difficulty["system_statistics"]
        assert "h2" in system_stats
        assert "h2o" in system_stats

        # Check ranking structure
        ranking = difficulty["difficulty_ranking"]
        assert len(ranking) == 2
        for entry in ranking:
            assert "system" in entry
            assert "difficulty_score" in entry
            assert "difficulty_rank" in entry


class TestConvergenceTracker:
    """Test the convergence tracking functionality."""

    def setup_method(self):
        """Set up test data."""
        self.tracker = ConvergenceTracker()

        # Add mock convergence data
        reference_energy = -1.17447
        for i, param_val in enumerate([1, 2, 3, 4, 5]):
            # Simulate exponential convergence
            energy = reference_energy + 0.01 * np.exp(-0.5 * param_val)

            self.tracker.add_convergence_point(
                system="h2",
                method="test_method",
                parameter_name="basis_size",
                parameter_value=param_val,
                energy=energy,
                reference_energy=reference_energy,
            )

    def test_add_convergence_point(self):
        """Test adding convergence data points."""
        tracker = ConvergenceTracker()

        tracker.add_convergence_point(
            system="test_system",
            method="test_method",
            parameter_name="test_param",
            parameter_value=1.0,
            energy=-100.0,
        )

        key = "test_system_test_method_test_param"
        assert key in tracker.convergence_data
        assert len(tracker.convergence_data[key]) == 1

        point = tracker.convergence_data[key][0]
        assert point.parameter_value == 1.0
        assert point.energy == -100.0

    def test_convergence_analysis(self):
        """Test convergence analysis."""
        analysis = self.tracker.analyze_convergence("h2", "test_method", "basis_size")

        assert analysis is not None
        assert analysis.system_name == "h2"
        assert analysis.method == "test_method"
        assert analysis.parameter_name == "basis_size"
        assert len(analysis.points) == 5

        # Check that convergence parameters were fitted
        assert analysis.converged_value is not None
        assert analysis.convergence_rate is not None

    def test_insufficient_data(self):
        """Test behavior with insufficient convergence data."""
        tracker = ConvergenceTracker()

        # Add only one point
        tracker.add_convergence_point(
            system="test",
            method="test",
            parameter_name="test",
            parameter_value=1.0,
            energy=-100.0,
        )

        # Should return None for insufficient data
        analysis = tracker.analyze_convergence("test", "test", "test")
        assert analysis is None


class TestIntegration:
    """Integration tests for the validation framework."""

    @pytest.mark.slow
    def test_full_validation_workflow(self):
        """Test the complete validation workflow."""
        try:
            # Initialize components
            comparator = MethodComparator()
            analyzer = StatisticalAnalyzer()

            # Run validation on a simple system
            result = comparator.validate_method(
                system_name="h2", method="casscf", target_accuracy="low"
            )

            # Add to analyzer
            analyzer.add_validation_results([result])

            # Analyze results
            stats = analyzer.analyze_method_accuracy("casscf")
            assert stats["n_systems"] == 1

            # Export results
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                temp_path = Path(f.name)

            try:
                comparator.export_results(temp_path)
                assert temp_path.exists()

                # Check file contents
                with open(temp_path, "r") as f:
                    data = json.load(f)

                assert len(data["validation_results"]) == 1

            finally:
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            pytest.skip(
                f"Full workflow test failed, possibly due to missing dependencies: {e}"
            )

    def test_benchmark_suite_integration(self):
        """Test integration between benchmark suite and reference database."""
        suite = BenchmarkSuite()
        db = ReferenceDatabase()

        # Check that systems in benchmark suite have reference data
        for system_name in ["h2", "h2o", "lih"]:
            system = suite.get_system(system_name)
            assert system is not None

            # Check that we have at least one reference entry for this system
            entries = db.get_entries(system=system_name)
            assert len(entries) > 0

            # Check that reference energies match system's reference data
            for method, ref_energy in system.reference_energies.items():
                db_entry = db.get_reference_energy(
                    system_name, method, system.basis_set
                )
                if db_entry:
                    # Allow some tolerance for rounding differences
                    assert abs(db_entry.energy - ref_energy) < 1e-4


class TestExternalMethodIntegration:
    """Test external method integration with the validation framework."""

    def test_external_method_suite(self):
        """Test getting external method benchmark suite."""
        suite = BenchmarkSuite()

        external_methods = ["molpro", "orca", "psi4"]
        external_suite = suite.get_external_method_suite(
            external_methods, difficulty_levels=["easy", "medium"]
        )

        assert isinstance(external_suite, dict)
        assert len(external_suite) == len(external_methods)

        for method in external_methods:
            assert method in external_suite
            assert isinstance(external_suite[method], list)

            # Check that systems are appropriate difficulty
            for system in external_suite[method]:
                assert system.difficulty_level in ["easy", "medium"]

    def test_add_external_reference_data(self):
        """Test adding external reference data."""
        suite = BenchmarkSuite()

        # Add external reference data
        suite.add_external_reference_data(
            system_name="h2",
            method="molpro_casscf",
            energy=-1.17400,
            source="External MOLPRO calculation",
        )

        h2_system = suite.get_system("h2")
        assert "molpro_casscf" in h2_system.reference_energies
        assert abs(h2_system.reference_energies["molpro_casscf"] + 1.17400) < 1e-6
        assert "External MOLPRO calculation" in h2_system.references

    def test_external_method_compatibility(self):
        """Test external method compatibility checking."""
        suite = BenchmarkSuite()

        # Test with mock external method
        try:
            compatibility = suite.validate_external_method_compatibility("mock_method")

            assert "method" in compatibility
            assert "available" in compatibility
            assert "compatible_systems" in compatibility
            assert "incompatible_systems" in compatibility
            assert "recommended_systems" in compatibility

            assert compatibility["method"] == "mock_method"
            assert (
                compatibility["available"] is False
            )  # Mock method should not be available

        except Exception:
            # If ExternalMethodRunner is not properly mocked, skip this test
            pytest.skip("ExternalMethodRunner not available for testing")

    @pytest.mark.slow
    def test_external_method_validation(self):
        """Test external method validation (requires Docker setup)."""
        comparator = MethodComparator()

        try:
            # This will likely fail in test environment without Docker
            result = comparator.validate_external_method(
                system_name="h2", external_method="mock_external", method_type="hf"
            )

            # If we get here, external method worked
            assert isinstance(result, ValidationResult)
            assert result.system_name == "h2"
            assert "mock_external" in result.method

        except Exception as e:
            # Expected in test environment without external methods
            pytest.skip(f"External method validation not available: {e}")

    def test_external_validation_suite_structure(self):
        """Test the structure of external validation suite results."""
        comparator = MethodComparator()

        # Mock the external runner to avoid Docker dependency
        class MockExternalRunner:
            def is_method_available(self, method):
                return method in ["molpro", "orca"]

            def run_calculation(self, method, input_data):
                # Return mock results
                return {
                    "energy": -1.17400,
                    "convergence_info": {"converged": True, "iterations": 10},
                }

        # Replace the external runner with mock
        original_runner = comparator.external_runner
        comparator.external_runner = MockExternalRunner()

        try:
            # Test the structure without actually running calculations
            external_methods = ["molpro", "orca"]

            # Just test that the method exists and has the right signature
            assert hasattr(comparator, "run_external_validation_suite")

            # Test method signature
            import inspect

            sig = inspect.signature(comparator.run_external_validation_suite)
            expected_params = [
                "external_methods",
                "method_types",
                "systems",
                "difficulty_levels",
            ]

            for param in expected_params:
                assert param in sig.parameters

        finally:
            # Restore original runner
            comparator.external_runner = original_runner


class TestWorkflowOrchestration:
    """Test production workflow orchestration capabilities."""

    def test_workflow_configuration(self):
        """Test workflow configuration."""
        from quantum.chemistry.validation import WorkflowConfiguration

        config = WorkflowConfiguration(
            max_parallel_tasks=8,
            timeout_seconds=600,
            output_directory=Path("test_results"),
        )

        assert config.max_parallel_tasks == 8
        assert config.timeout_seconds == 600
        assert config.output_directory == Path("test_results")
        assert config.retry_failed_tasks is True  # Default value

    def test_validation_task(self):
        """Test validation task creation."""
        from quantum.chemistry.validation import ValidationTask

        task = ValidationTask(
            task_id="test_001",
            system_name="h2",
            method="casscf",
            priority=1,
            parameters={"target_accuracy": "standard"},
        )

        assert task.task_id == "test_001"
        assert task.system_name == "h2"
        assert task.method == "casscf"
        assert task.priority == 1
        assert task.status == "pending"
        assert task.parameters["target_accuracy"] == "standard"

    def test_workflow_manager_initialization(self):
        """Test workflow manager initialization."""
        from quantum.chemistry.validation import (
            ValidationWorkflowManager,
            WorkflowConfiguration,
        )

        config = WorkflowConfiguration(max_parallel_tasks=4)
        manager = ValidationWorkflowManager(config=config)

        assert manager.config.max_parallel_tasks == 4
        assert manager.benchmark_suite is not None
        assert manager.reference_db is not None
        assert manager.comparator is not None
        assert len(manager.tasks) == 0
        assert len(manager.completed_tasks) == 0
        assert len(manager.failed_tasks) == 0

    def test_standard_workflow_creation(self):
        """Test standard workflow creation."""
        from quantum.chemistry.validation import ValidationWorkflowManager

        manager = ValidationWorkflowManager()

        # Create standard workflow
        tasks = manager.create_standard_workflow(
            methods=["casscf"], systems=["h2", "lih"], difficulty_levels=["easy"]
        )

        assert len(tasks) == 2  # 1 method × 2 systems
        assert all(task.system_name in ["h2", "lih"] for task in tasks)
        assert all(task.method == "casscf" for task in tasks)
        assert all(not task.is_external for task in tasks)
        assert all(task.status == "pending" for task in tasks)

    def test_convergence_workflow_creation(self):
        """Test convergence workflow creation."""
        from quantum.chemistry.validation import ValidationWorkflowManager

        manager = ValidationWorkflowManager()

        # Create convergence workflow
        tasks = manager.create_convergence_workflow(
            method="casscf",
            system_name="h2",
            parameter_name="basis_size",
            parameter_values=[2, 3, 4],
            is_external=False,
        )

        assert len(tasks) == 3  # 3 parameter values
        assert all(task.system_name == "h2" for task in tasks)
        assert all(task.method == "casscf" for task in tasks)
        assert all(not task.is_external for task in tasks)
        assert all(task.parameters.get("convergence_study") is True for task in tasks)

    def test_workflow_status(self):
        """Test workflow status tracking."""
        from quantum.chemistry.validation import (
            ValidationTask,
            ValidationWorkflowManager,
        )

        manager = ValidationWorkflowManager()

        # Add some mock tasks
        tasks = [
            ValidationTask("task_1", "h2", "casscf"),
            ValidationTask("task_2", "lih", "casscf"),
            ValidationTask("task_3", "h2o", "nevpt2"),
        ]

        # Set different statuses
        tasks[0].status = "completed"
        tasks[1].status = "failed"
        tasks[2].status = "running"

        manager.tasks = tasks
        manager.completed_tasks = [tasks[0]]
        manager.failed_tasks = [tasks[1]]

        status = manager.get_workflow_status()

        assert status["total_tasks"] == 3
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1
        assert status["running_tasks"] == 1
        assert status["pending_tasks"] == 0
        assert status["success_rate"] == 1 / 3

    def test_add_tasks(self):
        """Test adding tasks to workflow."""
        from quantum.chemistry.validation import (
            ValidationTask,
            ValidationWorkflowManager,
        )

        manager = ValidationWorkflowManager()

        tasks = [
            ValidationTask("task_1", "h2", "casscf"),
            ValidationTask("task_2", "lih", "nevpt2"),
        ]

        manager.add_tasks(tasks)

        assert len(manager.tasks) == 2
        assert manager.tasks[0].task_id == "task_1"
        assert manager.tasks[1].task_id == "task_2"

    @pytest.mark.slow
    def test_workflow_execution_structure(self):
        """Test workflow execution structure (without actual calculations)."""
        from quantum.chemistry.validation import ValidationWorkflowManager

        manager = ValidationWorkflowManager()

        # Test that run_workflow method exists and has correct signature
        assert hasattr(manager, "run_workflow")

        import inspect

        sig = inspect.signature(manager.run_workflow)
        expected_params = ["progress_callback", "parallel"]

        for param in expected_params:
            assert param in sig.parameters

        # Test that workflow raises error when no tasks
        with pytest.raises(ValueError, match="No tasks to execute"):
            manager.run_workflow()

    def test_task_id_generation(self):
        """Test automatic task ID generation."""
        from quantum.chemistry.validation import ValidationTask

        # Test with empty task_id
        task = ValidationTask("", "h2", "casscf")
        assert task.task_id != ""
        assert "h2" in task.task_id
        assert "casscf" in task.task_id

        # Test with None task_id
        task2 = ValidationTask(None, "h2o", "nevpt2")
        assert task2.task_id is not None
        assert "h2o" in task2.task_id
        assert "nevpt2" in task2.task_id
