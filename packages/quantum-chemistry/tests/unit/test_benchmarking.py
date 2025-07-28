"""
Tests for benchmarking and validation infrastructure.

This module contains tests for the benchmarking datasets, analysis tools,
and validation frameworks for multireference methods.
"""

import pytest
import numpy as np
from datetime import datetime

from quantum.chemistry.multireference.benchmarking import (
    BenchmarkDataset,
    BenchmarkDatasetBuilder,
    BenchmarkEntry,
    BenchmarkMolecule,
    BenchmarkAnalyzer,
    ValidationRunner,
    SystemType,
    create_standard_benchmark_datasets,
)


class TestBenchmarkMolecule:
    """Test benchmark molecule specification."""
    
    def test_benchmark_molecule_creation(self):
        """Test creating benchmark molecule."""
        mol = BenchmarkMolecule(
            name="water_test",
            atoms=[("O", 0.0, 0.0, 0.0), ("H", 0.757, 0.0, 0.586), ("H", -0.757, 0.0, 0.586)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.ORGANIC,
            source="test_case"
        )
        
        assert mol.name == "water_test"
        assert len(mol.atoms) == 3
        assert mol.charge == 0
        assert mol.multiplicity == 1
        assert mol.system_type == SystemType.ORGANIC
    
    def test_pyscf_molecule_conversion(self):
        """Test conversion to PySCF molecule."""
        mol = BenchmarkMolecule(
            name="h2_test",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test_case"
        )
        
        pyscf_mol = mol.to_pyscf_molecule()
        
        assert pyscf_mol.natm == 2
        assert pyscf_mol.charge == 0
        assert pyscf_mol.spin == 0  # multiplicity - 1
        assert pyscf_mol.basis == "sto-3g"
    
    def test_system_hash_generation(self):
        """Test system hash generation."""
        mol1 = BenchmarkMolecule(
            name="test1",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        mol2 = BenchmarkMolecule(
            name="test2",  # Different name
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        # Same system should have same hash despite different names
        assert mol1.get_system_hash() == mol2.get_system_hash()
        
        # Different basis should have different hash
        mol3 = BenchmarkMolecule(
            name="test3",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="cc-pVDZ",  # Different basis
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        assert mol1.get_system_hash() != mol3.get_system_hash()


class TestBenchmarkEntry:
    """Test benchmark entry functionality."""
    
    def test_benchmark_entry_creation(self):
        """Test creating benchmark entry."""
        mol = BenchmarkMolecule(
            name="h2",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        entry = BenchmarkEntry(
            system=mol,
            method="CASSCF",
            active_space_method="avas",
            n_active_electrons=2,
            n_active_orbitals=2,
            energy=-1.1167593,
            reference_energy=-1.1167500,
            computational_cost={"time_seconds": 1.0, "memory_mb": 100.0}
        )
        
        assert entry.system.name == "h2"
        assert entry.method == "CASSCF"
        assert entry.n_active_electrons == 2
        assert entry.n_active_orbitals == 2
        assert entry.energy == -1.1167593
    
    def test_error_calculation(self):
        """Test automatic error calculation."""
        mol = BenchmarkMolecule(
            name="test",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        entry = BenchmarkEntry(
            system=mol,
            method="CASSCF",
            active_space_method="avas",
            n_active_electrons=2,
            n_active_orbitals=2,
            energy=-1.1167593,
            reference_energy=-1.1167500
        )
        
        entry.calculate_errors()
        
        expected_abs_error = abs(-1.1167593 - (-1.1167500))
        assert abs(entry.absolute_error - expected_abs_error) < 1e-10
        
        expected_rel_error = (expected_abs_error / abs(-1.1167500)) * 100
        assert abs(entry.relative_error - expected_rel_error) < 1e-10


class TestBenchmarkDataset:
    """Test benchmark dataset functionality."""
    
    def create_sample_dataset(self):
        """Create sample dataset for testing."""
        dataset = BenchmarkDataset(
            name="test_dataset",
            description="Test dataset for validation"
        )
        
        # Create sample molecules
        h2_mol = BenchmarkMolecule(
            name="h2",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        h2o_mol = BenchmarkMolecule(
            name="h2o",
            atoms=[("O", 0.0, 0.0, 0.0), ("H", 0.757, 0.0, 0.586), ("H", -0.757, 0.0, 0.586)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.ORGANIC,
            source="test"
        )
        
        # Add entries
        entries = [
            BenchmarkEntry(
                system=h2_mol,
                method="CASSCF",
                active_space_method="avas",
                n_active_electrons=2,
                n_active_orbitals=2,
                energy=-1.1167593,
                reference_energy=-1.1167500
            ),
            BenchmarkEntry(
                system=h2o_mol,
                method="CASSCF",
                active_space_method="avas",
                n_active_electrons=8,
                n_active_orbitals=4,
                energy=-74.9629675,
                reference_energy=-74.9629600
            ),
            BenchmarkEntry(
                system=h2o_mol,
                method="NEVPT2",
                active_space_method="avas",
                n_active_electrons=8,
                n_active_orbitals=4,
                energy=-74.9680000,
                reference_energy=-74.9679900
            )
        ]
        
        for entry in entries:
            dataset.add_entry(entry)
        
        return dataset
    
    def test_dataset_creation(self):
        """Test creating benchmark dataset."""
        dataset = BenchmarkDataset(
            name="test",
            description="Test dataset"
        )
        
        assert dataset.name == "test"
        assert dataset.description == "Test dataset"
        assert len(dataset.entries) == 0
    
    def test_dataset_filtering(self):
        """Test dataset filtering functionality."""
        dataset = self.create_sample_dataset()
        
        # Filter by system type
        organic_dataset = dataset.filter_by_system_type(SystemType.ORGANIC)
        assert len(organic_dataset.entries) == 2  # H2O entries
        
        diatomic_dataset = dataset.filter_by_system_type(SystemType.DIATOMIC) 
        assert len(diatomic_dataset.entries) == 1  # H2 entry
        
        # Filter by method
        casscf_dataset = dataset.filter_by_method("CASSCF")
        assert len(casscf_dataset.entries) == 2  # H2 and H2O CASSCF
        
        nevpt2_dataset = dataset.filter_by_method("NEVPT2")
        assert len(nevpt2_dataset.entries) == 1  # H2O NEVPT2
    
    def test_dataset_statistics(self):
        """Test dataset statistics calculation."""
        dataset = self.create_sample_dataset()
        stats = dataset.get_statistics()
        
        assert stats["total_entries"] == 3
        assert stats["entries_with_reference"] == 3
        assert "CASSCF" in stats["methods"]
        assert "NEVPT2" in stats["methods"]
        assert stats["methods"]["CASSCF"] == 2
        assert stats["methods"]["NEVPT2"] == 1
        
        # Check error statistics
        assert "error_statistics" in stats
        assert "mean_absolute_error" in stats["error_statistics"]
        assert "rmse" in stats["error_statistics"]


class TestBenchmarkAnalyzer:
    """Test benchmark analysis functionality."""
    
    def create_sample_dataset_with_errors(self):
        """Create dataset with known error patterns for testing."""
        dataset = BenchmarkDataset(
            name="error_test_dataset",
            description="Dataset for testing error analysis"
        )
        
        # Create entries with systematic errors
        test_mol = BenchmarkMolecule(
            name="test_mol",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        # Method A: systematic overestimation
        for i in range(5):
            entry = BenchmarkEntry(
                system=test_mol,
                method="METHOD_A",
                active_space_method="avas",
                n_active_electrons=2,
                n_active_orbitals=2,
                energy=-1.0 + 0.1,  # Consistently overestimate by 0.1 H
                reference_energy=-1.0
            )
            dataset.add_entry(entry)
        
        # Method B: random errors around reference
        reference_energies = [-1.0, -1.1, -0.9, -1.05, -0.95]
        calculated_energies = [-1.02, -1.08, -0.88, -1.07, -0.97]
        
        for ref, calc in zip(reference_energies, calculated_energies):
            entry = BenchmarkEntry(
                system=test_mol,
                method="METHOD_B",
                active_space_method="avas",
                n_active_electrons=2,
                n_active_orbitals=2,
                energy=calc,
                reference_energy=ref
            )
            dataset.add_entry(entry)
        
        return dataset
    
    def test_error_statistics(self):
        """Test error statistics calculation."""
        dataset = self.create_sample_dataset_with_errors()
        analyzer = BenchmarkAnalyzer(dataset)
        
        # Test overall statistics
        stats = analyzer.calculate_error_statistics()
        assert "mean_absolute_error" in stats
        assert "rmse" in stats
        assert stats["n_points"] == 10
        
        # Test method-specific statistics
        method_a_stats = analyzer.calculate_error_statistics(method="METHOD_A")
        assert method_a_stats["n_points"] == 5
        
        method_b_stats = analyzer.calculate_error_statistics(method="METHOD_B") 
        assert method_b_stats["n_points"] == 5
    
    def test_method_comparison(self):
        """Test method comparison functionality."""
        dataset = self.create_sample_dataset_with_errors()
        analyzer = BenchmarkAnalyzer(dataset)
        
        comparison = analyzer.compare_methods(["METHOD_A", "METHOD_B"])
        
        assert "METHOD_A" in comparison
        assert "METHOD_B" in comparison
        assert "mean_absolute_error" in comparison["METHOD_A"]
        assert "rmse" in comparison["METHOD_B"]
    
    def test_method_ranking(self):
        """Test method ranking by error metrics."""
        dataset = self.create_sample_dataset_with_errors()
        analyzer = BenchmarkAnalyzer(dataset)
        
        ranking = analyzer.method_ranking(["METHOD_A", "METHOD_B"])
        
        assert len(ranking) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranking)
        
        # First method should have lower error (better ranking)
        assert ranking[0][1] <= ranking[1][1]
    
    def test_systematic_error_analysis(self):
        """Test systematic error and bias analysis."""
        dataset = self.create_sample_dataset_with_errors()
        analyzer = BenchmarkAnalyzer(dataset)
        
        # METHOD_A should show systematic bias (overestimation)
        analysis_a = analyzer.systematic_error_analysis("METHOD_A")
        
        assert "systematic_bias" in analysis_a
        assert "error_distribution" in analysis_a
        assert analysis_a["systematic_bias"]["bias_direction"] == "overestimate"
        
        # Test normality check
        assert "normally_distributed" in analysis_a["error_distribution"]


class TestValidationRunner:
    """Test validation runner functionality."""
    
    def test_validation_runner_initialization(self):
        """Test validation runner initialization."""
        runner = ValidationRunner(
            tolerance_tight=0.001,
            tolerance_loose=0.01,
            max_scf_cycles=50
        )
        
        assert runner.tolerance_tight == 0.001
        assert runner.tolerance_loose == 0.01
        assert runner.max_scf_cycles == 50
        assert runner.workflow is not None
    
    def test_molecule_validation(self):
        """Test single molecule validation."""
        runner = ValidationRunner()
        
        # Create simple test molecule
        mol = BenchmarkMolecule(
            name="h2_test",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        # Run validation
        results = runner.validate_molecule(mol, methods=["casscf"])
        
        # Check results structure
        assert "molecule" in results
        assert "scf_results" in results
        assert "validation_status" in results
        assert results["molecule"] == "h2_test"
        
        # SCF should converge for H2
        if results["scf_results"]:
            assert "converged" in results["scf_results"]
            assert "energy" in results["scf_results"]


class TestStandardDatasets:
    """Test standard benchmark dataset creation."""
    
    def test_create_standard_datasets(self):
        """Test creation of standard benchmark datasets."""
        datasets = create_standard_benchmark_datasets()
        
        assert isinstance(datasets, dict)
        assert "questdb" in datasets
        assert "transition_metals" in datasets
        assert "bond_dissociation" in datasets
        
        # Check dataset structure
        questdb = datasets["questdb"]
        assert isinstance(questdb, BenchmarkDataset)
        assert questdb.name == "questdb_subset"
        assert len(questdb.entries) > 0
    
    def test_benchmark_dataset_builder(self):
        """Test benchmark dataset builder functionality."""
        builder = BenchmarkDatasetBuilder()
        
        dataset = (builder
                  .set_metadata("test_built", "Test built dataset")
                  .add_questdb_subset()
                  .build())
        
        assert dataset.name == "test_built"
        assert dataset.description == "Test built dataset"
        assert len(dataset.entries) > 0
        
        # Check that QUESTDB molecules were added
        has_water = any(entry.system.name == "water_s1" for entry in dataset.entries)
        has_formaldehyde = any(entry.system.name == "formaldehyde_t1" for entry in dataset.entries)
        
        assert has_water or has_formaldehyde  # At least one should be present


class TestIntegrationWithMultireference:
    """Test integration with multireference methods."""
    
    def test_benchmark_entry_to_multireference_result(self):
        """Test conversion from benchmark entry to multireference result."""
        mol = BenchmarkMolecule(
            name="test",
            atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
            charge=0,
            multiplicity=1,
            basis_set="sto-3g",
            system_type=SystemType.DIATOMIC,
            source="test"
        )
        
        entry = BenchmarkEntry(
            system=mol,
            method="CASSCF",
            active_space_method="avas",
            n_active_electrons=2,
            n_active_orbitals=2,
            energy=-1.1167593,
            correlation_energy=-0.005,
            properties={"dipole": 0.0},
            computational_cost={"time_seconds": 1.0}
        )
        
        mr_result = entry.to_multireference_result()
        
        assert mr_result.method == "CASSCF"
        assert mr_result.energy == -1.1167593
        assert mr_result.correlation_energy == -0.005
        assert mr_result.n_active_electrons == 2
        assert mr_result.n_active_orbitals == 2
        assert mr_result.properties == {"dipole": 0.0}