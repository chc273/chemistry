"""
Comprehensive Benchmarking Suite for Quantum Chemistry Methods

This module provides a complete framework for large-scale benchmarking studies
including automated workflows, cross-method validation, and statistical analysis.
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from ..databases import DatabaseManager, DatabaseInterface, PropertyType
from ..databases.base import MolecularEntry
from ...active_space import (
    ActiveSpaceMethod, find_active_space_avas, find_active_space_apc,
    find_active_space_dmet, find_active_space_boys, ActiveSpaceResult
)
from ...multireference.methods.casscf import CASSCFMethod
from ...validation.comparison import MethodComparator, ValidationResult
from ...validation.statistical import StatisticalAnalyzer, ConvergenceTracker

logger = logging.getLogger(__name__)


class BenchmarkScope(Enum):
    """Scope of benchmarking study."""
    ACTIVE_SPACE_SELECTION = "active_space_selection"
    MULTIREFERENCE_METHODS = "multireference_methods"
    COMBINED_WORKFLOW = "combined_workflow"
    METHOD_COMPARISON = "method_comparison"
    CONVERGENCE_STUDY = "convergence_study"


class BenchmarkTarget(Enum):
    """Target property for benchmarking."""
    GROUND_STATE_ENERGY = "ground_state_energy"
    EXCITATION_ENERGY = "excitation_energy"
    BOND_DISSOCIATION_ENERGY = "bond_dissociation_energy"
    FORMATION_ENERGY = "formation_energy"
    MOLECULAR_PROPERTIES = "molecular_properties"


@dataclass
class BenchmarkConfiguration:
    """Configuration for comprehensive benchmarking studies."""
    
    # Study identification
    study_name: str
    description: str
    output_directory: Path
    
    # Scope and targets
    scope: BenchmarkScope
    targets: List[BenchmarkTarget]
    
    # Database selection
    databases: List[str] = field(default_factory=lambda: ["w4_11", "g2_97", "questdb"])
    max_molecules_per_db: Optional[int] = None
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium"])
    
    # Method selection
    active_space_methods: List[ActiveSpaceMethod] = field(default_factory=lambda: [
        ActiveSpaceMethod.AVAS, ActiveSpaceMethod.APC, ActiveSpaceMethod.DMET_CAS
    ])
    multireference_methods: List[str] = field(default_factory=lambda: ["casscf", "nevpt2"])
    
    # Computational parameters
    basis_sets: List[str] = field(default_factory=lambda: ["sto-3g", "cc-pvdz"])
    active_space_sizes: List[Tuple[int, int]] = field(default_factory=list)
    
    # Performance settings
    max_workers: int = field(default_factory=lambda: min(8, mp.cpu_count()))
    timeout_seconds: int = 3600  # 1 hour per calculation
    memory_limit_gb: int = 8
    
    # Quality control
    convergence_threshold: float = 1e-6
    max_iterations: int = 100
    require_convergence: bool = True
    
    # Statistical analysis
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    uncertainty_analysis: bool = True
    
    # Output control
    save_intermediate_results: bool = True
    generate_plots: bool = True
    create_report: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "hdf5"])


@dataclass
class BenchmarkResult:
    """Result from a single benchmark calculation."""
    
    # Identification
    calculation_id: str
    molecule_name: str
    database_source: str
    
    # Method details
    active_space_method: Optional[str] = None
    multireference_method: Optional[str] = None
    basis_set: str = "sto-3g"
    
    # Active space information
    active_space: Optional[ActiveSpaceResult] = None
    n_active_electrons: Optional[int] = None
    n_active_orbitals: Optional[int] = None
    
    # Results
    computed_energy: Optional[float] = None
    reference_energy: Optional[float] = None
    energy_error: Optional[float] = None
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    
    # Computational details
    wall_time: Optional[float] = None
    cpu_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    converged: bool = False
    iterations: Optional[int] = None
    
    # Error information
    error_message: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ActiveSpaceResult):
                result_dict[key] = {
                    "method": value.method,
                    "n_active_electrons": value.n_active_electrons,
                    "n_active_orbitals": value.n_active_orbitals,
                    "active_orbitals": value.active_orbitals,
                    "selection_info": value.selection_info
                }
            else:
                result_dict[key] = value
        return result_dict


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmarking suite for quantum chemistry methods."""
    
    def __init__(self, config: BenchmarkConfiguration):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.method_comparator = MethodComparator()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.convergence_tracker = ConvergenceTracker()
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.failed_calculations: List[Dict[str, Any]] = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized ComprehensiveBenchmarkSuite: {config.study_name}")
    
    def _setup_logging(self) -> None:
        """Setup logging for the benchmark suite."""
        log_file = self.output_dir / f"{self.config.study_name}_benchmark.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def prepare_molecular_dataset(self) -> List[MolecularEntry]:
        """Prepare the molecular dataset for benchmarking."""
        logger.info("Preparing molecular dataset...")
        
        molecules = []
        
        for db_name in self.config.databases:
            try:
                self.db_manager.load_database(db_name)
                database = self.db_manager.databases[db_name]
                
                # Get molecules from database
                db_molecules = database.get_all_molecules()
                
                # Filter by difficulty level
                filtered_molecules = [
                    mol for mol in db_molecules
                    if mol.computational_difficulty in self.config.difficulty_levels
                ]
                
                # Limit number per database if specified
                if self.config.max_molecules_per_db:
                    filtered_molecules = filtered_molecules[:self.config.max_molecules_per_db]
                
                molecules.extend(filtered_molecules)
                logger.info(f"Added {len(filtered_molecules)} molecules from {db_name}")
                
            except Exception as e:
                logger.error(f"Failed to load database {db_name}: {e}")
                continue
        
        # Remove duplicates based on formula
        unique_molecules = {}
        for mol in molecules:
            key = f"{mol.formula}_{mol.charge}_{mol.multiplicity}"
            if key not in unique_molecules:
                unique_molecules[key] = mol
        
        final_molecules = list(unique_molecules.values())
        logger.info(f"Final dataset: {len(final_molecules)} unique molecules")
        
        return final_molecules
    
    def run_active_space_benchmark(
        self, 
        molecules: List[MolecularEntry]
    ) -> List[BenchmarkResult]:
        """Run active space selection benchmarking."""
        logger.info("Starting active space selection benchmark...")
        
        # Prepare calculation tasks
        tasks = []
        for molecule in molecules:
            for as_method in self.config.active_space_methods:
                for basis_set in self.config.basis_sets:
                    task = {
                        "molecule": molecule,
                        "active_space_method": as_method,
                        "basis_set": basis_set,
                        "calculation_id": str(uuid.uuid4())
                    }
                    tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} active space selection tasks")
        
        # Execute tasks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_active_space_calculation, task): task
                for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), 
                             desc="Active space calculations"):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result:
                        results.append(result)
                except Exception as e:
                    error_info = {
                        "task": task,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    self.failed_calculations.append(error_info)
                    logger.warning(f"Failed calculation: {task['calculation_id']} - {e}")
        
        logger.info(f"Completed {len(results)} active space calculations")
        return results
    
    def _run_active_space_calculation(self, task: Dict[str, Any]) -> Optional[BenchmarkResult]:
        """Run a single active space selection calculation."""
        start_time = time.time()
        
        try:
            molecule = task["molecule"]
            as_method = task["active_space_method"]
            basis_set = task["basis_set"]
            calc_id = task["calculation_id"]
            
            # Create PySCF molecule
            mol = molecule.create_molecule(basis_set)
            
            # Run SCF calculation
            if molecule.multiplicity == 1:
                from pyscf import scf
                mf = scf.RHF(mol)
            else:
                from pyscf import scf
                mf = scf.UHF(mol)
            
            mf.kernel()
            
            if not mf.converged:
                return None
            
            # Run active space selection
            if as_method == ActiveSpaceMethod.AVAS:
                active_space = find_active_space_avas(mf)
            elif as_method == ActiveSpaceMethod.APC:
                active_space = find_active_space_apc(mf)
            elif as_method == ActiveSpaceMethod.DMET_CAS:
                active_space = find_active_space_dmet(mf)
            elif as_method == ActiveSpaceMethod.BOYS:
                active_space = find_active_space_boys(mf)
            else:
                logger.warning(f"Unknown active space method: {as_method}")
                return None
            
            # Get reference energy if available
            ref_energy = None
            energy_error = None
            
            if BenchmarkTarget.GROUND_STATE_ENERGY in self.config.targets:
                # Try to get reference atomization or formation energy
                ref_entry = molecule.get_reference_value(PropertyType.ATOMIZATION_ENERGY)
                if ref_entry is None:
                    ref_entry = molecule.get_reference_value(PropertyType.FORMATION_ENERGY)
                
                if ref_entry:
                    ref_energy = ref_entry.value
                    # For now, use SCF energy as computed energy
                    # In practice, would run multireference calculation
                    computed_energy = mf.e_tot
                    energy_error = computed_energy - ref_energy
            
            wall_time = time.time() - start_time
            
            # Create result
            result = BenchmarkResult(
                calculation_id=calc_id,
                molecule_name=molecule.name,
                database_source=molecule.database_id.split('_')[0],
                active_space_method=as_method.value,
                basis_set=basis_set,
                active_space=active_space,
                n_active_electrons=active_space.n_active_electrons,
                n_active_orbitals=active_space.n_active_orbitals,
                computed_energy=mf.e_tot,
                reference_energy=ref_energy,
                energy_error=energy_error,
                absolute_error=abs(energy_error) if energy_error else None,
                relative_error=energy_error / ref_energy if ref_energy else None,
                wall_time=wall_time,
                converged=mf.converged
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in active space calculation {calc_id}: {e}")
            return None
    
    def run_multireference_benchmark(
        self, 
        molecules: List[MolecularEntry],
        active_space_results: Optional[List[BenchmarkResult]] = None
    ) -> List[BenchmarkResult]:
        """Run multireference method benchmarking."""
        logger.info("Starting multireference method benchmark...")
        
        # If no active space results provided, run AVAS as default
        if active_space_results is None:
            logger.info("No active space results provided, running AVAS selection...")
            active_space_results = []
            for molecule in molecules:
                try:
                    mol = molecule.create_molecule("cc-pvdz")
                    from pyscf import scf
                    mf = scf.RHF(mol) if molecule.multiplicity == 1 else scf.UHF(mol)
                    mf.kernel()
                    
                    if mf.converged:
                        active_space = find_active_space_avas(mf)
                        result = BenchmarkResult(
                            calculation_id=str(uuid.uuid4()),
                            molecule_name=molecule.name,
                            database_source=molecule.database_id.split('_')[0],
                            active_space_method="avas",
                            basis_set="cc-pvdz",
                            active_space=active_space,
                            n_active_electrons=active_space.n_active_electrons,
                            n_active_orbitals=active_space.n_active_orbitals
                        )
                        active_space_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to get active space for {molecule.name}: {e}")
        
        # Prepare multireference calculation tasks
        tasks = []
        for as_result in active_space_results:
            # Find corresponding molecule
            molecule = None
            for mol in molecules:
                if mol.name == as_result.molecule_name:
                    molecule = mol
                    break
            
            if molecule is None:
                continue
            
            for mr_method in self.config.multireference_methods:
                for basis_set in self.config.basis_sets:
                    task = {
                        "molecule": molecule,
                        "active_space_result": as_result,
                        "multireference_method": mr_method,
                        "basis_set": basis_set,
                        "calculation_id": str(uuid.uuid4())
                    }
                    tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} multireference calculation tasks")
        
        # Execute tasks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_multireference_calculation, task): task
                for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), 
                             desc="Multireference calculations"):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result:
                        results.append(result)
                except Exception as e:
                    error_info = {
                        "task": task,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    self.failed_calculations.append(error_info)
                    logger.warning(f"Failed calculation: {task['calculation_id']} - {e}")
        
        logger.info(f"Completed {len(results)} multireference calculations")
        return results
    
    def _run_multireference_calculation(self, task: Dict[str, Any]) -> Optional[BenchmarkResult]:
        """Run a single multireference calculation."""
        start_time = time.time()
        
        try:
            molecule = task["molecule"]
            as_result = task["active_space_result"]
            mr_method = task["multireference_method"]
            basis_set = task["basis_set"]
            calc_id = task["calculation_id"]
            
            # Create PySCF molecule
            mol = molecule.create_molecule(basis_set)
            
            # Run SCF calculation
            if molecule.multiplicity == 1:
                from pyscf import scf
                mf = scf.RHF(mol)
            else:
                from pyscf import scf
                mf = scf.UHF(mol)
            
            mf.kernel()
            
            if not mf.converged:
                return None
            
            # Run multireference calculation
            computed_energy = None
            converged = False
            iterations = 0
            
            if mr_method.lower() == "casscf":
                casscf_method = CASSCFMethod()
                try:
                    mr_result = casscf_method.calculate(mf, as_result.active_space)
                    computed_energy = mr_result.total_energy
                    converged = mr_result.converged
                    iterations = mr_result.iterations
                except Exception as e:
                    logger.warning(f"CASSCF calculation failed: {e}")
                    return None
            
            # Get reference energy
            ref_energy = None
            energy_error = None
            
            ref_entry = molecule.get_reference_value(PropertyType.ATOMIZATION_ENERGY)
            if ref_entry is None:
                ref_entry = molecule.get_reference_value(PropertyType.FORMATION_ENERGY)
            
            if ref_entry and computed_energy is not None:
                ref_energy = ref_entry.value
                energy_error = computed_energy - ref_energy
            
            wall_time = time.time() - start_time
            
            # Create result
            result = BenchmarkResult(
                calculation_id=calc_id,
                molecule_name=molecule.name,
                database_source=molecule.database_id.split('_')[0],
                active_space_method=as_result.active_space_method,
                multireference_method=mr_method,
                basis_set=basis_set,
                active_space=as_result.active_space,
                n_active_electrons=as_result.n_active_electrons,
                n_active_orbitals=as_result.n_active_orbitals,
                computed_energy=computed_energy,
                reference_energy=ref_energy,
                energy_error=energy_error,
                absolute_error=abs(energy_error) if energy_error else None,
                relative_error=energy_error / ref_energy if ref_energy else None,
                wall_time=wall_time,
                converged=converged,
                iterations=iterations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multireference calculation {calc_id}: {e}")
            return None
    
    def run_convergence_study(
        self, 
        molecules: List[MolecularEntry], 
        parameter_name: str = "basis_set"
    ) -> List[BenchmarkResult]:
        """Run convergence study for specified parameter."""
        logger.info(f"Starting convergence study for {parameter_name}...")
        
        if parameter_name == "basis_set":
            parameter_values = ["sto-3g", "cc-pvdz", "cc-pvtz"]
        else:
            parameter_values = self.config.basis_sets
        
        results = []
        
        for molecule in molecules[:5]:  # Limit for convergence studies
            for method in self.config.active_space_methods[:2]:  # Limit methods
                for param_value in parameter_values:
                    try:
                        # Create molecule with current parameter
                        mol = molecule.create_molecule(param_value)
                        
                        # Run calculation
                        from pyscf import scf
                        mf = scf.RHF(mol) if molecule.multiplicity == 1 else scf.UHF(mol)
                        mf.kernel()
                        
                        if mf.converged:
                            # Get reference energy
                            ref_entry = molecule.get_reference_value(PropertyType.ATOMIZATION_ENERGY)
                            ref_energy = ref_entry.value if ref_entry else None
                            
                            # Add convergence point
                            if ref_energy:
                                self.convergence_tracker.add_convergence_point(
                                    system=molecule.name,
                                    method=method.value,
                                    parameter_name=parameter_name,
                                    parameter_value=len(param_value),  # Simplified metric
                                    energy=mf.e_tot,
                                    reference_energy=ref_energy
                                )
                            
                            # Create result
                            result = BenchmarkResult(
                                calculation_id=str(uuid.uuid4()),
                                molecule_name=molecule.name,
                                database_source=molecule.database_id.split('_')[0],
                                active_space_method=method.value,
                                basis_set=param_value,
                                computed_energy=mf.e_tot,
                                reference_energy=ref_energy,
                                energy_error=mf.e_tot - ref_energy if ref_energy else None,
                                converged=mf.converged
                            )
                            results.append(result)
                    
                    except Exception as e:
                        logger.warning(f"Convergence calculation failed: {e}")
                        continue
        
        logger.info(f"Completed convergence study with {len(results)} points")
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmarking study."""
        logger.info(f"Starting comprehensive benchmark: {self.config.study_name}")
        start_time = time.time()
        
        # Prepare dataset
        molecules = self.prepare_molecular_dataset()
        
        # Save configuration
        config_file = self.output_dir / "benchmark_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Run benchmarks based on scope
        all_results = []
        
        if self.config.scope in [BenchmarkScope.ACTIVE_SPACE_SELECTION, BenchmarkScope.COMBINED_WORKFLOW]:
            as_results = self.run_active_space_benchmark(molecules)
            all_results.extend(as_results)
            
            if self.config.save_intermediate_results:
                self._save_results(as_results, "active_space_results.json")
        
        if self.config.scope in [BenchmarkScope.MULTIREFERENCE_METHODS, BenchmarkScope.COMBINED_WORKFLOW]:
            mr_results = self.run_multireference_benchmark(
                molecules, 
                all_results if self.config.scope == BenchmarkScope.COMBINED_WORKFLOW else None
            )
            all_results.extend(mr_results)
            
            if self.config.save_intermediate_results:
                self._save_results(mr_results, "multireference_results.json")
        
        if self.config.scope == BenchmarkScope.CONVERGENCE_STUDY:
            conv_results = self.run_convergence_study(molecules)
            all_results.extend(conv_results)
            
            if self.config.save_intermediate_results:
                self._save_results(conv_results, "convergence_results.json")
        
        # Store results
        self.results = all_results
        
        # Run statistical analysis
        analysis_results = self.analyze_results()
        
        # Save final results
        self._save_final_results(analysis_results)
        
        total_time = time.time() - start_time
        
        summary = {
            "study_name": self.config.study_name,
            "total_molecules": len(molecules),
            "total_calculations": len(all_results),
            "failed_calculations": len(self.failed_calculations),
            "total_time_hours": total_time / 3600,
            "success_rate": len(all_results) / (len(all_results) + len(self.failed_calculations)),
            "output_directory": str(self.output_dir)
        }
        
        logger.info(f"Comprehensive benchmark completed: {summary}")
        return summary
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results with statistical methods."""
        logger.info("Analyzing benchmark results...")
        
        if not self.results:
            logger.warning("No results to analyze")
            return {}
        
        # Convert results to validation results for statistical analysis
        validation_results = []
        for result in self.results:
            if result.energy_error is not None:
                val_result = ValidationResult(
                    system_name=result.molecule_name,
                    method=result.active_space_method or result.multireference_method,
                    computed_value=result.computed_energy,
                    reference_value=result.reference_energy,
                    property_type="energy",
                    energy_error=result.energy_error,
                    converged=result.converged,
                    computational_time=result.wall_time,
                    additional_data={
                        "basis_set": result.basis_set,
                        "n_active_electrons": result.n_active_electrons,
                        "n_active_orbitals": result.n_active_orbitals
                    }
                )
                validation_results.append(val_result)
        
        # Add to statistical analyzer
        self.statistical_analyzer.add_validation_results(validation_results)
        
        # Analyze by method
        methods = list(set(r.active_space_method or r.multireference_method for r in self.results))
        method_analysis = self.statistical_analyzer.compare_method_performance(methods)
        
        # System difficulty analysis
        system_analysis = self.statistical_analyzer.analyze_system_difficulty()
        
        # Create comprehensive analysis
        analysis = {
            "method_performance": method_analysis,
            "system_difficulty": system_analysis,
            "convergence_analysis": self._analyze_convergence(),
            "timing_analysis": self._analyze_timing(),
            "error_analysis": self._analyze_errors()
        }
        
        return analysis
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        converged_count = sum(1 for r in self.results if r.converged)
        total_count = len(self.results)
        
        convergence_by_method = {}
        for result in self.results:
            method = result.active_space_method or result.multireference_method
            if method not in convergence_by_method:
                convergence_by_method[method] = {"converged": 0, "total": 0}
            
            convergence_by_method[method]["total"] += 1
            if result.converged:
                convergence_by_method[method]["converged"] += 1
        
        # Calculate convergence rates
        convergence_rates = {}
        for method, data in convergence_by_method.items():
            convergence_rates[method] = data["converged"] / data["total"] if data["total"] > 0 else 0
        
        return {
            "overall_convergence_rate": converged_count / total_count if total_count > 0 else 0,
            "convergence_by_method": convergence_rates,
            "total_calculations": total_count,
            "converged_calculations": converged_count
        }
    
    def _analyze_timing(self) -> Dict[str, Any]:
        """Analyze computational timing."""
        wall_times = [r.wall_time for r in self.results if r.wall_time is not None]
        
        if not wall_times:
            return {}
        
        wall_times = np.array(wall_times)
        
        timing_by_method = {}
        for result in self.results:
            if result.wall_time is None:
                continue
                
            method = result.active_space_method or result.multireference_method
            if method not in timing_by_method:
                timing_by_method[method] = []
            timing_by_method[method].append(result.wall_time)
        
        # Calculate statistics for each method
        method_timing_stats = {}
        for method, times in timing_by_method.items():
            times = np.array(times)
            method_timing_stats[method] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "median_time": float(np.median(times))
            }
        
        return {
            "overall_timing": {
                "mean_time": float(np.mean(wall_times)),
                "std_time": float(np.std(wall_times)),
                "total_time": float(np.sum(wall_times)),
                "min_time": float(np.min(wall_times)),
                "max_time": float(np.max(wall_times))
            },
            "timing_by_method": method_timing_stats
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [r.absolute_error for r in self.results if r.absolute_error is not None]
        
        if not errors:
            return {}
        
        errors = np.array(errors)
        
        # Error distribution
        error_ranges = {
            "excellent": np.sum(errors < 0.001),  # < 1 mHartree
            "good": np.sum((errors >= 0.001) & (errors < 0.005)),  # 1-5 mHartree
            "acceptable": np.sum((errors >= 0.005) & (errors < 0.02)),  # 5-20 mHartree
            "poor": np.sum(errors >= 0.02)  # > 20 mHartree
        }
        
        return {
            "error_statistics": {
                "mean_absolute_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "median_absolute_error": float(np.median(errors)),
                "max_absolute_error": float(np.max(errors)),
                "min_absolute_error": float(np.min(errors))
            },
            "error_distribution": error_ranges,
            "total_with_errors": len(errors)
        }
    
    def _save_results(self, results: List[BenchmarkResult], filename: str) -> None:
        """Save results to file."""
        output_file = self.output_dir / filename
        
        results_data = [result.to_dict() for result in results]
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _save_final_results(self, analysis: Dict[str, Any]) -> None:
        """Save final comprehensive results."""
        # Save all results
        self._save_results(self.results, "all_results.json")
        
        # Save analysis
        analysis_file = self.output_dir / "statistical_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save failed calculations
        if self.failed_calculations:
            failed_file = self.output_dir / "failed_calculations.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_calculations, f, indent=2, default=str)
        
        # Export in requested formats
        for format_type in self.config.export_formats:
            try:
                self._export_results(format_type)
            except Exception as e:
                logger.warning(f"Failed to export in {format_type} format: {e}")
        
        logger.info(f"Final results saved to {self.output_dir}")
    
    def _export_results(self, format_type: str) -> None:
        """Export results in specified format."""
        if format_type.lower() == "csv":
            # Convert results to DataFrame
            data = []
            for result in self.results:
                row = {
                    "molecule_name": result.molecule_name,
                    "database_source": result.database_source,
                    "active_space_method": result.active_space_method,
                    "multireference_method": result.multireference_method,
                    "basis_set": result.basis_set,
                    "n_active_electrons": result.n_active_electrons,
                    "n_active_orbitals": result.n_active_orbitals,
                    "computed_energy": result.computed_energy,
                    "reference_energy": result.reference_energy,
                    "energy_error": result.energy_error,
                    "absolute_error": result.absolute_error,
                    "relative_error": result.relative_error,
                    "wall_time": result.wall_time,
                    "converged": result.converged,
                    "iterations": result.iterations
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_file = self.output_dir / "benchmark_results.csv"
            df.to_csv(csv_file, index=False)
        
        elif format_type.lower() == "hdf5":
            data = []
            for result in self.results:
                row = {
                    "molecule_name": result.molecule_name,
                    "database_source": result.database_source,
                    "active_space_method": result.active_space_method,
                    "multireference_method": result.multireference_method,
                    "basis_set": result.basis_set,
                    "n_active_electrons": result.n_active_electrons,
                    "n_active_orbitals": result.n_active_orbitals,
                    "computed_energy": result.computed_energy,
                    "reference_energy": result.reference_energy,
                    "energy_error": result.energy_error,
                    "absolute_error": result.absolute_error,
                    "relative_error": result.relative_error,
                    "wall_time": result.wall_time,
                    "converged": result.converged,
                    "iterations": result.iterations
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            hdf5_file = self.output_dir / "benchmark_results.h5"
            df.to_hdf(hdf5_file, key='results', mode='w')
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the benchmark study."""
        if not self.results:
            return "No results available for reporting."
        
        # Calculate key statistics
        total_calculations = len(self.results)
        converged_calculations = sum(1 for r in self.results if r.converged)
        failed_calculations = len(self.failed_calculations)
        
        errors = [r.absolute_error for r in self.results if r.absolute_error is not None]
        mean_error = np.mean(errors) if errors else 0
        
        wall_times = [r.wall_time for r in self.results if r.wall_time is not None]
        total_time_hours = sum(wall_times) / 3600 if wall_times else 0
        
        # Generate report
        report = f"""
# Comprehensive Benchmark Report: {self.config.study_name}

## Summary Statistics
- **Total calculations**: {total_calculations}
- **Converged calculations**: {converged_calculations} ({converged_calculations/total_calculations*100:.1f}%)
- **Failed calculations**: {failed_calculations}
- **Mean absolute error**: {mean_error:.6f} Hartree
- **Total computational time**: {total_time_hours:.2f} hours

## Study Configuration
- **Scope**: {self.config.scope.value}
- **Databases**: {', '.join(self.config.databases)}
- **Active space methods**: {', '.join([m.value for m in self.config.active_space_methods])}
- **Multireference methods**: {', '.join(self.config.multireference_methods)}
- **Basis sets**: {', '.join(self.config.basis_sets)}

## Key Findings
{'- High convergence rate indicates robust computational setup' if converged_calculations/total_calculations > 0.9 else '- Some convergence issues detected'}
{'- Excellent accuracy achieved' if mean_error < 0.005 else '- Moderate accuracy achieved' if mean_error < 0.02 else '- Large errors detected'}
- Results saved to: {self.output_dir}

For detailed analysis, see the statistical analysis files in the output directory.
"""
        
        # Save report
        report_file = self.output_dir / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report