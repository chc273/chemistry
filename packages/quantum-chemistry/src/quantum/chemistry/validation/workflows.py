"""
Production workflow orchestration for quantum chemistry validation.

This module provides high-level workflow orchestration tools for running
comprehensive validation campaigns across multiple methods, systems, and
computational environments.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .benchmarks import BenchmarkSuite
from .comparison import MethodComparator, ValidationResult
from .reference_data import ReferenceDatabase
from .statistical import ConvergenceTracker, StatisticalAnalyzer


@dataclass
class ValidationTask:
    """A single validation task in a workflow."""

    task_id: str
    system_name: str
    method: str
    method_type: str = None  # For external methods
    is_external: bool = False
    priority: int = 1  # 1 = highest, 5 = lowest

    # Task parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Execution status
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results
    result: Optional[ValidationResult] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        """Initialize task timing."""
        if not self.task_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.task_id = f"{self.system_name}_{self.method}_{timestamp}"


@dataclass
class WorkflowConfiguration:
    """Configuration for validation workflows."""

    # Execution settings
    max_parallel_tasks: int = 4
    timeout_seconds: int = 300  # 5 minutes default
    retry_failed_tasks: bool = True
    max_retries: int = 2

    # Resource limits
    max_memory_gb: Optional[int] = None
    max_cpu_cores: Optional[int] = None

    # Output settings
    output_directory: Path = Path("validation_results")
    save_intermediate_results: bool = True
    compression_level: int = 6

    # Logging
    log_level: str = "INFO"
    detailed_logging: bool = False

    # Quality control
    error_threshold_mhartree: float = 10.0  # Fail if error > 10 mH
    skip_known_problematic: bool = True

    def __post_init__(self):
        """Initialize configuration."""
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)


class ValidationWorkflowManager:
    """Manager for coordinating large-scale validation workflows."""

    def __init__(
        self,
        config: Optional[WorkflowConfiguration] = None,
        benchmark_suite: Optional[BenchmarkSuite] = None,
        reference_db: Optional[ReferenceDatabase] = None,
    ):
        """Initialize workflow manager.

        Args:
            config: Workflow configuration
            benchmark_suite: Benchmark systems suite
            reference_db: Reference data database
        """
        self.config = config or WorkflowConfiguration()
        self.benchmark_suite = benchmark_suite or BenchmarkSuite()
        self.reference_db = reference_db or ReferenceDatabase()

        # Initialize components
        self.comparator = MethodComparator(self.benchmark_suite, self.reference_db)
        self.analyzer = StatisticalAnalyzer()
        self.convergence_tracker = ConvergenceTracker()

        # Task management
        self.tasks: List[ValidationTask] = []
        self.completed_tasks: List[ValidationTask] = []
        self.failed_tasks: List[ValidationTask] = []

        # Execution tracking
        self.workflow_start_time: Optional[datetime] = None
        self.workflow_end_time: Optional[datetime] = None
        self.total_execution_time: Optional[float] = None

        # Setup logging
        self._setup_logging()

        self.logger.info(
            f"Initialized ValidationWorkflowManager with {len(self.benchmark_suite.list_systems())} benchmark systems"
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger("ValidationWorkflow")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Create file handler
        log_file = self.config.output_directory / "validation_workflow.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def create_standard_workflow(
        self,
        methods: List[str],
        external_methods: List[str] = None,
        systems: List[str] = None,
        difficulty_levels: List[str] = None,
    ) -> List[ValidationTask]:
        """Create a standard validation workflow.

        Args:
            methods: List of internal methods to validate
            external_methods: List of external methods to validate
            systems: List of system names (if None, uses recommended systems)
            difficulty_levels: System difficulty levels to include

        Returns:
            List of validation tasks
        """
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium"]

        # Get systems to test
        if systems is None:
            benchmark_systems = self.benchmark_suite.get_validation_suite(
                difficulty_levels
            )
            systems = [s.name.lower() for s in benchmark_systems]

        tasks = []
        task_counter = 1

        # Create tasks for internal methods
        for method in methods:
            for system_name in systems:
                task = ValidationTask(
                    task_id=f"internal_{task_counter:04d}",
                    system_name=system_name,
                    method=method,
                    is_external=False,
                    priority=2,  # Medium priority for internal methods
                    parameters={"target_accuracy": "standard"},
                )
                tasks.append(task)
                task_counter += 1

        # Create tasks for external methods
        if external_methods:
            external_suite = self.benchmark_suite.get_external_method_suite(
                external_methods, difficulty_levels
            )

            method_types = {
                "molpro": ["casscf", "caspt2"],
                "orca": ["casscf", "nevpt2"],
                "psi4": ["casscf", "ccsd"],
                "gaussian": ["hf", "mp2"],
            }

            for external_method in external_methods:
                suitable_systems = external_suite.get(external_method, [])
                available_method_types = method_types.get(external_method, ["default"])

                for method_type in available_method_types:
                    for system in suitable_systems:
                        task = ValidationTask(
                            task_id=f"external_{task_counter:04d}",
                            system_name=system.name.lower(),
                            method=external_method,
                            method_type=method_type,
                            is_external=True,
                            priority=1,  # High priority for external methods
                            parameters={"max_iterations": 100},
                        )
                        tasks.append(task)
                        task_counter += 1

        self.logger.info(f"Created standard workflow with {len(tasks)} tasks")
        return tasks

    def create_convergence_workflow(
        self,
        method: str,
        system_name: str,
        parameter_name: str,
        parameter_values: List[float],
        is_external: bool = False,
        method_type: str = None,
    ) -> List[ValidationTask]:
        """Create a convergence study workflow.

        Args:
            method: Method to study
            system_name: System to use for convergence study
            parameter_name: Parameter to vary (e.g., 'cardinal_number', 'active_space_size')
            parameter_values: Values of the parameter to test
            is_external: Whether this is an external method
            method_type: Method type for external methods

        Returns:
            List of validation tasks for convergence study
        """
        tasks = []

        for i, param_value in enumerate(parameter_values):
            task = ValidationTask(
                task_id=f"conv_{method}_{system_name}_{parameter_name}_{i:03d}",
                system_name=system_name,
                method=method,
                method_type=method_type,
                is_external=is_external,
                priority=3,  # Medium-low priority for convergence studies
                parameters={
                    parameter_name: param_value,
                    "convergence_study": True,
                    "reference_value": parameter_values[-1]
                    if i == len(parameter_values) - 1
                    else None,
                },
            )
            tasks.append(task)

        self.logger.info(
            f"Created convergence workflow for {method}/{system_name} with {len(tasks)} points"
        )
        return tasks

    def add_tasks(self, tasks: List[ValidationTask]):
        """Add tasks to the workflow."""
        self.tasks.extend(tasks)
        self.logger.info(
            f"Added {len(tasks)} tasks to workflow (total: {len(self.tasks)})"
        )

    def run_workflow(
        self, progress_callback: Optional[Callable] = None, parallel: bool = True
    ) -> Dict[str, Any]:
        """Run the complete validation workflow.

        Args:
            progress_callback: Optional callback function for progress updates
            parallel: Whether to run tasks in parallel

        Returns:
            Dictionary with workflow results and statistics
        """
        if not self.tasks:
            raise ValueError(
                "No tasks to execute. Use add_tasks() or create_*_workflow() first."
            )

        self.workflow_start_time = datetime.now()
        self.logger.info(f"Starting workflow execution with {len(self.tasks)} tasks")

        if progress_callback:
            progress_callback(0, len(self.tasks), "Starting workflow...")

        try:
            if parallel and self.config.max_parallel_tasks > 1:
                self._run_parallel_workflow(progress_callback)
            else:
                self._run_sequential_workflow(progress_callback)

        except KeyboardInterrupt:
            self.logger.warning("Workflow interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
        finally:
            self.workflow_end_time = datetime.now()
            self.total_execution_time = (
                self.workflow_end_time - self.workflow_start_time
            ).total_seconds()

        # Analyze results
        results = self._analyze_workflow_results()

        # Save results
        self._save_workflow_results(results)

        self.logger.info(
            f"Workflow completed in {self.total_execution_time:.2f} seconds"
        )
        return results

    def _run_sequential_workflow(self, progress_callback: Optional[Callable] = None):
        """Run workflow tasks sequentially."""
        for i, task in enumerate(self.tasks):
            try:
                self._execute_task(task)
                if progress_callback:
                    progress_callback(
                        i + 1, len(self.tasks), f"Completed {task.task_id}"
                    )
            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                task.status = "failed"
                task.error_message = str(e)
                self.failed_tasks.append(task)

    def _run_parallel_workflow(self, progress_callback: Optional[Callable] = None):
        """Run workflow tasks in parallel."""
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_tasks) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_task, task): task for task in self.tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_count += 1

                try:
                    future.result()  # This will raise any exception from the task
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {e}")
                    task.status = "failed"
                    task.error_message = str(e)
                    self.failed_tasks.append(task)

                if progress_callback:
                    progress_callback(
                        completed_count,
                        len(self.tasks),
                        f"Completed {completed_count}/{len(self.tasks)} tasks",
                    )

    def _execute_task(self, task: ValidationTask):
        """Execute a single validation task."""
        task.status = "running"
        task.start_time = datetime.now()

        self.logger.debug(
            f"Executing task {task.task_id}: {task.method} on {task.system_name}"
        )

        try:
            if task.is_external:
                # Run external method validation
                result = self.comparator.validate_external_method(
                    system_name=task.system_name,
                    external_method=task.method,
                    method_type=task.method_type,
                    **task.parameters,
                )
            else:
                # Run internal method validation
                result = self.comparator.validate_method(
                    system_name=task.system_name, method=task.method, **task.parameters
                )

            task.result = result
            task.status = "completed"
            task.end_time = datetime.now()
            task.execution_time = (task.end_time - task.start_time).total_seconds()

            self.completed_tasks.append(task)

            # Check for quality control issues
            error_mhartree = abs(result.energy_error) * 1000
            if error_mhartree > self.config.error_threshold_mhartree:
                self.logger.warning(
                    f"Task {task.task_id} has large error: {error_mhartree:.2f} mH"
                )

            self.logger.debug(
                f"Task {task.task_id} completed successfully in {task.execution_time:.2f}s"
            )

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.end_time = datetime.now()
            task.execution_time = (task.end_time - task.start_time).total_seconds()

            self.failed_tasks.append(task)
            self.logger.error(
                f"Task {task.task_id} failed after {task.execution_time:.2f}s: {e}"
            )
            raise

    def _analyze_workflow_results(self) -> Dict[str, Any]:
        """Analyze workflow results and generate summary statistics."""

        # Collect all validation results
        all_results = [task.result for task in self.completed_tasks if task.result]

        if all_results:
            self.analyzer.add_validation_results(all_results)

        # Workflow statistics
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

        # Execution time statistics
        execution_times = [
            task.execution_time for task in self.completed_tasks if task.execution_time
        ]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )

        # Method performance analysis
        method_performance = {}
        methods = list(set(task.method for task in self.completed_tasks))

        for method in methods:
            method_tasks = [
                task for task in self.completed_tasks if task.method == method
            ]
            method_results = [task.result for task in method_tasks if task.result]

            if method_results:
                errors = [abs(r.energy_error) * 1000 for r in method_results]
                method_performance[method] = {
                    "tasks_completed": len(method_tasks),
                    "mean_error_mhartree": sum(errors) / len(errors),
                    "max_error_mhartree": max(errors),
                    "success_rate": len(method_results) / len(method_tasks),
                }

        # System difficulty analysis
        system_difficulty = {}
        systems = list(set(task.system_name for task in self.completed_tasks))

        for system in systems:
            system_tasks = [
                task for task in self.completed_tasks if task.system_name == system
            ]
            system_results = [task.result for task in system_tasks if task.result]

            if system_results:
                errors = [abs(r.energy_error) * 1000 for r in system_results]
                system_difficulty[system] = {
                    "tasks_completed": len(system_tasks),
                    "mean_error_mhartree": sum(errors) / len(errors),
                    "max_error_mhartree": max(errors),
                    "difficulty_score": sum(errors)
                    / len(errors),  # Simple difficulty metric
                }

        # Convergence analysis
        convergence_studies = {}
        convergence_tasks = [
            task
            for task in self.completed_tasks
            if task.result and task.parameters.get("convergence_study")
        ]

        # Group convergence tasks by method/system
        for task in convergence_tasks:
            key = f"{task.method}_{task.system_name}"
            if key not in convergence_studies:
                convergence_studies[key] = []
            convergence_studies[key].append(task)

        # Analyze each convergence study
        convergence_analyses = {}
        for key, tasks in convergence_studies.items():
            if len(tasks) >= 3:  # Need at least 3 points for convergence analysis
                method, system = key.split("_", 1)

                # Add to convergence tracker
                for task in tasks:
                    param_name = None
                    param_value = None
                    for param, value in task.parameters.items():
                        if param not in ["convergence_study", "reference_value"]:
                            param_name = param
                            param_value = value
                            break

                    if param_name and param_value and task.result:
                        self.convergence_tracker.add_convergence_point(
                            system=system,
                            method=method,
                            parameter_name=param_name,
                            parameter_value=param_value,
                            energy=task.result.calculated_energy,
                        )

                # Analyze convergence
                if param_name:
                    convergence_analysis = self.convergence_tracker.analyze_convergence(
                        system, method, param_name
                    )
                    if convergence_analysis:
                        convergence_analyses[key] = {
                            "converged_value": convergence_analysis.converged_value,
                            "convergence_rate": convergence_analysis.convergence_rate,
                            "is_converged": convergence_analysis.is_converged,
                            "fit_quality": convergence_analysis.fit_quality,
                        }

        return {
            "workflow_summary": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": success_rate,
                "total_execution_time": self.total_execution_time,
                "average_task_time": avg_execution_time,
                "start_time": self.workflow_start_time.isoformat(),
                "end_time": self.workflow_end_time.isoformat(),
            },
            "method_performance": method_performance,
            "system_difficulty": system_difficulty,
            "convergence_analyses": convergence_analyses,
            "failed_tasks": [
                {
                    "task_id": task.task_id,
                    "method": task.method,
                    "system": task.system_name,
                    "error": task.error_message,
                }
                for task in self.failed_tasks
            ],
            "validation_results": [
                {
                    "task_id": task.task_id,
                    "method": task.result.method,
                    "system": task.result.system_name,
                    "energy_error_mhartree": task.result.energy_error * 1000,
                    "error_magnitude": task.result.error_magnitude,
                    "execution_time": task.execution_time,
                }
                for task in self.completed_tasks
                if task.result
            ],
        }

    def _save_workflow_results(self, results: Dict[str, Any]):
        """Save workflow results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        results_file = (
            self.config.output_directory / f"workflow_results_{timestamp}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save detailed validation results
        if self.completed_tasks:
            detailed_results = []
            for task in self.completed_tasks:
                if task.result:
                    detailed_results.append(
                        {
                            "task_id": task.task_id,
                            "system_name": task.result.system_name,
                            "method": task.result.method,
                            "basis_set": task.result.basis_set,
                            "calculated_energy": task.result.calculated_energy,
                            "reference_energy": task.result.reference_energy,
                            "energy_error": task.result.energy_error,
                            "relative_error": task.result.relative_error,
                            "absolute_error": task.result.absolute_error,
                            "error_magnitude": task.result.error_magnitude,
                            "calculation_time": task.result.calculation_time,
                            "execution_time": task.execution_time,
                            "parameters": task.parameters,
                        }
                    )

            detailed_file = (
                self.config.output_directory / f"detailed_results_{timestamp}.json"
            )
            with open(detailed_file, "w") as f:
                json.dump(detailed_results, f, indent=2, default=str)

        # Export statistical analysis
        if self.analyzer.validation_results:
            stats_file = (
                self.config.output_directory / f"statistical_analysis_{timestamp}.json"
            )
            self.analyzer.export_statistical_report(stats_file)

        self.logger.info(f"Workflow results saved to {self.config.output_directory}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        running_tasks = len([t for t in self.tasks if t.status == "running"])
        pending_tasks = len([t for t in self.tasks if t.status == "pending"])

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "pending_tasks": pending_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "is_running": self.workflow_start_time is not None
            and self.workflow_end_time is None,
        }

    def retry_failed_tasks(self) -> Dict[str, Any]:
        """Retry failed tasks."""
        if not self.failed_tasks:
            self.logger.info("No failed tasks to retry")
            return {"retried_tasks": 0, "newly_completed": 0}

        self.logger.info(f"Retrying {len(self.failed_tasks)} failed tasks")

        # Reset failed tasks
        tasks_to_retry = self.failed_tasks.copy()
        self.failed_tasks.clear()

        # Reset task status
        for task in tasks_to_retry:
            task.status = "pending"
            task.error_message = None
            task.start_time = None
            task.end_time = None
            task.execution_time = None

        # Add back to task queue
        self.tasks = tasks_to_retry

        # Re-run workflow
        self.run_workflow(parallel=True)

        return {
            "retried_tasks": len(tasks_to_retry),
            "newly_completed": len(self.completed_tasks),
            "still_failed": len(self.failed_tasks),
        }
