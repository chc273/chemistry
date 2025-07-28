#!/usr/bin/env python3
"""
Production Workflow Orchestration Example

This example demonstrates the production-ready workflow orchestration system
for large-scale quantum chemistry method validation campaigns.
"""

import numpy as np
from pathlib import Path
import time
import asyncio

# Import the validation framework with workflow orchestration
from quantum.chemistry.validation import (
    BenchmarkSuite, ReferenceDatabase, MethodComparator,
    StatisticalAnalyzer, ConvergenceTracker,
    ValidationTask, WorkflowConfiguration, ValidationWorkflowManager
)
from quantum.chemistry.external import ExternalMethodRunner


def progress_callback(completed: int, total: int, message: str):
    """Progress callback for workflow execution."""
    percentage = (completed / total) * 100 if total > 0 else 0
    bar_length = 40
    filled_length = int(bar_length * completed // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r🚀 Progress: |{bar}| {percentage:.1f}% ({completed}/{total}) - {message}', end='', flush=True)
    
    if completed == total:
        print()  # New line when complete


def main():
    """Main production workflow orchestration example."""
    print("=" * 80)
    print("Production Quantum Chemistry Validation Workflow")
    print("=" * 80)
    
    # 1. Initialize workflow configuration
    print("\n1. Setting up production workflow configuration...")
    
    config = WorkflowConfiguration(
        max_parallel_tasks=6,  # Run up to 6 tasks in parallel
        timeout_seconds=600,   # 10 minute timeout per task
        retry_failed_tasks=True,
        max_retries=2,
        output_directory=Path("production_results"),
        save_intermediate_results=True,
        log_level="INFO",
        error_threshold_mhartree=5.0,  # Flag errors > 5 mH as concerning
        skip_known_problematic=True
    )
    
    print(f"  ✅ Configured for {config.max_parallel_tasks} parallel tasks")
    print(f"  📁 Output directory: {config.output_directory}")
    print(f"  ⏱️  Task timeout: {config.timeout_seconds}s")
    print(f"  🎯 Error threshold: {config.error_threshold_mhartree} mHartree")
    
    # 2. Initialize workflow manager
    print("\n2. Initializing workflow management system...")
    
    workflow_manager = ValidationWorkflowManager(config=config)
    
    print(f"  📊 Loaded {len(workflow_manager.benchmark_suite.list_systems())} benchmark systems")
    print(f"  📚 Reference database with {len(workflow_manager.reference_db.entries)} entries")
    
    # 3. Create standard validation workflow
    print("\n3. Creating standard validation workflow...")
    
    # Define methods to validate
    internal_methods = ['casscf', 'nevpt2']  # Start with core methods
    external_methods = []  # Will check availability
    
    # Check for available external methods
    external_runner = ExternalMethodRunner()
    potential_external_methods = ['molpro', 'orca', 'psi4']
    
    for method in potential_external_methods:
        if external_runner.is_method_available(method):
            external_methods.append(method)
            print(f"  🐳 External method available: {method.upper()}")
        else:
            print(f"  ❌ External method not available: {method.upper()}")
    
    # Create standard workflow
    standard_tasks = workflow_manager.create_standard_workflow(
        methods=internal_methods,
        external_methods=external_methods[:2] if external_methods else None,  # Limit to 2 for demo
        difficulty_levels=['easy', 'medium']
    )
    
    print(f"  📋 Created {len(standard_tasks)} standard validation tasks")
    
    # 4. Create convergence study workflows
    print("\n4. Creating convergence study workflows...")
    
    # Basis set convergence study for H2 + CASSCF
    convergence_tasks_basis = workflow_manager.create_convergence_workflow(
        method='casscf',
        system_name='h2',
        parameter_name='cardinal_number',
        parameter_values=[2, 3, 4, 5],  # cc-pVDZ, cc-pVTZ, cc-pVQZ, cc-pV5Z
        is_external=False
    )
    
    print(f"  📈 Basis set convergence study: {len(convergence_tasks_basis)} tasks")
    
    # Active space convergence study for H2O + CASSCF
    convergence_tasks_active = workflow_manager.create_convergence_workflow(
        method='casscf',
        system_name='h2o',
        parameter_name='active_orbitals',
        parameter_values=[4, 6, 8, 10],  # Different active space sizes
        is_external=False
    )
    
    print(f"  🎯 Active space convergence study: {len(convergence_tasks_active)} tasks")
    
    # External method convergence study (if available)
    external_convergence_tasks = []
    if external_methods:
        external_convergence_tasks = workflow_manager.create_convergence_workflow(
            method=external_methods[0],
            system_name='h2',
            parameter_name='basis_cardinal',
            parameter_values=[2, 3, 4],
            is_external=True,
            method_type='casscf'
        )
        print(f"  🐳 External method convergence: {len(external_convergence_tasks)} tasks")
    
    # 5. Add all tasks to workflow
    print("\n5. Assembling complete workflow...")
    
    all_tasks = standard_tasks + convergence_tasks_basis + convergence_tasks_active + external_convergence_tasks
    workflow_manager.add_tasks(all_tasks)
    
    total_tasks = len(all_tasks)
    standard_count = len(standard_tasks)
    convergence_count = len(convergence_tasks_basis) + len(convergence_tasks_active) + len(external_convergence_tasks)
    
    print(f"  📊 Total workflow: {total_tasks} tasks")
    print(f"    - Standard validation: {standard_count} tasks")
    print(f"    - Convergence studies: {convergence_count} tasks")
    print(f"    - Internal methods: {len([t for t in all_tasks if not t.is_external])} tasks")
    print(f"    - External methods: {len([t for t in all_tasks if t.is_external])} tasks")
    
    # 6. Display workflow preview
    print("\n6. Workflow execution preview...")
    
    # Group tasks by priority
    priority_groups = {}
    for task in all_tasks:
        if task.priority not in priority_groups:
            priority_groups[task.priority] = []
        priority_groups[task.priority].append(task)
    
    priority_names = {1: "High", 2: "Medium", 3: "Medium-Low", 4: "Low", 5: "Very Low"}
    
    for priority in sorted(priority_groups.keys()):
        tasks = priority_groups[priority]
        print(f"  🎖️  Priority {priority} ({priority_names.get(priority, 'Unknown')}): {len(tasks)} tasks")
        
        # Show sample tasks
        sample_size = min(3, len(tasks))
        for i, task in enumerate(tasks[:sample_size]):
            method_info = f"{task.method}"
            if task.method_type:
                method_info += f"/{task.method_type}"
            if task.is_external:
                method_info += " (external)"
                
            print(f"    • {task.system_name} + {method_info}")
            
        if len(tasks) > sample_size:
            print(f"    ... and {len(tasks) - sample_size} more")
    
    # 7. Execute workflow
    print(f"\n7. Executing production workflow...")
    print(f"  🚀 Starting parallel execution with {config.max_parallel_tasks} workers")
    print(f"  ⏱️  Estimated time: {total_tasks * 15 / config.max_parallel_tasks / 60:.1f} minutes")
    
    start_time = time.time()
    
    try:
        # For demo purposes, we'll simulate execution if no external methods are available
        if not external_methods and total_tasks > 20:
            print("\n  🎭 Simulating workflow execution (no external methods available)...")
            results = simulate_workflow_execution(workflow_manager, progress_callback)
        else:
            # Run actual workflow
            results = workflow_manager.run_workflow(
                progress_callback=progress_callback,
                parallel=True
            )
        
        execution_time = time.time() - start_time
        
    except KeyboardInterrupt:
        print(f"\n\n⛔ Workflow interrupted by user after {time.time() - start_time:.1f}s")
        return
    except Exception as e:
        print(f"\n\n❌ Workflow execution failed: {e}")
        print("This may be due to missing dependencies or configuration issues.")
        return
    
    # 8. Analyze results
    print(f"\n8. Analyzing workflow results...")
    
    workflow_summary = results['workflow_summary']
    method_performance = results['method_performance']
    system_difficulty = results['system_difficulty']
    convergence_analyses = results['convergence_analyses']
    
    print(f"\n📊 Execution Summary:")
    print(f"  • Total execution time: {workflow_summary['total_execution_time']:.1f}s")
    print(f"  • Tasks completed: {workflow_summary['completed_tasks']}/{workflow_summary['total_tasks']}")
    print(f"  • Success rate: {workflow_summary['success_rate']:.1%}")
    print(f"  • Average task time: {workflow_summary['average_task_time']:.1f}s")
    print(f"  • Failed tasks: {workflow_summary['failed_tasks']}")
    
    # 9. Method performance analysis
    if method_performance:
        print(f"\n🎯 Method Performance Analysis:")
        
        # Sort methods by accuracy
        sorted_methods = sorted(method_performance.items(), 
                              key=lambda x: x[1]['mean_error_mhartree'])
        
        for method, stats in sorted_methods[:5]:  # Show top 5 methods
            print(f"  🏆 {method.upper():>12}: "
                  f"MAE = {stats['mean_error_mhartree']:>6.2f} mH, "
                  f"MAX = {stats['max_error_mhartree']:>6.2f} mH, "
                  f"Success = {stats['success_rate']:>6.1%}")
    
    # 10. System difficulty analysis
    if system_difficulty:
        print(f"\n🎲 System Difficulty Analysis:")
        
        # Sort systems by difficulty
        sorted_systems = sorted(system_difficulty.items(),
                              key=lambda x: x[1]['difficulty_score'], reverse=True)
        
        for system, stats in sorted_systems:
            difficulty_level = "Hard" if stats['difficulty_score'] > 2.0 else \
                             "Medium" if stats['difficulty_score'] > 1.0 else "Easy"
            
            print(f"  🔬 {system.upper():>8}: "
                  f"Difficulty = {stats['difficulty_score']:>5.2f} ({difficulty_level}), "
                  f"Mean error = {stats['mean_error_mhartree']:>5.1f} mH")
    
    # 11. Convergence analysis
    if convergence_analyses:
        print(f"\n📈 Convergence Analysis:")
        
        for study_key, analysis in convergence_analyses.items():
            method, system = study_key.split('_', 1)
            convergence_status = "✅ Converged" if analysis['is_converged'] else "⚠️  Not converged"
            
            print(f"  {convergence_status} - {method.upper()}/{system.upper()}:")
            print(f"    • Converged value: {analysis['converged_value']:.6f} Hartree")
            print(f"    • Convergence rate: {analysis['convergence_rate']:.3f}")
            print(f"    • Fit quality (R²): {analysis['fit_quality']:.3f}")
    
    # 12. Quality control and recommendations
    print(f"\n🔍 Quality Control Analysis:")
    
    high_error_results = [r for r in results['validation_results'] 
                         if abs(r['energy_error_mhartree']) > config.error_threshold_mhartree]
    
    if high_error_results:
        print(f"  ⚠️  {len(high_error_results)} results exceed error threshold ({config.error_threshold_mhartree} mH):")
        for result in high_error_results[:3]:  # Show worst 3
            print(f"    • {result['method']}/{result['system']}: {result['energy_error_mhartree']:.1f} mH")
        
        if len(high_error_results) > 3:
            print(f"    ... and {len(high_error_results) - 3} more")
    else:
        print(f"  ✅ All results within acceptable error threshold")
    
    # 13. Export and archival
    print(f"\n13. Results export and archival...")
    
    # Get workflow status
    status = workflow_manager.get_workflow_status()
    
    print(f"  💾 Results exported to: {config.output_directory}")
    print(f"  📄 Generated files:")
    
    # List expected output files
    expected_files = [
        "workflow_results_*.json",
        "detailed_results_*.json", 
        "statistical_analysis_*.json",
        "validation_workflow.log"
    ]
    
    for file_pattern in expected_files:
        print(f"    • {file_pattern}")
    
    # 14. Recommendations for production use
    print(f"\n14. Production deployment recommendations...")
    
    if workflow_summary['success_rate'] > 0.9:
        print(f"  ✅ Excellent success rate - workflow is production-ready")
    elif workflow_summary['success_rate'] > 0.8:
        print(f"  ⚠️  Good success rate - consider investigating failed tasks")
    else:
        print(f"  🚨 Low success rate - workflow needs optimization")
    
    print(f"\n💡 Optimization suggestions:")
    
    if workflow_summary['average_task_time'] > 60:
        print(f"  • Consider using lower accuracy settings for faster execution")
    
    if workflow_summary['failed_tasks'] > 0:
        print(f"  • Review failed tasks and adjust error handling")
        print(f"  • Consider increasing timeout for complex calculations")
    
    if external_methods:
        print(f"  • External methods integration is active - ensure Docker containers are optimized")
    else:
        print(f"  • Consider setting up external methods for comprehensive validation")
    
    # 15. Workflow retry demonstration
    if workflow_summary['failed_tasks'] > 0:
        print(f"\n15. Demonstrating automatic retry capability...")
        
        print(f"  🔄 Retrying {workflow_summary['failed_tasks']} failed tasks...")
        
        try:
            retry_results = workflow_manager.retry_failed_tasks()
            
            print(f"  📊 Retry results:")
            print(f"    • Tasks retried: {retry_results['retried_tasks']}")
            print(f"    • Newly completed: {retry_results['newly_completed']}")
            print(f"    • Still failed: {retry_results['still_failed']}")
            
        except Exception as e:
            print(f"  ❌ Retry failed: {e}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎯 Production Workflow Summary")
    print(f"=" * 80)
    
    total_validations = workflow_summary['completed_tasks']
    total_methods = len(method_performance) if method_performance else 0
    total_systems = len(system_difficulty) if system_difficulty else 0
    
    print(f"\n🚀 Workflow Execution Completed Successfully!")
    print(f"  • Total validations: {total_validations}")
    print(f"  • Methods tested: {total_methods}")
    print(f"  • Systems tested: {total_systems}")
    print(f"  • Convergence studies: {len(convergence_analyses)}")
    print(f"  • Overall success rate: {workflow_summary['success_rate']:.1%}")
    print(f"  • Total execution time: {workflow_summary['total_execution_time']:.1f}s")
    
    if method_performance:
        best_method = min(method_performance.items(), key=lambda x: x[1]['mean_error_mhartree'])
        print(f"  • Best performing method: {best_method[0]} ({best_method[1]['mean_error_mhartree']:.2f} mH MAE)")
    
    print(f"\n📁 Results archived in: {config.output_directory.absolute()}")
    print(f"🎉 Production validation workflow is ready for deployment!")


def simulate_workflow_execution(workflow_manager, progress_callback):
    """Simulate workflow execution for demonstration purposes."""
    import random
    from datetime import datetime, timedelta
    
    total_tasks = len(workflow_manager.tasks)
    
    # Simulate task execution
    for i, task in enumerate(workflow_manager.tasks):
        # Simulate execution time
        time.sleep(random.uniform(0.1, 0.3))  # Quick simulation
        
        # Simulate task completion with realistic results
        task.status = "completed"
        task.start_time = workflow_manager.workflow_start_time or datetime.now()
        task.end_time = task.start_time + timedelta(seconds=random.uniform(5, 30))
        task.execution_time = (task.end_time - task.start_time).total_seconds()
        
        # Create mock validation result
        system = workflow_manager.benchmark_suite.get_system(task.system_name)
        if system:
            ref_energy = system.reference_energies.get('fci', -100.0)
            # Add realistic error
            error = random.normalvariate(0, 0.002)  # 2 mH standard deviation
            calc_energy = ref_energy + error
            
            # Get reference entry
            reference_entry = workflow_manager.reference_db.get_reference_energy(
                task.system_name, 'fci', system.basis_set
            )
            
            if reference_entry:
                from quantum.chemistry.validation.comparison import ValidationResult
                
                task.result = ValidationResult(
                    system_name=task.system_name,
                    method=f"{task.method}_{task.method_type}" if task.method_type else task.method,
                    basis_set=system.basis_set,
                    calculated_energy=calc_energy,
                    reference_energy=reference_entry.energy,
                    energy_error=calc_energy - reference_entry.energy,
                    relative_error=(calc_energy - reference_entry.energy) / abs(reference_entry.energy),
                    reference_source=reference_entry.source,
                    reference_uncertainty=reference_entry.uncertainty,
                    reference_quality=reference_entry.quality_level,
                    calculation_time=task.execution_time
                )
                
                workflow_manager.completed_tasks.append(task)
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            task.status = "failed"
            task.error_message = "Simulated calculation failure"
            workflow_manager.failed_tasks.append(task)
        
        # Update progress
        progress_callback(i + 1, total_tasks, f"Completed {task.task_id}")
    
    # Set workflow timing
    from datetime import datetime, timedelta
    workflow_manager.workflow_start_time = datetime.now() - timedelta(seconds=total_tasks * 0.2)
    workflow_manager.workflow_end_time = datetime.now()
    workflow_manager.total_execution_time = (workflow_manager.workflow_end_time - workflow_manager.workflow_start_time).total_seconds()
    
    # Return simulated results
    return workflow_manager._analyze_workflow_results()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Production workflow demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Production workflow demo failed with error: {e}")
        print("This may be due to missing dependencies or configuration issues.")
        print("Please ensure all validation framework components are properly installed.")