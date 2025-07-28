#!/usr/bin/env python3
"""
Cross-Method Validation Example

This example demonstrates the comprehensive validation framework for
quantum chemistry methods, including cross-method comparisons,
statistical analysis, and convergence tracking.
"""

import numpy as np
from pathlib import Path

# Import the validation framework
from quantum.chemistry.validation import (
    BenchmarkSuite, ReferenceDatabase,
    MethodComparator, StatisticalAnalyzer, ConvergenceTracker
)
from quantum.chemistry.multireference.base import MultireferenceMethodType


def main():
    """Main validation example."""
    print("=" * 60)
    print("Quantum Chemistry Cross-Method Validation Framework")
    print("=" * 60)
    
    # Initialize the validation framework
    print("\n1. Initializing validation framework...")
    benchmark_suite = BenchmarkSuite()
    reference_db = ReferenceDatabase()
    comparator = MethodComparator(benchmark_suite, reference_db)
    analyzer = StatisticalAnalyzer()
    
    # Display available benchmark systems
    print(f"\nAvailable benchmark systems: {len(benchmark_suite.list_systems())}")
    for system_name in benchmark_suite.list_systems():
        system = benchmark_suite.get_system(system_name)
        print(f"  • {system.name}: {system.description} ({system.difficulty_level})")
    
    # Display reference database statistics
    print(f"\nReference database statistics:")
    db_stats = reference_db.get_statistics()
    print(f"  • Total entries: {db_stats['total_entries']}")
    print(f"  • Unique systems: {db_stats['unique_systems']}")
    print(f"  • Unique methods: {db_stats['unique_methods']}")
    print(f"  • Verification rate: {db_stats['verification_rate']:.1%}")
    
    # Test 1: Single method validation
    print("\n" + "=" * 60)
    print("2. Single Method Validation")
    print("=" * 60)
    
    try:
        print("\nValidating CASSCF method on H2 system...")
        h2_result = comparator.validate_method(
            system_name='h2',
            method='casscf',
            target_accuracy='standard'
        )
        
        print(f"Results:")
        print(f"  • Calculated energy: {h2_result.calculated_energy:.6f} Hartree")
        print(f"  • Reference energy:  {h2_result.reference_energy:.6f} Hartree")
        print(f"  • Energy error:      {h2_result.energy_error*1000:.2f} mHartree")
        print(f"  • Error magnitude:   {h2_result.error_magnitude}")
        print(f"  • Reference source:  {h2_result.reference_source}")
        
        analyzer.add_validation_results([h2_result])
        
    except Exception as e:
        print(f"Single method validation failed: {e}")
        print("This may be due to missing external method dependencies.")
    
    # Test 2: Method comparison
    print("\n" + "=" * 60)
    print("3. Method Comparison")
    print("=" * 60)
    
    try:
        print("\nComparing CASSCF vs NEVPT2 on H2O system...")
        h2o_comparison = comparator.compare_methods(
            system_name='h2o',
            method1='casscf',
            method2='nevpt2',
            target_accuracy='standard'
        )
        
        print(f"Results:")
        print(f"  • CASSCF energy:     {h2o_comparison.energy1:.6f} Hartree")
        print(f"  • NEVPT2 energy:     {h2o_comparison.energy2:.6f} Hartree")
        print(f"  • Energy difference: {h2o_comparison.energy_difference*1000:.2f} mHartree")
        print(f"  • Agreement level:   {h2o_comparison.agreement_level}")
        
        analyzer.add_method_comparisons([h2o_comparison])
        
    except Exception as e:
        print(f"Method comparison failed: {e}")
        print("This may be due to missing external method dependencies.")
    
    # Test 3: Validation suite (limited to available methods)
    print("\n" + "=" * 60)
    print("4. Validation Suite")
    print("=" * 60)
    
    # Use only core methods that should always be available
    available_methods = ['casscf', 'nevpt2']
    easy_systems = ['h2', 'lih']  # Use only the easiest systems
    
    print(f"\nRunning validation suite on {len(easy_systems)} systems with {len(available_methods)} methods...")
    
    validation_results = {}
    for method in available_methods:
        method_results = []
        print(f"\nValidating {method.upper()}:")
        
        for system_name in easy_systems:
            try:
                result = comparator.validate_method(
                    system_name=system_name,
                    method=method,
                    target_accuracy='low'  # Use low accuracy for speed
                )
                method_results.append(result)
                
                print(f"  • {system_name:>4}: {result.error_magnitude:>10} "
                      f"(error: {result.energy_error*1000:>6.2f} mH)")
                
            except Exception as e:
                print(f"  • {system_name:>4}: {'FAILED':>10} ({str(e)[:30]}...)")
        
        if method_results:  # Only add if we have successful results
            validation_results[method] = method_results
            analyzer.add_validation_results(method_results)
    
    # Test 4: Statistical analysis
    print("\n" + "=" * 60)
    print("5. Statistical Analysis")
    print("=" * 60)
    
    if analyzer.validation_results:
        print(f"\nAnalyzing {len(analyzer.validation_results)} validation results...")
        
        # Analyze each method
        unique_methods = list(set(r.method for r in analyzer.validation_results))
        for method in unique_methods:
            print(f"\n{method.upper()} Statistics:")
            stats = analyzer.analyze_method_accuracy(method)
            
            if stats:
                print(f"  • Systems tested: {stats['n_systems']}")
                print(f"  • Mean absolute error: {stats['mean_absolute_error']*1000:.2f} mHartree")
                print(f"  • RMS error: {stats['rms_error']*1000:.2f} mHartree")
                print(f"  • Max absolute error: {stats['max_absolute_error']*1000:.2f} mHartree")
                
                # Error distribution
                error_dist = stats['error_distribution']
                total = sum(error_dist.values())
                if total > 0:
                    print(f"  • Error distribution:")
                    for level, count in error_dist.items():
                        percentage = count / total * 100
                        print(f"    - {level:>10}: {count:>2} ({percentage:>5.1f}%)")
        
        # Compare methods if we have multiple
        if len(unique_methods) > 1:
            print(f"\nMethod Performance Comparison:")
            comparison = analyzer.compare_method_performance(unique_methods)
            
            ranking = comparison['ranking']
            print(f"  Method ranking (by accuracy):")
            for i, entry in enumerate(ranking[:5]):  # Show top 5
                print(f"  {i+1}. {entry['method']:>8}: "
                      f"MAE = {entry['mean_absolute_error']*1000:>6.2f} mH, "
                      f"RMS = {entry['rms_error']*1000:>6.2f} mH")
    
    else:
        print("\nNo validation results available for statistical analysis.")
        print("This may be due to missing external method dependencies.")
    
    # Test 5: System difficulty analysis
    if analyzer.validation_results:
        print(f"\nSystem Difficulty Analysis:")
        difficulty = analyzer.analyze_system_difficulty()
        
        ranking = difficulty['difficulty_ranking']
        print(f"  System difficulty ranking:")
        for entry in ranking:
            print(f"  • {entry['system']:>6}: difficulty = {entry['difficulty_score']:>6.4f}, "
                  f"mean error = {entry['mean_error']*1000:>5.1f} mH")
    
    # Test 6: Convergence tracking example
    print("\n" + "=" * 60)
    print("6. Convergence Tracking Example")
    print("=" * 60)
    
    print("\nDemonstrating convergence tracking with simulated data...")
    tracker = ConvergenceTracker()
    
    # Simulate basis set convergence for H2 + CASSCF
    reference_energy = -1.13126  # Estimated converged value
    basis_sizes = [2, 3, 4, 5, 6]  # Cardinal numbers
    
    print(f"Simulating basis set convergence:")
    for cardinal in basis_sizes:
        # Simulate exponential convergence: E(X) = E_∞ + A * exp(-α * X)
        simulated_energy = reference_energy + 0.02 * np.exp(-0.8 * cardinal)
        
        tracker.add_convergence_point(
            system='h2',
            method='casscf',
            parameter_name='cardinal_number',
            parameter_value=cardinal,
            energy=simulated_energy,
            reference_energy=reference_energy
        )
        
        error_mh = (simulated_energy - reference_energy) * 1000
        print(f"  • cc-pV{cardinal}Z: E = {simulated_energy:.6f} H, "
              f"error = {error_mh:>6.2f} mH")
    
    # Analyze convergence
    convergence = tracker.analyze_convergence('h2', 'casscf', 'cardinal_number')
    if convergence:
        print(f"\nConvergence Analysis:")
        print(f"  • Fitted converged value: {convergence.converged_value:.6f} Hartree")
        print(f"  • Convergence rate: {convergence.convergence_rate:.3f}")
        print(f"  • Fit quality (R²): {convergence.fit_quality:.3f}")
        print(f"  • Is converged: {convergence.is_converged}")
    
    # Test 7: Export results
    print("\n" + "=" * 60)
    print("7. Exporting Results")
    print("=" * 60)
    
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Export validation results
    if comparator.validation_results or comparator.method_comparisons:
        validation_file = output_dir / "validation_results.json"
        comparator.export_results(validation_file)
        print(f"✓ Validation results exported to: {validation_file}")
    
    # Export statistical analysis
    if analyzer.validation_results:
        stats_file = output_dir / "statistical_analysis.json"
        analyzer.export_statistical_report(stats_file)
        print(f"✓ Statistical analysis exported to: {stats_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("8. Summary")
    print("=" * 60)
    
    total_validations = len(analyzer.validation_results)
    total_comparisons = len(analyzer.method_comparisons)
    
    print(f"\nValidation Framework Demo Completed!")
    print(f"  • Total validations performed: {total_validations}")
    print(f"  • Total method comparisons: {total_comparisons}")
    print(f"  • Convergence analyses: 1 (simulated)")
    
    if total_validations > 0:
        unique_methods = len(set(r.method for r in analyzer.validation_results))
        unique_systems = len(set(r.system_name for r in analyzer.validation_results))
        print(f"  • Unique methods tested: {unique_methods}")
        print(f"  • Unique systems tested: {unique_systems}")
        
        # Overall accuracy assessment
        all_errors = [abs(r.energy_error) * 1000 for r in analyzer.validation_results]
        if all_errors:
            mean_error = np.mean(all_errors)
            max_error = np.max(all_errors)
            print(f"  • Overall mean absolute error: {mean_error:.2f} mHartree")
            print(f"  • Overall max absolute error: {max_error:.2f} mHartree")
    
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nThe validation framework is ready for production use!")
    print("See the exported JSON files for detailed results and statistics.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation demo interrupted by user.")
    except Exception as e:
        print(f"\n\nValidation demo failed with error: {e}")
        print("This may be due to missing dependencies or configuration issues.")
        print("Please check that the quantum chemistry package is properly installed.")