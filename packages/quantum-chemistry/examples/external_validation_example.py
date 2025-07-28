#!/usr/bin/env python3
"""
External Methods Validation Example

This example demonstrates the extended validation framework for
external quantum chemistry packages using Docker containerization.
"""

import numpy as np
from pathlib import Path

# Import the validation framework
from quantum.chemistry.validation import (
    BenchmarkSuite, ReferenceDatabase,
    MethodComparator, StatisticalAnalyzer, ConvergenceTracker
)
from quantum.chemistry.external import ExternalMethodRunner


def main():
    """Main external validation example."""
    print("=" * 70)
    print("External Quantum Chemistry Methods Validation Framework")
    print("=" * 70)
    
    # Initialize the validation framework
    print("\n1. Initializing validation framework with external methods support...")
    benchmark_suite = BenchmarkSuite()
    reference_db = ReferenceDatabase()
    comparator = MethodComparator(benchmark_suite, reference_db)
    analyzer = StatisticalAnalyzer()
    external_runner = ExternalMethodRunner()
    
    # Check available external methods
    print(f"\n📦 Checking available external quantum chemistry packages...")
    available_external_methods = []
    potential_methods = ['molpro', 'orca', 'psi4', 'gaussian', 'nwchem']
    
    for method in potential_methods:
        if external_runner.is_method_available(method):
            available_external_methods.append(method)
            print(f"  ✅ {method.upper()}: Available")
        else:
            print(f"  ❌ {method.upper()}: Not available")
    
    if not available_external_methods:
        print("\n⚠️  No external methods available. Please ensure Docker containers are set up.")
        print("This example will demonstrate the framework with simulated results.")
        simulate_results = True
    else:
        simulate_results = False
    
    # Get method-specific benchmark systems
    print(f"\n2. Getting method-specific benchmark systems...")
    if available_external_methods:
        external_suite = benchmark_suite.get_external_method_suite(
            available_external_methods, 
            difficulty_levels=['easy', 'medium']
        )
        
        print(f"\nRecommended systems for each method:")
        for method, systems in external_suite.items():
            system_names = [s.name for s in systems]
            print(f"  • {method.upper()}: {', '.join(system_names)}")
    
    # Test 1: External method compatibility check
    print("\n" + "=" * 70)
    print("3. External Method Compatibility Analysis")
    print("=" * 70)
    
    for method in available_external_methods[:2]:  # Check first 2 methods
        print(f"\n🔍 Analyzing compatibility for {method.upper()}...")
        compatibility = benchmark_suite.validate_external_method_compatibility(method)
        
        print(f"  • Available: {compatibility['available']}")
        print(f"  • Compatible systems: {len(compatibility['compatible_systems'])}")
        print(f"  • Recommended systems: {', '.join(compatibility['recommended_systems'])}")
        
        if compatibility['incompatible_systems']:
            print(f"  • Incompatible systems: {', '.join(compatibility['incompatible_systems'])}")
    
    # Test 2: Single external method validation
    print("\n" + "=" * 70)
    print("4. Single External Method Validation")
    print("=" * 70)
    
    if not simulate_results and available_external_methods:
        # Use the first available external method
        test_method = available_external_methods[0]
        method_types = {
            'molpro': 'casscf',
            'orca': 'casscf',
            'psi4': 'casscf',
            'gaussian': 'hf'
        }
        test_method_type = method_types.get(test_method, 'hf')
        
        try:
            print(f"\n🧪 Validating {test_method.upper()}/{test_method_type} on H2...")
            result = comparator.validate_external_method(
                system_name='h2',
                external_method=test_method,
                method_type=test_method_type,
                max_iterations=50  # Use limited iterations for speed
            )
            
            print(f"Results:")
            print(f"  • Method: {result.method}")
            print(f"  • Calculated energy: {result.calculated_energy:.6f} Hartree")
            print(f"  • Reference energy:  {result.reference_energy:.6f} Hartree")
            print(f"  • Energy error:      {result.energy_error*1000:.2f} mHartree")
            print(f"  • Error magnitude:   {result.error_magnitude}")
            print(f"  • Calculation time:  {result.calculation_time:.2f} seconds")
            
            analyzer.add_validation_results([result])
            
        except Exception as e:
            print(f"External method validation failed: {e}")
            simulate_results = True
    
    # Test 3: External validation suite
    print("\n" + "=" * 70)
    print("5. External Methods Validation Suite")
    print("=" * 70)
    
    if not simulate_results and len(available_external_methods) >= 1:
        # Run validation suite for available external methods
        external_methods_to_test = available_external_methods[:2]  # Limit to 2 for demo
        
        # Define lightweight method types for each package
        lightweight_methods = {
            'molpro': ['hf', 'mp2'],
            'orca': ['hf', 'mp2'],  
            'psi4': ['hf', 'mp2'],
            'gaussian': ['hf'],
            'nwchem': ['hf']
        }
        
        try:
            print(f"\n🚀 Running validation suite for: {', '.join(external_methods_to_test)}")
            external_results = comparator.run_external_validation_suite(
                external_methods=external_methods_to_test,
                method_types=lightweight_methods,
                systems=['h2', 'lih'],  # Use fastest systems only
                max_iterations=30
            )
            
            # Add results to analyzer
            for method, results in external_results.items():
                if results:
                    analyzer.add_validation_results(results)
            
            print(f"\n📊 External validation suite completed!")
            print(f"  • Methods tested: {len(external_results)}")
            total_results = sum(len(results) for results in external_results.values())
            print(f"  • Total validations: {total_results}")
            
        except Exception as e:
            print(f"External validation suite failed: {e}")
            simulate_results = True
    
    # Test 4: Method comparison (external vs internal)
    print("\n" + "=" * 70)
    print("6. External vs Internal Method Comparison")
    print("=" * 70)
    
    if not simulate_results and available_external_methods:
        try:
            external_method = available_external_methods[0]
            print(f"\n⚖️  Comparing {external_method.upper()}/HF vs Internal CASSCF on H2...")
            
            # Run external method validation
            external_result = comparator.validate_external_method(
                system_name='h2',
                external_method=external_method,
                method_type='hf'
            )
            
            # Run internal method validation
            internal_result = comparator.validate_method(
                system_name='h2',
                method='casscf',
                target_accuracy='low'
            )
            
            print(f"Results:")
            print(f"  • {external_method.upper()}/HF energy: {external_result.calculated_energy:.6f} H")
            print(f"  • Internal CASSCF energy: {internal_result.calculated_energy:.6f} H")
            energy_diff = external_result.calculated_energy - internal_result.calculated_energy
            print(f"  • Energy difference: {energy_diff*1000:.2f} mHartree")
            
            analyzer.add_validation_results([external_result, internal_result])
            
        except Exception as e:
            print(f"Method comparison failed: {e}")
    
    # Simulate results if no external methods available
    if simulate_results:
        print("\n🎭 Simulating external method validation results...")
        
        # Create mock external validation results
        mock_external_results = []
        systems = ['h2', 'lih', 'h2o']
        methods = ['molpro_casscf', 'orca_nevpt2', 'psi4_ccsd']
        
        for i, (system, method) in enumerate(zip(systems, methods)):
            # Get reference system
            benchmark_system = benchmark_suite.get_system(system)
            if benchmark_system:
                # Simulate realistic energies with small errors
                ref_energy = benchmark_system.reference_energies.get('fci', -100.0)
                simulated_energy = ref_energy + np.random.normal(0, 0.002)  # 2 mH std dev
                
                reference_entry = reference_db.get_reference_energy(system, 'fci', 'cc-pVDZ')
                if reference_entry:
                    mock_result = comparator.ValidationResult(
                        system_name=system,
                        method=method,
                        basis_set='cc-pVDZ',
                        calculated_energy=simulated_energy,
                        reference_energy=reference_entry.energy,
                        energy_error=simulated_energy - reference_entry.energy,
                        relative_error=(simulated_energy - reference_entry.energy) / abs(reference_entry.energy),
                        reference_source=reference_entry.source,
                        reference_uncertainty=reference_entry.uncertainty,
                        reference_quality=reference_entry.quality_level,
                        calculation_time=np.random.uniform(5, 30)  # 5-30 seconds
                    )
                    mock_external_results.append(mock_result)
        
        if mock_external_results:
            analyzer.add_validation_results(mock_external_results)
            comparator.validation_results.extend(mock_external_results)
    
    # Test 5: Statistical analysis including external methods
    print("\n" + "=" * 70)
    print("7. Statistical Analysis (Including External Methods)")
    print("=" * 70)
    
    if analyzer.validation_results:
        print(f"\n📈 Analyzing {len(analyzer.validation_results)} validation results...")
        
        # Get all unique methods (internal + external)
        all_methods = list(set(r.method for r in analyzer.validation_results))
        external_methods_in_results = [m for m in all_methods if '_' in m or m.lower() in potential_methods]
        internal_methods_in_results = [m for m in all_methods if m not in external_methods_in_results]
        
        print(f"\nMethod categories:")
        print(f"  • Internal methods: {', '.join(internal_methods_in_results) if internal_methods_in_results else 'None'}")
        print(f"  • External methods: {', '.join(external_methods_in_results) if external_methods_in_results else 'None'}")
        
        # Analyze each method
        for method in all_methods[:5]:  # Limit to first 5 methods
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
        
        # Compare internal vs external method performance
        if external_methods_in_results and internal_methods_in_results:
            print(f"\n🔄 Internal vs External Method Comparison:")
            
            internal_errors = []
            external_errors = []
            
            for result in analyzer.validation_results:
                if result.method in internal_methods_in_results:
                    internal_errors.append(abs(result.energy_error) * 1000)
                elif result.method in external_methods_in_results:
                    external_errors.append(abs(result.energy_error) * 1000)
            
            if internal_errors and external_errors:
                print(f"  • Internal methods mean error: {np.mean(internal_errors):.2f} mHartree")
                print(f"  • External methods mean error: {np.mean(external_errors):.2f} mHartree")
                print(f"  • Internal methods std dev: {np.std(internal_errors):.2f} mHartree")
                print(f"  • External methods std dev: {np.std(external_errors):.2f} mHartree")
    
    # Test 6: Export results
    print("\n" + "=" * 70)
    print("8. Export Results and Docker Integration Info")
    print("=" * 70)
    
    output_dir = Path("external_validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Export validation results
    if comparator.validation_results:
        validation_file = output_dir / "external_validation_results.json"
        comparator.export_results(validation_file)
        print(f"✅ Validation results exported to: {validation_file}")
    
    # Export statistical analysis
    if analyzer.validation_results:
        stats_file = output_dir / "external_statistical_analysis.json"
        analyzer.export_statistical_report(stats_file)
        print(f"✅ Statistical analysis exported to: {stats_file}")
    
    # Export Docker integration status
    docker_status = {
        'available_external_methods': available_external_methods,
        'docker_status': external_runner.get_docker_status() if hasattr(external_runner, 'get_docker_status') else 'Unknown',
        'method_compatibility': {}
    }
    
    for method in potential_methods:
        if method in available_external_methods:
            compatibility = benchmark_suite.validate_external_method_compatibility(method)
            docker_status['method_compatibility'][method] = {
                'available': compatibility['available'],
                'compatible_systems': compatibility['compatible_systems'],
                'recommended_systems': compatibility['recommended_systems']
            }
    
    docker_file = output_dir / "docker_integration_status.json"
    import json
    with open(docker_file, 'w') as f:
        json.dump(docker_status, f, indent=2)
    print(f"✅ Docker integration status exported to: {docker_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("9. Summary") 
    print("=" * 70)
    
    total_validations = len(analyzer.validation_results)
    external_validations = len([r for r in analyzer.validation_results 
                              if '_' in r.method or r.method.lower() in potential_methods])
    
    print(f"\n🎯 External Methods Validation Framework Demo Completed!")
    print(f"  • Total validations performed: {total_validations}")
    print(f"  • External method validations: {external_validations}")
    print(f"  • Available external packages: {len(available_external_methods)}")
    print(f"  • Docker integration: {'Active' if available_external_methods else 'Inactive'}")
    
    if total_validations > 0:
        all_errors = [abs(r.energy_error) * 1000 for r in analyzer.validation_results]
        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        print(f"  • Overall mean absolute error: {mean_error:.2f} mHartree") 
        print(f"  • Overall max absolute error: {max_error:.2f} mHartree")
    
    print(f"\nResults and Docker status saved to: {output_dir.absolute()}")
    
    if not available_external_methods:
        print(f"\n💡 To enable external methods validation:")
        print(f"  1. Ensure Docker is installed and running")
        print(f"  2. Build quantum chemistry Docker containers")
        print(f"  3. Configure the ExternalMethodRunner with container images")
        print(f"  4. Re-run this example to test actual external methods")
    else:
        print(f"\n🚀 External methods validation is ready for production use!")
        print(f"Available packages: {', '.join(available_external_methods)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ External validation demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ External validation demo failed with error: {e}")
        print("This may be due to missing external method dependencies or Docker configuration.")
        print("Please check that external quantum chemistry packages are properly set up.")