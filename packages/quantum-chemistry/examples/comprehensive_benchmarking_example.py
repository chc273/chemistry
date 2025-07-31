"""
Comprehensive Benchmarking Example

This example demonstrates the complete benchmarking workflow including:
- Database integration and molecular dataset preparation
- Large-scale automated benchmarking studies
- Advanced statistical analysis with uncertainty quantification
- Publication-quality visualization
- Automated report generation

This serves as both documentation and a practical guide for users wanting
to conduct publication-ready benchmarking studies.
"""

import logging
from pathlib import Path
import numpy as np

# Import the comprehensive benchmarking suite
from quantum.chemistry.validation.databases import DatabaseManager
from quantum.chemistry.validation.benchmarks import (
    ComprehensiveBenchmarkSuite, BenchmarkConfiguration,
    BenchmarkScope, BenchmarkTarget
)
from quantum.chemistry.validation.benchmarks.statistical_analysis import (
    AdvancedStatisticalAnalyzer, UncertaintyQuantifier
)
from quantum.chemistry.validation.benchmarks.visualization import PublicationQualityPlotter
from quantum.chemistry.validation.reporting import (
    ReportGenerator, ReportConfiguration
)
from quantum.chemistry.active_space import ActiveSpaceMethod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_database_integration():
    """Demonstrate database integration capabilities."""
    
    print("\n" + "="*60)
    print("DATABASE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Load all databases
    print("Loading quantum chemistry databases...")
    db_manager.load_all_databases()
    
    # Get database statistics
    stats = db_manager.get_all_database_stats()
    
    print("\nDatabase Statistics:")
    print("-" * 40)
    for db_name, db_stats in stats.items():
        print(f"{db_name.upper()}:")
        print(f"  - Total molecules: {db_stats.total_molecules}")
        print(f"  - Unique formulas: {db_stats.unique_formulas}")
        print(f"  - Multireference systems: {db_stats.multireference_systems}")
        print(f"  - Element coverage: {len(db_stats.element_coverage)} elements")
        print()
    
    # Demonstrate cross-database search
    print("Cross-database search for H2O:")
    h2o_results = db_manager.search_by_formula("H2O")
    for db_name, molecules in h2o_results.items():
        print(f"  - {db_name}: {len(molecules)} molecules")
    
    # Find common molecules
    print("\nFinding molecules common across databases...")
    common_molecules = db_manager.find_common_molecules()
    print(f"Found {len(common_molecules)} molecules in multiple databases")
    
    # Create unified benchmark set
    print("\nCreating unified benchmark set...")
    benchmark_set = db_manager.create_unified_benchmark_set(
        target_size=50,
        difficulty_levels=["easy", "medium"],
        max_multireference_character=0.3
    )
    print(f"Created benchmark set with {len(benchmark_set)} molecules")
    
    return db_manager, benchmark_set


def demonstrate_comprehensive_benchmarking():
    """Demonstrate comprehensive benchmarking capabilities."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    # Create benchmark configuration
    config = BenchmarkConfiguration(
        study_name="example_benchmark_study",
        description="Comprehensive benchmarking of active space selection methods",
        output_directory=Path("./benchmark_results"),
        
        # Study scope and targets
        scope=BenchmarkScope.ACTIVE_SPACE_SELECTION,
        targets=[BenchmarkTarget.GROUND_STATE_ENERGY],
        
        # Database selection
        databases=["w4_11", "g2_97"],
        max_molecules_per_db=10,  # Limited for example
        difficulty_levels=["easy", "medium"],
        
        # Method selection
        active_space_methods=[
            ActiveSpaceMethod.AVAS,
            ActiveSpaceMethod.APC,
            ActiveSpaceMethod.DMET_CAS
        ],
        basis_sets=["sto-3g", "cc-pvdz"],
        
        # Computational parameters
        max_workers=2,  # Limited for example
        timeout_seconds=300,  # 5 minutes
        memory_limit_gb=4,
        
        # Quality control
        convergence_threshold=1e-6,
        require_convergence=True,
        
        # Statistical analysis
        confidence_level=0.95,
        bootstrap_samples=1000,
        uncertainty_analysis=True,
        
        # Output control
        save_intermediate_results=True,
        generate_plots=True,
        create_report=True,
        export_formats=["json", "csv", "hdf5"]
    )
    
    print(f"Benchmark Configuration:")
    print(f"  - Study: {config.study_name}")
    print(f"  - Scope: {config.scope.value}")
    print(f"  - Databases: {config.databases}")
    print(f"  - Methods: {[m.value for m in config.active_space_methods]}")
    print(f"  - Basis sets: {config.basis_sets}")
    print(f"  - Max molecules per DB: {config.max_molecules_per_db}")
    
    # Initialize benchmarking suite
    suite = ComprehensiveBenchmarkSuite(config)
    
    # Note: In practice, you would run the full benchmark
    # For this example, we'll demonstrate the workflow without actual calculations
    print(f"\nBenchmarking suite initialized successfully!")
    print(f"Output directory: {config.output_directory}")
    
    # Demonstrate dataset preparation
    print("\nPreparing molecular dataset...")
    molecules = suite.prepare_molecular_dataset()
    print(f"Prepared dataset with {len(molecules)} molecules")
    
    # Show sample molecules
    print("\nSample molecules in dataset:")
    for i, mol in enumerate(molecules[:5]):
        print(f"  {i+1}. {mol.name} ({mol.formula}) - {mol.computational_difficulty}")
    
    return suite, config


def demonstrate_statistical_analysis():
    """Demonstrate advanced statistical analysis capabilities."""
    
    print("\n" + "="*60)
    print("ADVANCED STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Generate sample data for demonstration
    np.random.seed(42)  # For reproducible results
    
    # Simulate error data for different methods
    method_errors = {
        "AVAS": np.random.normal(0.0, 0.002, 100),
        "APC": np.random.normal(0.001, 0.003, 100), 
        "DMET-CAS": np.random.normal(-0.0005, 0.0015, 100)
    }
    
    print("Simulated error data:")
    for method, errors in method_errors.items():
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        print(f"  {method}: MAE = {mae:.6f}, RMSE = {rmse:.6f} Hartree")
    
    # Initialize advanced statistical analyzer
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Perform comprehensive error analysis for each method
    print("\nComprehensive Error Analysis:")
    print("-" * 40)
    
    method_metrics = {}
    for method, errors in method_errors.items():
        # Simulate computed vs reference values
        reference_values = np.random.normal(-1.0, 0.1, len(errors))
        computed_values = reference_values + errors
        
        metrics = analyzer.comprehensive_error_analysis(
            computed_values, reference_values
        )
        method_metrics[method] = metrics
        
        print(f"\n{method}:")
        print(f"  - MAE: {metrics.mae:.6f} ± {(metrics.mae_ci[1] - metrics.mae_ci[0])/2:.6f} Hartree")
        print(f"  - RMSE: {metrics.rmse:.6f} ± {(metrics.rmse_ci[1] - metrics.rmse_ci[0])/2:.6f} Hartree")
        print(f"  - R²: {metrics.r2:.4f}")
        print(f"  - Max Error: {metrics.max_error:.6f} Hartree")
        print(f"  - Error Distribution: {'Normal' if metrics.is_normal else 'Non-normal'}")
        print(f"  - Samples: {metrics.n_samples}")
    
    # Perform method comparison
    print("\nMethod Comparison Analysis:")
    print("-" * 30)
    
    comparison_results = analyzer.bayesian_method_comparison(method_errors)
    
    if comparison_results["analysis_type"] == "bayesian":
        print("Bayesian Analysis Results:")
        for comparison, results in comparison_results["pairwise_comparisons"].items():
            method1, method2 = comparison.split("_vs_")
            prob = results["prob_method1_better"]
            print(f"  - P({method1} better than {method2}): {prob:.3f}")
    else:
        print("Frequentist Analysis Results:")
        print(f"  - ANOVA F-statistic: {comparison_results['anova_f_statistic']:.4f}")
        print(f"  - ANOVA p-value: {comparison_results['anova_p_value']:.6f}")
    
    # Outlier detection
    print("\nOutlier Detection:")
    print("-" * 20)
    
    for method, errors in method_errors.items():
        outlier_info = analyzer.outlier_detection(errors, method="iqr")
        print(f"  {method}: {outlier_info['n_outliers']} outliers ({outlier_info['outlier_fraction']:.1%})")
    
    # Uncertainty quantification
    print("\nUncertainty Quantification:")
    print("-" * 30)
    
    quantifier = UncertaintyQuantifier()
    
    # Example with uncertainties
    computed = np.random.normal(-1.0, 0.1, 50)
    reference = np.random.normal(-1.05, 0.05, 50)
    computed_unc = np.full(50, 0.01)  # 10 mHartree uncertainty
    
    unc_analysis = quantifier.propagate_uncertainties(
        computed, reference, computed_unc
    )
    
    print(f"  - Total Uncertainty: {unc_analysis.total_uncertainty:.6f} Hartree")
    print(f"  - Systematic Component: {unc_analysis.systematic_uncertainty:.6f} Hartree")
    print(f"  - Statistical Component: {unc_analysis.statistical_uncertainty:.6f} Hartree")
    print(f"  - Coverage Probability: {unc_analysis.coverage_probability:.3f}")
    
    return method_metrics, comparison_results


def demonstrate_visualization():
    """Demonstrate publication-quality visualization capabilities."""
    
    print("\n" + "="*60)
    print("PUBLICATION-QUALITY VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize plotter
    plotter = PublicationQualityPlotter(style="publication")
    
    # Generate sample data
    np.random.seed(42)
    method_errors = {
        "AVAS": np.random.normal(0.0, 0.002, 100),
        "APC": np.random.normal(0.001, 0.003, 100),
        "DMET-CAS": np.random.normal(-0.0005, 0.0015, 100)
    }
    
    # Create mock error metrics
    from unittest.mock import Mock
    method_metrics = {}
    for method, errors in method_errors.items():
        metrics = Mock()
        metrics.mae = np.mean(np.abs(errors))
        metrics.rmse = np.sqrt(np.mean(errors**2))
        metrics.r2 = 0.95 + np.random.random() * 0.04
        metrics.max_error = np.max(np.abs(errors))
        metrics.n_samples = len(errors)
        metrics.mae_ci = (metrics.mae * 0.9, metrics.mae * 1.1)
        metrics.rmse_ci = (metrics.rmse * 0.9, metrics.rmse * 1.1)
        method_metrics[method] = metrics
    
    # Create output directory
    viz_dir = Path("./visualization_examples")
    viz_dir.mkdir(exist_ok=True)
    
    print(f"Creating visualization examples in {viz_dir}")
    
    # 1. Error distribution analysis
    print("  - Creating error distribution plots...")
    error_fig = plotter.plot_error_distribution(
        method_errors,
        output_path=viz_dir / "error_distribution_example.png",
        title="Active Space Method Error Distribution Analysis"
    )
    
    # 2. Method comparison
    print("  - Creating method comparison plots...")
    comparison_fig = plotter.plot_method_comparison(
        method_metrics,
        output_path=viz_dir / "method_comparison_example.png",
        title="Active Space Method Performance Comparison"
    )
    
    # 3. Parity plots
    print("  - Creating parity plots...")
    parity_data = {}
    for method, errors in method_errors.items():
        reference = np.random.normal(-1.0, 0.1, len(errors))
        computed = reference + errors
        parity_data[method] = {
            "computed": computed,
            "reference": reference
        }
    
    parity_fig = plotter.plot_parity_plots(
        parity_data,
        output_path=viz_dir / "parity_plots_example.png",
        title="Computed vs Reference Energy Correlation"
    )
    
    # 4. Active space analysis
    print("  - Creating active space analysis plots...")
    active_space_data = {
        "AVAS": {
            "n_active_electrons": 6,
            "n_active_orbitals": 6,
            "selection_time": 2.3
        },
        "APC": {
            "n_active_electrons": 8,
            "n_active_orbitals": 8,
            "selection_time": 1.8
        },
        "DMET-CAS": {
            "n_active_electrons": 4,
            "n_active_orbitals": 4,
            "selection_time": 5.1
        }
    }
    
    as_fig = plotter.plot_active_space_analysis(
        active_space_data,
        output_path=viz_dir / "active_space_analysis_example.png",
        title="Active Space Selection Analysis"
    )
    
    # 5. Interactive dashboard (if Plotly available)
    print("  - Creating interactive dashboard...")
    benchmark_data = {
        "method_errors": method_errors,
        "method_metrics": method_metrics,
        "timing_data": {
            method: {"mean_time": np.random.uniform(1, 10)} 
            for method in method_errors.keys()
        }
    }
    
    dashboard_file = plotter.create_interactive_dashboard(
        benchmark_data,
        output_path=viz_dir / "interactive_dashboard_example.html"
    )
    
    if dashboard_file:
        print(f"    Interactive dashboard created: {dashboard_file}")
    
    # 6. Complete figure set
    print("  - Creating complete publication figure set...")
    figure_files = plotter.create_publication_figure_set(
        benchmark_data,
        viz_dir,
        study_name="example_study"
    )
    
    print(f"\nCreated {len(figure_files)} visualization files:")
    for file_path in figure_files:
        print(f"  - {file_path.name}")
    
    return viz_dir, figure_files


def demonstrate_report_generation():
    """Demonstrate automated report generation capabilities."""
    
    print("\n" + "="*60)
    print("AUTOMATED REPORT GENERATION DEMONSTRATION")
    print("="*60)
    
    # Create report configuration
    report_config = ReportConfiguration(
        title="Comprehensive Benchmarking of Active Space Selection Methods",
        authors=[
            "Research Scientist",
            "Senior Researcher", 
            "Principal Investigator"
        ],
        affiliation="Quantum Chemistry Research Institute",
        abstract="""
This study presents a comprehensive benchmarking analysis of active space 
selection methods for multireference quantum chemistry calculations. We 
evaluate the performance of AVAS, APC, and DMET-CAS methods across multiple 
standard databases including W4-11 and G2/97 test sets. Statistical analysis 
with uncertainty quantification provides robust method comparisons suitable 
for method selection guidelines.
        """.strip(),
        keywords=[
            "quantum chemistry", "multireference", "active space selection",
            "benchmarking", "CASSCF", "AVAS", "APC", "DMET"
        ],
        output_formats=["html", "markdown"],  # Skip PDF for example
        output_directory=Path("./report_examples"),
        
        # Content settings
        include_methodology=True,
        include_statistical_analysis=True,
        include_convergence_analysis=True,
        include_figures=True,
        include_tables=True,
        include_supplementary=True,
        
        # Formatting
        decimal_places=6,
        figure_format="png"
    )
    
    print(f"Report Configuration:")
    print(f"  - Title: {report_config.title}")
    print(f"  - Authors: {len(report_config.authors)} authors")
    print(f"  - Output formats: {report_config.output_formats}")
    print(f"  - Output directory: {report_config.output_directory}")
    
    # Initialize report generator
    generator = ReportGenerator(report_config)
    
    # Create sample benchmark results
    from quantum.chemistry.validation.benchmarks import BenchmarkResult
    
    sample_results = []
    molecules = ["H2", "H2O", "NH3", "CH4", "HF"]
    methods = ["avas", "apc", "dmet"]
    
    for i, (mol, method) in enumerate(zip(molecules * 3, methods * 5)):
        result = BenchmarkResult(
            calculation_id=f"example_{i:03d}",
            molecule_name=mol,
            database_source="w4_11",
            active_space_method=method,
            basis_set="cc-pvdz",
            n_active_electrons=4 + i % 3,
            n_active_orbitals=4 + i % 3,
            computed_energy=-1.0 - i * 0.1,
            reference_energy=-1.05 - i * 0.1,
            energy_error=0.05 + (i % 3) * 0.01,
            absolute_error=0.05 + (i % 3) * 0.01,
            relative_error=0.05,
            wall_time=10.0 + i * 2,
            converged=True,
            iterations=25 + i % 10
        )
        sample_results.append(result)
    
    # Create sample error metrics
    from unittest.mock import Mock
    error_metrics = {}
    
    for method in methods:
        metrics = Mock()
        base_error = 0.001 + hash(method) % 1000 / 100000
        metrics.mae = base_error
        metrics.rmse = base_error * 1.2
        metrics.max_error = base_error * 5
        metrics.r2 = 0.95 - hash(method) % 100 / 1000
        metrics.n_samples = 50
        metrics.mae_ci = (metrics.mae * 0.9, metrics.mae * 1.1)
        metrics.rmse_ci = (metrics.rmse * 0.9, metrics.rmse * 1.1)
        metrics.bias_ci = (-0.001, 0.001)
        metrics.is_normal = True
        metrics.mean_error = 0.0
        metrics.std_error = metrics.mae * 0.8
        error_metrics[method] = metrics
    
    # Sample statistical analysis
    statistical_analysis = {
        "method_performance": {
            "method_statistics": {
                method: metrics.summary_dict 
                for method, metrics in error_metrics.items()
            },
            "ranking": [
                {"method": method, "rank": i+1, "mae": error_metrics[method].mae}
                for i, method in enumerate(methods)
            ]
        },
        "system_difficulty": {
            "difficulty_ranking": [
                {"system": mol, "difficulty_score": 0.1 + i * 0.05}
                for i, mol in enumerate(molecules)
            ]
        }
    }
    
    # Generate comprehensive report
    print(f"\nGenerating comprehensive reports...")
    
    generated_files = generator.generate_comprehensive_report(
        sample_results,
        error_metrics,
        statistical_analysis,
        figure_paths=None  # Would include actual figure paths in practice
    )
    
    print(f"\nGenerated report files:")
    for format_type, file_path in generated_files.items():
        print(f"  - {format_type.upper()}: {file_path}")
    
    # Demonstrate individual formatters
    print(f"\nDemonstrating formatting utilities:")
    
    # LaTeX formatting
    from quantum.chemistry.validation.reporting import LaTeXFormatter
    latex_formatter = LaTeXFormatter()
    
    sample_number = 0.001234
    formatted_latex = latex_formatter.format_number_with_uncertainty(
        sample_number, sample_number * 0.1, "hartree"
    )
    print(f"  - LaTeX formatted number: {formatted_latex}")
    
    # HTML formatting
    from quantum.chemistry.validation.reporting import HTMLFormatter
    html_formatter = HTMLFormatter()
    
    formatted_html = html_formatter.format_number_with_uncertainty(
        sample_number, sample_number * 0.1, "hartree"
    )
    print(f"  - HTML formatted number: {formatted_html}")
    
    return generated_files


def main():
    """Run complete benchmarking demonstration."""
    
    print("COMPREHENSIVE QUANTUM CHEMISTRY BENCHMARKING SUITE")
    print("=" * 60)
    print("This example demonstrates all major features of the enhanced")
    print("benchmarking framework for publication-ready studies.")
    print("=" * 60)
    
    try:
        # 1. Database Integration
        db_manager, benchmark_molecules = demonstrate_database_integration()
        
        # 2. Comprehensive Benchmarking
        suite, config = demonstrate_comprehensive_benchmarking()
        
        # 3. Statistical Analysis
        method_metrics, comparison_results = demonstrate_statistical_analysis()
        
        # 4. Visualization
        viz_dir, figure_files = demonstrate_visualization()
        
        # 5. Report Generation
        report_files = demonstrate_report_generation()
        
        # Summary
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        
        print(f"✓ Database Integration: {len(db_manager.databases)} databases loaded")
        print(f"✓ Benchmarking Suite: Configuration created for {len(benchmark_molecules)} molecules")
        print(f"✓ Statistical Analysis: {len(method_metrics)} methods analyzed")
        print(f"✓ Visualization: {len(figure_files)} figures created")
        print(f"✓ Report Generation: {len(report_files)} reports generated")
        
        print(f"\nOutput directories:")
        print(f"  - Benchmark results: {config.output_directory}")
        print(f"  - Visualizations: {viz_dir}")
        print(f"  - Reports: {report_files.get('html', 'N/A').parent if 'html' in report_files else 'N/A'}")
        
        print(f"\nThis demonstration shows the complete workflow for:")
        print(f"  • Database-driven molecular dataset preparation")
        print(f"  • Automated large-scale benchmarking studies") 
        print(f"  • Advanced statistical analysis with uncertainty quantification")
        print(f"  • Publication-quality visualization")
        print(f"  • Automated scientific report generation")
        
        print(f"\nFor production use:")
        print(f"  1. Increase max_molecules_per_db for comprehensive studies")
        print(f"  2. Add more active space and multireference methods")
        print(f"  3. Include larger basis sets and convergence studies")
        print(f"  4. Enable PDF report generation with LaTeX")
        print(f"  5. Use high-performance computing resources")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    main()