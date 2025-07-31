"""
Validation and benchmarking framework for quantum chemistry methods.

This module provides comprehensive tools for validating quantum chemistry calculations
against reference data, comparing results between different methods, and orchestrating
large-scale validation workflows with statistical analysis.

Enhanced Features:
- Database integration (W4-11, G2/97, QUESTDB, TMC-151)
- Comprehensive benchmarking suite
- Advanced statistical analysis with uncertainty quantification
- Publication-quality visualization
- Automated report generation (PDF, HTML, Markdown)
"""

# Legacy components
from .benchmarks import BenchmarkSystem, BenchmarkSuite
from .comparison import MethodComparator, ValidationResult, MethodComparison
from .reference_data import ReferenceDatabase, ReferenceEntry
from .statistical import (
    StatisticalAnalyzer, ConvergenceTracker, 
    ConvergencePoint, ConvergenceAnalysis
)
from .workflows import (
    ValidationTask, WorkflowConfiguration, ValidationWorkflowManager
)

# Enhanced database integration
from .databases import (
    DatabaseManager, DatabaseInterface, MolecularEntry, ReferenceDataEntry,
    W4_11Database, G2_97Database, QuestDBDatabase, TMC151Database,
    PropertyType, BasisSetType
)

# Temporarily disabled - circular import and missing files
# TODO: Re-enable once issues are resolved
# from .benchmarks import (
#     ComprehensiveBenchmarkSuite, BenchmarkConfiguration, BenchmarkResult,
#     BenchmarkScope, BenchmarkTarget
# )
# from .benchmarks.statistical_analysis import (
#     AdvancedStatisticalAnalyzer, UncertaintyQuantifier, ErrorMetrics,
#     UncertaintyAnalysis
# )
# from .benchmarks.visualization import PublicationQualityPlotter

# Automated report generation
from .reporting import (
    ReportGenerator, ReportConfiguration, LaTeXFormatter, HTMLFormatter
)

__all__ = [
    # Legacy benchmark systems
    'BenchmarkSystem',
    'BenchmarkSuite',
    
    # Method comparison and validation
    'MethodComparator',
    'ValidationResult',
    'MethodComparison',
    
    # Reference data management
    'ReferenceDatabase',
    'ReferenceEntry',
    
    # Statistical analysis and convergence
    'StatisticalAnalyzer',
    'ConvergenceTracker',
    'ConvergencePoint',
    'ConvergenceAnalysis',
    
    # Production workflow orchestration
    'ValidationTask',
    'WorkflowConfiguration', 
    'ValidationWorkflowManager',
    
    # Enhanced database integration
    'DatabaseManager',
    'DatabaseInterface',
    'MolecularEntry',
    'ReferenceDataEntry',
    'W4_11Database',
    'G2_97Database', 
    'QuestDBDatabase',
    'TMC151Database',
    'PropertyType',
    'BasisSetType',
    
    # Temporarily disabled - circular import and missing files
    # 'ComprehensiveBenchmarkSuite',
    # 'BenchmarkConfiguration', 
    # 'BenchmarkResult',
    # 'BenchmarkScope',
    # 'BenchmarkTarget',
    # 'AdvancedStatisticalAnalyzer',
    # 'UncertaintyQuantifier',
    # 'ErrorMetrics', 
    # 'UncertaintyAnalysis',
    # 'PublicationQualityPlotter',
    
    # Automated report generation
    'ReportGenerator',
    'ReportConfiguration',
    'LaTeXFormatter',
    'HTMLFormatter',
]