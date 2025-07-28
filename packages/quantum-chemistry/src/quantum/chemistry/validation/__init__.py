"""
Validation and benchmarking framework for quantum chemistry methods.

This module provides comprehensive tools for validating quantum chemistry calculations
against reference data, comparing results between different methods, and orchestrating
large-scale validation workflows with statistical analysis.
"""

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

__all__ = [
    # Benchmark systems
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
]