"""
Performance benchmarking and validation tools for GPU acceleration.

This module provides comprehensive benchmarking capabilities to measure
and validate GPU acceleration performance gains across different
quantum chemistry methods and system sizes.
"""

from __future__ import annotations

from .performance_tests import (
    GPUBenchmarkSuite,
    PerformanceComparison,
    BenchmarkResult,
    ValidationResult,
)

__all__ = [
    "GPUBenchmarkSuite",
    "PerformanceComparison", 
    "BenchmarkResult",
    "ValidationResult",
]