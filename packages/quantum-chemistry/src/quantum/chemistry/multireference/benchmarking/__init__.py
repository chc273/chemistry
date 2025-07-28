"""
Benchmarking and validation infrastructure for multireference methods.

This module provides tools for systematic benchmarking, dataset management,
and statistical analysis of multireference method accuracy.
"""

from .datasets import (
    BenchmarkDataset,
    BenchmarkDatasetBuilder,
    BenchmarkEntry,
    BenchmarkMolecule,
    SystemType,
    create_standard_benchmark_datasets,
)
from .analysis import BenchmarkAnalyzer
from .validation import ValidationRunner

__all__ = [
    "BenchmarkDataset",
    "BenchmarkDatasetBuilder", 
    "BenchmarkEntry",
    "BenchmarkMolecule",
    "SystemType",
    "create_standard_benchmark_datasets",
    "BenchmarkAnalyzer",
    "ValidationRunner",
]