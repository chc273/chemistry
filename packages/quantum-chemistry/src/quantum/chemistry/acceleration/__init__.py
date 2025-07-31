"""
GPU acceleration framework for quantum chemistry calculations.

This module provides GPU acceleration capabilities for multireference quantum
chemistry methods, including hardware detection, memory management, and
optimized implementations of computational kernels.

Key Components:
- GPUAccelerationManager: Hardware detection and resource management
- GPU-accelerated method implementations with automatic fallback
- Memory optimization for large active spaces
- Performance benchmarking and validation tools

The framework maintains full API compatibility with existing CPU implementations
while providing significant performance improvements for suitable systems.
"""

from __future__ import annotations

from .gpu_manager import GPUAccelerationManager, GPUCapabilities, GPUMemoryInfo
from .memory_optimizer import MemoryOptimizer, MemoryPartitionStrategy
from .pyscf_gpu_integration import PySCFGPUBackend

__all__ = [
    "GPUAccelerationManager",
    "GPUCapabilities", 
    "GPUMemoryInfo",
    "MemoryOptimizer",
    "MemoryPartitionStrategy",
    "PySCFGPUBackend",
]

# Version info
__version__ = "0.1.0"