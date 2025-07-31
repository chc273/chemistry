"""
GPU hardware detection and resource management for quantum chemistry calculations.

This module provides comprehensive GPU resource management including hardware
detection, memory allocation, and performance optimization for quantum chemistry
workloads with automatic fallback to CPU when necessary.
"""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GPUBackendType(str, Enum):
    """Available GPU backend types."""
    
    CUDA = "cuda"
    ROCM = "rocm"  
    OPENCL = "opencl"
    NONE = "none"


@dataclass
class GPUMemoryInfo:
    """GPU memory information container."""
    
    total_mb: float
    available_mb: float
    used_mb: float
    reserved_mb: float = 0.0
    
    @property
    def utilization(self) -> float:
        """Memory utilization percentage."""
        return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0.0
    
    @property
    def available_ratio(self) -> float:
        """Fraction of total memory available."""
        return self.available_mb / self.total_mb if self.total_mb > 0 else 0.0


@dataclass  
class GPUCapabilities:
    """GPU hardware capabilities and features."""
    
    device_id: int
    name: str
    backend: GPUBackendType
    compute_capability: Tuple[int, int]
    memory_info: GPUMemoryInfo
    max_threads_per_block: int
    max_blocks_per_grid: int
    warp_size: int
    supports_double_precision: bool
    supports_tensor_cores: bool = False
    pci_bus_id: Optional[str] = None
    
    @property
    def compute_capability_str(self) -> str:
        """Compute capability as string (e.g., '8.0')."""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
    
    def is_suitable_for_quantum_chemistry(self) -> bool:
        """Check if GPU is suitable for quantum chemistry calculations."""
        # Require double precision support and reasonable compute capability
        return (
            self.supports_double_precision and
            self.compute_capability >= (3, 5) and  # Minimum for good performance
            self.memory_info.total_mb >= 1024  # At least 1GB memory
        )


class GPUAccelerationManager:
    """
    Central manager for GPU acceleration in quantum chemistry calculations.
    
    This class handles hardware detection, resource allocation, memory management,
    and provides a unified interface for GPU-accelerated quantum chemistry methods.
    """
    
    def __init__(self, 
                 auto_detect: bool = True,
                 preferred_backend: Optional[GPUBackendType] = None,
                 memory_fraction: float = 0.8):
        """
        Initialize GPU acceleration manager.
        
        Args:
            auto_detect: Automatically detect available GPU hardware
            preferred_backend: Preferred GPU backend (CUDA, ROCm, etc.)
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.preferred_backend = preferred_backend
        self.memory_fraction = max(0.1, min(1.0, memory_fraction))
        self.available_gpus: List[GPUCapabilities] = []
        self.selected_gpu: Optional[GPUCapabilities] = None
        self.pyscf_gpu_available: bool = False
        
        # Memory management state
        self._memory_pool: Dict[int, float] = {}  # device_id -> allocated MB
        self._memory_reservations: Dict[str, Tuple[int, float]] = {}  # reservation_id -> (device, mb)
        
        if auto_detect:
            self.detect_gpu_hardware()
            self.detect_pyscf_gpu_support()
    
    def detect_gpu_hardware(self) -> List[GPUCapabilities]:
        """
        Detect available GPU hardware and capabilities.
        
        Returns:
            List of detected GPU capabilities
        """
        self.available_gpus.clear()
        
        # Try CUDA detection first
        cuda_gpus = self._detect_cuda_gpus()
        self.available_gpus.extend(cuda_gpus)
        
        # Try ROCm detection  
        rocm_gpus = self._detect_rocm_gpus()
        self.available_gpus.extend(rocm_gpus)
        
        # Select best GPU automatically
        if self.available_gpus:
            self.selected_gpu = self._select_best_gpu()
            logger.info(f"Selected GPU: {self.selected_gpu.name} ({self.selected_gpu.backend})")
        else:
            logger.warning("No suitable GPUs detected")
            
        return self.available_gpus
    
    def _detect_cuda_gpus(self) -> List[GPUCapabilities]:
        """Detect CUDA-capable GPUs."""
        gpus = []
        
        try:
            # Try importing CUDA libraries
            import cupy as cp
            
            # Get number of devices
            num_devices = cp.cuda.runtime.getDeviceCount()
            
            for device_id in range(num_devices):
                with cp.cuda.Device(device_id):
                    # Get device properties
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    
                    # Get memory info
                    meminfo = cp.cuda.runtime.memGetInfo()
                    memory_info = GPUMemoryInfo(
                        total_mb=meminfo[1] / (1024**2),
                        available_mb=meminfo[0] / (1024**2),
                        used_mb=(meminfo[1] - meminfo[0]) / (1024**2)
                    )
                    
                    # Create capabilities object
                    capabilities = GPUCapabilities(
                        device_id=device_id,
                        name=props['name'].decode('utf-8'),
                        backend=GPUBackendType.CUDA,
                        compute_capability=(props['major'], props['minor']),
                        memory_info=memory_info,
                        max_threads_per_block=props['maxThreadsPerBlock'],
                        max_blocks_per_grid=props['maxGridSize'][0],
                        warp_size=props['warpSize'],
                        supports_double_precision=True,  # Modern GPUs support FP64
                        supports_tensor_cores=props['major'] >= 7,  # Volta and newer
                        pci_bus_id=f"{props['pciBusID']:02x}:{props['pciDeviceID']:02x}.0"
                    )
                    
                    gpus.append(capabilities)
                    logger.info(f"Detected CUDA GPU {device_id}: {capabilities.name}")
                    
        except ImportError:
            logger.debug("CuPy not available, CUDA detection skipped")
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
            
        return gpus
    
    def _detect_rocm_gpus(self) -> List[GPUCapabilities]:
        """Detect ROCm-capable GPUs."""
        gpus = []
        
        try:
            # Try ROCm detection via rocm-smi
            result = subprocess.run(
                ['rocm-smi', '--showid', '--showproductname', '--showmeminfo', '--csv'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        device_id = int(parts[0])
                        name = parts[1].strip()
                        total_mem = float(parts[2]) if parts[2].strip() else 1024.0
                        used_mem = float(parts[3]) if parts[3].strip() else 0.0
                        
                        memory_info = GPUMemoryInfo(
                            total_mb=total_mem,
                            available_mb=total_mem - used_mem,
                            used_mb=used_mem
                        )
                        
                        capabilities = GPUCapabilities(
                            device_id=device_id,
                            name=name,
                            backend=GPUBackendType.ROCM,
                            compute_capability=(9, 0),  # Assume modern ROCm GPU
                            memory_info=memory_info,
                            max_threads_per_block=1024,
                            max_blocks_per_grid=65535,
                            warp_size=64,  # AMD wavefront size
                            supports_double_precision=True
                        )
                        
                        gpus.append(capabilities)
                        logger.info(f"Detected ROCm GPU {device_id}: {name}")
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("ROCm detection failed or rocm-smi not available")
        except Exception as e:
            logger.warning(f"ROCm detection error: {e}")
            
        return gpus
    
    def _select_best_gpu(self) -> Optional[GPUCapabilities]:
        """Select the best available GPU based on capabilities."""
        if not self.available_gpus:
            return None
            
        # Filter for quantum chemistry suitability
        suitable_gpus = [gpu for gpu in self.available_gpus if gpu.is_suitable_for_quantum_chemistry()]
        
        if not suitable_gpus:
            logger.warning("No GPUs suitable for quantum chemistry found")
            return None
        
        # Prefer backend if specified
        if self.preferred_backend:
            backend_gpus = [gpu for gpu in suitable_gpus if gpu.backend == self.preferred_backend]
            if backend_gpus:
                suitable_gpus = backend_gpus
        
        # Select GPU with most available memory
        return max(suitable_gpus, key=lambda gpu: gpu.memory_info.available_mb)
    
    def detect_pyscf_gpu_support(self) -> bool:
        """
        Detect if PySCF GPU modules are available.
        
        Returns:
            True if PySCF GPU support is available
        """
        try:
            # Try importing PySCF GPU modules
            import pyscf.gpu
            self.pyscf_gpu_available = True
            logger.info("PySCF GPU support detected")
            
        except ImportError:
            self.pyscf_gpu_available = False
            logger.info("PySCF GPU support not available")
            
        return self.pyscf_gpu_available
    
    def estimate_memory_requirement(self,
                                  n_active_orbitals: int,
                                  n_active_electrons: int,
                                  basis_size: int,
                                  method: str = "casscf") -> float:
        """
        Estimate GPU memory requirement for calculation.
        
        Args:
            n_active_orbitals: Number of active orbitals
            n_active_electrons: Number of active electrons  
            basis_size: Total basis set size
            method: Quantum chemistry method
            
        Returns:
            Estimated memory requirement in MB
        """
        # Base memory estimates for different components
        # All estimates include safety margins
        
        if method.lower() == "casscf":
            # CASSCF memory scaling: O(N^4) for active space + O(M^2) for basis
            active_memory = (n_active_orbitals ** 4) * 8e-6 * 1.5  # 50% margin
            basis_memory = (basis_size ** 2) * 8e-6 * 2.0  # 100% margin
            
        elif method.lower() == "nevpt2":
            # NEVPT2 additional memory for correlation
            active_memory = (n_active_orbitals ** 4) * 8e-6 * 2.0
            basis_memory = (basis_size ** 2) * 8e-6 * 3.0
            correlation_memory = (n_active_orbitals ** 3) * basis_size * 8e-6
            basis_memory += correlation_memory
            
        else:
            # Default conservative estimate
            active_memory = (n_active_orbitals ** 4) * 8e-6 * 2.0
            basis_memory = (basis_size ** 2) * 8e-6 * 2.5
        
        total_memory = active_memory + basis_memory
        
        # Add overhead for GPU framework and intermediate arrays  
        overhead = max(500.0, total_memory * 0.2)  # At least 500MB or 20%
        
        return total_memory + overhead
    
    def can_fit_on_gpu(self,
                      n_active_orbitals: int,
                      n_active_electrons: int, 
                      basis_size: int,
                      method: str = "casscf") -> bool:
        """
        Check if calculation can fit on selected GPU.
        
        Args:
            n_active_orbitals: Number of active orbitals
            n_active_electrons: Number of active electrons
            basis_size: Total basis set size
            method: Quantum chemistry method
            
        Returns:
            True if calculation fits on GPU
        """
        if not self.selected_gpu:
            return False
            
        required_memory = self.estimate_memory_requirement(
            n_active_orbitals, n_active_electrons, basis_size, method
        )
        
        available_memory = self.selected_gpu.memory_info.available_mb * self.memory_fraction
        
        return required_memory <= available_memory
    
    def reserve_gpu_memory(self, 
                          reservation_id: str,
                          memory_mb: float,
                          device_id: Optional[int] = None) -> bool:
        """
        Reserve GPU memory for calculation.
        
        Args:
            reservation_id: Unique identifier for reservation
            memory_mb: Amount of memory to reserve in MB
            device_id: Specific device ID (uses selected GPU if None)
            
        Returns:
            True if reservation successful
        """
        if device_id is None:
            if not self.selected_gpu:
                return False
            device_id = self.selected_gpu.device_id
        
        # Check if enough memory is available
        total_reserved = self._memory_pool.get(device_id, 0.0)
        gpu = next((gpu for gpu in self.available_gpus if gpu.device_id == device_id), None)
        
        if not gpu:
            return False
            
        available = gpu.memory_info.available_mb * self.memory_fraction - total_reserved
        
        if memory_mb > available:
            logger.warning(f"Insufficient GPU memory: requested {memory_mb:.1f}MB, "
                          f"available {available:.1f}MB")
            return False
        
        # Make reservation
        self._memory_reservations[reservation_id] = (device_id, memory_mb)
        self._memory_pool[device_id] = total_reserved + memory_mb
        
        logger.info(f"Reserved {memory_mb:.1f}MB on GPU {device_id}")
        return True
    
    def release_gpu_memory(self, reservation_id: str) -> bool:
        """
        Release reserved GPU memory.
        
        Args:
            reservation_id: Reservation identifier to release
            
        Returns:
            True if release successful
        """
        if reservation_id not in self._memory_reservations:
            return False
            
        device_id, memory_mb = self._memory_reservations[reservation_id]
        
        # Update memory pool
        current_reserved = self._memory_pool.get(device_id, 0.0)
        self._memory_pool[device_id] = max(0.0, current_reserved - memory_mb)
        
        # Remove reservation
        del self._memory_reservations[reservation_id]
        
        logger.info(f"Released {memory_mb:.1f}MB from GPU {device_id}")
        return True
    
    def get_gpu_status(self) -> Dict[str, any]:
        """
        Get comprehensive GPU status information.
        
        Returns:
            Dictionary with GPU status details
        """
        status = {
            'gpus_available': len(self.available_gpus),
            'selected_gpu': None,
            'pyscf_gpu_support': self.pyscf_gpu_available,
            'memory_reservations': len(self._memory_reservations),
            'gpu_details': []
        }
        
        if self.selected_gpu:
            status['selected_gpu'] = {
                'device_id': self.selected_gpu.device_id,
                'name': self.selected_gpu.name,
                'backend': self.selected_gpu.backend.value,
                'compute_capability': self.selected_gpu.compute_capability_str,
                'memory_total_mb': self.selected_gpu.memory_info.total_mb,
                'memory_available_mb': self.selected_gpu.memory_info.available_mb,
                'memory_reserved_mb': self._memory_pool.get(self.selected_gpu.device_id, 0.0)
            }
        
        for gpu in self.available_gpus:
            gpu_info = {
                'device_id': gpu.device_id,
                'name': gpu.name,
                'backend': gpu.backend.value,
                'suitable_for_qc': gpu.is_suitable_for_quantum_chemistry(),
                'memory_total_mb': gpu.memory_info.total_mb,
                'memory_available_mb': gpu.memory_info.available_mb
            }
            status['gpu_details'].append(gpu_info)
        
        return status
    
    def set_gpu_device(self, device_id: int) -> bool:
        """
        Set active GPU device.
        
        Args:
            device_id: GPU device ID to select
            
        Returns:
            True if device set successfully
        """
        gpu = next((gpu for gpu in self.available_gpus if gpu.device_id == device_id), None)
        
        if not gpu:
            logger.error(f"GPU device {device_id} not found")
            return False
            
        if not gpu.is_suitable_for_quantum_chemistry():
            logger.warning(f"GPU device {device_id} may not be suitable for quantum chemistry")
        
        self.selected_gpu = gpu
        logger.info(f"Selected GPU device {device_id}: {gpu.name}")
        return True
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.selected_gpu is not None and self.selected_gpu.is_suitable_for_quantum_chemistry()
    
    def get_recommended_block_size(self, matrix_size: int) -> int:
        """
        Get recommended block size for matrix operations.
        
        Args:
            matrix_size: Size of matrices to process
            
        Returns:
            Recommended block size for optimal performance
        """
        if not self.selected_gpu:
            return min(1000, matrix_size)  # Conservative CPU fallback
        
        # Base block size on GPU memory and compute capability
        available_memory_mb = self.selected_gpu.memory_info.available_mb * self.memory_fraction
        
        # Estimate memory per block (rough approximation)
        memory_per_element = 8e-6  # 8 bytes per double precision element in MB
        max_block_from_memory = int(np.sqrt(available_memory_mb / memory_per_element / 4))  # Factor of 4 for safety
        
        # Base on compute capability
        if self.selected_gpu.compute_capability >= (7, 0):  # Volta and newer
            base_block_size = 2048
        elif self.selected_gpu.compute_capability >= (6, 0):  # Pascal
            base_block_size = 1024
        else:  # Older architectures
            base_block_size = 512
        
        # Take minimum of constraints
        recommended_size = min(base_block_size, max_block_from_memory, matrix_size)
        
        # Ensure minimum block size for efficiency
        return max(256, recommended_size)


# Singleton instance for global access
_gpu_manager_instance: Optional[GPUAccelerationManager] = None


def get_gpu_manager() -> GPUAccelerationManager:
    """Get global GPU acceleration manager instance."""
    global _gpu_manager_instance
    
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUAccelerationManager()
    
    return _gpu_manager_instance


def reset_gpu_manager() -> None:
    """Reset global GPU manager instance (mainly for testing)."""
    global _gpu_manager_instance
    _gpu_manager_instance = None