"""
GPU-accelerated implementations of multireference quantum chemistry methods.

This module provides GPU-accelerated versions of existing multireference methods,
maintaining full API compatibility while providing significant performance
improvements for suitable systems.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from quantum.chemistry.multireference.base import (
    MultireferenceMethod,
    MultireferenceMethodType,
    MultireferenceResult
)
from quantum.chemistry.multireference.methods.casscf import CASSCFMethod, NEVPT2Method

from .gpu_manager import GPUAccelerationManager, get_gpu_manager
from .memory_optimizer import MemoryOptimizer, MemoryPartitionStrategy
from .pyscf_gpu_integration import PySCFGPUBackend

logger = logging.getLogger(__name__)


class GPUAcceleratedMethod:
    """
    Mixin class for GPU acceleration capabilities.
    
    This mixin provides common GPU acceleration functionality that can be
    mixed into existing multireference method implementations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize GPU acceleration capabilities."""
        super().__init__(*args, **kwargs)
        
        # Extract GPU-specific parameters
        self.use_gpu = kwargs.get('use_gpu', True)
        self.gpu_memory_fraction = kwargs.get('gpu_memory_fraction', 0.8)
        self.partition_strategy = kwargs.get('partition_strategy', MemoryPartitionStrategy.ADAPTIVE)
        
        # Initialize GPU components
        self.gpu_manager = get_gpu_manager()
        self.memory_optimizer = MemoryOptimizer(self.gpu_manager)
        self.gpu_backend = PySCFGPUBackend(self.gpu_manager, self.memory_optimizer)
        
        # Performance tracking
        self.gpu_stats = {
            'gpu_used': False,
            'memory_partitions': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'data_transfer_time': 0.0
        }
    
    def _should_use_gpu(self,
                       scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                       active_space: ActiveSpaceResult) -> bool:
        """
        Determine if GPU acceleration should be used for this calculation.
        
        Args:
            scf_obj: SCF object
            active_space: Active space information
            
        Returns:
            True if GPU should be used
        """
        if not self.use_gpu:
            return False
            
        if not self.gpu_manager.is_gpu_available():
            logger.debug("GPU not available for acceleration")
            return False
        
        # Check if calculation fits on GPU
        basis_size = scf_obj.mol.nao_nr()
        method_name = self._get_method_type().value
        
        if self.gpu_manager.can_fit_on_gpu(
            active_space.n_active_electrons,
            active_space.n_active_orbitals,
            basis_size,
            method_name
        ):
            return True
        
        # Check if memory partitioning is feasible
        array_shapes = self._estimate_array_shapes(active_space, basis_size)
        estimates = self.memory_optimizer.estimate_performance_improvement(array_shapes, method_name)
        
        # Use GPU if expected speedup > 1.5x
        best_performance = max(estimates.values()) if estimates else 1.0
        return best_performance > 1.5
    
    def _estimate_array_shapes(self,
                              active_space: ActiveSpaceResult,
                              basis_size: int) -> list[Tuple[int, ...]]:
        """Estimate array shapes for memory planning."""
        n_active = active_space.n_active_orbitals
        
        shapes = [
            (basis_size, basis_size),  # Fock matrix
            (basis_size, n_active),    # MO coefficients
            (n_active, n_active, n_active, n_active),  # Active space integrals
        ]
        
        return shapes
    
    def _create_calculation_id(self) -> str:
        """Create unique calculation ID."""
        return f"{self._get_method_type().value}_{uuid.uuid4().hex[:8]}"
    
    def _record_gpu_stats(self, 
                         gpu_used: bool,
                         gpu_time: float = 0.0,
                         cpu_time: float = 0.0,
                         partitions: int = 0) -> None:
        """Record GPU performance statistics."""
        self.gpu_stats.update({
            'gpu_used': gpu_used,
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'memory_partitions': partitions
        })


class GPUCASSCFMethod(GPUAcceleratedMethod, CASSCFMethod):
    """
    GPU-accelerated CASSCF implementation.
    
    This class extends the base CASSCF method with GPU acceleration capabilities,
    including automatic memory management and performance optimization.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize GPU-accelerated CASSCF method.
        
        Args:
            **kwargs: Method parameters including GPU-specific options
        """
        super().__init__(**kwargs)
        
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform GPU-accelerated CASSCF calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with CASSCF results including GPU performance data
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for GPU CASSCF calculation")
        
        calculation_id = self._create_calculation_id()
        should_use_gpu = self._should_use_gpu(scf_obj, active_space)
        
        start_time = time.time()
        
        try:
            if should_use_gpu:
                result = self._gpu_casscf_calculation(scf_obj, active_space, calculation_id, **kwargs)
            else:
                logger.info("Using CPU fallback for CASSCF calculation")
                result = self._cpu_casscf_calculation(scf_obj, active_space, **kwargs)
            
            # Add GPU performance information
            result.computational_cost.update({
                'gpu_acceleration_used': should_use_gpu,
                'gpu_stats': self.gpu_stats.copy()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"GPU CASSCF calculation failed: {e}")
            logger.info("Falling back to CPU calculation")
            
            # Clean up GPU resources
            self.memory_optimizer.cleanup_calculation(calculation_id)
            
            # Fall back to CPU
            result = self._cpu_casscf_calculation(scf_obj, active_space, **kwargs)
            result.computational_cost['gpu_acceleration_used'] = False
            result.computational_cost['gpu_fallback_reason'] = str(e)
            
            return result
    
    def _gpu_casscf_calculation(self,
                               scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                               active_space: ActiveSpaceResult,
                               calculation_id: str,
                               **kwargs) -> MultireferenceResult:
        """Perform GPU-accelerated CASSCF calculation."""
        
        gpu_start_time = time.time()
        
        # Create memory partition plan if needed
        array_shapes = self._estimate_array_shapes(active_space, scf_obj.mol.nao_nr())
        partition_plan = self.memory_optimizer.create_partition_plan(
            calculation_id,
            array_shapes,
            method="casscf",
            strategy=self.partition_strategy
        )
        
        # Reserve GPU memory
        memory_reservation_id = f"{calculation_id}_memory"
        memory_required = self.gpu_manager.estimate_memory_requirement(
            active_space.n_active_orbitals,
            active_space.n_active_electrons,
            scf_obj.mol.nao_nr(),
            "casscf"
        )
        
        memory_reserved = self.gpu_manager.reserve_gpu_memory(
            memory_reservation_id, memory_required
        )
        
        if not memory_reserved:
            raise RuntimeError("Failed to reserve GPU memory for CASSCF calculation")
        
        try:
            with self.gpu_backend.gpu_context():
                # Apply GPU acceleration to SCF object
                gpu_scf_obj = self.gpu_backend.accelerate_scf(scf_obj, enable_gpu=True)
                
                # Set up GPU-accelerated CASSCF
                from pyscf import mcscf
                casscf_obj = mcscf.CASSCF(gpu_scf_obj,
                                        active_space.n_active_orbitals,
                                        active_space.n_active_electrons)
                
                # Apply GPU acceleration to CASSCF
                gpu_casscf_obj = self.gpu_backend.accelerate_casscf(
                    casscf_obj, 
                    enable_gpu=True,
                    partition_strategy=self.partition_strategy
                )
                
                # Configure CASSCF parameters
                gpu_casscf_obj.max_cycle_macro = self.max_cycle
                gpu_casscf_obj.conv_tol = self.conv_tol
                gpu_casscf_obj.conv_tol_grad = self.conv_tol_grad
                
                # Apply additional configuration
                for key, value in kwargs.items():
                    if hasattr(gpu_casscf_obj, key):
                        setattr(gpu_casscf_obj, key, value)
                
                # Run GPU CASSCF calculation
                gpu_casscf_obj.kernel(active_space.orbital_coefficients)
                
                # Extract results
                cas_energy = gpu_casscf_obj.e_tot
                correlation_energy = cas_energy - scf_obj.e_tot
                
                # Natural orbitals and occupation numbers
                natural_orbitals, occupation_numbers = mcscf.addons.make_natural_orbitals(gpu_casscf_obj)
                
                # Convergence information
                convergence_info = {
                    'converged': gpu_casscf_obj.converged,
                    'iterations': getattr(gpu_casscf_obj, 'niter', None),
                    'energy_gradient': getattr(gpu_casscf_obj, 'de', None),
                    'orbital_gradient': getattr(gpu_casscf_obj, 'max_orb_grad', None),
                    'gpu_accelerated': True
                }
                
                gpu_time = time.time() - gpu_start_time
                
                # Record performance statistics
                self._record_gpu_stats(
                    gpu_used=True,
                    gpu_time=gpu_time,
                    partitions=partition_plan.num_blocks
                )
                
                # Computational cost information
                computational_cost = {
                    'wall_time': gpu_time,
                    'gpu_time': gpu_time,
                    'cpu_time': 0.0,
                    'memory_mb': memory_required,
                    'memory_partitions': partition_plan.num_blocks
                }
                
                result = MultireferenceResult(
                    method="GPU-CASSCF",
                    energy=cas_energy,
                    correlation_energy=correlation_energy,
                    active_space_info={
                        'n_electrons': active_space.n_active_electrons,
                        'n_orbitals': active_space.n_active_orbitals,
                        'selection_method': active_space.method,
                        'gpu_accelerated': True,
                        'memory_strategy': partition_plan.strategy.value
                    },
                    n_active_electrons=active_space.n_active_electrons,
                    n_active_orbitals=active_space.n_active_orbitals,
                    convergence_info=convergence_info,
                    computational_cost=computational_cost,
                    natural_orbitals=natural_orbitals,
                    occupation_numbers=occupation_numbers,
                    basis_set=scf_obj.mol.basis,
                    software_version="PySCF-GPU"
                )
                
                logger.info(f"GPU CASSCF calculation completed in {gpu_time:.2f}s")
                return result
                
        finally:
            # Clean up resources
            self.gpu_manager.release_gpu_memory(memory_reservation_id)
            self.memory_optimizer.cleanup_calculation(calculation_id)
    
    def _cpu_casscf_calculation(self,
                               scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                               active_space: ActiveSpaceResult,
                               **kwargs) -> MultireferenceResult:
        """Fallback CPU CASSCF calculation."""
        
        cpu_start_time = time.time()
        
        # Use parent class CPU implementation
        result = super(CASSCFMethod, self).calculate(scf_obj, active_space, **kwargs)
        
        cpu_time = time.time() - cpu_start_time
        
        # Record CPU statistics
        self._record_gpu_stats(gpu_used=False, cpu_time=cpu_time)
        
        # Update computational cost
        result.computational_cost.update({
            'wall_time': cpu_time,
            'cpu_time': cpu_time,
            'gpu_time': 0.0
        })
        
        return result
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost with GPU acceleration considered.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates including GPU acceleration factors
        """
        # Get base CPU cost estimate
        cpu_cost = super().estimate_cost(n_electrons, n_orbitals, basis_size, **kwargs)
        
        if not self.gpu_manager.is_gpu_available():
            return cpu_cost
        
        # Estimate GPU speedup
        if self.gpu_manager.can_fit_on_gpu(n_electrons, n_orbitals, basis_size, "casscf"):
            # Direct GPU calculation
            gpu_speedup = 8.0  # Assume 8x speedup for GPU
            gpu_cost = {
                'memory_mb': cpu_cost['memory_mb'] * 1.2,  # Slightly more memory on GPU
                'time_seconds': cpu_cost['time_seconds'] / gpu_speedup,
                'disk_mb': cpu_cost['disk_mb'] * 0.5  # Less disk I/O on GPU
            }
        else:
            # Memory partitioned calculation
            array_shapes = [(n_orbitals, n_orbitals, n_orbitals, n_orbitals)]
            estimates = self.memory_optimizer.estimate_performance_improvement(array_shapes, "casscf")
            best_performance = max(estimates.values()) if estimates else 2.0
            
            gpu_cost = {
                'memory_mb': cpu_cost['memory_mb'],
                'time_seconds': cpu_cost['time_seconds'] / best_performance,
                'disk_mb': cpu_cost['disk_mb']
            }
        
        return gpu_cost


class GPUNEVPT2Method(GPUAcceleratedMethod, NEVPT2Method):
    """
    GPU-accelerated NEVPT2 implementation.
    
    This class extends the base NEVPT2 method with GPU acceleration for both
    the CASSCF step and the perturbative correction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize GPU-accelerated NEVPT2 method.
        
        Args:
            **kwargs: Method parameters including GPU-specific options
        """
        super().__init__(**kwargs)
    
    def calculate(self,
                 scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                 active_space: ActiveSpaceResult,
                 **kwargs) -> MultireferenceResult:
        """
        Perform GPU-accelerated NEVPT2 calculation.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            **kwargs: Additional calculation parameters
            
        Returns:
            MultireferenceResult with NEVPT2 results including GPU performance data
        """
        if not self.validate_input(scf_obj, active_space):
            raise ValueError("Invalid input parameters for GPU NEVPT2 calculation")
        
        calculation_id = self._create_calculation_id()
        should_use_gpu = self._should_use_gpu(scf_obj, active_space)
        
        try:
            if should_use_gpu:
                result = self._gpu_nevpt2_calculation(scf_obj, active_space, calculation_id, **kwargs)
            else:
                logger.info("Using CPU fallback for NEVPT2 calculation")
                result = self._cpu_nevpt2_calculation(scf_obj, active_space, **kwargs)
            
            # Add GPU performance information
            result.computational_cost.update({
                'gpu_acceleration_used': should_use_gpu,
                'gpu_stats': self.gpu_stats.copy()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"GPU NEVPT2 calculation failed: {e}")
            logger.info("Falling back to CPU calculation")
            
            # Clean up GPU resources
            self.memory_optimizer.cleanup_calculation(calculation_id)
            
            # Fall back to CPU
            result = self._cpu_nevpt2_calculation(scf_obj, active_space, **kwargs)
            result.computational_cost['gpu_acceleration_used'] = False
            result.computational_cost['gpu_fallback_reason'] = str(e)
            
            return result
    
    def _gpu_nevpt2_calculation(self,
                               scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                               active_space: ActiveSpaceResult,
                               calculation_id: str,
                               **kwargs) -> MultireferenceResult:
        """Perform GPU-accelerated NEVPT2 calculation."""
        
        gpu_start_time = time.time()
        
        # Estimate memory requirements for NEVPT2 (larger than CASSCF)
        basis_size = scf_obj.mol.nao_nr()
        memory_required = self.gpu_manager.estimate_memory_requirement(
            active_space.n_active_orbitals,
            active_space.n_active_electrons,
            basis_size,
            "nevpt2"
        )
        
        # Reserve GPU memory
        memory_reservation_id = f"{calculation_id}_memory"
        memory_reserved = self.gpu_manager.reserve_gpu_memory(
            memory_reservation_id, memory_required
        )
        
        if not memory_reserved:
            raise RuntimeError("Failed to reserve GPU memory for NEVPT2 calculation")
        
        try:
            with self.gpu_backend.gpu_context():
                # First perform GPU-accelerated CASSCF
                gpu_casscf_method = GPUCASSCFMethod(
                    max_cycle=self.max_cycle,
                    conv_tol=self.conv_tol,
                    conv_tol_grad=self.conv_tol_grad,
                    use_gpu=True,
                    partition_strategy=self.partition_strategy
                )
                
                casscf_result = gpu_casscf_method.calculate(scf_obj, active_space, **kwargs)
                
                # Apply GPU acceleration to SCF
                gpu_scf_obj = self.gpu_backend.accelerate_scf(scf_obj, enable_gpu=True)
                
                # Set up CASSCF for NEVPT2
                from pyscf import mcscf, mrpt
                
                casscf_obj = mcscf.CASSCF(gpu_scf_obj,
                                        active_space.n_active_orbitals,
                                        active_space.n_active_electrons)
                
                # Apply GPU acceleration
                gpu_casscf_obj = self.gpu_backend.accelerate_casscf(casscf_obj, enable_gpu=True)
                
                # Configure and run CASSCF
                gpu_casscf_obj.max_cycle_macro = self.max_cycle
                gpu_casscf_obj.conv_tol = self.conv_tol
                gpu_casscf_obj.conv_tol_grad = self.conv_tol_grad
                gpu_casscf_obj.kernel(active_space.orbital_coefficients)
                
                # Run NEVPT2 calculation
                # Note: NEVPT2 GPU acceleration depends on PySCF implementation
                try:
                    if self.nevpt2_type.lower() == "sc":
                        pt2_correction = mrpt.NEVPT(gpu_casscf_obj).kernel()
                    else:
                        pt2_correction = mrpt.NEVPT(gpu_casscf_obj).kernel()
                except Exception as nevpt_error:
                    logger.warning(f"GPU NEVPT2 failed, using CPU for PT2 correction: {nevpt_error}")
                    # Fall back to CPU for NEVPT2 part only
                    cpu_casscf = mcscf.CASSCF(scf_obj, active_space.n_active_orbitals, active_space.n_active_electrons)
                    cpu_casscf.kernel(active_space.orbital_coefficients)
                    pt2_correction = mrpt.NEVPT(cpu_casscf).kernel()
                
                # Calculate total NEVPT2 energy
                nevpt2_energy = gpu_casscf_obj.e_tot + pt2_correction
                nevpt2_correlation = nevpt2_energy - scf_obj.e_tot
                
                gpu_time = time.time() - gpu_start_time
                
                # Record performance statistics
                self._record_gpu_stats(
                    gpu_used=True,
                    gpu_time=gpu_time
                )
                
                # Create result
                result = MultireferenceResult(
                    method="GPU-CASSCF+NEVPT2",
                    energy=nevpt2_energy,
                    correlation_energy=nevpt2_correlation,
                    active_space_info={
                        'n_electrons': active_space.n_active_electrons,
                        'n_orbitals': active_space.n_active_orbitals,
                        'selection_method': active_space.method,
                        'pt2_correction': pt2_correction,
                        'nevpt2_type': self.nevpt2_type,
                        'gpu_accelerated': True
                    },
                    n_active_electrons=active_space.n_active_electrons,
                    n_active_orbitals=active_space.n_active_orbitals,
                    convergence_info=casscf_result.convergence_info,
                    computational_cost={
                        'wall_time': gpu_time,
                        'gpu_time': gpu_time,
                        'memory_mb': memory_required
                    },
                    natural_orbitals=casscf_result.natural_orbitals,
                    occupation_numbers=casscf_result.occupation_numbers,
                    basis_set=scf_obj.mol.basis,
                    software_version="PySCF-GPU"
                )
                
                logger.info(f"GPU NEVPT2 calculation completed in {gpu_time:.2f}s")
                return result
                
        finally:
            # Clean up resources
            self.gpu_manager.release_gpu_memory(memory_reservation_id)
            self.memory_optimizer.cleanup_calculation(calculation_id)
    
    def _cpu_nevpt2_calculation(self,
                               scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                               active_space: ActiveSpaceResult,
                               **kwargs) -> MultireferenceResult:
        """Fallback CPU NEVPT2 calculation."""
        
        cpu_start_time = time.time()
        
        # Use parent class CPU implementation
        result = super(NEVPT2Method, self).calculate(scf_obj, active_space, **kwargs)
        
        cpu_time = time.time() - cpu_start_time
        
        # Record CPU statistics
        self._record_gpu_stats(gpu_used=False, cpu_time=cpu_time)
        
        # Update computational cost
        result.computational_cost.update({
            'wall_time': cpu_time,
            'cpu_time': cpu_time,
            'gpu_time': 0.0
        })
        
        return result
    
    def estimate_cost(self,
                     n_electrons: int,
                     n_orbitals: int,
                     basis_size: int,
                     **kwargs) -> Dict[str, float]:
        """
        Estimate computational cost for GPU NEVPT2.
        
        Args:
            n_electrons: Number of active electrons
            n_orbitals: Number of active orbitals
            basis_size: Total basis set size
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost estimates including GPU acceleration
        """
        # Get base NEVPT2 cost estimate
        cpu_cost = super().estimate_cost(n_electrons, n_orbitals, basis_size, **kwargs)
        
        if not self.gpu_manager.is_gpu_available():
            return cpu_cost
        
        # Estimate GPU speedup for NEVPT2
        if self.gpu_manager.can_fit_on_gpu(n_electrons, n_orbitals, basis_size, "nevpt2"):
            # NEVPT2 has less GPU speedup than CASSCF due to algorithmic complexity
            gpu_speedup = 4.0  # Conservative estimate
            gpu_cost = {
                'memory_mb': cpu_cost['memory_mb'] * 1.5,  # More memory for correlation
                'time_seconds': cpu_cost['time_seconds'] / gpu_speedup,
                'disk_mb': cpu_cost['disk_mb'] * 0.3  # Much less disk I/O
            }
        else:
            # Memory partitioned calculation - limited speedup
            gpu_cost = {
                'memory_mb': cpu_cost['memory_mb'],
                'time_seconds': cpu_cost['time_seconds'] / 2.0,  # Modest speedup
                'disk_mb': cpu_cost['disk_mb']
            }
        
        return gpu_cost


# Factory functions for creating GPU-accelerated methods
def create_gpu_casscf_method(**kwargs) -> GPUCASSCFMethod:
    """
    Create GPU-accelerated CASSCF method.
    
    Args:
        **kwargs: Method configuration parameters
        
    Returns:
        GPU-accelerated CASSCF method instance
    """
    return GPUCASSCFMethod(**kwargs)


def create_gpu_nevpt2_method(**kwargs) -> GPUNEVPT2Method:
    """
    Create GPU-accelerated NEVPT2 method.
    
    Args:
        **kwargs: Method configuration parameters
        
    Returns:
        GPU-accelerated NEVPT2 method instance
    """
    return GPUNEVPT2Method(**kwargs)


def create_optimal_gpu_method(scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
                             active_space: ActiveSpaceResult,
                             method_type: MultireferenceMethodType,
                             **kwargs) -> MultireferenceMethod:
    """
    Create optimal GPU-accelerated method based on system characteristics.
    
    Args:
        scf_obj: SCF object for system analysis
        active_space: Active space information
        method_type: Desired method type
        **kwargs: Method configuration
        
    Returns:
        Optimal GPU-accelerated method instance
    """
    # Analyze system to determine best GPU strategy
    gpu_manager = get_gpu_manager()
    
    if not gpu_manager.is_gpu_available():
        logger.info("GPU not available, creating CPU method")
        kwargs['use_gpu'] = False
    else:
        # Determine optimal memory strategy
        basis_size = scf_obj.mol.nao_nr()
        can_fit = gpu_manager.can_fit_on_gpu(
            active_space.n_active_electrons,
            active_space.n_active_orbitals,
            basis_size,
            method_type.value
        )
        
        if can_fit:
            kwargs['partition_strategy'] = MemoryPartitionStrategy.BLOCK_CYCLIC
        else:
            kwargs['partition_strategy'] = MemoryPartitionStrategy.ADAPTIVE
    
    # Create appropriate method
    if method_type == MultireferenceMethodType.CASSCF:
        return create_gpu_casscf_method(**kwargs)
    elif method_type == MultireferenceMethodType.NEVPT2:
        return create_gpu_nevpt2_method(**kwargs)
    else:
        raise ValueError(f"GPU acceleration not available for method: {method_type}")


# Export GPU method registry for integration with base framework
GPU_METHOD_REGISTRY = {
    MultireferenceMethodType.CASSCF: GPUCASSCFMethod,
    MultireferenceMethodType.NEVPT2: GPUNEVPT2Method,
}