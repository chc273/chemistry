"""
PySCF GPU backend integration for accelerated quantum chemistry calculations.

This module provides seamless integration with PySCF's GPU modules, enabling
GPU acceleration of SCF, integral computation, and multireference methods
while maintaining compatibility with the existing CPU-based workflow.
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyscf import gto, scf
    from pyscf.mcscf import casscf

from .gpu_manager import GPUAccelerationManager, get_gpu_manager
from .memory_optimizer import MemoryOptimizer, MemoryPartitionStrategy

logger = logging.getLogger(__name__)


class PySCFGPUBackend:
    """
    PySCF GPU backend integration and management.
    
    This class provides a unified interface for GPU-accelerated PySCF calculations,
    handling device setup, memory management, and automatic fallback to CPU.
    """
    
    def __init__(self, 
                 gpu_manager: Optional[GPUAccelerationManager] = None,
                 memory_optimizer: Optional[MemoryOptimizer] = None):
        """
        Initialize PySCF GPU backend.
        
        Args:
            gpu_manager: GPU acceleration manager
            memory_optimizer: Memory optimization manager
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.memory_optimizer = memory_optimizer or MemoryOptimizer(self.gpu_manager)
        
        # Check PySCF GPU availability
        self.gpu_available = self._check_pyscf_gpu_support()
        self.current_device = None
        
        # Performance tracking
        self.performance_stats: Dict[str, List[float]] = {
            'scf_time': [],
            'integral_time': [],
            'casscf_time': []
        }
        
    def _check_pyscf_gpu_support(self) -> bool:
        """Check if PySCF GPU modules are available and functional."""
        try:
            # Try importing various PySCF GPU modules
            gpu_modules = []
            
            try:
                from pyscf.gpu import df
                gpu_modules.append("density_fitting")
            except ImportError:
                pass
                
            try:
                from pyscf.gpu import scf as gpu_scf
                gpu_modules.append("scf")
            except ImportError:
                pass
                
            try:
                from pyscf.gpu import mcscf as gpu_mcscf 
                gpu_modules.append("mcscf")
            except ImportError:
                pass
            
            if gpu_modules:
                logger.info(f"PySCF GPU modules available: {', '.join(gpu_modules)}")
                return True
            else:
                logger.warning("No PySCF GPU modules found")
                return False
                
        except Exception as e:
            logger.warning(f"PySCF GPU support check failed: {e}")
            return False
    
    @contextmanager
    def gpu_context(self, device_id: Optional[int] = None):
        """
        Context manager for GPU device selection and cleanup.
        
        Args:
            device_id: Specific GPU device ID (uses selected GPU if None)
        """
        if not self.gpu_available or not self.gpu_manager.is_gpu_available():
            logger.debug("GPU context requested but GPU not available, using CPU")
            yield False
            return
            
        old_device = self.current_device
        
        try:
            # Set GPU device
            if device_id is not None:
                success = self.gpu_manager.set_gpu_device(device_id)
                if not success:
                    logger.warning(f"Failed to set GPU device {device_id}, falling back to CPU")
                    yield False
                    return
            
            # Configure GPU environment
            if self.gpu_manager.selected_gpu:
                device_id = self.gpu_manager.selected_gpu.device_id
                self._configure_gpu_environment(device_id)
                self.current_device = device_id
                
                logger.debug(f"Entered GPU context on device {device_id}")
                yield True
            else:
                yield False
                
        except Exception as e:
            logger.error(f"GPU context error: {e}")
            yield False
            
        finally:
            # Cleanup and restore
            self.current_device = old_device
            self._cleanup_gpu_resources()
    
    def _configure_gpu_environment(self, device_id: int) -> None:
        """Configure GPU environment for PySCF calculations."""
        try:
            # Set device for CuPy if available
            if self.gpu_manager.selected_gpu.backend.value == "cuda":
                try:
                    import cupy as cp
                    cp.cuda.Device(device_id).use()
                    
                    # Set memory pool to limit fragmentation
                    memory_limit = int(self.gpu_manager.selected_gpu.memory_info.available_mb * 
                                     self.gpu_manager.memory_fraction * 1024**2)
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=memory_limit)
                    
                    logger.debug(f"Configured CUDA device {device_id} with {memory_limit/1024**2:.1f}MB limit")
                    
                except ImportError:
                    logger.warning("CuPy not available for GPU configuration")
                    
        except Exception as e:
            logger.warning(f"GPU environment configuration failed: {e}")
    
    def _cleanup_gpu_resources(self) -> None:
        """Clean up GPU resources and free memory."""
        try:
            if self.gpu_manager.selected_gpu and self.gpu_manager.selected_gpu.backend.value == "cuda":
                try:
                    import cupy as cp
                    
                    # Free memory pool
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    
                    # Clear cache
                    cp.fuse.clear_memo()
                    
                except ImportError:
                    pass
                    
        except Exception as e:
            logger.debug(f"GPU cleanup warning: {e}")
    
    def accelerate_scf(self, 
                      scf_obj: 'scf.hf.SCF',
                      enable_gpu: bool = True) -> 'scf.hf.SCF':
        """
        Apply GPU acceleration to SCF calculation.
        
        Args:
            scf_obj: PySCF SCF object
            enable_gpu: Enable GPU acceleration if available
            
        Returns:
            GPU-accelerated SCF object or original if GPU unavailable
        """
        if not enable_gpu or not self.gpu_available or not self.gpu_manager.is_gpu_available():
            logger.debug("GPU acceleration not applied to SCF")
            return scf_obj
        
        try:
            # Check memory requirements
            mol = scf_obj.mol
            basis_size = mol.nao_nr()
            
            # Estimate SCF memory requirement (rough)
            scf_memory_mb = (basis_size ** 2) * 8e-6 * 4  # Fock, density, overlap, etc.
            
            if not self.gpu_manager.can_fit_on_gpu(0, 0, basis_size, "scf"):
                logger.warning("SCF calculation too large for GPU memory, using CPU")
                return scf_obj
            
            # Apply GPU acceleration
            from pyscf.gpu import scf as gpu_scf
            
            if hasattr(scf_obj, 'density_fit') and scf_obj.density_fit:
                # Density fitting SCF
                gpu_scf_obj = gpu_scf.density_fit(scf_obj)
            else:
                # Regular SCF
                gpu_scf_obj = gpu_scf.SCF(scf_obj.mol)
                gpu_scf_obj.__dict__.update(scf_obj.__dict__)
            
            logger.info(f"Applied GPU acceleration to {type(scf_obj).__name__}")
            return gpu_scf_obj
            
        except ImportError:
            logger.warning("PySCF GPU SCF module not available")
            return scf_obj
        except Exception as e:
            logger.error(f"Failed to apply GPU acceleration to SCF: {e}")
            return scf_obj
    
    def accelerate_casscf(self,
                         casscf_obj: 'casscf.CASSCF',
                         enable_gpu: bool = True,
                         partition_strategy: Optional[MemoryPartitionStrategy] = None) -> 'casscf.CASSCF':
        """
        Apply GPU acceleration to CASSCF calculation.
        
        Args:
            casscf_obj: PySCF CASSCF object
            enable_gpu: Enable GPU acceleration if available
            partition_strategy: Memory partitioning strategy for large active spaces
            
        Returns:
            GPU-accelerated CASSCF object or original if GPU unavailable
        """
        if not enable_gpu or not self.gpu_available or not self.gpu_manager.is_gpu_available():
            logger.debug("GPU acceleration not applied to CASSCF")
            return casscf_obj
        
        try:
            # Check memory requirements
            n_active_orbitals = casscf_obj.ncas
            n_active_electrons = casscf_obj.nelecas
            basis_size = casscf_obj.mol.nao_nr()
            
            if not self.gpu_manager.can_fit_on_gpu(n_active_electrons, n_active_orbitals, basis_size, "casscf"):
                logger.info("CASSCF calculation requires memory partitioning")
                
                # Create memory partition plan
                array_shapes = [
                    (n_active_orbitals, n_active_orbitals, n_active_orbitals, n_active_orbitals),  # 4-center integrals
                    (basis_size, basis_size),  # Fock matrix
                    (basis_size, n_active_orbitals)  # MO coefficients
                ]
                
                partition_plan = self.memory_optimizer.create_partition_plan(
                    f"casscf_{id(casscf_obj)}",
                    array_shapes,
                    method="casscf",
                    strategy=partition_strategy
                )
                
                if len(partition_plan.gpu_blocks) == 0:
                    logger.warning("No blocks fit on GPU, using CPU")
                    return casscf_obj
                
                logger.info(f"Using memory partitioning: {partition_plan.num_blocks} blocks, "
                           f"{len(partition_plan.gpu_blocks)} on GPU")
            
            # Apply GPU acceleration  
            from pyscf.gpu import mcscf as gpu_mcscf
            
            gpu_casscf_obj = gpu_mcscf.CASSCF(casscf_obj._scf, casscf_obj.ncas, casscf_obj.nelecas)
            
            # Copy configuration
            for attr in ['conv_tol', 'conv_tol_grad', 'max_cycle_macro', 'max_cycle_micro']:
                if hasattr(casscf_obj, attr):
                    setattr(gpu_casscf_obj, attr, getattr(casscf_obj, attr))
            
            logger.info("Applied GPU acceleration to CASSCF")
            return gpu_casscf_obj
            
        except ImportError:
            logger.warning("PySCF GPU MCSCF module not available")
            return casscf_obj
        except Exception as e:
            logger.error(f"Failed to apply GPU acceleration to CASSCF: {e}")
            return casscf_obj
    
    def accelerate_integral_transformation(self,
                                         mol: 'gto.Mole',
                                         mo_coeffs: np.ndarray,
                                         enable_gpu: bool = True) -> Dict[str, np.ndarray]:
        """
        GPU-accelerated integral transformation.
        
        Args:
            mol: PySCF molecule object
            mo_coeffs: MO coefficients for transformation
            enable_gpu: Enable GPU acceleration if available
            
        Returns:
            Dictionary with transformed integrals
        """
        if not enable_gpu or not self.gpu_available or not self.gpu_manager.is_gpu_available():
            return self._cpu_integral_transformation(mol, mo_coeffs)
        
        start_time = time.time()
        
        try:
            with self.gpu_context():
                # Use GPU-accelerated integral transformation
                from pyscf.gpu import ao2mo
                
                # Estimate memory requirements
                n_mo = mo_coeffs.shape[1]
                integral_memory_mb = (n_mo ** 4) * 8e-6
                
                if integral_memory_mb > self.gpu_manager.selected_gpu.memory_info.available_mb * 0.8:
                    logger.info("Using blocked integral transformation for large system")
                    return self._blocked_gpu_integral_transformation(mol, mo_coeffs)
                else:
                    # Direct transformation
                    eri_mo = ao2mo.full(mol, mo_coeffs)
                    
                    result = {'eri_mo': eri_mo}
                    
                    self.performance_stats['integral_time'].append(time.time() - start_time)
                    logger.info(f"GPU integral transformation completed in {time.time() - start_time:.2f}s")
                    
                    return result
                    
        except Exception as e:
            logger.error(f"GPU integral transformation failed: {e}")
            return self._cpu_integral_transformation(mol, mo_coeffs)
    
    def _blocked_gpu_integral_transformation(self,
                                           mol: 'gto.Mole',
                                           mo_coeffs: np.ndarray) -> Dict[str, np.ndarray]:
        """GPU integral transformation with memory blocking."""
        
        try:
            # Create partition plan for integral arrays
            n_mo = mo_coeffs.shape[1]
            array_shapes = [(n_mo, n_mo, n_mo, n_mo)]
            
            partition_plan = self.memory_optimizer.create_partition_plan(
                f"integrals_{id(mol)}",
                array_shapes,
                method="integrals",
                strategy=MemoryPartitionStrategy.BLOCK_CYCLIC
            )
            
            # Implement blocked transformation
            block_size = self.gpu_manager.get_recommended_block_size(n_mo)
            eri_mo = np.zeros((n_mo, n_mo, n_mo, n_mo))
            
            from pyscf.gpu import ao2mo
            
            for i_start in range(0, n_mo, block_size):
                i_end = min(i_start + block_size, n_mo)
                
                for j_start in range(0, n_mo, block_size):
                    j_end = min(j_start + block_size, n_mo)
                    
                    # Transform block
                    mo_block = mo_coeffs[:, i_start:i_end]
                    eri_block = ao2mo.general(mol, [mo_block, mo_coeffs, mo_coeffs, mo_coeffs])
                    
                    # Reshape and store
                    eri_block = eri_block.reshape(i_end-i_start, n_mo, n_mo, n_mo)
                    eri_mo[i_start:i_end, :, :, :] = eri_block
            
            return {'eri_mo': eri_mo}
            
        except Exception as e:
            logger.error(f"Blocked GPU integral transformation failed: {e}")
            return self._cpu_integral_transformation(mol, mo_coeffs)
    
    def _cpu_integral_transformation(self,
                                   mol: 'gto.Mole', 
                                   mo_coeffs: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback CPU integral transformation."""
        
        start_time = time.time()
        
        try:
            from pyscf import ao2mo
            
            eri_mo = ao2mo.full(mol, mo_coeffs)
            
            self.performance_stats['integral_time'].append(time.time() - start_time)
            logger.info(f"CPU integral transformation completed in {time.time() - start_time:.2f}s")
            
            return {'eri_mo': eri_mo}
            
        except Exception as e:
            logger.error(f"CPU integral transformation failed: {e}")
            return {}
    
    def benchmark_gpu_performance(self,
                                test_molecules: List['gto.Mole'],
                                methods: List[str] = ['scf', 'casscf']) -> Dict[str, Dict[str, float]]:
        """
        Benchmark GPU vs CPU performance for different methods.
        
        Args:
            test_molecules: List of test molecules
            methods: Methods to benchmark
            
        Returns:
            Benchmark results dictionary
        """
        results = {}
        
        for mol in test_molecules:
            mol_name = f"mol_{mol.natm}atoms"
            results[mol_name] = {}
            
            for method in methods:
                if method == 'scf':
                    results[mol_name][method] = self._benchmark_scf(mol)
                elif method == 'casscf':
                    results[mol_name][method] = self._benchmark_casscf(mol)
        
        return results
    
    def _benchmark_scf(self, mol: 'gto.Mole') -> Dict[str, float]:
        """Benchmark SCF performance."""
        from pyscf import scf
        
        # CPU benchmark
        scf_cpu = scf.RHF(mol)
        cpu_start = time.time()
        scf_cpu.kernel()
        cpu_time = time.time() - cpu_start
        
        # GPU benchmark
        gpu_time = cpu_time  # Default to same time
        
        if self.gpu_available and self.gpu_manager.is_gpu_available():
            try:
                scf_gpu = self.accelerate_scf(scf.RHF(mol))
                
                gpu_start = time.time()
                scf_gpu.kernel()
                gpu_time = time.time() - gpu_start
                
            except Exception as e:
                logger.warning(f"GPU SCF benchmark failed: {e}")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time, 
            'speedup': speedup
        }
    
    def _benchmark_casscf(self, mol: 'gto.Mole') -> Dict[str, float]:
        """Benchmark CASSCF performance."""
        from pyscf import scf, mcscf
        
        # Run SCF first
        mf = scf.RHF(mol)
        mf.kernel()
        
        # Small active space for benchmarking
        n_active_orbitals = min(6, mol.nao_nr())
        n_active_electrons = min(6, mol.nelectron)
        
        # CPU benchmark
        casscf_cpu = mcscf.CASSCF(mf, n_active_orbitals, n_active_electrons)
        casscf_cpu.max_cycle_macro = 5  # Limit iterations for benchmarking
        
        cpu_start = time.time()
        casscf_cpu.kernel()
        cpu_time = time.time() - cpu_start
        
        # GPU benchmark
        gpu_time = cpu_time  # Default to same time
        
        if self.gpu_available and self.gpu_manager.is_gpu_available():
            try:
                casscf_gpu = self.accelerate_casscf(mcscf.CASSCF(mf, n_active_orbitals, n_active_electrons))
                casscf_gpu.max_cycle_macro = 5
                
                gpu_start = time.time()
                casscf_gpu.kernel()
                gpu_time = time.time() - gpu_start
                
            except Exception as e:
                logger.warning(f"GPU CASSCF benchmark failed: {e}")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        }
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance statistics summary."""
        summary = {
            'gpu_available': self.gpu_available,
            'gpu_active': self.gpu_manager.is_gpu_available(),
            'current_device': self.current_device,
            'statistics': {}
        }
        
        for method, times in self.performance_stats.items():
            if times:
                summary['statistics'][method] = {
                    'count': len(times),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
        
        return summary
    
    def clear_performance_stats(self) -> None:
        """Clear performance statistics."""
        for key in self.performance_stats:
            self.performance_stats[key].clear()


# Factory functions for easy integration
def create_gpu_scf(mol: 'gto.Mole', 
                  method: str = 'rhf',
                  enable_gpu: bool = True) -> 'scf.hf.SCF':
    """
    Create GPU-accelerated SCF object.
    
    Args:
        mol: PySCF molecule
        method: SCF method ('rhf', 'uhf', 'rohf')
        enable_gpu: Enable GPU acceleration
        
    Returns:
        SCF object with GPU acceleration if available
    """
    from pyscf import scf
    
    # Create base SCF object
    if method.lower() == 'rhf':
        scf_obj = scf.RHF(mol)
    elif method.lower() == 'uhf':
        scf_obj = scf.UHF(mol)
    elif method.lower() == 'rohf':
        scf_obj = scf.ROHF(mol)
    else:
        raise ValueError(f"Unsupported SCF method: {method}")
    
    # Apply GPU acceleration
    if enable_gpu:
        backend = PySCFGPUBackend()
        scf_obj = backend.accelerate_scf(scf_obj)
    
    return scf_obj


def create_gpu_casscf(scf_obj: 'scf.hf.SCF',
                     n_active_orbitals: int,
                     n_active_electrons: Union[int, Tuple[int, int]],
                     enable_gpu: bool = True) -> 'casscf.CASSCF':
    """
    Create GPU-accelerated CASSCF object.
    
    Args:
        scf_obj: Converged SCF object
        n_active_orbitals: Number of active orbitals
        n_active_electrons: Number of active electrons
        enable_gpu: Enable GPU acceleration
        
    Returns:
        CASSCF object with GPU acceleration if available
    """
    from pyscf import mcscf
    
    # Create base CASSCF object
    casscf_obj = mcscf.CASSCF(scf_obj, n_active_orbitals, n_active_electrons)
    
    # Apply GPU acceleration
    if enable_gpu:
        backend = PySCFGPUBackend()
        casscf_obj = backend.accelerate_casscf(casscf_obj)
    
    return casscf_obj