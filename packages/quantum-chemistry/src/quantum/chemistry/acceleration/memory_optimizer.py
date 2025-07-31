"""
Memory optimization algorithms for large active spaces in GPU-accelerated calculations.

This module provides sophisticated memory partitioning and optimization strategies
to handle quantum chemistry calculations that exceed available GPU memory through
block-wise processing and hybrid CPU/GPU algorithms.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .gpu_manager import GPUAccelerationManager, GPUCapabilities

logger = logging.getLogger(__name__)


class MemoryPartitionStrategy(str, Enum):
    """Available memory partitioning strategies."""
    
    BLOCK_CYCLIC = "block_cyclic"          # Block-cyclic distribution
    ROW_WISE = "row_wise"                  # Row-wise partitioning
    COLUMN_WISE = "column_wise"            # Column-wise partitioning
    HIERARCHICAL = "hierarchical"          # Multi-level hierarchical
    ADAPTIVE = "adaptive"                  # Adaptive based on access patterns
    STREAMING = "streaming"                # Streaming with overlap


@dataclass
class MemoryBlock:
    """Memory block descriptor for partitioned computations."""
    
    block_id: int
    start_idx: Tuple[int, ...]
    end_idx: Tuple[int, ...]
    size_bytes: int
    location: str  # "gpu", "cpu", "disk"
    priority: float = 1.0
    access_count: int = 0
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Block shape in each dimension."""
        return tuple(end - start for start, end in zip(self.start_idx, self.end_idx))
    
    @property
    def size_elements(self) -> int:
        """Number of elements in block."""
        return np.prod(self.shape)


@dataclass
class MemoryPartitionPlan:
    """Complete memory partitioning plan for a calculation."""
    
    strategy: MemoryPartitionStrategy
    blocks: List[MemoryBlock]
    total_memory_required: float  # MB
    gpu_memory_budget: float      # MB
    cpu_memory_budget: float      # MB
    estimated_performance_factor: float = 1.0  # Relative to full GPU
    
    @property
    def num_blocks(self) -> int:
        """Total number of blocks."""
        return len(self.blocks)
    
    @property
    def gpu_blocks(self) -> List[MemoryBlock]:
        """Blocks assigned to GPU memory."""
        return [block for block in self.blocks if block.location == "gpu"]
    
    @property
    def cpu_blocks(self) -> List[MemoryBlock]:
        """Blocks assigned to CPU memory."""
        return [block for block in self.blocks if block.location == "cpu"]


class MemoryPartitioner(ABC):
    """Abstract base class for memory partitioning algorithms."""
    
    @abstractmethod
    def partition(self,
                 array_shapes: List[Tuple[int, ...]],
                 gpu_memory_mb: float,
                 cpu_memory_mb: float,
                 access_pattern: Optional[str] = None) -> MemoryPartitionPlan:
        """
        Create memory partition plan for given arrays and constraints.
        
        Args:
            array_shapes: List of array shapes to partition
            gpu_memory_mb: Available GPU memory in MB
            cpu_memory_mb: Available CPU memory in MB
            access_pattern: Expected access pattern hint
            
        Returns:
            Memory partition plan
        """
        pass


class BlockCyclicPartitioner(MemoryPartitioner):
    """Block-cyclic memory partitioning for balanced load distribution."""
    
    def __init__(self, block_size: Optional[int] = None):
        """
        Initialize block-cyclic partitioner.
        
        Args:
            block_size: Fixed block size (auto-determined if None)
        """
        self.block_size = block_size
    
    def partition(self,
                 array_shapes: List[Tuple[int, ...]],
                 gpu_memory_mb: float,
                 cpu_memory_mb: float,
                 access_pattern: Optional[str] = None) -> MemoryPartitionPlan:
        """Create block-cyclic partition plan."""
        
        # Calculate total memory requirement
        total_elements = sum(np.prod(shape) for shape in array_shapes)
        total_memory_mb = total_elements * 8e-6  # 8 bytes per double
        
        # Determine optimal block size if not specified
        if self.block_size is None:
            # Target blocks that fit in GPU memory with room for computation
            target_block_memory = gpu_memory_mb * 0.3  # Use 30% of GPU memory per block
            target_elements = int(target_block_memory / 8e-6)
            
            # Find block size that divides arrays reasonably
            max_dim = max(max(shape) for shape in array_shapes)
            self.block_size = min(int(np.sqrt(target_elements)), max_dim // 4)
            self.block_size = max(256, self.block_size)  # Minimum block size
        
        blocks = []
        block_id = 0
        
        for array_idx, shape in enumerate(array_shapes):
            if len(shape) == 2:  # Matrix
                rows, cols = shape
                
                for i in range(0, rows, self.block_size):
                    for j in range(0, cols, self.block_size):
                        end_i = min(i + self.block_size, rows)
                        end_j = min(j + self.block_size, cols)
                        
                        block_elements = (end_i - i) * (end_j - j)
                        block_bytes = block_elements * 8
                        
                        # Assign location based on memory availability
                        location = "gpu" if len([b for b in blocks if b.location == "gpu"]) * block_bytes < gpu_memory_mb * 1e6 * 0.8 else "cpu"
                        
                        block = MemoryBlock(
                            block_id=block_id,
                            start_idx=(i, j),
                            end_idx=(end_i, end_j),
                            size_bytes=block_bytes,
                            location=location
                        )
                        
                        blocks.append(block)
                        block_id += 1
                        
            elif len(shape) == 4:  # 4D tensor (common in quantum chemistry)
                n1, n2, n3, n4 = shape
                
                # Use smaller blocks for 4D arrays due to memory scaling
                tensor_block_size = max(32, self.block_size // 4)
                
                for i in range(0, n1, tensor_block_size):
                    for j in range(0, n2, tensor_block_size):
                        for k in range(0, n3, tensor_block_size):
                            for l in range(0, n4, tensor_block_size):
                                end_i = min(i + tensor_block_size, n1)
                                end_j = min(j + tensor_block_size, n2)
                                end_k = min(k + tensor_block_size, n3)
                                end_l = min(l + tensor_block_size, n4)
                                
                                block_elements = (end_i - i) * (end_j - j) * (end_k - k) * (end_l - l)
                                block_bytes = block_elements * 8
                                
                                location = "gpu" if len([b for b in blocks if b.location == "gpu"]) * block_bytes < gpu_memory_mb * 1e6 * 0.6 else "cpu"
                                
                                block = MemoryBlock(
                                    block_id=block_id,
                                    start_idx=(i, j, k, l),
                                    end_idx=(end_i, end_j, end_k, end_l),
                                    size_bytes=block_bytes,
                                    location=location
                                )
                                
                                blocks.append(block)
                                block_id += 1
        
        # Calculate performance estimate
        gpu_ratio = len([b for b in blocks if b.location == "gpu"]) / len(blocks)
        performance_factor = gpu_ratio * 10 + (1 - gpu_ratio) * 1  # Assume 10x GPU speedup
        
        return MemoryPartitionPlan(
            strategy=MemoryPartitionStrategy.BLOCK_CYCLIC,
            blocks=blocks,
            total_memory_required=total_memory_mb,
            gpu_memory_budget=gpu_memory_mb,
            cpu_memory_budget=cpu_memory_mb,
            estimated_performance_factor=performance_factor
        )


class AdaptivePartitioner(MemoryPartitioner):
    """Adaptive memory partitioning based on access patterns and system characteristics."""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 history_window: int = 100):
        """
        Initialize adaptive partitioner.
        
        Args:
            learning_rate: Rate for adapting to access patterns
            history_window: Number of recent accesses to consider
        """
        self.learning_rate = learning_rate
        self.history_window = history_window
        self.access_history: List[Tuple[int, float]] = []  # (block_id, timestamp)
    
    def partition(self,
                 array_shapes: List[Tuple[int, ...]],
                 gpu_memory_mb: float,
                 cpu_memory_mb: float,
                 access_pattern: Optional[str] = None) -> MemoryPartitionPlan:
        """Create adaptive partition plan based on learned access patterns."""
        
        # Start with block-cyclic as baseline
        baseline_partitioner = BlockCyclicPartitioner()
        baseline_plan = baseline_partitioner.partition(array_shapes, gpu_memory_mb, cpu_memory_mb)
        
        # Adjust based on access patterns if available
        if access_pattern and self.access_history:
            self._optimize_block_placement(baseline_plan)
        
        return baseline_plan
    
    def _optimize_block_placement(self, plan: MemoryPartitionPlan) -> None:
        """Optimize block placement based on access history."""
        
        # Calculate block access frequencies
        block_frequencies = {}
        for block_id, timestamp in self.access_history[-self.history_window:]:
            block_frequencies[block_id] = block_frequencies.get(block_id, 0) + 1
        
        # Sort blocks by access frequency
        blocks_by_frequency = sorted(plan.blocks, 
                                   key=lambda b: block_frequencies.get(b.block_id, 0), 
                                   reverse=True)
        
        # Reassign high-frequency blocks to GPU
        gpu_memory_used = 0
        for block in blocks_by_frequency:
            if gpu_memory_used + block.size_bytes <= plan.gpu_memory_budget * 1e6 * 0.8:
                block.location = "gpu"
                gpu_memory_used += block.size_bytes
            else:
                block.location = "cpu"
    
    def record_access(self, block_id: int, timestamp: float) -> None:
        """Record block access for learning."""
        self.access_history.append((block_id, timestamp))
        
        # Maintain history window
        if len(self.access_history) > self.history_window * 2:
            self.access_history = self.access_history[-self.history_window:]


class StreamingPartitioner(MemoryPartitioner):
    """Streaming memory partitioner with CPU/GPU overlap."""
    
    def __init__(self, 
                 stream_buffer_size: int = 2,
                 prefetch_blocks: int = 1):
        """
        Initialize streaming partitioner.
        
        Args:
            stream_buffer_size: Number of blocks to buffer
            prefetch_blocks: Number of blocks to prefetch
        """
        self.stream_buffer_size = stream_buffer_size
        self.prefetch_blocks = prefetch_blocks
    
    def partition(self,
                 array_shapes: List[Tuple[int, ...]],
                 gpu_memory_mb: float,
                 cpu_memory_mb: float,
                 access_pattern: Optional[str] = None) -> MemoryPartitionPlan:
        """Create streaming partition plan with overlap optimization."""
        
        # Calculate optimal streaming block size
        target_block_memory = gpu_memory_mb / (self.stream_buffer_size + self.prefetch_blocks)
        target_elements = int(target_block_memory * 1e6 / 8)
        
        blocks = []
        block_id = 0
        
        for shape in array_shapes:
            if len(shape) == 2:
                rows, cols = shape
                
                # Calculate streaming block dimensions
                block_rows = min(rows, int(np.sqrt(target_elements * rows / cols)))
                block_cols = min(cols, target_elements // block_rows)
                
                for i in range(0, rows, block_rows):
                    for j in range(0, cols, block_cols):
                        end_i = min(i + block_rows, rows)
                        end_j = min(j + block_cols, cols)
                        
                        block_elements = (end_i - i) * (end_j - j)
                        block_bytes = block_elements * 8
                        
                        # First few blocks go to GPU, rest to CPU for streaming
                        location = "gpu" if block_id < self.stream_buffer_size else "cpu"
                        
                        block = MemoryBlock(
                            block_id=block_id,
                            start_idx=(i, j),
                            end_idx=(end_i, end_j),
                            size_bytes=block_bytes,
                            location=location,
                            priority=1.0 / (block_id + 1)  # Higher priority for earlier blocks
                        )
                        
                        blocks.append(block)
                        block_id += 1
        
        total_memory_mb = sum(block.size_bytes for block in blocks) / 1e6
        
        # Performance estimate accounts for streaming overlap
        overlap_efficiency = 0.8  # Assume 80% overlap efficiency
        performance_factor = self.stream_buffer_size + overlap_efficiency * (len(blocks) - self.stream_buffer_size)
        
        return MemoryPartitionPlan(
            strategy=MemoryPartitionStrategy.STREAMING,
            blocks=blocks,
            total_memory_required=total_memory_mb,
            gpu_memory_budget=gpu_memory_mb,
            cpu_memory_budget=cpu_memory_mb,
            estimated_performance_factor=performance_factor
        )


class MemoryOptimizer:
    """
    Central memory optimization manager for GPU-accelerated quantum chemistry calculations.
    
    This class coordinates different partitioning strategies and provides unified
    memory management for large-scale calculations that exceed GPU memory limits.
    """
    
    def __init__(self, 
                 gpu_manager: Optional[GPUAccelerationManager] = None,
                 default_strategy: MemoryPartitionStrategy = MemoryPartitionStrategy.ADAPTIVE):
        """
        Initialize memory optimizer.
        
        Args:
            gpu_manager: GPU acceleration manager (uses global if None)
            default_strategy: Default partitioning strategy
        """
        from .gpu_manager import get_gpu_manager
        
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.default_strategy = default_strategy
        
        # Initialize partitioners
        self.partitioners: Dict[MemoryPartitionStrategy, MemoryPartitioner] = {
            MemoryPartitionStrategy.BLOCK_CYCLIC: BlockCyclicPartitioner(),
            MemoryPartitionStrategy.ADAPTIVE: AdaptivePartitioner(),
            MemoryPartitionStrategy.STREAMING: StreamingPartitioner()
        }
        
        # Active partition plans
        self.active_plans: Dict[str, MemoryPartitionPlan] = {}
        
    def create_partition_plan(self,
                            calculation_id: str,
                            array_shapes: List[Tuple[int, ...]],
                            method: str = "casscf",
                            strategy: Optional[MemoryPartitionStrategy] = None,
                            access_pattern: Optional[str] = None) -> MemoryPartitionPlan:
        """
        Create optimized memory partition plan for calculation.
        
        Args:
            calculation_id: Unique identifier for calculation
            array_shapes: Shapes of arrays to partition
            method: Quantum chemistry method
            strategy: Partitioning strategy (uses default if None)
            access_pattern: Expected access pattern hint
            
        Returns:
            Optimized memory partition plan
        """
        strategy = strategy or self.default_strategy
        
        # Get memory constraints
        if self.gpu_manager.selected_gpu:
            gpu_memory_mb = self.gpu_manager.selected_gpu.memory_info.available_mb * self.gpu_manager.memory_fraction
        else:
            gpu_memory_mb = 0.0
            
        # Estimate available CPU memory (conservative)
        cpu_memory_mb = 4096.0  # 4GB default, could be system-dependent
        
        # Create partition plan
        partitioner = self.partitioners[strategy]
        plan = partitioner.partition(array_shapes, gpu_memory_mb, cpu_memory_mb, access_pattern)
        
        # Store active plan
        self.active_plans[calculation_id] = plan
        
        logger.info(f"Created {strategy} partition plan for {calculation_id}: "
                   f"{plan.num_blocks} blocks, {len(plan.gpu_blocks)} on GPU")
        
        return plan
    
    def estimate_performance_improvement(self,
                                       array_shapes: List[Tuple[int, ...]],
                                       method: str = "casscf") -> Dict[str, float]:
        """
        Estimate performance improvement for different strategies.
        
        Args:
            array_shapes: Array shapes to analyze
            method: Quantum chemistry method
            
        Returns:
            Dictionary of strategy -> performance factor estimates
        """
        estimates = {}
        
        if not self.gpu_manager.selected_gpu:
            return {"cpu_only": 1.0}
        
        gpu_memory_mb = self.gpu_manager.selected_gpu.memory_info.available_mb * self.gpu_manager.memory_fraction
        cpu_memory_mb = 4096.0
        
        for strategy, partitioner in self.partitioners.items():
            try:
                plan = partitioner.partition(array_shapes, gpu_memory_mb, cpu_memory_mb)
                estimates[strategy.value] = plan.estimated_performance_factor
            except Exception as e:
                logger.warning(f"Could not estimate performance for {strategy}: {e}")
                estimates[strategy.value] = 1.0
        
        return estimates
    
    def recommend_strategy(self,
                          array_shapes: List[Tuple[int, ...]],
                          method: str = "casscf",
                          priority: str = "performance") -> MemoryPartitionStrategy:
        """
        Recommend optimal partitioning strategy.
        
        Args:
            array_shapes: Array shapes to analyze
            method: Quantum chemistry method
            priority: Optimization priority ("performance", "memory", "balanced")
            
        Returns:
            Recommended partitioning strategy
        """
        if not self.gpu_manager.is_gpu_available():
            return MemoryPartitionStrategy.BLOCK_CYCLIC  # CPU fallback
        
        # Get performance estimates
        estimates = self.estimate_performance_improvement(array_shapes, method)
        
        # Calculate total memory requirement
        total_elements = sum(np.prod(shape) for shape in array_shapes)
        total_memory_mb = total_elements * 8e-6
        
        gpu_memory_mb = self.gpu_manager.selected_gpu.memory_info.available_mb * self.gpu_manager.memory_fraction
        
        # Decision logic based on priority and system characteristics
        if priority == "performance":
            # Choose strategy with highest performance estimate
            best_strategy = max(estimates.items(), key=lambda x: x[1])[0]
            return MemoryPartitionStrategy(best_strategy)
            
        elif priority == "memory":
            # Choose strategy that minimizes memory fragmentation
            if total_memory_mb <= gpu_memory_mb:
                return MemoryPartitionStrategy.BLOCK_CYCLIC
            else:
                return MemoryPartitionStrategy.STREAMING
                
        else:  # balanced
            # Balance performance and memory efficiency
            if total_memory_mb <= gpu_memory_mb * 0.8:  # Fits comfortably
                return MemoryPartitionStrategy.BLOCK_CYCLIC
            elif total_memory_mb <= gpu_memory_mb * 2:  # Moderate overflow
                return MemoryPartitionStrategy.ADAPTIVE
            else:  # Large overflow
                return MemoryPartitionStrategy.STREAMING
    
    def get_memory_statistics(self, calculation_id: str) -> Dict[str, any]:
        """
        Get memory usage statistics for active calculation.
        
        Args:
            calculation_id: Calculation identifier
            
        Returns:
            Memory usage statistics
        """
        if calculation_id not in self.active_plans:
            return {}
        
        plan = self.active_plans[calculation_id]
        
        gpu_memory_used = sum(block.size_bytes for block in plan.gpu_blocks) / 1e6
        cpu_memory_used = sum(block.size_bytes for block in plan.cpu_blocks) / 1e6
        
        return {
            'strategy': plan.strategy.value,
            'total_blocks': plan.num_blocks,
            'gpu_blocks': len(plan.gpu_blocks),
            'cpu_blocks': len(plan.cpu_blocks),
            'gpu_memory_used_mb': gpu_memory_used,
            'cpu_memory_used_mb': cpu_memory_used,
            'total_memory_required_mb': plan.total_memory_required,
            'gpu_memory_budget_mb': plan.gpu_memory_budget,
            'estimated_performance_factor': plan.estimated_performance_factor
        }
    
    def cleanup_calculation(self, calculation_id: str) -> None:
        """
        Clean up resources for completed calculation.
        
        Args:
            calculation_id: Calculation identifier to clean up
        """
        if calculation_id in self.active_plans:
            del self.active_plans[calculation_id]
            logger.info(f"Cleaned up memory plan for {calculation_id}")
    
    def get_optimization_report(self) -> Dict[str, any]:
        """
        Generate comprehensive memory optimization report.
        
        Returns:
            Detailed optimization report
        """
        report = {
            'gpu_available': self.gpu_manager.is_gpu_available(),
            'active_calculations': len(self.active_plans),
            'default_strategy': self.default_strategy.value,
            'available_strategies': list(self.partitioners.keys()),
            'gpu_status': self.gpu_manager.get_gpu_status() if self.gpu_manager.is_gpu_available() else None,
            'memory_statistics': {}
        }
        
        # Add statistics for active calculations
        for calc_id in self.active_plans:
            report['memory_statistics'][calc_id] = self.get_memory_statistics(calc_id)
        
        return report