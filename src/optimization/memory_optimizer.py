"""Advanced memory optimization for neuromorphic computing."""

import torch
import numpy as np
import gc
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import weakref
from collections import defaultdict, deque
import psutil
import pickle
import tempfile
import os


class MemoryTier(Enum):
    """Memory hierarchy tiers."""
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    STORAGE = "storage"


class OptimizationStrategy(Enum):
    """Memory optimization strategies."""
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    TENSOR_COMPRESSION = "tensor_compression"
    MEMORY_POOLING = "memory_pooling"
    SPARSE_STORAGE = "sparse_storage"
    QUANTIZATION = "quantization"
    OFFLOADING = "offloading"


@dataclass
class MemoryUsage:
    """Memory usage information."""
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_bytes: int
    utilization_percent: float
    fragmentation_percent: float


@dataclass
class OptimizationResult:
    """Result of memory optimization."""
    strategy: OptimizationStrategy
    memory_saved_bytes: int
    performance_impact: float  # Negative = slowdown, positive = speedup
    success: bool
    details: Optional[Dict[str, Any]] = None


class TensorPool:
    """Memory pool for tensor reuse."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize tensor pool.
        
        Args:
            device: Device for tensor allocation
        """
        self.device = device
        self.pools: Dict[Tuple[tuple, torch.dtype], deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_tensors': 0,
            'memory_saved_bytes': 0
        }
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor from pool or newly created
        """
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].popleft()
                tensor.zero_()  # Clear contents
                self.stats['reuses'] += 1
                return tensor
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self.stats['allocations'] += 1
                self.stats['total_tensors'] += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return
        """
        if tensor.device.type != self.device.split(':')[0]:
            return  # Wrong device
        
        key = (tuple(tensor.shape), tensor.dtype)
        
        with self.lock:
            # Limit pool size per shape/dtype
            if len(self.pools[key]) < 10:
                self.pools[key].append(tensor.detach())
                memory_size = tensor.numel() * tensor.element_size()
                self.stats['memory_saved_bytes'] += memory_size
    
    def clear(self):
        """Clear all pools."""
        with self.lock:
            self.pools.clear()
            self.stats['total_tensors'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pools_count': len(self.pools),
                'total_tensors_in_pools': sum(len(pool) for pool in self.pools.values()),
                **self.stats.copy()
            }


class SparseStorageManager:
    """Manager for sparse tensor storage optimization."""
    
    def __init__(self, sparsity_threshold: float = 0.7):
        """Initialize sparse storage manager.
        
        Args:
            sparsity_threshold: Minimum sparsity to use sparse storage
        """
        self.sparsity_threshold = sparsity_threshold
        self.sparse_tensors: Dict[int, Tuple[torch.Tensor, Dict]] = {}
        self.compression_stats = {
            'tensors_compressed': 0,
            'memory_saved_bytes': 0,
            'compression_ratio': 0.0
        }
    
    def try_sparse_conversion(self, tensor: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor]]:
        """Try to convert tensor to sparse format.
        
        Args:
            tensor: Dense tensor to convert
            
        Returns:
            Tuple of (success, sparse_tensor)
        """
        if tensor.numel() == 0:
            return False, None
        
        # Calculate sparsity
        zero_elements = (tensor == 0).sum().item()
        sparsity = zero_elements / tensor.numel()
        
        if sparsity < self.sparsity_threshold:
            return False, None
        
        try:
            # Convert to COO format
            sparse_tensor = tensor.to_sparse()
            
            # Calculate memory savings
            dense_size = tensor.numel() * tensor.element_size()
            sparse_size = sparse_tensor._nnz() * tensor.element_size() * (tensor.dim() + 1)
            
            if sparse_size < dense_size:
                # Store metadata
                tensor_id = id(tensor)
                self.sparse_tensors[tensor_id] = (sparse_tensor, {
                    'original_shape': tensor.shape,
                    'sparsity': sparsity,
                    'memory_saved': dense_size - sparse_size
                })
                
                self.compression_stats['tensors_compressed'] += 1
                self.compression_stats['memory_saved_bytes'] += dense_size - sparse_size
                
                return True, sparse_tensor
            
        except Exception:
            pass
        
        return False, None
    
    def get_dense_tensor(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Convert sparse tensor back to dense.
        
        Args:
            sparse_tensor: Sparse tensor
            
        Returns:
            Dense tensor
        """
        return sparse_tensor.to_dense()
    
    def cleanup_references(self):
        """Clean up dead tensor references."""
        dead_ids = []
        for tensor_id in self.sparse_tensors:
            # Check if original tensor still exists
            # This is a simplified check - in practice, you'd use weak references
            pass
        
        for tensor_id in dead_ids:
            del self.sparse_tensors[tensor_id]


class QuantizationManager:
    """Manager for tensor quantization."""
    
    def __init__(self):
        """Initialize quantization manager."""
        self.quantized_tensors: Dict[int, Tuple[torch.Tensor, Dict]] = {}
        self.quantization_stats = {
            'tensors_quantized': 0,
            'memory_saved_bytes': 0,
            'average_precision_loss': 0.0
        }
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        method: str = "linear"
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Quantize tensor to reduce memory usage.
        
        Args:
            tensor: Tensor to quantize
            bits: Number of bits for quantization
            method: Quantization method
            
        Returns:
            Tuple of (success, quantized_tensor)
        """
        if tensor.dtype not in [torch.float32, torch.float64]:
            return False, None
        
        try:
            if method == "linear":
                # Simple linear quantization
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                
                if tensor_min == tensor_max:
                    # Constant tensor
                    return False, None
                
                # Quantize to specified bits
                scale = (tensor_max - tensor_min) / (2 ** bits - 1)
                zero_point = tensor_min
                
                quantized = torch.round((tensor - zero_point) / scale).clamp(0, 2 ** bits - 1)
                
                # Convert to appropriate integer type
                if bits <= 8:
                    quantized = quantized.to(torch.uint8)
                elif bits <= 16:
                    quantized = quantized.to(torch.int16)
                else:
                    return False, None
                
                # Calculate memory savings
                original_size = tensor.numel() * tensor.element_size()
                quantized_size = quantized.numel() * quantized.element_size()
                
                if quantized_size < original_size:
                    # Store quantization parameters
                    tensor_id = id(tensor)
                    self.quantized_tensors[tensor_id] = (quantized, {
                        'scale': scale.item(),
                        'zero_point': zero_point.item(),
                        'original_dtype': tensor.dtype,
                        'original_shape': tensor.shape,
                        'memory_saved': original_size - quantized_size
                    })
                    
                    self.quantization_stats['tensors_quantized'] += 1
                    self.quantization_stats['memory_saved_bytes'] += original_size - quantized_size
                    
                    return True, quantized
            
        except Exception:
            pass
        
        return False, None
    
    def dequantize_tensor(self, tensor_id: int) -> Optional[torch.Tensor]:
        """Dequantize tensor back to original precision.
        
        Args:
            tensor_id: ID of quantized tensor
            
        Returns:
            Dequantized tensor or None
        """
        if tensor_id not in self.quantized_tensors:
            return None
        
        quantized, params = self.quantized_tensors[tensor_id]
        
        # Dequantize
        dequantized = quantized.to(params['original_dtype'])
        dequantized = dequantized * params['scale'] + params['zero_point']
        
        return dequantized


class MemoryOffloader:
    """Offload tensors to different memory tiers."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize memory offloader.
        
        Args:
            temp_dir: Temporary directory for storage offloading
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.offloaded_tensors: Dict[int, Dict] = {}
        self.offload_stats = {
            'tensors_offloaded': 0,
            'bytes_offloaded': 0,
            'storage_files': 0
        }
    
    def offload_to_storage(self, tensor: torch.Tensor) -> bool:
        """Offload tensor to storage.
        
        Args:
            tensor: Tensor to offload
            
        Returns:
            True if successful
        """
        try:
            tensor_id = id(tensor)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                dir=self.temp_dir,
                delete=False,
                suffix='.tensor'
            )
            
            # Save tensor
            torch.save(tensor, temp_file.name)
            temp_file.close()
            
            # Store metadata
            self.offloaded_tensors[tensor_id] = {
                'file_path': temp_file.name,
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
                'size_bytes': tensor.numel() * tensor.element_size()
            }
            
            self.offload_stats['tensors_offloaded'] += 1
            self.offload_stats['bytes_offloaded'] += tensor.numel() * tensor.element_size()
            self.offload_stats['storage_files'] += 1
            
            return True
            
        except Exception:
            return False
    
    def load_from_storage(self, tensor_id: int) -> Optional[torch.Tensor]:
        """Load tensor from storage.
        
        Args:
            tensor_id: ID of offloaded tensor
            
        Returns:
            Loaded tensor or None
        """
        if tensor_id not in self.offloaded_tensors:
            return None
        
        try:
            metadata = self.offloaded_tensors[tensor_id]
            tensor = torch.load(metadata['file_path'])
            return tensor
            
        except Exception:
            return None
    
    def cleanup_storage(self, tensor_id: int):
        """Clean up storage for tensor.
        
        Args:
            tensor_id: ID of tensor to clean up
        """
        if tensor_id in self.offloaded_tensors:
            metadata = self.offloaded_tensors[tensor_id]
            try:
                os.unlink(metadata['file_path'])
                self.offload_stats['storage_files'] -= 1
            except OSError:
                pass
            
            del self.offloaded_tensors[tensor_id]
            self.offload_stats['tensors_offloaded'] -= 1


class MemoryOptimizer:
    """Main memory optimization coordinator."""
    
    def __init__(
        self,
        target_memory_usage: float = 0.8,
        optimization_interval: float = 60.0,
        enable_automatic_optimization: bool = True
    ):
        """Initialize memory optimizer.
        
        Args:
            target_memory_usage: Target memory usage ratio (0-1)
            optimization_interval: Seconds between automatic optimizations
            enable_automatic_optimization: Enable automatic optimization
        """
        self.target_memory_usage = target_memory_usage
        self.optimization_interval = optimization_interval
        self.enable_automatic_optimization = enable_automatic_optimization
        
        # Optimization components
        self.tensor_pool = TensorPool()
        self.sparse_manager = SparseStorageManager()
        self.quantization_manager = QuantizationManager()
        self.offloader = MemoryOffloader()
        
        # Statistics and monitoring
        self.optimization_history: List[Dict] = []
        self.memory_history: deque = deque(maxlen=1000)
        
        # Automatic optimization
        self._optimization_thread: Optional[threading.Thread] = None
        self._stop_optimization = threading.Event()
        
        # Device detection
        self.has_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.has_cuda else "cpu"
        
        if self.has_cuda:
            self.tensor_pool = TensorPool("cuda")
    
    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage.
        
        Returns:
            Memory usage information
        """
        if self.has_cuda:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - reserved
        else:
            # System memory
            memory = psutil.virtual_memory()
            allocated = memory.used
            reserved = memory.used
            total = memory.total
            free = memory.available
        
        utilization = (reserved / total) * 100 if total > 0 else 0
        fragmentation = ((reserved - allocated) / reserved) * 100 if reserved > 0 else 0
        
        return MemoryUsage(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_bytes=free,
            total_bytes=total,
            utilization_percent=utilization,
            fragmentation_percent=fragmentation
        )
    
    def should_optimize(self) -> bool:
        """Check if optimization is needed.
        
        Returns:
            True if optimization should be performed
        """
        memory_usage = self.get_memory_usage()
        current_usage_ratio = memory_usage.reserved_bytes / memory_usage.total_bytes
        
        return current_usage_ratio > self.target_memory_usage
    
    def optimize_memory(self, strategies: Optional[List[OptimizationStrategy]] = None) -> List[OptimizationResult]:
        """Perform memory optimization.
        
        Args:
            strategies: Specific strategies to apply (all if None)
            
        Returns:
            List of optimization results
        """
        if strategies is None:
            strategies = list(OptimizationStrategy)
        
        results = []
        initial_memory = self.get_memory_usage()
        
        for strategy in strategies:
            try:
                result = self._apply_strategy(strategy)
                results.append(result)
                
                if result.success and result.memory_saved_bytes > 0:
                    print(f"Applied {strategy.value}: saved {result.memory_saved_bytes / 1024**2:.1f}MB")
                
            except Exception as e:
                results.append(OptimizationResult(
                    strategy=strategy,
                    memory_saved_bytes=0,
                    performance_impact=0.0,
                    success=False,
                    details={'error': str(e)}
                ))
        
        final_memory = self.get_memory_usage()
        total_saved = initial_memory.reserved_bytes - final_memory.reserved_bytes
        
        # Record optimization session
        self.optimization_history.append({
            'timestamp': time.time(),
            'strategies_applied': [s.value for s in strategies],
            'memory_saved_bytes': total_saved,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'results': results
        })
        
        return results
    
    def _apply_strategy(self, strategy: OptimizationStrategy) -> OptimizationResult:
        """Apply specific optimization strategy.
        
        Args:
            strategy: Optimization strategy to apply
            
        Returns:
            Optimization result
        """
        if strategy == OptimizationStrategy.MEMORY_POOLING:
            return self._optimize_with_pooling()
        
        elif strategy == OptimizationStrategy.SPARSE_STORAGE:
            return self._optimize_with_sparse_storage()
        
        elif strategy == OptimizationStrategy.QUANTIZATION:
            return self._optimize_with_quantization()
        
        elif strategy == OptimizationStrategy.OFFLOADING:
            return self._optimize_with_offloading()
        
        elif strategy == OptimizationStrategy.GRADIENT_CHECKPOINTING:
            return self._optimize_with_checkpointing()
        
        elif strategy == OptimizationStrategy.TENSOR_COMPRESSION:
            return self._optimize_with_compression()
        
        else:
            return OptimizationResult(
                strategy=strategy,
                memory_saved_bytes=0,
                performance_impact=0.0,
                success=False,
                details={'error': 'Strategy not implemented'}
            )
    
    def _optimize_with_pooling(self) -> OptimizationResult:
        """Optimize using tensor pooling."""
        initial_stats = self.tensor_pool.get_stats()
        
        # Force garbage collection to clean up unused tensors
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()
        
        final_stats = self.tensor_pool.get_stats()
        memory_saved = final_stats.get('memory_saved_bytes', 0) - initial_stats.get('memory_saved_bytes', 0)
        
        return OptimizationResult(
            strategy=OptimizationStrategy.MEMORY_POOLING,
            memory_saved_bytes=memory_saved,
            performance_impact=0.1,  # Small speedup from reuse
            success=True,
            details=final_stats
        )
    
    def _optimize_with_sparse_storage(self) -> OptimizationResult:
        """Optimize using sparse tensor storage."""
        # This would require integration with actual tensor tracking
        # For now, return a placeholder result
        
        return OptimizationResult(
            strategy=OptimizationStrategy.SPARSE_STORAGE,
            memory_saved_bytes=0,
            performance_impact=0.0,
            success=True,
            details={'message': 'Sparse storage optimization placeholder'}
        )
    
    def _optimize_with_quantization(self) -> OptimizationResult:
        """Optimize using tensor quantization."""
        # This would require integration with model parameters
        # For now, return a placeholder result
        
        return OptimizationResult(
            strategy=OptimizationStrategy.QUANTIZATION,
            memory_saved_bytes=0,
            performance_impact=-0.05,  # Small slowdown from quantization
            success=True,
            details={'message': 'Quantization optimization placeholder'}
        )
    
    def _optimize_with_offloading(self) -> OptimizationResult:
        """Optimize using memory offloading."""
        # This would require integration with tensor lifecycle management
        
        return OptimizationResult(
            strategy=OptimizationStrategy.OFFLOADING,
            memory_saved_bytes=0,
            performance_impact=-0.2,  # Slowdown from storage I/O
            success=True,
            details={'message': 'Offloading optimization placeholder'}
        )
    
    def _optimize_with_checkpointing(self) -> OptimizationResult:
        """Optimize using gradient checkpointing."""
        # This would be integrated with model training
        
        return OptimizationResult(
            strategy=OptimizationStrategy.GRADIENT_CHECKPOINTING,
            memory_saved_bytes=0,
            performance_impact=-0.15,  # Slowdown from recomputation
            success=True,
            details={'message': 'Gradient checkpointing placeholder'}
        )
    
    def _optimize_with_compression(self) -> OptimizationResult:
        """Optimize using tensor compression."""
        # This would use compression algorithms on tensors
        
        return OptimizationResult(
            strategy=OptimizationStrategy.TENSOR_COMPRESSION,
            memory_saved_bytes=0,
            performance_impact=-0.1,  # Slowdown from compression/decompression
            success=True,
            details={'message': 'Tensor compression placeholder'}
        )
    
    def start_automatic_optimization(self):
        """Start automatic memory optimization."""
        if self._optimization_thread and self._optimization_thread.is_alive():
            return
        
        self._stop_optimization.clear()
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="MemoryOptimizer",
            daemon=True
        )
        self._optimization_thread.start()
        
        print("Automatic memory optimization started")
    
    def stop_automatic_optimization(self):
        """Stop automatic memory optimization."""
        if self._optimization_thread:
            self._stop_optimization.set()
            self._optimization_thread.join(timeout=5)
        
        print("Automatic memory optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while not self._stop_optimization.is_set():
            try:
                # Record current memory usage
                memory_usage = self.get_memory_usage()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'usage': memory_usage
                })
                
                # Check if optimization is needed
                if self.should_optimize():
                    print(f"Memory usage {memory_usage.utilization_percent:.1f}% - starting optimization")
                    results = self.optimize_memory()
                    
                    successful_optimizations = [r for r in results if r.success]
                    total_saved = sum(r.memory_saved_bytes for r in successful_optimizations)
                    
                    if total_saved > 0:
                        print(f"Optimization completed: saved {total_saved / 1024**2:.1f}MB")
                
            except Exception as e:
                print(f"Error in memory optimization loop: {e}")
            
            # Wait for next optimization cycle
            self._stop_optimization.wait(self.optimization_interval)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report.
        
        Returns:
            Optimization report
        """
        current_memory = self.get_memory_usage()
        
        # Calculate total savings across all optimizations
        total_saved = sum(
            session.get('memory_saved_bytes', 0)
            for session in self.optimization_history
        )
        
        # Recent memory trend
        recent_memory = list(self.memory_history)[-10:] if self.memory_history else []
        memory_trend = 0.0
        if len(recent_memory) >= 2:
            first_usage = recent_memory[0]['usage'].utilization_percent
            last_usage = recent_memory[-1]['usage'].utilization_percent
            memory_trend = last_usage - first_usage
        
        return {
            'current_memory': current_memory,
            'total_optimizations': len(self.optimization_history),
            'total_memory_saved_mb': total_saved / 1024**2,
            'memory_trend_percent': memory_trend,
            'automatic_optimization_active': (
                self._optimization_thread and self._optimization_thread.is_alive()
            ),
            'optimization_components': {
                'tensor_pool': self.tensor_pool.get_stats(),
                'sparse_storage': self.sparse_manager.compression_stats,
                'quantization': self.quantization_manager.quantization_stats,
                'offloading': self.offloader.offload_stats
            },
            'recent_optimizations': self.optimization_history[-5:] if self.optimization_history else []
        }
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history.clear()
        self.memory_history.clear()
        
        # Clear component statistics
        self.tensor_pool.clear()
        self.sparse_manager.compression_stats = {
            'tensors_compressed': 0,
            'memory_saved_bytes': 0,
            'compression_ratio': 0.0
        }
    
    def __enter__(self):
        """Context manager entry."""
        if self.enable_automatic_optimization:
            self.start_automatic_optimization()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enable_automatic_optimization:
            self.stop_automatic_optimization()


# Global memory optimizer instance
_global_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer


def optimize_memory(strategies: Optional[List[OptimizationStrategy]] = None) -> List[OptimizationResult]:
    """Optimize memory using global optimizer."""
    return get_memory_optimizer().optimize_memory(strategies)


def get_memory_usage() -> MemoryUsage:
    """Get current memory usage."""
    return get_memory_optimizer().get_memory_usage()


def start_memory_monitoring():
    """Start automatic memory monitoring."""
    get_memory_optimizer().start_automatic_optimization()


def stop_memory_monitoring():
    """Stop automatic memory monitoring."""
    get_memory_optimizer().stop_automatic_optimization()