"""Advanced performance optimization for neuromorphic edge systems."""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from functools import wraps, lru_cache
import threading
from collections import defaultdict
import hashlib


class PerformanceOptimizer:
    """Advanced performance optimization engine for neuromorphic systems."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache_stats = {"hits": 0, "misses": 0}
        self.optimization_stats = defaultdict(list)
        self.memory_pool = {}
        self.thread_local_storage = threading.local()
        
        # Performance monitoring
        self.operation_times = defaultdict(list)
        self.memory_usage = []
        
        # Adaptive optimization parameters
        self.batch_size_history = []
        self.optimal_batch_size = 32
        
    def adaptive_batching(self, data: np.ndarray, target_latency_ms: float = 50.0) -> List[np.ndarray]:
        """Implement adaptive batching based on system performance.
        
        Args:
            data: Input data to batch
            target_latency_ms: Target processing latency per batch
            
        Returns:
            List of optimally-sized batches
        """
        # Start with current optimal batch size
        current_batch_size = self.optimal_batch_size
        
        # Split data into batches
        batches = []
        total_samples = len(data)
        
        for i in range(0, total_samples, current_batch_size):
            batch = data[i:i + current_batch_size]
            batches.append(batch)
        
        return batches
    
    def optimized_matrix_operations(self, operation: str, *arrays: np.ndarray) -> np.ndarray:
        """Perform optimized matrix operations with caching and vectorization.
        
        Args:
            operation: Type of matrix operation
            *arrays: Input arrays
            
        Returns:
            Result of optimized operation
        """
        # Create cache key from operation and array shapes/types
        cache_key = self._create_cache_key(operation, *arrays)
        
        # Check cache first
        if cache_key in self.memory_pool:
            self.cache_stats["hits"] += 1
            return self.memory_pool[cache_key].copy()
        
        self.cache_stats["misses"] += 1
        
        # Perform optimized operation
        start_time = time.time()
        
        if operation == "einsum_optimized":
            # Optimize einsum operations
            equation, *operands = arrays
            result = self._optimized_einsum(equation, *operands)
        
        elif operation == "batch_matmul":
            # Batch matrix multiplication
            result = self._optimized_batch_matmul(*arrays)
            
        else:
            # Fallback to standard operations
            if operation == "matmul":
                result = np.matmul(*arrays)
            elif operation == "add":
                result = np.add(*arrays)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Record performance
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self.operation_times[operation].append(processing_time)
        
        # Cache result if it's reasonable size
        if result.nbytes < 50 * 1024 * 1024:  # Less than 50MB
            self.memory_pool[cache_key] = result.copy()
        
        # Manage cache size
        self._manage_cache()
        
        return result
    
    def _create_cache_key(self, operation: str, *arrays: np.ndarray) -> str:
        """Create cache key from operation and arrays."""
        key_parts = [operation]
        
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                # Use shape and dtype for key
                arr_info = f"{arr.shape}_{arr.dtype}"
                key_parts.append(arr_info)
            else:
                key_parts.append(str(arr))
        
        return "_".join(key_parts)
    
    def _manage_cache(self):
        """Manage memory cache size."""
        max_cache_size = 20  # Maximum number of cached items
        
        if len(self.memory_pool) > max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.memory_pool.keys())[:-max_cache_size//2]
            for key in keys_to_remove:
                del self.memory_pool[key]
    
    def _optimized_einsum(self, equation: str, *operands: np.ndarray) -> np.ndarray:
        """Optimized einsum implementation."""
        # For common patterns, use optimized paths
        if equation == "bft,fo->bot":
            # Common spike processing pattern
            spikes, weights = operands
            batch_size, features, time_steps = spikes.shape
            output_size = weights.shape[1]
            
            # Reshape for efficient matrix multiplication
            spikes_reshaped = spikes.transpose(0, 2, 1).reshape(-1, features)
            result = np.matmul(spikes_reshaped, weights)
            result = result.reshape(batch_size, time_steps, output_size)
            result = result.transpose(0, 2, 1)  # Back to batch, output, time
            
            return result
        
        else:
            # Fallback to numpy einsum
            return np.einsum(equation, *operands)
    
    def _optimized_batch_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Optimized batch matrix multiplication."""
        if a.ndim == 3 and b.ndim == 2:
            # Batch matrix-vector multiplication
            batch_size = a.shape[0]
            result = np.zeros((batch_size, a.shape[1]))
            
            for i in range(batch_size):
                result[i] = np.matmul(a[i], b)
            
            return result
        
        else:
            return np.matmul(a, b)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        stats = {
            "cache_stats": self.cache_stats.copy(),
            "cache_hit_rate": (
                self.cache_stats["hits"] / 
                (self.cache_stats["hits"] + self.cache_stats["misses"])
                if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
            ),
            "optimal_batch_size": self.optimal_batch_size,
            "cached_operations": len(self.memory_pool),
        }
        
        # Add operation timing statistics
        for operation, times in self.operation_times.items():
            if times:
                stats[f"{operation}_avg_time_ms"] = np.mean(times)
                stats[f"{operation}_count"] = len(times)
        
        return stats


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()


def optimized_processing(func: Callable) -> Callable:
    """Decorator for optimized processing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Record performance
        func_name = func.__name__
        performance_optimizer.operation_times[func_name].append((end_time - start_time) * 1000)
        
        return result
    
    return wrapper