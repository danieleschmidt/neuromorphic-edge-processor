"""Advanced optimization algorithms for neuromorphic edge processors."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict, deque
from dataclasses import dataclass
import pickle

from ..utils.logging import get_logger
from ..monitoring.health_monitor import HealthMonitor


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_pruning: bool = False
    pruning_threshold: float = 0.01
    enable_gradient_compression: bool = False
    compression_ratio: float = 0.1
    enable_mixed_precision: bool = False
    use_gpu_acceleration: bool = True
    batch_size_optimization: bool = True
    memory_mapping: bool = True


class AdaptiveCache:
    """Adaptive caching system for computation results."""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.size_tracker = {}
        self.total_size = 0
        self.lock = threading.RLock()
        self.logger = get_logger("optimization.cache")
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _compute_key(self, *args, **kwargs) -> str:
        """Compute cache key from arguments."""
        import hashlib
        
        def serialize_obj(obj):
            if isinstance(obj, torch.Tensor):
                return f"tensor_{obj.shape}_{obj.dtype}_{obj.sum().item():.6f}"
            elif isinstance(obj, (list, tuple)):
                return str([serialize_obj(item) for item in obj])
            elif isinstance(obj, dict):
                return str({k: serialize_obj(v) for k, v in sorted(obj.items())})
            else:
                return str(obj)
        
        key_data = {
            'args': [serialize_obj(arg) for arg in args],
            'kwargs': {k: serialize_obj(v) for k, v in sorted(kwargs.items())}
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default estimate
    
    def _evict_lru(self):
        """Evict least recently used items."""
        if not self.cache:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        # Evict oldest items until we're under the size limit
        target_size = self.max_size_bytes * 0.8  # Leave some headroom
        
        for key in sorted_keys:
            if self.total_size <= target_size:
                break
            
            self._remove_item(key)
            self.evictions += 1
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            size = self.size_tracker.get(key, 0)
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            if key in self.size_tracker:
                del self.size_tracker[key]
            self.total_size -= size
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self.lock:
            size = self._estimate_size(value)
            
            # Don't cache items that are too large
            if size > self.max_size_bytes * 0.5:
                return False
            
            # Remove existing item if present
            if key in self.cache:
                self._remove_item(key)
            
            # Evict if necessary
            while self.total_size + size > self.max_size_bytes:
                if not self.cache:  # No items to evict
                    return False
                self._evict_lru()
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.size_tracker[key] = size
            self.total_size += size
            
            return True
    
    def cached_call(self, func: Callable, *args, **kwargs):
        """Execute function with caching."""
        key = self._compute_key(func.__name__, *args, **kwargs)
        
        # Try cache first
        result = self.get(key)
        if result is not None:
            return result
        
        # Compute and cache
        result = func(*args, **kwargs)
        self.put(key, result)
        
        return result
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.size_tracker.clear()
            self.total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache),
                "total_size_mb": self.total_size / 1024 / 1024,
                "utilization": self.total_size / self.max_size_bytes
            }


class ModelOptimizer:
    """Advanced model optimization with multiple strategies."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = get_logger("optimization.model")
        
        # Initialize caching
        if self.config.enable_caching:
            self.cache = AdaptiveCache(self.config.cache_size_mb)
        else:
            self.cache = None
        
        # Optimization statistics
        self.optimization_stats = {
            "model_compressions": 0,
            "quantizations_applied": 0,
            "pruning_operations": 0,
            "cache_hits": 0,
            "total_optimizations": 0
        }
    
    def optimize_model(self, model: nn.Module, optimization_level: str = "balanced") -> nn.Module:
        """Apply comprehensive model optimizations.
        
        Args:
            model: PyTorch model to optimize
            optimization_level: "conservative", "balanced", "aggressive"
            
        Returns:
            Optimized model
        """
        optimized_model = model
        
        # Configure optimization based on level
        if optimization_level == "conservative":
            enable_pruning = False
            enable_quantization = False
        elif optimization_level == "balanced":
            enable_pruning = self.config.enable_pruning
            enable_quantization = self.config.enable_quantization
        else:  # aggressive
            enable_pruning = True
            enable_quantization = True
        
        # Apply optimizations
        if enable_quantization:
            optimized_model = self._apply_quantization(optimized_model)
        
        if enable_pruning:
            optimized_model = self._apply_pruning(optimized_model)
        
        # Compile model for faster execution
        if hasattr(torch, 'compile'):
            try:
                optimized_model = torch.compile(optimized_model)
                self.logger.info("Applied torch.compile optimization")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
        
        self.optimization_stats["total_optimizations"] += 1
        
        return optimized_model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        try:
            # Dynamic quantization for inference
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            self.optimization_stats["quantizations_applied"] += 1
            self.logger.info("Applied dynamic quantization")
            
            return quantized_model
            
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model."""
        try:
            import torch.nn.utils.prune as prune
            
            # Prune linear layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')  # Make pruning permanent
            
            self.optimization_stats["pruning_operations"] += 1
            self.logger.info("Applied L1 unstructured pruning")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def optimize_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        max_memory_mb: float = 1000
    ) -> int:
        """Find optimal batch size for given constraints."""
        
        if not self.config.batch_size_optimization:
            return 32  # Default batch size
        
        optimal_batch_size = 1
        test_input_base = torch.randn(1, *input_shape[1:]).to(device)
        model = model.to(device)
        model.eval()
        
        # Binary search for optimal batch size
        low, high = 1, 512
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test batch size
                test_input = test_input_base.repeat(mid, *[1] * (len(input_shape) - 1))
                
                # Measure memory usage
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(test_input)
                
                if device.startswith('cuda'):
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    # Rough estimate for CPU
                    peak_memory_mb = test_input.numel() * 4 / 1024 / 1024  # 4 bytes per float
                
                if peak_memory_mb <= max_memory_mb:
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                high = mid - 1
        
        self.logger.info(f"Optimal batch size: {optimal_batch_size}")
        return max(1, optimal_batch_size)
    
    def create_cached_forward(self, model: nn.Module) -> Callable:
        """Create cached version of model forward pass."""
        
        if not self.cache:
            return model.forward
        
        def cached_forward(*args, **kwargs):
            return self.cache.cached_call(model.forward, *args, **kwargs)
        
        return cached_forward


class ConcurrentProcessor:
    """Concurrent processing system for neuromorphic computations."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_multiprocessing: bool = False,
        chunk_size: int = 1000
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.enable_multiprocessing = enable_multiprocessing
        self.chunk_size = chunk_size
        self.logger = get_logger("optimization.concurrent")
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU-intensive tasks (optional)
        self.process_pool = None
        if enable_multiprocessing:
            try:
                self.process_pool = mp.Pool(processes=mp.cpu_count())
            except Exception as e:
                self.logger.warning(f"Failed to create process pool: {e}")
    
    def process_batch_concurrent(
        self,
        model: nn.Module,
        data_batches: List[torch.Tensor],
        device: str = "cpu"
    ) -> List[torch.Tensor]:
        """Process multiple batches concurrently."""
        
        if len(data_batches) == 1:
            # Single batch - no need for concurrency
            return [model(data_batches[0])]
        
        model.share_memory()  # Enable sharing between threads
        results = [None] * len(data_batches)
        
        def process_single_batch(batch_idx: int, batch_data: torch.Tensor):
            """Process single batch."""
            try:
                batch_data = batch_data.to(device)
                with torch.no_grad():
                    result = model(batch_data)
                return batch_idx, result.cpu()
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                return batch_idx, None
        
        # Submit all tasks
        futures = {}
        for i, batch in enumerate(data_batches):
            future = self.thread_pool.submit(process_single_batch, i, batch)
            futures[future] = i
        
        # Collect results
        completed = 0
        for future in as_completed(futures):
            batch_idx, result = future.result()
            results[batch_idx] = result
            completed += 1
            
            if completed % 10 == 0:
                self.logger.debug(f"Completed {completed}/{len(data_batches)} batches")
        
        return [r for r in results if r is not None]
    
    def parallel_spike_processing(
        self,
        spike_data: List[torch.Tensor],
        processing_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process spike data in parallel."""
        
        if not spike_data:
            return []
        
        # Split data into chunks
        chunks = [
            spike_data[i:i + self.chunk_size]
            for i in range(0, len(spike_data), self.chunk_size)
        ]
        
        def process_chunk(chunk):
            """Process chunk of spike data."""
            return [processing_func(data, **kwargs) for data in chunk]
        
        # Process chunks concurrently
        futures = []
        for chunk in chunks:
            future = self.thread_pool.submit(process_chunk, chunk)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)
        
        return all_results
    
    def concurrent_model_evaluation(
        self,
        models: Dict[str, nn.Module],
        test_data: List[torch.Tensor],
        device: str = "cpu"
    ) -> Dict[str, List[torch.Tensor]]:
        """Evaluate multiple models concurrently."""
        
        results = {}
        
        def evaluate_model(model_name: str, model: nn.Module):
            """Evaluate single model."""
            model_results = []
            model = model.to(device)
            model.eval()
            
            for data in test_data:
                data = data.to(device)
                with torch.no_grad():
                    output = model(data)
                model_results.append(output.cpu())
            
            return model_name, model_results
        
        # Submit evaluation tasks
        futures = {}
        for name, model in models.items():
            future = self.thread_pool.submit(evaluate_model, name, model)
            futures[future] = name
        
        # Collect results
        for future in as_completed(futures):
            model_name, model_results = future.result()
            results[model_name] = model_results
        
        return results
    
    def shutdown(self):
        """Shutdown concurrent processing pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()


class ResourcePool:
    """Resource pooling for efficient memory and compute management."""
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.tensor_pool = deque()
        self.model_pool = {}
        self.lock = threading.Lock()
        self.logger = get_logger("optimization.resource_pool")
        
        # Pool statistics
        self.allocations = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        
        with self.lock:
            # Look for compatible tensor in pool
            for i, (pooled_tensor, pooled_shape, pooled_dtype) in enumerate(self.tensor_pool):
                if pooled_shape == shape and pooled_dtype == dtype:
                    # Remove from pool and return
                    del self.tensor_pool[i]
                    self.pool_hits += 1
                    pooled_tensor.zero_()  # Clear tensor
                    return pooled_tensor
            
            # No compatible tensor found, allocate new one
            self.pool_misses += 1
            self.allocations += 1
            return torch.zeros(shape, dtype=dtype)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool."""
        
        with self.lock:
            if len(self.tensor_pool) < self.pool_size:
                self.tensor_pool.append((tensor, tensor.shape, tensor.dtype))
    
    def get_model_copy(self, model_name: str, model_class: type, *args, **kwargs) -> nn.Module:
        """Get model copy from pool."""
        
        with self.lock:
            if model_name not in self.model_pool:
                self.model_pool[model_name] = deque()
            
            pool = self.model_pool[model_name]
            
            if pool:
                # Get from pool
                model = pool.popleft()
                self.pool_hits += 1
                return model
            else:
                # Create new model
                model = model_class(*args, **kwargs)
                self.pool_misses += 1
                self.allocations += 1
                return model
    
    def return_model(self, model_name: str, model: nn.Module):
        """Return model to pool."""
        
        with self.lock:
            if model_name not in self.model_pool:
                self.model_pool[model_name] = deque()
            
            pool = self.model_pool[model_name]
            if len(pool) < self.pool_size:
                # Reset model state if possible
                if hasattr(model, 'reset_state'):
                    model.reset_state(1)  # Reset to single batch
                
                pool.append(model)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        
        with self.lock:
            total_accesses = self.pool_hits + self.pool_misses
            hit_rate = self.pool_hits / total_accesses if total_accesses > 0 else 0
            
            return {
                "allocations": self.allocations,
                "pool_hits": self.pool_hits,
                "pool_misses": self.pool_misses,
                "hit_rate": hit_rate,
                "tensor_pool_size": len(self.tensor_pool),
                "model_pools": {name: len(pool) for name, pool in self.model_pool.items()}
            }
    
    def clear_pools(self):
        """Clear all resource pools."""
        
        with self.lock:
            self.tensor_pool.clear()
            self.model_pool.clear()


class AdvancedOptimizer:
    """Main optimization coordinator combining all optimization strategies."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = get_logger("optimization.advanced")
        
        # Initialize optimization components
        self.model_optimizer = ModelOptimizer(self.config)
        self.concurrent_processor = ConcurrentProcessor(
            enable_multiprocessing=False  # Keep simple for now
        )
        self.resource_pool = ResourcePool()
        
        # Health monitoring
        self.health_monitor = None
        if hasattr(self.config, 'enable_monitoring') and self.config.enable_monitoring:
            self.health_monitor = HealthMonitor(auto_start=True)
        
        # Optimization history
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
    
    def optimize_system(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        optimization_level: str = "balanced"
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Perform comprehensive system optimization.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for optimization profiling
            optimization_level: Level of optimization to apply
            
        Returns:
            Tuple of (optimized_model, optimization_metrics)
        """
        
        start_time = time.time()
        self.logger.info(f"Starting system optimization (level: {optimization_level})")
        
        # Store original model performance
        original_metrics = self._benchmark_model(model, sample_input)
        
        # Apply model optimizations
        optimized_model = self.model_optimizer.optimize_model(model, optimization_level)
        
        # Optimize batch size
        optimal_batch_size = self.model_optimizer.optimize_batch_size(
            optimized_model, sample_input.shape
        )
        
        # Create cached forward pass if caching enabled
        if self.config.enable_caching:
            cached_forward = self.model_optimizer.create_cached_forward(optimized_model)
            optimized_model.cached_forward = cached_forward
        
        # Benchmark optimized model
        optimized_metrics = self._benchmark_model(optimized_model, sample_input)
        
        # Calculate optimization improvements
        optimization_time = time.time() - start_time
        
        improvements = {
            "inference_speedup": original_metrics["inference_time"] / optimized_metrics["inference_time"],
            "memory_reduction": 1.0 - (optimized_metrics["memory_usage"] / original_metrics["memory_usage"]),
            "optimization_time": optimization_time,
            "optimal_batch_size": optimal_batch_size,
            "cache_stats": self.model_optimizer.cache.get_stats() if self.model_optimizer.cache else {},
            "original_metrics": original_metrics,
            "optimized_metrics": optimized_metrics
        }
        
        # Store optimization history
        optimization_record = {
            "timestamp": time.time(),
            "optimization_level": optimization_level,
            "improvements": improvements,
            "model_params": sum(p.numel() for p in optimized_model.parameters())
        }
        
        self.optimization_history.append(optimization_record)
        
        self.logger.info(
            f"Optimization complete: {improvements['inference_speedup']:.2f}x speedup, "
            f"{improvements['memory_reduction']:.1%} memory reduction"
        )
        
        return optimized_model, improvements
    
    def _benchmark_model(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """Benchmark model performance."""
        
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_input)
        
        # Benchmark inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                output = model(sample_input)
        inference_time = (time.time() - start_time) / 10
        
        # Estimate memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        activation_memory = output.numel() * output.element_size()
        total_memory = param_memory + activation_memory
        
        return {
            "inference_time": inference_time,
            "memory_usage": total_memory,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "output_size": output.numel()
        }
    
    def process_batch_optimized(
        self,
        model: nn.Module,
        data_batches: List[torch.Tensor],
        device: str = "cpu"
    ) -> List[torch.Tensor]:
        """Process batches with full optimization pipeline."""
        
        # Use concurrent processing if multiple batches
        if len(data_batches) > 1:
            return self.concurrent_processor.process_batch_concurrent(
                model, data_batches, device
            )
        else:
            # Single batch - process normally
            model = model.to(device)
            with torch.no_grad():
                result = model(data_batches[0].to(device))
            return [result.cpu()]
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        cache_stats = self.model_optimizer.cache.get_stats() if self.model_optimizer.cache else {}
        pool_stats = self.resource_pool.get_pool_stats()
        
        report = {
            "config": {
                "enable_caching": self.config.enable_caching,
                "cache_size_mb": self.config.cache_size_mb,
                "enable_quantization": self.config.enable_quantization,
                "enable_pruning": self.config.enable_pruning,
                "batch_size_optimization": self.config.batch_size_optimization
            },
            "cache_statistics": cache_stats,
            "resource_pool_statistics": pool_stats,
            "model_optimization_stats": self.model_optimizer.optimization_stats,
            "optimization_history": self.optimization_history[-10:],  # Last 10 optimizations
            "performance_summary": self._compute_performance_summary()
        }
        
        return report
    
    def _compute_performance_summary(self) -> Dict[str, Any]:
        """Compute performance summary from history."""
        
        if not self.optimization_history:
            return {}
        
        speedups = [opt["improvements"]["inference_speedup"] for opt in self.optimization_history]
        memory_reductions = [opt["improvements"]["memory_reduction"] for opt in self.optimization_history]
        
        return {
            "average_speedup": np.mean(speedups),
            "best_speedup": max(speedups),
            "average_memory_reduction": np.mean(memory_reductions),
            "best_memory_reduction": max(memory_reductions),
            "total_optimizations": len(self.optimization_history)
        }
    
    def clear_caches(self):
        """Clear all optimization caches."""
        if self.model_optimizer.cache:
            self.model_optimizer.cache.clear()
        
        self.resource_pool.clear_pools()
        
        self.logger.info("Cleared all optimization caches")
    
    def shutdown(self):
        """Shutdown optimization system."""
        self.concurrent_processor.shutdown()
        
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        self.logger.info("Advanced optimizer shutdown complete")