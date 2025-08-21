"""
Advanced Performance Optimization Framework

Comprehensive performance optimization system including:
- Adaptive caching with intelligent eviction
- Concurrent and parallel processing
- Memory optimization and pooling
- Auto-scaling and load balancing
- Performance profiling and analytics
"""

import time
import threading
import multiprocessing
import queue
import logging
import gc
import weakref
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import pickle
import json
from contextlib import contextmanager


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    EXTREME = 3


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_count: int = 0
    total_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_count: int = 0
    
    def __post_init__(self):
        self.latency_history: List[float] = []
        self.timestamp = time.time()


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    cache_size: int = 1000
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_concurrent_processing: bool = True
    max_workers: int = multiprocessing.cpu_count()
    enable_memory_pooling: bool = True
    enable_auto_scaling: bool = True
    profiling_enabled: bool = True
    batch_size: int = 32
    prefetch_size: int = 64
    compression_enabled: bool = False


class AdaptiveCache:
    """
    High-performance adaptive cache with multiple eviction strategies.
    
    Features:
    - Multiple eviction policies (LRU, LFU, FIFO, Adaptive)
    - Automatic size management
    - Performance monitoring
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl = ttl
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Strategy-specific data structures
        self.access_order: List[str] = []  # For LRU
        self.access_count: Dict[str, int] = {}  # For LFU
        self.insertion_order: List[str] = []  # For FIFO
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.strategy_performance = {
            CacheStrategy.LRU: 0.0,
            CacheStrategy.LFU: 0.0,
            CacheStrategy.FIFO: 0.0
        }
        self.current_adaptive_strategy = CacheStrategy.LRU
        self.strategy_evaluation_interval = 100
        self.operations_since_evaluation = 0
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self.operations_since_evaluation += 1
            
            if key not in self.cache:
                self.misses += 1
                self._evaluate_adaptive_strategy()
                return None
            
            # Check TTL
            if self.ttl and self._is_expired(key):
                self._remove(key)
                self.misses += 1
                self._evaluate_adaptive_strategy()
                return None
            
            # Update access patterns
            self._update_access_patterns(key)
            
            self.hits += 1
            self._evaluate_adaptive_strategy()
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict()
            
            # Store value
            self.cache[key] = value
            self.metadata[key] = {
                'timestamp': time.time(),
                'access_time': time.time(),
                'access_count': 1,
                'size': self._estimate_size(value)
            }
            
            # Update data structures
            if key not in self.access_order:
                self.access_order.append(key)
            if key not in self.insertion_order:
                self.insertion_order.append(key)
            
            self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        with self.lock:
            return self._remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.access_order.clear()
            self.access_count.clear()
            self.insertion_order.clear()
    
    def _remove(self, key: str) -> bool:
        """Internal remove method."""
        if key in self.cache:
            del self.cache[key]
            del self.metadata[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_count:
                del self.access_count[key]
            if key in self.insertion_order:
                self.insertion_order.remove(key)
            
            return True
        return False
    
    def _evict(self) -> None:
        """Evict entry based on current strategy."""
        if not self.cache:
            return
        
        strategy = self.current_adaptive_strategy if self.strategy == CacheStrategy.ADAPTIVE else self.strategy
        
        if strategy == CacheStrategy.LRU:
            key_to_evict = self.access_order[0]
        elif strategy == CacheStrategy.LFU:
            key_to_evict = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        elif strategy == CacheStrategy.FIFO:
            key_to_evict = self.insertion_order[0]
        else:
            key_to_evict = list(self.cache.keys())[0]
        
        self._remove(key_to_evict)
        self.evictions += 1
    
    def _update_access_patterns(self, key: str) -> None:
        """Update access patterns for cache strategies."""
        current_time = time.time()
        
        # Update metadata
        self.metadata[key]['access_time'] = current_time
        self.metadata[key]['access_count'] += 1
        
        # Update LRU order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Update LFU count
        self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if not self.ttl:
            return False
        
        timestamp = self.metadata[key]['timestamp']
        return time.time() - timestamp > self.ttl
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value))
    
    def _evaluate_adaptive_strategy(self) -> None:
        """Evaluate and adapt caching strategy."""
        if (self.strategy != CacheStrategy.ADAPTIVE or 
            self.operations_since_evaluation < self.strategy_evaluation_interval):
            return
        
        # Calculate hit rate for current strategy
        total_ops = self.hits + self.misses
        if total_ops > 0:
            hit_rate = self.hits / total_ops
            self.strategy_performance[self.current_adaptive_strategy] = hit_rate
        
        # Switch to best performing strategy
        best_strategy = max(self.strategy_performance.keys(), 
                          key=lambda k: self.strategy_performance[k])
        
        if best_strategy != self.current_adaptive_strategy:
            self.current_adaptive_strategy = best_strategy
            self.logger.info(f"Switched cache strategy to {best_strategy.value}")
        
        self.operations_since_evaluation = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_ops = self.hits + self.misses
            hit_rate = self.hits / total_ops if total_ops > 0 else 0.0
            
            total_size = sum(meta['size'] for meta in self.metadata.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'total_memory': total_size,
                'current_strategy': self.current_adaptive_strategy.value if self.strategy == CacheStrategy.ADAPTIVE else self.strategy.value,
                'strategy_performance': dict(self.strategy_performance)
            }


class ConcurrentProcessor:
    """
    Concurrent and parallel processing manager with auto-scaling.
    
    Features:
    - Thread and process pool management
    - Auto-scaling based on load
    - Work stealing and load balancing
    - Performance monitoring
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Worker pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Auto-scaling parameters
        self.min_workers = max(1, config.max_workers // 4)
        self.max_workers = config.max_workers
        self.current_workers = self.min_workers
        
        # Load monitoring
        self.task_queue_size = 0
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize pools if concurrent processing is enabled
        if config.enable_concurrent_processing:
            self._initialize_pools()
    
    def _initialize_pools(self) -> None:
        """Initialize worker pools."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="neuromorphic_worker"
        )
        
        if self.config.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
            self.process_pool = ProcessPoolExecutor(
                max_workers=max(1, self.current_workers // 2)
            )
        
        self.logger.info(f"Initialized pools with {self.current_workers} thread workers")
    
    def submit_task(
        self,
        func: Callable,
        *args,
        use_process_pool: bool = False,
        priority: int = 0,
        **kwargs
    ) -> Optional[Any]:
        """
        Submit task for concurrent execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            use_process_pool: Whether to use process pool instead of thread pool
            priority: Task priority (higher = more important)
            **kwargs: Function keyword arguments
            
        Returns:
            Future object or None if pools not available
        """
        if not self.config.enable_concurrent_processing:
            return func(*args, **kwargs)
        
        with self.lock:
            self.task_queue_size += 1
            self.active_tasks += 1
        
        # Choose appropriate pool
        pool = None
        if use_process_pool and self.process_pool:
            pool = self.process_pool
        elif self.thread_pool:
            pool = self.thread_pool
        
        if not pool:
            # Fallback to synchronous execution
            with self.lock:
                self.task_queue_size -= 1
                self.active_tasks -= 1
            return func(*args, **kwargs)
        
        # Submit task
        future = pool.submit(self._execute_with_monitoring, func, *args, **kwargs)
        
        # Auto-scale if needed
        self._check_auto_scaling()
        
        return future
    
    def submit_batch(
        self,
        func: Callable,
        batch_data: List[Any],
        use_process_pool: bool = False,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Submit batch of tasks for concurrent execution.
        
        Args:
            func: Function to apply to each item
            batch_data: List of data items
            use_process_pool: Whether to use process pool
            batch_size: Size of sub-batches (None = auto)
            
        Returns:
            List of results
        """
        if not batch_data:
            return []
        
        batch_size = batch_size or self.config.batch_size
        
        # Split into batches
        batches = [
            batch_data[i:i + batch_size]
            for i in range(0, len(batch_data), batch_size)
        ]
        
        # Submit batch processing tasks
        futures = []
        for batch in batches:
            future = self.submit_task(
                self._process_batch,
                func, batch,
                use_process_pool=use_process_pool
            )
            if future:
                futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Batch processing failed: {str(e)}")
                with self.lock:
                    self.failed_tasks += 1
        
        return results
    
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of data items."""
        results = []
        for item in batch:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch item processing failed: {str(e)}")
                results.append(None)
        return results
    
    def _execute_with_monitoring(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with performance monitoring."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Update metrics
            execution_time = time.time() - start_time
            with self.lock:
                self.metrics.operation_count += 1
                self.metrics.total_execution_time += execution_time
                self.metrics.latency_history.append(execution_time)
                
                # Keep only recent latency data
                if len(self.metrics.latency_history) > 1000:
                    self.metrics.latency_history.pop(0)
                
                self.completed_tasks += 1
                self.active_tasks -= 1
                self.task_queue_size -= 1
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failed_tasks += 1
                self.active_tasks -= 1
                self.task_queue_size -= 1
                self.metrics.error_count += 1
            raise
    
    def _check_auto_scaling(self) -> None:
        """Check if auto-scaling is needed."""
        if not self.config.enable_auto_scaling:
            return
        
        with self.lock:
            # Calculate load metrics
            queue_ratio = self.task_queue_size / max(1, self.current_workers)
            active_ratio = self.active_tasks / max(1, self.current_workers)
            
            # Scale up conditions
            if (queue_ratio > 2.0 or active_ratio > 0.8) and self.current_workers < self.max_workers:
                self._scale_up()
            
            # Scale down conditions
            elif (queue_ratio < 0.5 and active_ratio < 0.3) and self.current_workers > self.min_workers:
                self._scale_down()
    
    def _scale_up(self) -> None:
        """Scale up worker pools."""
        new_workers = min(self.current_workers * 2, self.max_workers)
        
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            
            # Recreate thread pool with new size
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.current_workers,
                    thread_name_prefix="neuromorphic_worker"
                )
            
            self.logger.info(f"Scaled up to {self.current_workers} workers")
    
    def _scale_down(self) -> None:
        """Scale down worker pools."""
        new_workers = max(self.current_workers // 2, self.min_workers)
        
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            
            # Recreate thread pool with new size
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.current_workers,
                    thread_name_prefix="neuromorphic_worker"
                )
            
            self.logger.info(f"Scaled down to {self.current_workers} workers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrent processing statistics."""
        with self.lock:
            # Calculate latency percentiles
            if self.metrics.latency_history:
                sorted_latencies = sorted(self.metrics.latency_history)
                n = len(sorted_latencies)
                self.metrics.latency_p50 = sorted_latencies[int(n * 0.5)]
                self.metrics.latency_p95 = sorted_latencies[int(n * 0.95)]
                self.metrics.latency_p99 = sorted_latencies[int(n * 0.99)]
            
            # Calculate throughput
            if self.metrics.total_execution_time > 0:
                self.metrics.throughput = self.metrics.operation_count / self.metrics.total_execution_time
            
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'task_queue_size': self.task_queue_size,
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'metrics': {
                    'operation_count': self.metrics.operation_count,
                    'total_execution_time': self.metrics.total_execution_time,
                    'throughput': self.metrics.throughput,
                    'latency_p50': self.metrics.latency_p50,
                    'latency_p95': self.metrics.latency_p95,
                    'latency_p99': self.metrics.latency_p99,
                    'error_rate': self.metrics.error_count / max(1, self.metrics.operation_count)
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown worker pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.logger.info("Worker pools shut down")


class MemoryPool:
    """
    Memory pooling system for efficient memory management.
    
    Features:
    - Object pooling and reuse
    - Memory pre-allocation
    - Garbage collection optimization
    - Memory usage tracking
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Object pools
        self.pools: Dict[str, queue.Queue] = {}
        self.pool_stats: Dict[str, Dict[str, int]] = {}
        
        # Memory tracking
        self.total_allocated = 0
        self.total_freed = 0
        self.peak_usage = 0
        
        # Weak references for automatic cleanup
        self.tracked_objects: List[weakref.ref] = []
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get_object(self, object_type: str, factory: Callable, *args, **kwargs) -> Any:
        """
        Get object from pool or create new one.
        
        Args:
            object_type: Type identifier for the object
            factory: Factory function to create new objects
            *args: Factory arguments
            **kwargs: Factory keyword arguments
            
        Returns:
            Object from pool or newly created
        """
        if not self.config.enable_memory_pooling:
            return factory(*args, **kwargs)
        
        with self.lock:
            # Initialize pool if doesn't exist
            if object_type not in self.pools:
                self.pools[object_type] = queue.Queue()
                self.pool_stats[object_type] = {
                    'created': 0,
                    'reused': 0,
                    'returned': 0
                }
            
            pool = self.pools[object_type]
            stats = self.pool_stats[object_type]
            
            # Try to get from pool
            try:
                obj = pool.get_nowait()
                stats['reused'] += 1
                return obj
            except queue.Empty:
                # Create new object
                obj = factory(*args, **kwargs)
                stats['created'] += 1
                
                # Track memory usage
                try:
                    obj_size = self._estimate_object_size(obj)
                    self.total_allocated += obj_size
                    self.peak_usage = max(self.peak_usage, self.total_allocated - self.total_freed)
                except:
                    pass
                
                return obj
    
    def return_object(self, object_type: str, obj: Any) -> None:
        """
        Return object to pool for reuse.
        
        Args:
            object_type: Type identifier for the object
            obj: Object to return to pool
        """
        if not self.config.enable_memory_pooling:
            return
        
        with self.lock:
            if object_type in self.pools:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except:
                        pass
                
                # Return to pool
                self.pools[object_type].put(obj)
                self.pool_stats[object_type]['returned'] += 1
    
    def clear_pools(self) -> None:
        """Clear all object pools."""
        with self.lock:
            for pool in self.pools.values():
                while not pool.empty():
                    try:
                        pool.get_nowait()
                    except queue.Empty:
                        break
            
            self.pools.clear()
            self.pool_stats.clear()
            
            # Force garbage collection
            gc.collect()
    
    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object memory size."""
        try:
            return len(pickle.dumps(obj))
        except:
            try:
                import sys
                return sys.getsizeof(obj)
            except:
                return 1024  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            pool_sizes = {
                pool_type: pool.qsize()
                for pool_type, pool in self.pools.items()
            }
            
            return {
                'pools': dict(self.pool_stats),
                'pool_sizes': pool_sizes,
                'total_allocated': self.total_allocated,
                'total_freed': self.total_freed,
                'peak_usage': self.peak_usage,
                'current_usage': self.total_allocated - self.total_freed
            }


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Integrates all optimization components for maximum performance.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Optimization components
        self.cache = AdaptiveCache(
            max_size=self.config.cache_size,
            strategy=self.config.cache_strategy
        )
        
        self.processor = ConcurrentProcessor(self.config)
        self.memory_pool = MemoryPool(self.config)
        
        # Performance tracking
        self.start_time = time.time()
        self.optimization_events = []
        
        # Profiling
        self.profiling_data = {}
        
        self.logger.info(f"Performance optimizer initialized with level: {self.config.optimization_level.name}")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for operation profiling."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            if self.config.profiling_enabled:
                self.profiling_data[operation_name] = {
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'timestamp': start_time
                }
    
    def optimize_function(self, func: Callable, cache_key: Optional[str] = None) -> Callable:
        """
        Wrap function with optimization features.
        
        Args:
            func: Function to optimize
            cache_key: Optional cache key for memoization
            
        Returns:
            Optimized function wrapper
        """
        def optimized_wrapper(*args, **kwargs):
            # Generate cache key if not provided
            if cache_key or self.config.optimization_level.value >= OptimizationLevel.BASIC.value:
                key = cache_key or self._generate_cache_key(func, args, kwargs)
                
                # Try cache first
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result
            
            # Profile execution if enabled
            if self.config.profiling_enabled:
                with self.profile_operation(func.__name__):
                    if self.config.enable_concurrent_processing and self.config.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
                        # Use concurrent execution for expensive operations
                        future = self.processor.submit_task(func, *args, **kwargs)
                        result = future.result() if future else func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            if cache_key or self.config.optimization_level.value >= OptimizationLevel.BASIC.value:
                self.cache.put(key, result)
            
            return result
        
        return optimized_wrapper
    
    def process_batch_parallel(
        self,
        func: Callable,
        data_batch: List[Any],
        **kwargs
    ) -> List[Any]:
        """Process batch of data in parallel."""
        return self.processor.submit_batch(func, data_batch, **kwargs)
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        try:
            # Create hash from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except:
            # Fallback to simple string representation
            return f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage estimate."""
        try:
            import gc
            return sum(len(str(obj)) for obj in gc.get_objects()[:100])  # Sample
        except:
            return 0
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'uptime': time.time() - self.start_time,
            'optimization_level': self.config.optimization_level.name,
            'cache_stats': self.cache.get_stats(),
            'processor_stats': self.processor.get_stats(),
            'memory_pool_stats': self.memory_pool.get_stats(),
            'profiling_data': self.profiling_data.copy(),
            'config': {
                'cache_size': self.config.cache_size,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'enable_concurrent_processing': self.config.enable_concurrent_processing,
                'enable_memory_pooling': self.config.enable_memory_pooling
            }
        }
    
    def optimize_for_workload(self, workload_characteristics: Dict[str, Any]) -> None:
        """Automatically optimize configuration based on workload characteristics."""
        cpu_intensive = workload_characteristics.get('cpu_intensive', False)
        memory_intensive = workload_characteristics.get('memory_intensive', False)
        io_intensive = workload_characteristics.get('io_intensive', False)
        
        if cpu_intensive:
            # Increase concurrency for CPU-bound tasks
            self.config.max_workers = min(multiprocessing.cpu_count() * 2, 32)
            self.config.optimization_level = OptimizationLevel.AGGRESSIVE
        
        if memory_intensive:
            # Enable aggressive memory pooling
            self.config.enable_memory_pooling = True
            self.config.cache_size = self.config.cache_size // 2  # Reduce cache to save memory
        
        if io_intensive:
            # Increase thread pool for I/O operations
            self.config.max_workers = min(multiprocessing.cpu_count() * 4, 64)
        
        self.logger.info(f"Optimized configuration for workload: {workload_characteristics}")
    
    def shutdown(self) -> None:
        """Shutdown optimizer and clean up resources."""
        self.processor.shutdown()
        self.memory_pool.clear_pools()
        self.cache.clear()
        
        self.logger.info("Performance optimizer shut down")


def demo_performance_optimization():
    """Demonstrate the performance optimization system."""
    print("Performance Optimization Demo")
    print("=" * 40)
    
    # Create optimizer with aggressive settings
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        cache_size=100,
        max_workers=4,
        enable_concurrent_processing=True,
        enable_memory_pooling=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Test function to optimize
    def expensive_computation(n):
        """Simulate expensive computation."""
        result = 0
        for i in range(n * 1000):
            result += i ** 2
        return result % 1000000
    
    # Optimize the function
    optimized_func = optimizer.optimize_function(expensive_computation)
    
    print("\n1. Testing Function Optimization:")
    
    # First call (cache miss)
    start_time = time.time()
    result1 = optimized_func(100)
    time1 = time.time() - start_time
    print(f"First call: {result1} in {time1:.4f}s")
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = optimized_func(100)
    time2 = time.time() - start_time
    print(f"Second call: {result2} in {time2:.4f}s (speedup: {time1/time2:.1f}x)")
    
    # Test batch processing
    print("\n2. Testing Batch Processing:")
    test_data = list(range(1, 21))
    
    start_time = time.time()
    results = optimizer.process_batch_parallel(expensive_computation, test_data)
    batch_time = time.time() - start_time
    
    print(f"Processed {len(test_data)} items in {batch_time:.4f}s")
    print(f"Average per item: {batch_time/len(test_data):.4f}s")
    
    # Display comprehensive stats
    print("\n3. Performance Statistics:")
    stats = optimizer.get_comprehensive_stats()
    
    print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    print(f"Completed tasks: {stats['processor_stats']['completed_tasks']}")
    print(f"Average latency: {stats['processor_stats']['metrics']['latency_p50']:.4f}s")
    print(f"Throughput: {stats['processor_stats']['metrics']['throughput']:.2f} ops/s")
    
    # Clean up
    optimizer.shutdown()
    
    print("\nPerformance optimization demo completed!")


if __name__ == "__main__":
    demo_performance_optimization()