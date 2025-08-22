"""Adaptive caching system for neuromorphic computing operations."""

import time
import threading
from typing import Any, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import hashlib
import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu" 
    ADAPTIVE = "adaptive"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    computation_time: float = 0.0
    cache_hits: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate entry size."""
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size(self.value)
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes."""
        try:
            # Try pickle size as approximation
            return len(pickle.dumps(obj))
        except:
            # Fallback size estimation
            if hasattr(obj, 'nbytes'):
                return obj.nbytes
            elif hasattr(obj, '__len__'):
                return len(obj) * 8  # Rough estimate
            else:
                return 64  # Default estimate


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_computation_time_saved: float = 0.0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class AdaptiveCache:
    """Adaptive cache with multiple eviction policies and performance tracking."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        ttl_seconds: float = 3600.0,
        eviction_policy: str = "adaptive",
        enable_persistence: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for entries
            eviction_policy: 'lru', 'lfu', 'adaptive', 'ttl'
            enable_persistence: Whether to persist cache to disk
            cache_dir: Directory for persistent cache
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        self.enable_persistence = enable_persistence
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_frequency: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Access pattern learning
        self._access_patterns: Dict[str, list] = defaultdict(list)
        self._adaptive_weights: Dict[str, float] = {}
        
        # Persistence
        if enable_persistence and cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # Create a hashable representation
        key_data = {
            'func_name': func.__name__,
            'func_module': getattr(func, '__module__', ''),
            'args': self._serialize_args(args),
            'kwargs': self._serialize_args(kwargs)
        }
        
        # Create hash
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_args(self, args) -> str:
        """Serialize arguments for hashing."""
        if isinstance(args, (tuple, list)):
            return str([self._serialize_single_arg(arg) for arg in args])
        elif isinstance(args, dict):
            return str({k: self._serialize_single_arg(v) for k, v in args.items()})
        else:
            return self._serialize_single_arg(args)
    
    def _serialize_single_arg(self, arg) -> str:
        """Serialize a single argument."""
        if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
            # NumPy/JAX array
            return f"array_{arg.shape}_{arg.dtype}_{hash(arg.tobytes())}"
        elif isinstance(arg, (int, float, str, bool, type(None))):
            return str(arg)
        elif hasattr(arg, '__dict__'):
            # Objects with attributes
            return str(sorted(arg.__dict__.items()))
        else:
            return str(type(arg).__name__)
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        with self._lock:
            self.stats.total_requests += 1
            
            if key not in self._cache:
                self.stats.cache_misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                self.stats.cache_misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            entry.cache_hits += 1
            self._access_frequency[key] += 1
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            # Record access pattern
            self._record_access_pattern(key)
            
            self.stats.cache_hits += 1
            self.stats.total_computation_time_saved += entry.computation_time
            
            return entry
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put entry into cache."""
        with self._lock:
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                computation_time=computation_time
            )
            
            # Check if we need to evict
            self._ensure_capacity()
            
            self._cache[key] = entry
            self.stats.memory_usage_bytes += entry.size_bytes
            
            # Update adaptive weights
            self._update_adaptive_weights(key, computation_time)
            
            # Persist if enabled
            if self.enable_persistence:
                self._persist_entry(key, entry)
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed capacity limits."""
        # Size limit
        while len(self._cache) >= self.max_size:
            self._evict_entry()
        
        # Memory limit
        while self.stats.memory_usage_bytes >= self.max_memory_bytes:
            self._evict_entry()
    
    def _evict_entry(self) -> None:
        """Evict an entry based on the eviction policy."""
        if not self._cache:
            return
        
        if self.eviction_policy == "lru":
            key = next(iter(self._cache))
        elif self.eviction_policy == "lfu":
            key = min(self._cache.keys(), key=lambda k: self._access_frequency[k])
        elif self.eviction_policy == "ttl":
            key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        elif self.eviction_policy == "adaptive":
            key = self._adaptive_eviction()
        else:
            key = next(iter(self._cache))  # Default to LRU
        
        self._remove_entry(key)
        self.stats.evictions += 1
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on access patterns and computation time."""
        scores = {}
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Calculate adaptive score
            age_factor = current_time - entry.timestamp
            access_factor = entry.access_count
            computation_factor = entry.computation_time
            frequency_factor = self._access_frequency[key]
            
            # Adaptive weight (learned from access patterns)
            adaptive_weight = self._adaptive_weights.get(key, 1.0)
            
            # Higher score = more likely to evict
            score = (age_factor / (access_factor + 1)) / (computation_factor + 0.1) / (frequency_factor + 1) / adaptive_weight
            scores[key] = score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats.memory_usage_bytes -= entry.size_bytes
            
            if key in self._access_frequency:
                del self._access_frequency[key]
            
            if key in self._adaptive_weights:
                del self._adaptive_weights[key]
    
    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for adaptive learning."""
        current_time = time.time()
        self._access_patterns[key].append(current_time)
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff_time
        ]
    
    def _update_adaptive_weights(self, key: str, computation_time: float) -> None:
        """Update adaptive weights based on computation patterns."""
        if key not in self._adaptive_weights:
            self._adaptive_weights[key] = 1.0
        
        # Increase weight for expensive computations
        if computation_time > 0.1:  # More than 100ms
            self._adaptive_weights[key] *= 1.1
        
        # Increase weight for frequently accessed items
        access_count = len(self._access_patterns.get(key, []))
        if access_count > 10:
            self._adaptive_weights[key] *= 1.05
        
        # Bound the weights
        self._adaptive_weights[key] = min(self._adaptive_weights[key], 10.0)
    
    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        if self.cache_dir:
            try:
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception:
                pass  # Silently fail on persistence errors
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_frequency.clear()
            self._access_patterns.clear()
            self._adaptive_weights.clear()
            self.stats = CacheStats()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'total_entries': len(self._cache),
                'memory_usage_mb': self.stats.memory_usage_bytes / (1024 * 1024),
                'hit_rate': self.stats.hit_rate,
                'miss_rate': self.stats.miss_rate,
                'total_requests': self.stats.total_requests,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'evictions': self.stats.evictions,
                'time_saved_seconds': self.stats.total_computation_time_saved,
                'average_entry_size_kb': (
                    self.stats.memory_usage_bytes / len(self._cache) / 1024
                    if self._cache else 0
                ),
                'adaptive_weights_count': len(self._adaptive_weights)
            }


def cached_computation(
    cache: Optional[AdaptiveCache] = None,
    ttl: Optional[float] = None,
    cache_key_func: Optional[Callable] = None
):
    """Decorator for caching expensive computations."""
    def decorator(func: Callable) -> Callable:
        nonlocal cache
        
        if cache is None:
            cache = AdaptiveCache(max_size=100, max_memory_mb=50.0)
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = cache._generate_key(func, args, kwargs)
            
            # Try to get from cache
            entry = cache.get(key)
            if entry is not None:
                return entry.value
            
            # Compute and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            cache.put(key, result, computation_time)
            return result
        
        # Add cache control methods
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.cache_stats = cache.get_statistics
        
        return wrapper
    
    return decorator


# Global cache instances
computation_cache = AdaptiveCache(
    max_size=500,
    max_memory_mb=200.0,
    eviction_policy="adaptive"
)

model_cache = AdaptiveCache(
    max_size=50,
    max_memory_mb=100.0,
    eviction_policy="lfu",
    enable_persistence=True,
    cache_dir="model_cache"
)