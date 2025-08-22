"""Adaptive caching system optimized for neuromorphic computing workloads."""

import torch
import numpy as np
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import pickle
import weakref
import psutil
from pathlib import Path


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    TTL = "ttl"           # Time To Live
    SIZE_AWARE = "size_aware"  # Size-aware eviction


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    hit_rate: float = 0.0
    computation_cost: float = 0.0


class NeuromorphicCache:
    """High-performance adaptive cache for neuromorphic computations.
    
    Features:
    - Multiple cache replacement policies
    - Adaptive sizing based on memory pressure
    - Spike pattern-aware caching
    - Hierarchical cache levels (L1/L2)
    - Thread-safe operations
    - Automatic cache warmup
    """
    
    def __init__(
        self,
        max_size_mb: float = 1024.0,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        l1_ratio: float = 0.1,  # L1 cache as fraction of total
        enable_persistence: bool = True,
        cache_dir: Optional[str] = None,
        adaptive_threshold: float = 0.8,
        ttl_seconds: float = 3600.0
    ):
        """Initialize neuromorphic cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            policy: Cache replacement policy
            l1_ratio: L1 cache size as fraction of total cache
            enable_persistence: Enable persistent cache storage
            cache_dir: Directory for persistent cache
            adaptive_threshold: Memory usage threshold for adaptive scaling
            ttl_seconds: Default TTL for cache entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.policy = policy
        self.l1_ratio = l1_ratio
        self.enable_persistence = enable_persistence
        self.adaptive_threshold = adaptive_threshold
        self.default_ttl = ttl_seconds
        
        # L1 (fast, small) and L2 (slower, larger) caches
        self.l1_max_size = int(self.max_size_bytes * l1_ratio)
        self.l2_max_size = self.max_size_bytes - self.l1_max_size
        
        # Cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Metadata
        self.current_size = 0
        self.l1_size = 0
        self.l2_size = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'evictions': 0,
            'promotions': 0,
            'demotions': 0
        }
        
        # Access patterns for adaptive policy
        self.access_patterns = defaultdict(list)
        self.frequency_counter = defaultdict(int)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Persistent storage
        if self.enable_persistence:
            self.cache_dir = Path(cache_dir or "./neuromorphic_cache")
            self.cache_dir.mkdir(exist_ok=True)
            self._load_persistent_cache()
        
        # Memory monitoring
        self.memory_monitor_interval = 10.0  # seconds
        self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring for adaptive scaling."""
        def monitor_memory():
            while True:
                try:
                    memory_percent = psutil.virtual_memory().percent
                    
                    if memory_percent > self.adaptive_threshold * 100:
                        # High memory pressure - reduce cache size
                        self._adaptive_downsize(memory_percent / 100.0)
                    elif memory_percent < (self.adaptive_threshold - 0.1) * 100:
                        # Low memory pressure - can increase cache size
                        self._adaptive_upsize()
                    
                    time.sleep(self.memory_monitor_interval)
                    
                except Exception:
                    # Continue monitoring even if there are errors
                    time.sleep(self.memory_monitor_interval)
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
    
    def _compute_cache_key(self, key_data: Union[str, Dict, torch.Tensor, np.ndarray]) -> str:
        """Compute a hash key for cache storage."""
        if isinstance(key_data, str):
            return key_data
        
        # Create a hashable representation
        if isinstance(key_data, torch.Tensor):
            # Use tensor properties for key
            key_str = f"tensor_{key_data.shape}_{key_data.dtype}_{key_data.device}_{key_data.sum().item()}"
        elif isinstance(key_data, np.ndarray):
            key_str = f"array_{key_data.shape}_{key_data.dtype}_{key_data.sum()}"
        elif isinstance(key_data, dict):
            # Sort dict for consistent hashing
            key_str = str(sorted(key_data.items()))
        else:
            key_str = str(key_data)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # Fallback to pickle serialization size
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default estimate
    
    def put(
        self,
        key: Union[str, Any],
        value: Any,
        ttl: Optional[float] = None,
        computation_cost: float = 0.0,
        force_l1: bool = False
    ) -> bool:
        """Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            computation_cost: Cost of computing this value (for priority)
            force_l1: Force storage in L1 cache
            
        Returns:
            True if stored successfully
        """
        cache_key = self._compute_cache_key(key)
        size_bytes = self._estimate_size(value)
        
        # Check if entry fits in cache
        if size_bytes > self.max_size_bytes:
            return False
        
        with self.lock:
            # Remove existing entry if present
            self._remove_key(cache_key)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl,
                computation_cost=computation_cost
            )
            
            # Decide cache level
            target_l1 = (
                force_l1 or
                size_bytes <= self.l1_max_size // 10 or  # Small objects
                computation_cost > 1.0  # High computation cost
            )
            
            if target_l1 and size_bytes <= self.l1_max_size:
                # Try to fit in L1
                self._ensure_space(self.l1_cache, size_bytes, self.l1_max_size, "l1")
                self.l1_cache[cache_key] = entry
                self.l1_size += size_bytes
            else:
                # Put in L2
                self._ensure_space(self.l2_cache, size_bytes, self.l2_max_size, "l2")
                self.l2_cache[cache_key] = entry
                self.l2_size += size_bytes
            
            self.current_size += size_bytes
            
            # Update access patterns
            self.frequency_counter[cache_key] = 1
            
            # Persist if enabled
            if self.enable_persistence:
                self._persist_entry(entry)
            
            return True
    
    def get(self, key: Union[str, Any]) -> Optional[Any]:
        """Retrieve a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._compute_cache_key(key)
        
        with self.lock:
            # Check L1 cache first
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(entry, "l1")
                    self.stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                entry.hit_rate = self._calculate_hit_rate(cache_key)
                
                # Move to end (most recently used)
                self.l1_cache.move_to_end(cache_key)
                
                self.stats['hits'] += 1
                self.stats['l1_hits'] += 1
                self.frequency_counter[cache_key] += 1
                
                return entry.value
            
            # Check L2 cache
            if cache_key in self.l2_cache:
                entry = self.l2_cache[cache_key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(entry, "l2")
                    self.stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                entry.hit_rate = self._calculate_hit_rate(cache_key)
                
                # Consider promoting to L1 if frequently accessed
                if self._should_promote(entry):
                    self._promote_to_l1(cache_key, entry)
                else:
                    self.l2_cache.move_to_end(cache_key)
                
                self.stats['hits'] += 1
                self.stats['l2_hits'] += 1
                self.frequency_counter[cache_key] += 1
                
                return entry.value
            
            # Not found
            self.stats['misses'] += 1
            return None
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        
        return time.time() - entry.timestamp > entry.ttl
    
    def _should_promote(self, entry: CacheEntry) -> bool:
        """Determine if L2 entry should be promoted to L1."""
        if entry.size_bytes > self.l1_max_size // 5:  # Too large for L1
            return False
        
        # Promote if:
        # 1. High access frequency
        # 2. High computation cost
        # 3. Good hit rate
        return (
            entry.access_count >= 5 or
            entry.computation_cost > 1.0 or
            entry.hit_rate > 0.8
        )
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1."""
        if entry.size_bytes > self.l1_max_size:
            return
        
        with self.lock:
            # Remove from L2
            if key in self.l2_cache:
                del self.l2_cache[key]
                self.l2_size -= entry.size_bytes
            
            # Ensure space in L1
            self._ensure_space(self.l1_cache, entry.size_bytes, self.l1_max_size, "l1")
            
            # Add to L1
            self.l1_cache[key] = entry
            self.l1_size += entry.size_bytes
            
            self.stats['promotions'] += 1
    
    def _demote_to_l2(self, key: str, entry: CacheEntry):
        """Demote entry from L1 to L2."""
        with self.lock:
            # Remove from L1
            if key in self.l1_cache:
                del self.l1_cache[key]
                self.l1_size -= entry.size_bytes
            
            # Ensure space in L2
            self._ensure_space(self.l2_cache, entry.size_bytes, self.l2_max_size, "l2")
            
            # Add to L2
            self.l2_cache[key] = entry
            self.l2_size += entry.size_bytes
            
            self.stats['demotions'] += 1
    
    def _ensure_space(self, cache: OrderedDict, size_needed: int, max_size: int, cache_level: str):
        """Ensure sufficient space in cache by evicting entries."""
        current_size = self.l1_size if cache_level == "l1" else self.l2_size
        
        while current_size + size_needed > max_size and cache:
            evicted_key = self._select_eviction_candidate(cache)
            if evicted_key:
                entry = cache[evicted_key]
                self._remove_entry(entry, cache_level)
                current_size -= entry.size_bytes
                self.stats['evictions'] += 1
            else:
                break
    
    def _select_eviction_candidate(self, cache: OrderedDict) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            # Least recently used (first in OrderedDict)
            return next(iter(cache))
        
        elif self.policy == CachePolicy.LFU:
            # Least frequently used
            min_freq = float('inf')
            candidate = None
            
            for key, entry in cache.items():
                if entry.access_count < min_freq:
                    min_freq = entry.access_count
                    candidate = key
            
            return candidate
        
        elif self.policy == CachePolicy.TTL:
            # Earliest expiring entry
            min_ttl = float('inf')
            candidate = None
            
            for key, entry in cache.items():
                remaining_ttl = entry.ttl - (time.time() - entry.timestamp)
                if remaining_ttl < min_ttl:
                    min_ttl = remaining_ttl
                    candidate = key
            
            return candidate
        
        elif self.policy == CachePolicy.SIZE_AWARE:
            # Largest entry with low access frequency
            best_score = float('inf')
            candidate = None
            
            for key, entry in cache.items():
                score = entry.size_bytes / (entry.access_count + 1)
                if score < best_score:
                    best_score = score
                    candidate = key
            
            return candidate
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy considering multiple factors
            best_score = float('inf')
            candidate = None
            
            for key, entry in cache.items():
                # Score based on:
                # - Time since last access (higher = worse)
                # - Access frequency (lower = worse)
                # - Size (larger = worse for eviction)
                # - Computation cost (higher = better to keep)
                
                time_factor = time.time() - entry.last_access
                freq_factor = 1.0 / (entry.access_count + 1)
                size_factor = entry.size_bytes / 1024.0  # KB
                cost_factor = 1.0 / (entry.computation_cost + 1)
                
                score = (time_factor * freq_factor * size_factor * cost_factor)
                
                if score < best_score:
                    best_score = score
                    candidate = key
            
            return candidate
        
        else:
            # Default to LRU
            return next(iter(cache))
    
    def _remove_key(self, key: str):
        """Remove key from both cache levels."""
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            self._remove_entry(entry, "l1")
        
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            self._remove_entry(entry, "l2")
    
    def _remove_entry(self, entry: CacheEntry, cache_level: str):
        """Remove entry from specified cache level."""
        if cache_level == "l1":
            if entry.key in self.l1_cache:
                del self.l1_cache[entry.key]
                self.l1_size -= entry.size_bytes
        else:
            if entry.key in self.l2_cache:
                del self.l2_cache[entry.key]
                self.l2_size -= entry.size_bytes
        
        self.current_size -= entry.size_bytes
    
    def _calculate_hit_rate(self, key: str) -> float:
        """Calculate hit rate for a specific key."""
        total_accesses = self.frequency_counter[key]
        if total_accesses == 0:
            return 0.0
        
        # Simplified hit rate calculation
        return min(1.0, total_accesses / 10.0)
    
    def _adaptive_downsize(self, memory_pressure: float):
        """Reduce cache size under memory pressure."""
        if memory_pressure > 0.9:
            reduction_factor = 0.5
        elif memory_pressure > 0.85:
            reduction_factor = 0.7
        else:
            reduction_factor = 0.8
        
        target_size = int(self.max_size_bytes * reduction_factor)
        
        with self.lock:
            # Aggressively evict from L2 first
            while self.current_size > target_size and self.l2_cache:
                evicted_key = self._select_eviction_candidate(self.l2_cache)
                if evicted_key:
                    entry = self.l2_cache[evicted_key]
                    self._remove_entry(entry, "l2")
                else:
                    break
            
            # Then from L1 if necessary
            while self.current_size > target_size and self.l1_cache:
                evicted_key = self._select_eviction_candidate(self.l1_cache)
                if evicted_key:
                    entry = self.l1_cache[evicted_key]
                    self._remove_entry(entry, "l1")
                else:
                    break
    
    def _adaptive_upsize(self):
        """Increase cache size when memory pressure is low."""
        # Could implement cache size expansion here
        pass
    
    def _persist_entry(self, entry: CacheEntry):
        """Persist cache entry to disk."""
        if not self.enable_persistence:
            return
        
        try:
            file_path = self.cache_dir / f"{entry.key}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'value': entry.value,
                    'timestamp': entry.timestamp,
                    'access_count': entry.access_count,
                    'ttl': entry.ttl
                }, f)
        except Exception:
            # Fail silently for persistence errors
            pass
    
    def _load_persistent_cache(self):
        """Load persistent cache from disk."""
        if not self.enable_persistence or not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if entry is still valid
                if data['ttl'] and time.time() - data['timestamp'] > data['ttl']:
                    cache_file.unlink()  # Remove expired entry
                    continue
                
                # Recreate entry
                entry = CacheEntry(
                    key=cache_file.stem,
                    value=data['value'],
                    timestamp=data['timestamp'],
                    access_count=data['access_count'],
                    ttl=data['ttl'],
                    size_bytes=self._estimate_size(data['value'])
                )
                
                # Add to L2 cache (L1 will be populated on access)
                if entry.size_bytes <= self.l2_max_size:
                    self.l2_cache[entry.key] = entry
                    self.l2_size += entry.size_bytes
                    self.current_size += entry.size_bytes
                
            except Exception:
                # Remove corrupted cache files
                try:
                    cache_file.unlink()
                except:
                    pass
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l1_size = 0
            self.l2_size = 0
            self.current_size = 0
            self.frequency_counter.clear()
            self.access_patterns.clear()
            
            # Reset statistics
            for key in self.stats:
                self.stats[key] = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(1, total_requests)
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'l1_hit_rate': self.stats['l1_hits'] / max(1, total_requests),
            'l2_hit_rate': self.stats['l2_hits'] / max(1, total_requests),
            'current_size_mb': self.current_size / (1024 * 1024),
            'l1_size_mb': self.l1_size / (1024 * 1024),
            'l2_size_mb': self.l2_size / (1024 * 1024),
            'l1_entries': len(self.l1_cache),
            'l2_entries': len(self.l2_cache),
            'memory_usage_percent': (self.current_size / self.max_size_bytes) * 100
        }
    
    def optimize_cache(self):
        """Perform cache optimization based on access patterns."""
        with self.lock:
            # Analyze access patterns and adjust placement
            for key in list(self.l2_cache.keys()):
                entry = self.l2_cache[key]
                
                # Promote frequently accessed small entries
                if (entry.access_count >= 10 and 
                    entry.size_bytes <= self.l1_max_size // 10):
                    self._promote_to_l1(key, entry)
            
            # Demote infrequently accessed L1 entries
            for key in list(self.l1_cache.keys()):
                entry = self.l1_cache[key]
                
                if (entry.access_count < 3 and 
                    time.time() - entry.last_access > 300):  # 5 minutes
                    self._demote_to_l2(key, entry)