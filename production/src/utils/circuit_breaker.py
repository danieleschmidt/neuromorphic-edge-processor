"""Circuit breaker pattern for resilient neuromorphic operations."""

import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from collections import deque
import statistics


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures to open circuit
    success_threshold: int = 3          # Number of successes to close circuit (in half-open)
    timeout_seconds: float = 60.0       # Time to wait before trying again
    monitoring_window_seconds: float = 300.0  # Window for failure rate calculation
    failure_rate_threshold: float = 0.5  # Failure rate to open circuit (0.0-1.0)
    min_calls_threshold: int = 10       # Minimum calls before calculating failure rate
    half_open_max_calls: int = 5        # Max calls allowed in half-open state


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: str
    failure_count: int
    success_count: int
    total_calls: int
    failure_rate: float
    last_failure_time: Optional[float]
    last_state_change: float
    time_to_retry: Optional[float]


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, stats: Optional[CircuitBreakerStats] = None):
        super().__init__(message)
        self.stats = stats


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._last_state_change = time.time()
        self._lock = threading.RLock()
        
        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        
        # Call history for failure rate calculation
        self._call_history = deque(maxlen=1000)  # Store last 1000 calls
        
        # Callbacks
        self._on_open_callbacks: List[Callable] = []
        self._on_close_callbacks: List[Callable] = []
        self._on_half_open_callbacks: List[Callable] = []
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Any exception from the wrapped function
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    # Circuit is open, reject call
                    time_to_retry = self._time_until_retry()
                    stats = self.get_stats()
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry in {time_to_retry:.1f} seconds.",
                        stats=stats
                    )
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Too many calls in half-open state
                    stats = self.get_stats()
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN and at call limit.",
                        stats=stats
                    )
                
                self._half_open_calls += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._record_success(start_time)
            return result
        
        except Exception as e:
            self._record_failure(start_time, e)
            raise
    
    def _record_success(self, start_time: float):
        """Record successful call."""
        with self._lock:
            end_time = time.time()
            duration = end_time - start_time
            
            # Record call in history
            self._call_history.append({
                'timestamp': end_time,
                'success': True,
                'duration': duration
            })
            
            self._success_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Check if we should close the circuit
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                    self._reset_counts()
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _record_failure(self, start_time: float, exception: Exception):
        """Record failed call."""
        with self._lock:
            end_time = time.time()
            duration = end_time - start_time
            
            # Record call in history
            self._call_history.append({
                'timestamp': end_time,
                'success': False,
                'duration': duration,
                'exception': str(exception)
            })
            
            self._failure_count += 1
            self._last_failure_time = end_time
            
            if self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()
            
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
                self._reset_counts()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        # Check failure count threshold
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        recent_calls = self._get_recent_calls()
        if len(recent_calls) >= self.config.min_calls_threshold:
            failure_rate = self._calculate_failure_rate(recent_calls)
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset (transition to half-open)."""
        if self._last_failure_time is None:
            return True
        
        return (time.time() - self._last_failure_time) >= self.config.timeout_seconds
    
    def _get_recent_calls(self) -> List[Dict[str, Any]]:
        """Get calls within the monitoring window."""
        current_time = time.time()
        cutoff_time = current_time - self.config.monitoring_window_seconds
        
        return [
            call for call in self._call_history
            if call['timestamp'] > cutoff_time
        ]
    
    def _calculate_failure_rate(self, calls: List[Dict[str, Any]]) -> float:
        """Calculate failure rate from call history."""
        if not calls:
            return 0.0
        
        failed_calls = sum(1 for call in calls if not call['success'])
        return failed_calls / len(calls)
    
    def _time_until_retry(self) -> float:
        """Calculate time until retry is allowed."""
        if self._last_failure_time is None:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        remaining = self.config.timeout_seconds - elapsed
        return max(0.0, remaining)
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._last_state_change = time.time()
            self._trigger_callbacks(self._on_open_callbacks)
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        if self._state != CircuitState.HALF_OPEN:
            self._state = CircuitState.HALF_OPEN
            self._last_state_change = time.time()
            self._half_open_calls = 0
            self._success_count = 0  # Reset success count for half-open evaluation
            self._trigger_callbacks(self._on_half_open_callbacks)
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._last_state_change = time.time()
            self._trigger_callbacks(self._on_close_callbacks)
    
    def _reset_counts(self):
        """Reset failure and success counts."""
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    def _trigger_callbacks(self, callbacks: List[Callable]):
        """Trigger state change callbacks."""
        for callback in callbacks:
            try:
                callback(self)
            except Exception as e:
                # Don't let callback errors affect circuit breaker operation
                pass
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            recent_calls = self._get_recent_calls()
            failure_rate = self._calculate_failure_rate(recent_calls)
            
            return CircuitBreakerStats(
                state=self._state.value,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=len(self._call_history),
                failure_rate=failure_rate,
                last_failure_time=self._last_failure_time,
                last_state_change=self._last_state_change,
                time_to_retry=self._time_until_retry() if self._state == CircuitState.OPEN else None
            )
    
    def force_open(self):
        """Force circuit breaker to OPEN state."""
        with self._lock:
            self._transition_to_open()
            self._last_failure_time = time.time()
    
    def force_close(self):
        """Force circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to_closed()
            self._reset_counts()
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._last_failure_time = None
            self._last_state_change = time.time()
            self._reset_counts()
            self._call_history.clear()
    
    def on_open(self, callback: Callable):
        """Register callback for circuit opening."""
        self._on_open_callbacks.append(callback)
    
    def on_close(self, callback: Callable):
        """Register callback for circuit closing."""
        self._on_close_callbacks.append(callback)
    
    def on_half_open(self, callback: Callable):
        """Register callback for circuit half-open."""
        self._on_half_open_callbacks.append(callback)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration (only used when creating new breaker)
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def remove_breaker(self, name: str):
        """Remove circuit breaker from registry."""
        with self._lock:
            self._breakers.pop(name, None)
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_breaker_names(self) -> List[str]:
        """Get names of all registered circuit breakers."""
        with self._lock:
            return list(self._breakers.keys())


# Global registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get circuit breaker from global registry."""
    return _global_registry.get_breaker(name, config)


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None
):
    """Decorator for adding circuit breaker to functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        config: Circuit breaker configuration
    """
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__qualname__}"
        breaker = get_circuit_breaker(breaker_name, config)
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.circuit_breaker = breaker
        
        return wrapper
    
    return decorator


def get_all_circuit_stats() -> Dict[str, CircuitBreakerStats]:
    """Get statistics for all circuit breakers."""
    return _global_registry.get_all_stats()


def reset_all_circuits():
    """Reset all circuit breakers."""
    _global_registry.reset_all()


# Neuromorphic-specific circuit breaker configurations
NEUROMORPHIC_CONFIGS = {
    'model_inference': CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=30.0,
        failure_rate_threshold=0.3,
        min_calls_threshold=5
    ),
    'spike_processing': CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=60.0,
        failure_rate_threshold=0.4,
        min_calls_threshold=10
    ),
    'learning_update': CircuitBreakerConfig(
        failure_threshold=2,
        timeout_seconds=120.0,
        failure_rate_threshold=0.2,
        min_calls_threshold=3
    ),
    'data_loading': CircuitBreakerConfig(
        failure_threshold=10,
        timeout_seconds=30.0,
        failure_rate_threshold=0.5,
        min_calls_threshold=20
    ),
    'gpu_operations': CircuitBreakerConfig(
        failure_threshold=2,
        timeout_seconds=60.0,
        failure_rate_threshold=0.2,
        min_calls_threshold=5
    )
}


def get_neuromorphic_circuit_breaker(operation_type: str) -> CircuitBreaker:
    """Get preconfigured circuit breaker for neuromorphic operations.
    
    Args:
        operation_type: Type of neuromorphic operation
        
    Returns:
        Configured CircuitBreaker instance
    """
    config = NEUROMORPHIC_CONFIGS.get(operation_type)
    return get_circuit_breaker(f"neuromorphic.{operation_type}", config)