"""Robust error handling for neuromorphic computing systems."""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Optional, Dict, Union
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    input_shapes: Dict[str, Any]
    timestamp: float
    error_type: str
    error_message: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


class NeuromorphicError(Exception):
    """Base exception for neuromorphic computing errors."""
    def __init__(self, message: str, error_code: str = None, context: ErrorContext = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context


class ModelError(NeuromorphicError):
    """Errors related to neural models."""
    pass


class AlgorithmError(NeuromorphicError):
    """Errors related to algorithms."""
    pass


class DataError(NeuromorphicError):
    """Errors related to data processing."""
    pass


class ResourceError(NeuromorphicError):
    """Errors related to resource constraints."""
    pass


class ErrorRecoveryManager:
    """Manages error recovery strategies for neuromorphic systems."""
    
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {
            'dimension_mismatch': self._recover_dimension_mismatch,
            'memory_error': self._recover_memory_error,
            'numerical_instability': self._recover_numerical_instability,
            'resource_exhaustion': self._recover_resource_exhaustion,
        }
        self.logger = logging.getLogger(__name__)
    
    def _recover_dimension_mismatch(self, func_args: tuple, func_kwargs: dict, error: Exception) -> Any:
        """Attempt to recover from dimension mismatch errors."""
        self.logger.warning(f"Attempting dimension mismatch recovery: {error}")
        
        # Try to reshape inputs to compatible dimensions
        if func_args:
            new_args = []
            for arg in func_args:
                if hasattr(arg, 'shape') and hasattr(arg, 'reshape'):
                    # Try to flatten if multi-dimensional
                    if len(arg.shape) > 2:
                        new_args.append(arg.reshape(arg.shape[0], -1))
                    else:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            return tuple(new_args), func_kwargs
        
        return func_args, func_kwargs
    
    def _recover_memory_error(self, func_args: tuple, func_kwargs: dict, error: Exception) -> Any:
        """Attempt to recover from memory errors by reducing batch size."""
        self.logger.warning(f"Attempting memory error recovery: {error}")
        
        # Reduce batch size if present
        new_kwargs = func_kwargs.copy()
        if 'batch_size' in new_kwargs:
            new_kwargs['batch_size'] = max(1, new_kwargs['batch_size'] // 2)
            self.logger.info(f"Reduced batch size to {new_kwargs['batch_size']}")
        
        # Try to process data in smaller chunks
        if func_args:
            new_args = []
            for arg in func_args:
                if hasattr(arg, 'shape') and len(arg.shape) > 0 and arg.shape[0] > 1:
                    # Take only first half of batch
                    half_size = arg.shape[0] // 2
                    new_args.append(arg[:half_size])
                else:
                    new_args.append(arg)
            return tuple(new_args), new_kwargs
        
        return func_args, new_kwargs
    
    def _recover_numerical_instability(self, func_args: tuple, func_kwargs: dict, error: Exception) -> Any:
        """Attempt to recover from numerical instability."""
        self.logger.warning(f"Attempting numerical instability recovery: {error}")
        
        # Add numerical stability improvements
        new_kwargs = func_kwargs.copy()
        
        # Reduce learning rate if present
        if 'learning_rate' in new_kwargs:
            new_kwargs['learning_rate'] *= 0.5
            self.logger.info(f"Reduced learning rate to {new_kwargs['learning_rate']}")
        
        # Add epsilon for numerical stability
        if 'epsilon' not in new_kwargs:
            new_kwargs['epsilon'] = 1e-8
        
        return func_args, new_kwargs
    
    def _recover_resource_exhaustion(self, func_args: tuple, func_kwargs: dict, error: Exception) -> Any:
        """Attempt to recover from resource exhaustion."""
        self.logger.warning(f"Attempting resource exhaustion recovery: {error}")
        
        # Reduce computational complexity
        new_kwargs = func_kwargs.copy()
        
        # Reduce duration or time steps
        if 'duration' in new_kwargs:
            new_kwargs['duration'] = min(100.0, new_kwargs['duration'] * 0.5)
        
        if 'time_steps' in new_kwargs:
            new_kwargs['time_steps'] = max(10, new_kwargs['time_steps'] // 2)
        
        return func_args, new_kwargs
    
    def classify_error(self, error: Exception, function_name: str) -> str:
        """Classify error type for appropriate recovery strategy."""
        error_msg = str(error).lower()
        
        if 'shape' in error_msg or 'dimension' in error_msg or 'size' in error_msg:
            return 'dimension_mismatch'
        elif 'memory' in error_msg or 'out of memory' in error_msg:
            return 'memory_error'
        elif 'nan' in error_msg or 'inf' in error_msg or 'overflow' in error_msg:
            return 'numerical_instability'
        elif 'resource' in error_msg or 'timeout' in error_msg:
            return 'resource_exhaustion'
        else:
            return 'unknown'
    
    def log_error(self, context: ErrorContext):
        """Log error with context information."""
        self.error_log.append(context)
        self.logger.error(
            f"Error in {context.function_name}: {context.error_type} - {context.error_message}"
        )
        self.logger.error(f"Input shapes: {context.input_shapes}")


def robust_execution(max_retries: int = 2, fallback_value: Any = None):
    """Decorator for robust execution with error recovery."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recovery_manager = ErrorRecoveryManager()
            logger = logging.getLogger(func.__name__)
            
            # Extract input shapes for context
            input_shapes = {}
            for i, arg in enumerate(args):
                if hasattr(arg, 'shape'):
                    input_shapes[f'arg_{i}'] = arg.shape
            for key, value in kwargs.items():
                if hasattr(value, 'shape'):
                    input_shapes[key] = value.shape
            
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt} for {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Successfully recovered after {attempt} attempts")
                    
                    return result
                
                except Exception as e:
                    last_error = e
                    error_type = recovery_manager.classify_error(e, func.__name__)
                    
                    # Create error context
                    context = ErrorContext(
                        function_name=func.__name__,
                        input_shapes=input_shapes,
                        timestamp=time.time(),
                        error_type=error_type,
                        error_message=str(e),
                        recovery_attempted=attempt < max_retries
                    )
                    
                    if attempt < max_retries:
                        # Attempt recovery
                        try:
                            if error_type in recovery_manager.recovery_strategies:
                                new_args, new_kwargs = recovery_manager.recovery_strategies[error_type](
                                    args, kwargs, e
                                )
                                args = new_args
                                kwargs = new_kwargs
                                context.recovery_attempted = True
                                logger.info(f"Applied recovery strategy for {error_type}")
                            else:
                                logger.warning(f"No recovery strategy for error type: {error_type}")
                        except Exception as recovery_error:
                            logger.error(f"Recovery attempt failed: {recovery_error}")
                    
                    recovery_manager.log_error(context)
            
            # All retries exhausted
            logger.error(f"All retry attempts exhausted for {func.__name__}")
            
            if fallback_value is not None:
                logger.info(f"Returning fallback value for {func.__name__}")
                return fallback_value
            
            # Re-raise the last error with additional context
            raise ModelError(
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                error_code="MAX_RETRIES_EXCEEDED",
                context=context
            ) from last_error
        
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__name__)
            
            # Validate arguments
            for i, (validator_name, validator_func) in enumerate(validators.items()):
                try:
                    if i < len(args):
                        if not validator_func(args[i]):
                            raise DataError(
                                f"Validation failed for argument {i} in {func.__name__}: {validator_name}",
                                error_code="VALIDATION_FAILED"
                            )
                    elif validator_name in kwargs:
                        if not validator_func(kwargs[validator_name]):
                            raise DataError(
                                f"Validation failed for parameter {validator_name} in {func.__name__}",
                                error_code="VALIDATION_FAILED"
                            )
                except Exception as e:
                    logger.error(f"Validation error in {func.__name__}: {e}")
                    raise
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Common validators
def is_positive(value) -> bool:
    """Check if value is positive."""
    return value > 0


def is_array_like(value) -> bool:
    """Check if value is array-like."""
    return hasattr(value, '__len__') or hasattr(value, 'shape')


def has_finite_values(value) -> bool:
    """Check if array has finite values only."""
    if hasattr(value, 'shape'):
        if hasattr(value, 'isfinite'):  # JAX arrays
            return jnp.all(jnp.isfinite(value))
        else:  # NumPy arrays
            return np.all(np.isfinite(value))
    return True


def is_in_range(min_val: float, max_val: float):
    """Create validator for range checking."""
    def validator(value) -> bool:
        if hasattr(value, 'shape'):
            return jnp.all((value >= min_val) & (value <= max_val))
        else:
            return min_val <= value <= max_val
    return validator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half_open
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half_open'
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise ResourceError(
                    "Circuit breaker is open - service temporarily unavailable",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
                self.logger.info("Circuit breaker closed - service recovered")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


# Global circuit breaker instance
neural_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)


class RobustErrorHandler:
    """Comprehensive error handling for neuromorphic systems."""
    
    def __init__(self):
        self.recovery_manager = RecoveryManager()
        self.circuit_breaker = CircuitBreaker()
        self.logger = logging.getLogger(__name__)
    
    def handle_with_recovery(self, func: Callable, *args, **kwargs):
        """Handle function execution with automatic recovery."""
        return self.circuit_breaker.call(func, *args, **kwargs)
    
    def validate_and_execute(self, func: Callable, validators: dict, *args, **kwargs):
        """Validate inputs and execute function with error handling."""
        # Apply validation
        validate_inputs(validators)(func)(*args, **kwargs)
    
    def get_error_stats(self):
        """Get error handling statistics."""
        return {
            'circuit_breaker_state': self.circuit_breaker.state,
            'failure_count': self.circuit_breaker.failure_count,
            'recovery_strategies': len(self.recovery_manager.strategies)
        }