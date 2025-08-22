"""Comprehensive error handling and recovery mechanisms for neuromorphic systems."""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
import numpy as np
import logging
import traceback
import functools
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import threading
import sys
import warnings


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    COMPUTATIONAL = "computational"
    MEMORY = "memory"
    HARDWARE = "hardware"
    NETWORK = "network"
    INPUT_VALIDATION = "input_validation"
    MODEL = "model"
    INFERENCE = "inference"
    SYSTEM = "system"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


class NeuromorphicError(Exception):
    """Base exception for neuromorphic computing errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recoverable = recoverable


class SpikeProcessingError(NeuromorphicError):
    """Errors related to spike processing."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.COMPUTATIONAL, **kwargs)


class ModelInferenceError(NeuromorphicError):
    """Errors during model inference."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.INFERENCE, **kwargs)


class MemoryError(NeuromorphicError):
    """Memory-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MEMORY, severity=ErrorSeverity.HIGH, **kwargs)


class HardwareError(NeuromorphicError):
    """Hardware-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.HARDWARE, severity=ErrorSeverity.CRITICAL, **kwargs)


class ErrorHandler:
    """Comprehensive error handling system with recovery mechanisms."""
    
    def __init__(
        self,
        max_retry_attempts: int = 3,
        retry_delay: float = 1.0,
        enable_automatic_recovery: bool = True,
        log_errors: bool = True,
        error_history_size: int = 1000
    ):
        """Initialize error handler.
        
        Args:
            max_retry_attempts: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            enable_automatic_recovery: Enable automatic recovery attempts
            log_errors: Enable error logging
            error_history_size: Maximum number of errors to keep in history
        """
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.enable_automatic_recovery = enable_automatic_recovery
        self.log_errors = log_errors
        self.error_history_size = error_history_size
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.log_errors:
            self._setup_error_logging()
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def _setup_error_logging(self):
        """Setup error logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.ERROR)
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for different error categories."""
        # Memory errors
        self.register_recovery_strategy(
            ErrorCategory.MEMORY,
            self._memory_cleanup_strategy
        )
        self.register_recovery_strategy(
            ErrorCategory.MEMORY,
            self._reduce_batch_size_strategy
        )
        
        # Computational errors
        self.register_recovery_strategy(
            ErrorCategory.COMPUTATIONAL,
            self._numerical_stabilization_strategy
        )
        self.register_recovery_strategy(
            ErrorCategory.COMPUTATIONAL,
            self._fallback_computation_strategy
        )
        
        # Model errors
        self.register_recovery_strategy(
            ErrorCategory.MODEL,
            self._model_reset_strategy
        )
        
        # Hardware errors
        self.register_recovery_strategy(
            ErrorCategory.HARDWARE,
            self._device_fallback_strategy
        )
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[ErrorInfo, Any], Tuple[bool, Any]]
    ):
        """Register a recovery strategy for a specific error category.
        
        Args:
            category: Error category to handle
            strategy: Recovery function that takes (error_info, context) and returns (success, result)
        """
        with self.lock:
            if category not in self.recovery_strategies:
                self.recovery_strategies[category] = []
            self.recovery_strategies[category].append(strategy)
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None
    ) -> ErrorInfo:
        """Handle an error and attempt recovery if enabled.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            severity: Override error severity
            category: Override error category
            
        Returns:
            ErrorInfo object with details about the error and recovery attempt
        """
        # Generate unique error ID
        error_id = f"error_{int(time.time() * 1000000)}_{id(exception)}"
        
        # Classify error if not provided
        if isinstance(exception, NeuromorphicError):
            severity = severity or exception.severity
            category = category or exception.category
            context = {**(exception.context or {}), **(context or {})}
        else:
            severity = severity or self._classify_error_severity(exception)
            category = category or self._classify_error_category(exception)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # Store error in history
        with self.lock:
            self.error_history.append(error_info)
            if len(self.error_history) > self.error_history_size:
                self.error_history.pop(0)
            
            # Update error counts
            error_key = f"{category.value}:{type(exception).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        if self.log_errors:
            self.logger.error(
                f"[{severity.value.upper()}] {category.value}: {error_info.message}",
                extra={"error_id": error_id, "context": context}
            )
        
        # Attempt recovery if enabled and error is recoverable
        if (self.enable_automatic_recovery and 
            severity != ErrorSeverity.CRITICAL and
            category in self.recovery_strategies):
            
            recovery_success = self._attempt_recovery(error_info, context)
            error_info.recovery_attempted = True
            error_info.recovery_successful = recovery_success
        
        return error_info
    
    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type."""
        if isinstance(exception, MemoryError):
            return ErrorSeverity.HIGH
        elif TORCH_AVAILABLE and hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(exception, torch.cuda.OutOfMemoryError):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (RuntimeError, SystemError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _classify_error_category(self, exception: Exception) -> ErrorCategory:
        """Classify error category based on exception type."""
        if isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY
        elif TORCH_AVAILABLE and hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(exception, torch.cuda.OutOfMemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(exception, RuntimeError):
            if "CUDA" in str(exception):
                return ErrorCategory.HARDWARE
            else:
                return ErrorCategory.COMPUTATIONAL
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.INPUT_VALIDATION
        else:
            return ErrorCategory.SYSTEM
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Optional[Dict[str, Any]]) -> bool:
        """Attempt to recover from an error using registered strategies.
        
        Args:
            error_info: Error information
            context: Recovery context
            
        Returns:
            True if recovery was successful
        """
        strategies = self.recovery_strategies.get(error_info.category, [])
        
        for i, strategy in enumerate(strategies):
            try:
                self.logger.info(f"Attempting recovery strategy {i+1}/{len(strategies)} for {error_info.error_id}")
                
                success, result = strategy(error_info, context)
                
                if success:
                    error_info.recovery_method = strategy.__name__
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return True
                
            except Exception as recovery_exception:
                self.logger.warning(
                    f"Recovery strategy {strategy.__name__} failed: {recovery_exception}"
                )
                continue
        
        self.logger.warning(f"All recovery strategies failed for {error_info.error_id}")
        return False
    
    def _memory_cleanup_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Clean up memory and force garbage collection."""
        try:
            import gc
            
            # Clear PyTorch cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory info
            if TORCH_AVAILABLE and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                self.logger.info(f"Memory after cleanup - Allocated: {allocated/1024**2:.2f}MB, Cached: {cached/1024**2:.2f}MB")
            
            return True, "Memory cleanup completed"
            
        except Exception as e:
            return False, f"Memory cleanup failed: {e}"
    
    def _reduce_batch_size_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Suggest batch size reduction."""
        try:
            if context and isinstance(context, dict):
                current_batch_size = context.get('batch_size', 32)
                new_batch_size = max(1, current_batch_size // 2)
                
                context['suggested_batch_size'] = new_batch_size
                
                return True, f"Suggested batch size reduction: {current_batch_size} -> {new_batch_size}"
            
            return False, "No batch size context available"
            
        except Exception as e:
            return False, f"Batch size strategy failed: {e}"
    
    def _numerical_stabilization_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Apply numerical stabilization techniques."""
        try:
            # Add epsilon to avoid division by zero
            if context and isinstance(context, dict):
                context['numerical_epsilon'] = 1e-8
                context['use_gradient_clipping'] = True
                context['max_gradient_norm'] = 1.0
                
                return True, "Numerical stabilization parameters set"
            
            return False, "No context for numerical stabilization"
            
        except Exception as e:
            return False, f"Numerical stabilization failed: {e}"
    
    def _fallback_computation_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Switch to fallback computation method."""
        try:
            if context and isinstance(context, dict):
                context['use_fallback_computation'] = True
                context['reduce_precision'] = True
                
                return True, "Fallback computation mode enabled"
            
            return False, "No context for fallback computation"
            
        except Exception as e:
            return False, f"Fallback computation failed: {e}"
    
    def _model_reset_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Reset model state."""
        try:
            if context and hasattr(context, 'reset_state'):
                context.reset_state()
                return True, "Model state reset"
            
            return False, "No model reset method available"
            
        except Exception as e:
            return False, f"Model reset failed: {e}"
    
    def _device_fallback_strategy(self, error_info: ErrorInfo, context: Any) -> Tuple[bool, Any]:
        """Recovery strategy: Fall back to CPU if GPU fails."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available() and "CUDA" in error_info.message:
                if context and isinstance(context, dict):
                    context['device'] = 'cpu'
                    context['gpu_fallback_applied'] = True
                    
                    return True, "GPU -> CPU fallback applied"
            
            return False, "Device fallback not applicable"
            
        except Exception as e:
            return False, f"Device fallback failed: {e}"
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        error_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Safely execute a function with error handling and retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            max_retries: Override max retry attempts
            retry_delay: Override retry delay
            error_context: Additional error context
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        max_retries = max_retries or self.max_retry_attempts
        retry_delay = retry_delay or self.retry_delay
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Handle the error
                error_info = self.handle_error(
                    e,
                    context={
                        **(error_context or {}),
                        'function': func.__name__ if hasattr(func, '__name__') else str(func),
                        'attempt': attempt + 1,
                        'max_attempts': max_retries + 1
                    }
                )
                
                # If this was the last attempt, re-raise
                if attempt == max_retries:
                    if self.log_errors:
                        self.logger.error(f"All {max_retries + 1} attempts failed for {func.__name__ if hasattr(func, '__name__') else 'function'}")
                    break
                
                # Wait before retry
                if retry_delay > 0:
                    time.sleep(retry_delay)
                
                # Log retry attempt
                if self.log_errors:
                    self.logger.info(f"Retrying {func.__name__ if hasattr(func, '__name__') else 'function'} (attempt {attempt + 2}/{max_retries + 1})")
        
        # Re-raise the last exception
        raise last_exception
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends.
        
        Returns:
            Dictionary containing error statistics
        """
        with self.lock:
            if not self.error_history:
                return {"total_errors": 0, "error_counts": {}, "recovery_rate": 0.0}
            
            total_errors = len(self.error_history)
            recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
            successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
            
            # Error counts by category and severity
            category_counts = {}
            severity_counts = {}
            
            for error in self.error_history:
                category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            # Recent errors (last hour)
            recent_threshold = time.time() - 3600
            recent_errors = [e for e in self.error_history if e.timestamp > recent_threshold]
            
            return {
                "total_errors": total_errors,
                "recent_errors_1h": len(recent_errors),
                "recovery_attempts": recovery_attempts,
                "successful_recoveries": successful_recoveries,
                "recovery_rate": successful_recoveries / max(1, recovery_attempts),
                "error_counts_by_type": dict(self.error_counts),
                "error_counts_by_category": category_counts,
                "error_counts_by_severity": severity_counts,
                "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
            }
    
    def clear_error_history(self):
        """Clear error history and reset counters."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
        
        if self.log_errors:
            self.logger.info("Error history cleared")


# Decorator for automatic error handling
def handle_errors(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    error_context: Optional[Dict[str, Any]] = None,
    severity: Optional[ErrorSeverity] = None,
    category: Optional[ErrorCategory] = None
):
    """Decorator for automatic error handling and retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)
        error_context: Additional error context
        severity: Override error severity
        category: Override error category
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error handler
            if not hasattr(wrapper, '_error_handler'):
                wrapper._error_handler = ErrorHandler()
            
            # Add function context
            context = {
                **(error_context or {}),
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
            
            return wrapper._error_handler.safe_execute(
                func, *args,
                max_retries=max_retries,
                retry_delay=retry_delay,
                error_context=context,
                **kwargs
            )
        
        return wrapper
    return decorator


@contextmanager
def error_context(
    handler: ErrorHandler,
    context: Optional[Dict[str, Any]] = None,
    suppress_errors: bool = False
):
    """Context manager for error handling.
    
    Args:
        handler: ErrorHandler instance
        context: Additional error context
        suppress_errors: Whether to suppress errors (return None on error)
    """
    try:
        yield
    except Exception as e:
        error_info = handler.handle_error(e, context=context)
        
        if not suppress_errors:
            raise
        
        # Return None if errors are suppressed
        return None