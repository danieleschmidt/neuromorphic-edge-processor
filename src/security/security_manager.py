"""Security manager for neuromorphic computing operations."""

import torch
import numpy as np
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import wraps
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_input_size: int = 10000
    max_sequence_length: int = 5000
    max_memory_mb: float = 1000.0
    allowed_dtypes: List[str] = None
    enable_input_sanitization: bool = True
    enable_output_validation: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    log_security_events: bool = True
    
    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = ['float32', 'float64', 'int32', 'int64']


class SecurityManager:
    """Comprehensive security manager for neuromorphic operations.
    
    Provides input validation, sanitization, rate limiting, memory monitoring,
    and security logging for all neuromorphic computing operations.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security manager.
        
        Args:
            config: Security configuration settings
        """
        self.config = config if config is not None else SecurityConfig()
        self.logger = self._setup_logger()
        
        # Rate limiting
        self.request_history: Dict[str, List[float]] = {}
        
        # Memory tracking
        self.memory_usage_history: List[float] = []
        
        # Security event logging
        self.security_events: List[Dict] = []
        
        # Hash cache for input validation
        self.input_hash_cache: Dict[str, bool] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup security event logger."""
        logger = logging.getLogger('neuromorphic_security')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray], 
                      source: str = "unknown") -> bool:
        """Validate input data for security compliance.
        
        Args:
            data: Input tensor or array to validate
            source: Source identifier for logging
            
        Returns:
            True if input is valid, False otherwise
        """
        try:
            # Convert to torch tensor if numpy
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            
            if not isinstance(data, torch.Tensor):
                self._log_security_event("invalid_input_type", {
                    "source": source,
                    "type": str(type(data)),
                    "reason": "Input must be torch.Tensor or numpy.ndarray"
                })
                return False
            
            # Check data type
            if str(data.dtype).replace('torch.', '') not in self.config.allowed_dtypes:
                self._log_security_event("invalid_dtype", {
                    "source": source,
                    "dtype": str(data.dtype),
                    "allowed": self.config.allowed_dtypes
                })
                return False
            
            # Check size constraints
            if data.numel() > self.config.max_input_size:
                self._log_security_event("input_size_exceeded", {
                    "source": source,
                    "size": data.numel(),
                    "max_allowed": self.config.max_input_size
                })
                return False
            
            # Check sequence length if 3D
            if data.dim() == 3 and data.shape[-1] > self.config.max_sequence_length:
                self._log_security_event("sequence_length_exceeded", {
                    "source": source,
                    "length": data.shape[-1],
                    "max_allowed": self.config.max_sequence_length
                })
                return False
            
            # Check for NaN/Inf values
            if torch.isnan(data).any() or torch.isinf(data).any():
                self._log_security_event("invalid_values", {
                    "source": source,
                    "has_nan": torch.isnan(data).any().item(),
                    "has_inf": torch.isinf(data).any().item()
                })
                return False
            
            # Check value ranges (prevent adversarial inputs)
            if torch.abs(data).max() > 1e6:
                self._log_security_event("extreme_values", {
                    "source": source,
                    "max_abs_value": torch.abs(data).max().item(),
                    "threshold": 1e6
                })
                return False
            
            return True
            
        except Exception as e:
            self._log_security_event("validation_error", {
                "source": source,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def sanitize_input(self, data: torch.Tensor, method: str = "clip") -> torch.Tensor:
        """Sanitize input data to prevent adversarial attacks.
        
        Args:
            data: Input tensor to sanitize
            method: Sanitization method ("clip", "normalize", "filter")
            
        Returns:
            Sanitized tensor
        """
        if not self.config.enable_input_sanitization:
            return data
        
        sanitized = data.clone()
        
        try:
            if method == "clip":
                # Clip extreme values
                sanitized = torch.clamp(sanitized, -100.0, 100.0)
                
            elif method == "normalize":
                # Normalize to reasonable range
                if sanitized.std() > 0:
                    sanitized = (sanitized - sanitized.mean()) / sanitized.std()
                    sanitized = torch.clamp(sanitized, -5.0, 5.0)
                    
            elif method == "filter":
                # Apply median filter to remove spikes
                if sanitized.dim() >= 2:
                    # Simple median filter approximation
                    kernel_size = 3
                    padding = kernel_size // 2
                    
                    if sanitized.dim() == 2:
                        # Apply to 2D data
                        padded = torch.nn.functional.pad(sanitized, (padding, padding, padding, padding), 'reflect')
                        filtered = torch.zeros_like(sanitized)
                        
                        for i in range(sanitized.shape[0]):
                            for j in range(sanitized.shape[1]):
                                window = padded[i:i+kernel_size, j:j+kernel_size]
                                filtered[i, j] = torch.median(window)
                        
                        sanitized = filtered
            
            # Replace NaN/Inf with zeros
            sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=100.0, neginf=-100.0)
            
            self._log_security_event("input_sanitized", {
                "method": method,
                "original_shape": list(data.shape),
                "changes_made": not torch.allclose(data, sanitized, atol=1e-6)
            })
            
            return sanitized
            
        except Exception as e:
            self.logger.warning(f"Sanitization failed: {e}. Returning clamped input.")
            return torch.clamp(data, -100.0, 100.0)
    
    def check_memory_usage(self, threshold_mb: Optional[float] = None) -> bool:
        """Check current memory usage against limits.
        
        Args:
            threshold_mb: Memory threshold in MB (uses config default if None)
            
        Returns:
            True if memory usage is acceptable
        """
        threshold = threshold_mb if threshold_mb is not None else self.config.max_memory_mb
        
        try:
            # Get memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024**2
            else:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / 1024**2
            
            self.memory_usage_history.append(memory_mb)
            
            # Keep only recent history
            if len(self.memory_usage_history) > 1000:
                self.memory_usage_history = self.memory_usage_history[-1000:]
            
            if memory_mb > threshold:
                self._log_security_event("memory_limit_exceeded", {
                    "current_mb": memory_mb,
                    "threshold_mb": threshold,
                    "peak_mb": max(self.memory_usage_history[-100:]) if self.memory_usage_history else memory_mb
                })
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")
            return True  # Fail open for monitoring
    
    def rate_limit_check(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits.
        
        Args:
            client_id: Identifier for the client/source
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Initialize client history if needed
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        # Remove old requests outside window
        self.request_history[client_id] = [
            t for t in self.request_history[client_id] if t > window_start
        ]
        
        # Check if limit exceeded
        if len(self.request_history[client_id]) >= self.config.rate_limit_requests:
            self._log_security_event("rate_limit_exceeded", {
                "client_id": client_id,
                "requests_in_window": len(self.request_history[client_id]),
                "limit": self.config.rate_limit_requests,
                "window_seconds": self.config.rate_limit_window
            })
            return False
        
        # Record this request
        self.request_history[client_id].append(current_time)
        return True
    
    def validate_output(self, output: torch.Tensor, expected_shape: Optional[Tuple] = None) -> bool:
        """Validate model output for anomalies.
        
        Args:
            output: Model output tensor
            expected_shape: Expected output shape
            
        Returns:
            True if output is valid
        """
        if not self.config.enable_output_validation:
            return True
        
        try:
            # Check for NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                self._log_security_event("invalid_output_values", {
                    "has_nan": torch.isnan(output).any().item(),
                    "has_inf": torch.isinf(output).any().item(),
                    "shape": list(output.shape)
                })
                return False
            
            # Check expected shape
            if expected_shape is not None and tuple(output.shape) != expected_shape:
                self._log_security_event("unexpected_output_shape", {
                    "actual_shape": list(output.shape),
                    "expected_shape": list(expected_shape)
                })
                return False
            
            # Check for extreme values
            max_val = torch.abs(output).max().item()
            if max_val > 1e10:
                self._log_security_event("extreme_output_values", {
                    "max_abs_value": max_val,
                    "shape": list(output.shape)
                })
                return False
            
            return True
            
        except Exception as e:
            self._log_security_event("output_validation_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def compute_input_hash(self, data: torch.Tensor) -> str:
        """Compute hash of input data for caching/tracking.
        
        Args:
            data: Input tensor
            
        Returns:
            SHA-256 hash of the data
        """
        # Convert to bytes for hashing
        data_bytes = data.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for monitoring and analysis.
        
        Args:
            event_type: Type of security event
            details: Event details dictionary
        """
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        if self.config.log_security_events:
            self.logger.warning(f"Security Event - {event_type}: {details}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Returns:
            Security report dictionary
        """
        current_time = time.time()
        recent_events = [
            e for e in self.security_events 
            if current_time - e["timestamp"] < 3600  # Last hour
        ]
        
        event_counts = {}
        for event in recent_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "timestamp": current_time,
            "total_events_last_hour": len(recent_events),
            "event_type_counts": event_counts,
            "memory_usage_stats": {
                "current_mb": self.memory_usage_history[-1] if self.memory_usage_history else 0.0,
                "average_mb": np.mean(self.memory_usage_history) if self.memory_usage_history else 0.0,
                "peak_mb": max(self.memory_usage_history) if self.memory_usage_history else 0.0
            },
            "rate_limiting": {
                "active_clients": len(self.request_history),
                "total_requests_tracked": sum(len(reqs) for reqs in self.request_history.values())
            },
            "configuration": {
                "max_input_size": self.config.max_input_size,
                "max_sequence_length": self.config.max_sequence_length,
                "max_memory_mb": self.config.max_memory_mb,
                "rate_limit": self.config.rate_limit_requests
            }
        }
    
    def secure_operation(self, operation_name: str, client_id: str = "default"):
        """Decorator factory for securing operations.
        
        Args:
            operation_name: Name of the operation being secured
            client_id: Client identifier for rate limiting
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Rate limiting check
                if not self.rate_limit_check(client_id):
                    raise SecurityError(f"Rate limit exceeded for {client_id}")
                
                # Memory check
                if not self.check_memory_usage():
                    raise SecurityError("Memory usage limit exceeded")
                
                # Input validation for tensor arguments
                for arg in args:
                    if isinstance(arg, (torch.Tensor, np.ndarray)):
                        if not self.validate_input(arg, operation_name):
                            raise SecurityError(f"Input validation failed for {operation_name}")
                
                # Execute operation
                try:
                    result = func(*args, **kwargs)
                    
                    # Output validation
                    if isinstance(result, torch.Tensor):
                        if not self.validate_output(result):
                            raise SecurityError(f"Output validation failed for {operation_name}")
                    
                    return result
                    
                except Exception as e:
                    self._log_security_event("operation_error", {
                        "operation": operation_name,
                        "client_id": client_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    raise
            
            return wrapper
        return decorator


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Global security manager instance
_global_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


def configure_security(config: SecurityConfig):
    """Configure global security manager."""
    global _global_security_manager
    _global_security_manager = SecurityManager(config)