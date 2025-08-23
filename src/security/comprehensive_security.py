"""Comprehensive security module for neuromorphic edge systems."""

import numpy as np
import hashlib
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path


class SecurityValidator:
    """Advanced security validation for neuromorphic systems."""
    
    def __init__(self, max_spike_rate: float = 1000.0, max_memory_mb: float = 512.0):
        """Initialize security validator.
        
        Args:
            max_spike_rate: Maximum allowed spike rate (Hz)
            max_memory_mb: Maximum memory usage allowed (MB)
        """
        self.max_spike_rate = max_spike_rate
        self.max_memory_mb = max_memory_mb
        self.logger = self._setup_security_logger()
        
        # Rate limiting
        self.request_history = {}
        self.max_requests_per_minute = 100
        
        # Input sanitization
        self.safe_dtypes = {np.float32, np.float64, np.int32, np.int64, np.bool_}
        
    def _setup_security_logger(self) -> logging.Logger:
        """Set up security-focused logging."""
        logger = logging.getLogger('neuromorphic_security')
        logger.setLevel(logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def sanitize_input_data(self, data: np.ndarray, max_value: float = 100.0) -> np.ndarray:
        """Sanitize input data to prevent malicious inputs.
        
        Args:
            data: Input data array
            max_value: Maximum allowed value
            
        Returns:
            Sanitized data array
            
        Raises:
            SecurityError: If data fails security checks
        """
        try:
            # Check data type
            if data.dtype not in self.safe_dtypes:
                self.logger.warning(f"Potentially unsafe data type: {data.dtype}")
                data = data.astype(np.float32)
            
            # Check for malicious patterns
            if np.any(np.isnan(data)):
                self.logger.error("NaN values detected - potential attack vector")
                data = np.nan_to_num(data, nan=0.0)
            
            if np.any(np.isinf(data)):
                self.logger.error("Infinite values detected - potential DoS attack")
                data = np.nan_to_num(data, posinf=max_value, neginf=-max_value)
            
            # Value range validation
            if np.any(np.abs(data) > max_value):
                self.logger.warning(f"Values exceed safe range: max={np.abs(data).max()}")
                data = np.clip(data, -max_value, max_value)
            
            # Memory usage check
            memory_mb = data.nbytes / (1024 * 1024)
            if memory_mb > self.max_memory_mb:
                raise SecurityError(f"Input data too large: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Input sanitization failed: {str(e)}")
            raise SecurityError(f"Failed to sanitize input data: {str(e)}")
    
    def validate_spike_patterns(self, spikes: np.ndarray) -> bool:
        """Validate spike patterns for suspicious activity.
        
        Args:
            spikes: Spike data to validate
            
        Returns:
            True if patterns are safe
            
        Raises:
            SecurityError: If malicious patterns detected
        """
        try:
            # Calculate spike rate
            if spikes.ndim >= 3:
                spike_counts = spikes.sum(axis=(0, 2))  # Sum over batch and time
                time_duration = spikes.shape[-1]  # Time steps
                spike_rates = spike_counts * 1000.0 / time_duration  # Assume 1ms time steps
            else:
                spike_rates = spikes.sum(axis=-1)  # Simple sum
            
            # Check for abnormally high spike rates (potential DoS)
            max_rate = np.max(spike_rates)
            if max_rate > self.max_spike_rate:
                self.logger.error(f"Abnormally high spike rate detected: {max_rate:.1f}Hz")
                raise SecurityError(f"Spike rate {max_rate:.1f}Hz exceeds safe limit {self.max_spike_rate}Hz")
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            self.logger.error(f"Spike pattern validation failed: {str(e)}")
            return False
    
    def create_secure_hash(self, data: Union[np.ndarray, str, bytes]) -> str:
        """Create secure hash of data for integrity verification.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex string of hash
        """
        try:
            hasher = hashlib.sha256()
            
            if isinstance(data, np.ndarray):
                hasher.update(data.tobytes())
            elif isinstance(data, str):
                hasher.update(data.encode('utf-8'))
            else:
                hasher.update(data)
            
            return hasher.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Hash creation failed: {str(e)}")
            return ""


class SecurityError(Exception):
    """Security-related exception."""
    pass


# Global security validator
security_validator = SecurityValidator()


def secure_neural_processing(func):
    """Decorator for secure neural processing operations."""
    def wrapper(*args, **kwargs):
        try:
            # Basic security checks
            for arg in args:
                if isinstance(arg, np.ndarray):
                    security_validator.sanitize_input_data(arg)
            
            result = func(*args, **kwargs)
            
            # Validate output if it's an array
            if isinstance(result, np.ndarray):
                security_validator.sanitize_input_data(result)
            elif isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], np.ndarray):
                    security_validator.sanitize_input_data(result[0])
            
            return result
            
        except SecurityError:
            raise
        except Exception as e:
            security_validator.logger.error(f"Security wrapper failed: {str(e)}")
            raise SecurityError(f"Secure processing failed: {str(e)}")
    
    return wrapper