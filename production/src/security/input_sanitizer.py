"""Advanced input sanitization and validation for neuromorphic systems."""

import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import re
import warnings


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Any
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class SecurityLimits:
    """Security limits for neuromorphic computations."""
    
    # Array size limits
    MAX_ARRAY_ELEMENTS = 1e8  # 100M elements
    MAX_MEMORY_MB = 1000  # 1GB
    
    # Numerical limits
    MAX_FLOAT_VALUE = 1e6
    MIN_FLOAT_VALUE = -1e6
    MAX_INT_VALUE = 2**31 - 1
    MIN_INT_VALUE = -(2**31)
    
    # Spike train limits
    MAX_SPIKE_RATE = 1000.0  # Hz
    MAX_DURATION = 10000.0  # ms
    MAX_NEURONS = 100000
    
    # Network limits
    MAX_LAYERS = 100
    MAX_CONNECTIONS = 1e7
    
    # String limits (for parameters)
    MAX_STRING_LENGTH = 1000
    ALLOWED_STRING_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.\/\s]*$')


class InputSanitizer:
    """Comprehensive input sanitization for neuromorphic computing."""
    
    def __init__(self, strict_mode: bool = True, auto_fix: bool = True):
        """
        Initialize input sanitizer.
        
        Args:
            strict_mode: If True, reject inputs that exceed limits
            auto_fix: If True, attempt to fix issues automatically
        """
        self.strict_mode = strict_mode
        self.auto_fix = auto_fix
        self.limits = SecurityLimits()
    
    def sanitize_array(self, array: Union[np.ndarray, jnp.ndarray, list], 
                      name: str = "array") -> ValidationResult:
        """Sanitize array inputs."""
        warnings_list = []
        errors = []
        metadata = {}
        
        # Convert to numpy for validation
        if isinstance(array, list):
            array = np.array(array)
        elif hasattr(array, '__array__'):
            array = np.asarray(array)
        
        # Check array size
        total_elements = array.size if hasattr(array, 'size') else len(array)
        metadata['total_elements'] = total_elements
        metadata['shape'] = getattr(array, 'shape', len(array))
        metadata['dtype'] = getattr(array, 'dtype', type(array))
        
        if total_elements > self.limits.MAX_ARRAY_ELEMENTS:
            error_msg = f"{name} has {total_elements} elements, exceeding limit of {self.limits.MAX_ARRAY_ELEMENTS}"
            if self.strict_mode:
                errors.append(error_msg)
                return ValidationResult(False, array, warnings_list, errors, metadata)
            else:
                warnings_list.append(error_msg)
                if self.auto_fix:
                    # Truncate array
                    if hasattr(array, 'shape') and len(array.shape) > 0:
                        max_first_dim = int(self.limits.MAX_ARRAY_ELEMENTS / np.prod(array.shape[1:]))
                        array = array[:max_first_dim]
                        warnings_list.append(f"Truncated {name} to {array.shape}")
        
        # Check memory usage estimate
        if hasattr(array, 'dtype'):
            memory_mb = array.nbytes / (1024 * 1024)
            metadata['memory_mb'] = memory_mb
            
            if memory_mb > self.limits.MAX_MEMORY_MB:
                error_msg = f"{name} requires {memory_mb:.1f} MB, exceeding limit of {self.limits.MAX_MEMORY_MB} MB"
                if self.strict_mode:
                    errors.append(error_msg)
                    return ValidationResult(False, array, warnings_list, errors, metadata)
                else:
                    warnings_list.append(error_msg)
        
        # Check for NaN and infinity
        if hasattr(array, 'dtype') and np.issubdtype(array.dtype, np.floating):
            if np.any(np.isnan(array)):
                error_msg = f"{name} contains NaN values"
                if self.auto_fix:
                    array = np.nan_to_num(array, nan=0.0)
                    warnings_list.append(f"Replaced NaN values in {name} with 0.0")
                else:
                    errors.append(error_msg)
            
            if np.any(np.isinf(array)):
                error_msg = f"{name} contains infinite values"
                if self.auto_fix:
                    array = np.nan_to_num(array, posinf=self.limits.MAX_FLOAT_VALUE, 
                                        neginf=self.limits.MIN_FLOAT_VALUE)
                    warnings_list.append(f"Clipped infinite values in {name}")
                else:
                    errors.append(error_msg)
        
        # Check value ranges for floating point
        if hasattr(array, 'dtype') and np.issubdtype(array.dtype, np.floating):
            if np.any(array > self.limits.MAX_FLOAT_VALUE) or np.any(array < self.limits.MIN_FLOAT_VALUE):
                if self.auto_fix:
                    array = np.clip(array, self.limits.MIN_FLOAT_VALUE, self.limits.MAX_FLOAT_VALUE)
                    warnings_list.append(f"Clipped {name} values to safe range")
                else:
                    errors.append(f"{name} contains values outside safe range")
        
        # Convert back to JAX array if input was JAX
        if 'jax' in str(type(array)):
            array = jnp.asarray(array)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, array, warnings_list, errors, metadata)
    
    def sanitize_spike_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Sanitize spike-related parameters."""
        warnings_list = []
        errors = []
        metadata = {}
        sanitized_params = params.copy()
        
        # Check firing rates
        if 'firing_rate' in params:
            rate = params['firing_rate']
            if hasattr(rate, '__iter__'):
                max_rate = np.max(rate)
            else:
                max_rate = rate
            
            if max_rate > self.limits.MAX_SPIKE_RATE:
                if self.auto_fix:
                    if hasattr(rate, '__iter__'):
                        sanitized_params['firing_rate'] = np.clip(rate, 0, self.limits.MAX_SPIKE_RATE)
                    else:
                        sanitized_params['firing_rate'] = min(rate, self.limits.MAX_SPIKE_RATE)
                    warnings_list.append(f"Clipped firing rate to {self.limits.MAX_SPIKE_RATE} Hz")
                else:
                    errors.append(f"Firing rate {max_rate} Hz exceeds limit of {self.limits.MAX_SPIKE_RATE} Hz")
        
        # Check duration
        if 'duration' in params:
            duration = params['duration']
            if duration > self.limits.MAX_DURATION:
                if self.auto_fix:
                    sanitized_params['duration'] = self.limits.MAX_DURATION
                    warnings_list.append(f"Clipped duration to {self.limits.MAX_DURATION} ms")
                else:
                    errors.append(f"Duration {duration} ms exceeds limit of {self.limits.MAX_DURATION} ms")
        
        # Check neuron counts
        for param_name in ['n_neurons', 'num_neurons', 'neurons']:
            if param_name in params:
                n_neurons = params[param_name]
                if n_neurons > self.limits.MAX_NEURONS:
                    if self.auto_fix:
                        sanitized_params[param_name] = self.limits.MAX_NEURONS
                        warnings_list.append(f"Clipped {param_name} to {self.limits.MAX_NEURONS}")
                    else:
                        errors.append(f"{param_name} {n_neurons} exceeds limit of {self.limits.MAX_NEURONS}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_params, warnings_list, errors, metadata)
    
    def sanitize_network_topology(self, topology: Dict[str, Any]) -> ValidationResult:
        """Sanitize network topology parameters."""
        warnings_list = []
        errors = []
        metadata = {}
        sanitized_topology = topology.copy()
        
        # Check layer sizes
        if 'layer_sizes' in topology:
            layer_sizes = topology['layer_sizes']
            
            # Check number of layers
            if len(layer_sizes) > self.limits.MAX_LAYERS:
                if self.auto_fix:
                    sanitized_topology['layer_sizes'] = layer_sizes[:self.limits.MAX_LAYERS]
                    warnings_list.append(f"Truncated network to {self.limits.MAX_LAYERS} layers")
                else:
                    errors.append(f"Network has {len(layer_sizes)} layers, exceeding limit of {self.limits.MAX_LAYERS}")
            
            # Check individual layer sizes
            for i, size in enumerate(layer_sizes):
                if size > self.limits.MAX_NEURONS:
                    if self.auto_fix:
                        sanitized_topology['layer_sizes'][i] = self.limits.MAX_NEURONS
                        warnings_list.append(f"Clipped layer {i} size to {self.limits.MAX_NEURONS}")
                    else:
                        errors.append(f"Layer {i} size {size} exceeds neuron limit")
            
            # Estimate total connections
            total_connections = 0
            for i in range(len(layer_sizes) - 1):
                total_connections += layer_sizes[i] * layer_sizes[i + 1]
            
            metadata['estimated_connections'] = total_connections
            
            if total_connections > self.limits.MAX_CONNECTIONS:
                error_msg = f"Network topology would create {total_connections} connections, exceeding limit"
                if self.strict_mode:
                    errors.append(error_msg)
                else:
                    warnings_list.append(error_msg)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized_topology, warnings_list, errors, metadata)
    
    def sanitize_string_parameter(self, value: str, name: str = "parameter") -> ValidationResult:
        """Sanitize string parameters."""
        warnings_list = []
        errors = []
        metadata = {'original_length': len(value)}
        
        # Check length
        if len(value) > self.limits.MAX_STRING_LENGTH:
            if self.auto_fix:
                value = value[:self.limits.MAX_STRING_LENGTH]
                warnings_list.append(f"Truncated {name} to {self.limits.MAX_STRING_LENGTH} characters")
            else:
                errors.append(f"{name} length {len(value)} exceeds limit of {self.limits.MAX_STRING_LENGTH}")
        
        # Check for allowed characters
        if not self.limits.ALLOWED_STRING_PATTERN.match(value):
            if self.auto_fix:
                # Remove invalid characters
                value = re.sub(r'[^a-zA-Z0-9_\-\.\/\s]', '', value)
                warnings_list.append(f"Removed invalid characters from {name}")
            else:
                errors.append(f"{name} contains invalid characters")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, value, warnings_list, errors, metadata)
    
    def comprehensive_sanitize(self, inputs: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive sanitization of all inputs."""
        all_warnings = []
        all_errors = []
        all_metadata = {}
        sanitized_inputs = {}
        
        for key, value in inputs.items():
            if hasattr(value, 'shape') or isinstance(value, (list, tuple)):
                # Array-like input
                result = self.sanitize_array(value, name=key)
                sanitized_inputs[key] = result.sanitized_input
                all_warnings.extend(result.warnings)
                all_errors.extend(result.errors)
                all_metadata[key] = result.metadata
                
            elif isinstance(value, str):
                # String parameter
                result = self.sanitize_string_parameter(value, name=key)
                sanitized_inputs[key] = result.sanitized_input
                all_warnings.extend(result.warnings)
                all_errors.extend(result.errors)
                all_metadata[key] = result.metadata
                
            elif isinstance(value, dict):
                # Dictionary parameters (e.g., topology, config)
                if 'layer_sizes' in value or 'layers' in value:
                    result = self.sanitize_network_topology(value)
                else:
                    result = self.sanitize_spike_parameters(value)
                sanitized_inputs[key] = result.sanitized_input
                all_warnings.extend(result.warnings)
                all_errors.extend(result.errors)
                all_metadata[key] = result.metadata
                
            else:
                # Pass through other types (numbers, booleans, etc.)
                sanitized_inputs[key] = value
        
        is_valid = len(all_errors) == 0
        return ValidationResult(is_valid, sanitized_inputs, all_warnings, all_errors, all_metadata)
    
    def sanitize_input(self, input_data):
        """Sanitize input data using validation and fixing."""
        # Simple input sanitization for compatibility
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', 'script', 'alert', 'hack']
            sanitized = input_data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized
        elif hasattr(input_data, 'shape'):  # numpy/jax array
            # Basic array sanitization
            import numpy as np
            if np.isnan(input_data).any():
                input_data = np.nan_to_num(input_data)
            return input_data
        else:
            return input_data


def sanitize_input(strict_mode: bool = True, auto_fix: bool = True):
    """Decorator for automatic input sanitization."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            sanitizer = InputSanitizer(strict_mode=strict_mode, auto_fix=auto_fix)
            
            # Prepare inputs for sanitization
            inputs = {}
            
            # Add keyword arguments
            inputs.update(kwargs)
            
            # Add positional arguments with generic names
            for i, arg in enumerate(args[1:], 1):  # Skip self
                inputs[f'arg_{i}'] = arg
            
            # Sanitize inputs
            result = sanitizer.comprehensive_sanitize(inputs)
            
            # Issue warnings
            for warning in result.warnings:
                warnings.warn(f"Input sanitization warning in {func.__name__}: {warning}")
            
            # Raise errors if validation failed
            if not result.is_valid:
                error_msg = f"Input validation failed in {func.__name__}: " + "; ".join(result.errors)
                raise ValueError(error_msg)
            
            # Reconstruct arguments
            new_kwargs = {}
            new_args = list(args)
            
            for key, value in result.sanitized_input.items():
                if key.startswith('arg_'):
                    arg_index = int(key.split('_')[1])
                    if arg_index < len(new_args):
                        new_args[arg_index] = value
                else:
                    new_kwargs[key] = value
            
            return func(*new_args, **new_kwargs)
        
        return wrapper
    return decorator


# Create global sanitizer instance
global_sanitizer = InputSanitizer(strict_mode=False, auto_fix=True)