"""Comprehensive input validation for neuromorphic systems."""

# import torch  # Optional dependency - will be imported when needed
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import re
import hashlib
import json
import sys
from pathlib import Path
# from ..utils.logging import SecurityLogger  # Optional dependency


class ValidationError(Exception):
    """Exception raised when input validation fails."""
    def __init__(self, message: str, violation_type: str = "validation_failure", details: Dict = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class InputValidator:
    """Comprehensive input validation for neuromorphic computing systems.
    
    Validates inputs against size limits, format requirements, content policies,
    and security constraints to prevent injection attacks and system abuse.
    """
    
    def __init__(
        self,
        max_input_size_mb: float = 100.0,
        max_neurons: int = 100000,
        max_time_steps: int = 10000,
        allowed_file_extensions: List[str] = None,
        enable_content_filtering: bool = True,
        log_violations: bool = True
    ):
        """Initialize input validator.
        
        Args:
            max_input_size_mb: Maximum input size in megabytes
            max_neurons: Maximum number of neurons allowed
            max_time_steps: Maximum time steps allowed  
            allowed_file_extensions: List of allowed file extensions
            enable_content_filtering: Enable content filtering
            log_violations: Log validation violations
        """
        self.max_input_size_mb = max_input_size_mb
        self.max_neurons = max_neurons
        self.max_time_steps = max_time_steps
        self.allowed_extensions = allowed_file_extensions or ['.h5', '.npz', '.json', '.yaml', '.yml']
        self.enable_content_filtering = enable_content_filtering
        self.log_violations = log_violations
        
        # Initialize security logger
        if self.log_violations:
            try:
                from ..utils.logging import SecurityLogger
                self.security_logger = SecurityLogger()
            except ImportError:
                self.security_logger = None
        
        # Malicious patterns to detect
        self.malicious_patterns = [
            r'__import__',  # Python imports
            r'eval\s*\(',   # eval() calls
            r'exec\s*\(',   # exec() calls
            r'subprocess',  # subprocess calls
            r'os\.system',  # OS command execution
            r'rm\s+-rf',    # Dangerous shell commands
            r'DROP\s+TABLE',  # SQL injection
            r'<script',     # XSS attempts
            r'javascript:',  # JavaScript injection
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
    
    def validate_tensor_input(self, tensor, input_name: str = "tensor") -> bool:
        """Validate tensor inputs for neuromorphic processing.
        
        Args:
            tensor: Input tensor to validate
            input_name: Name of input for error reporting
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Import torch when needed
            try:
                import torch
            except ImportError:
                torch = None
            
            # Check tensor is valid
            if torch and hasattr(tensor, 'dtype'):  # PyTorch tensor
                if not isinstance(tensor, torch.Tensor):
                    raise ValidationError(
                        f"{input_name} must be a torch.Tensor, got {type(tensor)}",
                        "invalid_type",
                        {"expected": "torch.Tensor", "actual": str(type(tensor))}
                    )
                
                # Check for NaN or infinite values
                if torch.isnan(tensor).any():
                    raise ValidationError(
                        f"{input_name} contains NaN values",
                        "invalid_values",
                        {"contains_nan": True}
                    )
                
                if torch.isinf(tensor).any():
                    raise ValidationError(
                        f"{input_name} contains infinite values", 
                        "invalid_values",
                        {"contains_inf": True}
                    )
                    
                total_elements = tensor.numel()
                tensor_shape = tensor.shape
                tensor_dim = tensor.dim()
            elif hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):  # NumPy array
                if np.isnan(tensor).any():
                    raise ValidationError(
                        f"{input_name} contains NaN values",
                        "invalid_values",
                        {"contains_nan": True}
                    )
                
                if np.isinf(tensor).any():
                    raise ValidationError(
                        f"{input_name} contains infinite values",
                        "invalid_values", 
                        {"contains_inf": True}
                    )
                    
                total_elements = tensor.size
                tensor_shape = tensor.shape
                tensor_dim = tensor.ndim
            else:
                raise ValidationError(
                    f"{input_name} must be a tensor (PyTorch or NumPy), got {type(tensor)}",
                    "invalid_type",
                    {"expected": "tensor", "actual": str(type(tensor))}
                )
            
            # Check tensor dimensions
            if tensor_dim > 4:  # [batch, neurons, time] or [batch, channels, height, width]
                raise ValidationError(
                    f"{input_name} has too many dimensions: {tensor_dim} (max 4)",
                    "dimension_limit",
                    {"dimensions": tensor_dim, "max_allowed": 4}
                )
            
            # Check size limits
            size_mb = total_elements * 4 / 1024 / 1024  # Assume 4 bytes per float
            
            if size_mb > self.max_input_size_mb:
                raise ValidationError(
                    f"{input_name} size {size_mb:.2f}MB exceeds limit {self.max_input_size_mb}MB",
                    "size_limit",
                    {"size_mb": size_mb, "limit_mb": self.max_input_size_mb}
                )
            
            # Check neuromorphic-specific constraints
            if tensor_dim >= 2:  # Has neuron dimension
                num_neurons = tensor_shape[-2] if tensor_dim >= 2 else tensor_shape[-1]
                if num_neurons > self.max_neurons:
                    raise ValidationError(
                        f"{input_name} has {num_neurons} neurons, exceeds limit {self.max_neurons}",
                        "neuron_limit",
                        {"neurons": num_neurons, "limit": self.max_neurons}
                    )
            
            if tensor_dim >= 3:  # Has time dimension
                time_steps = tensor_shape[-1]
                if time_steps > self.max_time_steps:
                    raise ValidationError(
                        f"{input_name} has {time_steps} time steps, exceeds limit {self.max_time_steps}",
                        "time_limit", 
                        {"time_steps": time_steps, "limit": self.max_time_steps}
                    )
            
            # Check value ranges for spike data (skip for now to avoid tensor operations)
            # if tensor_dtype == torch.bool or (tensor.min() >= 0 and tensor.max() <= 1):
            #     # Likely spike data - should be binary
            #     unique_values = torch.unique(tensor)
            #     if len(unique_values) > 2 or not all(v in [0, 1] for v in unique_values):
            #         if self.enable_content_filtering:
            #             raise ValidationError(
            #                 f"{input_name} appears to be spike data but has non-binary values",
            #                 "invalid_spike_format",
            #                 {"unique_values": unique_values.tolist()[:10]}
            #             )
            
            if self.log_violations and self.security_logger:
                self.security_logger.log_input_validation("tensor", True, {
                    "input_name": input_name,
                    "shape": list(tensor_shape),
                    "size_mb": size_mb
                })
            
            return True
            
        except ValidationError:
            if self.log_violations:
                self.security_logger.log_input_validation("tensor", False, {
                    "input_name": input_name,
                    "error": str(sys.exc_info()[1])
                })
            raise
    
    def validate_config_input(self, config: Dict[str, Any], config_name: str = "config") -> bool:
        """Validate configuration dictionary inputs.
        
        Args:
            config: Configuration dictionary to validate
            config_name: Name of config for error reporting
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not isinstance(config, dict):
                raise ValidationError(
                    f"{config_name} must be a dictionary, got {type(config)}",
                    "invalid_type"
                )
            
            # Check JSON serializable
            try:
                json.dumps(config, default=str)
            except (TypeError, ValueError) as e:
                raise ValidationError(
                    f"{config_name} must be JSON serializable: {e}",
                    "serialization_error"
                )
            
            # Check for malicious content in string values
            if self.enable_content_filtering:
                self._scan_for_malicious_content(config, config_name)
            
            # Validate specific config parameters
            self._validate_config_parameters(config, config_name)
            
            if self.log_violations:
                self.security_logger.log_input_validation("config", True, {
                    "config_name": config_name,
                    "num_keys": len(config)
                })
            
            return True
            
        except ValidationError:
            if self.log_violations:
                self.security_logger.log_input_validation("config", False, {
                    "config_name": config_name,
                    "error": str(sys.exc_info()[1])
                })
            raise
    
    def validate_file_path(self, file_path: Union[str, Path], operation: str = "read") -> bool:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            operation: Operation being performed ('read', 'write')
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            path = Path(file_path)
            
            # Check for path traversal attempts
            if ".." in str(path):
                raise ValidationError(
                    f"Path traversal detected in {file_path}",
                    "path_traversal",
                    {"path": str(file_path)}
                )
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_extensions:
                raise ValidationError(
                    f"File extension {path.suffix} not allowed",
                    "invalid_extension",
                    {"extension": path.suffix, "allowed": self.allowed_extensions}
                )
            
            # Check path length
            if len(str(path)) > 1000:
                raise ValidationError(
                    f"Path too long: {len(str(path))} characters (max 1000)",
                    "path_length",
                    {"length": len(str(path))}
                )
            
            # Check for suspicious file names
            suspicious_names = ['passwd', 'shadow', '.env', 'id_rsa', 'authorized_keys']
            if any(sus in path.name.lower() for sus in suspicious_names):
                raise ValidationError(
                    f"Suspicious file name: {path.name}",
                    "suspicious_filename",
                    {"filename": path.name}
                )
            
            # For write operations, check if directory is writable
            if operation == "write" and path.exists():
                if not path.parent.exists() or not path.parent.is_dir():
                    raise ValidationError(
                        f"No write permission for directory: {path.parent}",
                        "permission_denied",
                        {"directory": str(path.parent)}
                    )
            
            if self.log_violations:
                self.security_logger.log_input_validation("file_path", True, {
                    "path": str(file_path),
                    "operation": operation,
                    "extension": path.suffix
                })
            
            return True
            
        except ValidationError:
            if self.log_violations:
                self.security_logger.log_input_validation("file_path", False, {
                    "path": str(file_path),
                    "operation": operation,
                    "error": str(sys.exc_info()[1])
                })
            raise
    
    def _scan_for_malicious_content(self, obj: Any, path: str = ""):
        """Recursively scan object for malicious content patterns."""
        
        if isinstance(obj, str):
            # Check for malicious patterns
            for pattern in self.compiled_patterns:
                if pattern.search(obj):
                    raise ValidationError(
                        f"Malicious pattern detected in {path}: {pattern.pattern}",
                        "malicious_content",
                        {"pattern": pattern.pattern, "path": path}
                    )
            
            # Check for excessively long strings (possible DoS)
            if len(obj) > 10000:
                raise ValidationError(
                    f"String too long in {path}: {len(obj)} characters",
                    "string_length",
                    {"length": len(obj), "path": path}
                )
        
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._scan_for_malicious_content(key, f"{new_path}[key]")
                self._scan_for_malicious_content(value, new_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._scan_for_malicious_content(item, new_path)
    
    def _validate_config_parameters(self, config: Dict[str, Any], config_name: str):
        """Validate specific configuration parameters."""
        
        # Check for reasonable parameter ranges
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                raise ValidationError(
                    f"Invalid learning_rate in {config_name}: {lr}",
                    "invalid_parameter",
                    {"parameter": "learning_rate", "value": lr}
                )
        
        if "batch_size" in config:
            bs = config["batch_size"] 
            if not isinstance(bs, int) or bs <= 0 or bs > 10000:
                raise ValidationError(
                    f"Invalid batch_size in {config_name}: {bs}",
                    "invalid_parameter",
                    {"parameter": "batch_size", "value": bs}
                )
        
        if "num_epochs" in config:
            epochs = config["num_epochs"]
            if not isinstance(epochs, int) or epochs <= 0 or epochs > 100000:
                raise ValidationError(
                    f"Invalid num_epochs in {config_name}: {epochs}",
                    "invalid_parameter", 
                    {"parameter": "num_epochs", "value": epochs}
                )
        
        # Check for suspicious system-level configurations
        dangerous_keys = ["__builtins__", "__globals__", "__locals__", "sys", "os", "subprocess"]
        for key in dangerous_keys:
            if key in config:
                raise ValidationError(
                    f"Dangerous configuration key in {config_name}: {key}",
                    "dangerous_config",
                    {"key": key}
                )
    
    def validate_model_input(self, model_data: Any, model_name: str = "model") -> bool:
        """Validate model data for loading/saving.
        
        Args:
            model_data: Model data to validate (state dict, weights, etc.)
            model_name: Name of model for error reporting
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(model_data, dict):
                # Validate state dict
                total_params = 0
                for key, value in model_data.items():
                    if not isinstance(key, str):
                        raise ValidationError(
                            f"Non-string key in {model_name}: {type(key)}",
                            "invalid_key_type"
                        )
                    
                    if isinstance(value, torch.Tensor):
                        # Validate tensor
                        self.validate_tensor_input(value, f"{model_name}.{key}")
                        total_params += value.numel()
                    
                    # Check for suspicious keys
                    if any(sus in key.lower() for sus in ['backdoor', 'trojan', 'malicious']):
                        raise ValidationError(
                            f"Suspicious parameter name in {model_name}: {key}",
                            "suspicious_parameter",
                            {"parameter": key}
                        )
                
                # Check total model size
                model_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float
                if model_size_mb > self.max_input_size_mb * 10:  # Allow larger models
                    raise ValidationError(
                        f"Model {model_name} too large: {model_size_mb:.2f}MB",
                        "model_size_limit",
                        {"size_mb": model_size_mb}
                    )
            
            if self.log_violations:
                self.security_logger.log_input_validation("model", True, {
                    "model_name": model_name,
                    "type": str(type(model_data))
                })
            
            return True
            
        except ValidationError:
            if self.log_violations:
                self.security_logger.log_input_validation("model", False, {
                    "model_name": model_name,
                    "error": str(sys.exc_info()[1])
                })
            raise
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation settings and statistics."""
        return {
            "max_input_size_mb": self.max_input_size_mb,
            "max_neurons": self.max_neurons,
            "max_time_steps": self.max_time_steps,
            "allowed_extensions": self.allowed_extensions,
            "content_filtering_enabled": self.enable_content_filtering,
            "logging_enabled": self.log_violations,
            "malicious_patterns": len(self.malicious_patterns)
        }
    
    def update_security_config(self, **kwargs):
        """Update security configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Recompile patterns if updated
        if 'malicious_patterns' in kwargs:
            self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]