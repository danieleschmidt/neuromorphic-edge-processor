"""Security configuration for neuromorphic systems."""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class SecurityConfig:
    """Centralized security configuration management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize security configuration.
        
        Args:
            config_path: Path to custom config file (optional)
        """
        self.config_path = config_path
        self._load_default_config()
        
        if config_path and config_path.exists():
            self._load_custom_config(config_path)
    
    def _load_default_config(self):
        """Load default security settings."""
        # Input validation limits
        self.max_input_size_mb = float(os.getenv('MAX_INPUT_SIZE_MB', '100'))
        self.max_neurons = int(os.getenv('MAX_NEURONS', '100000'))
        self.max_time_steps = int(os.getenv('MAX_TIME_STEPS', '10000'))
        self.max_model_size_mb = float(os.getenv('MAX_MODEL_SIZE_MB', '1000'))
        
        # File system security
        self.allowed_file_extensions = ['.h5', '.npz', '.json', '.yaml', '.yml', '.pt', '.pth']
        self.max_path_length = 1000
        self.allow_path_traversal = False
        
        # Content filtering
        self.enable_content_filtering = os.getenv('ENABLE_CONTENT_FILTERING', 'true').lower() == 'true'
        self.scan_for_malicious_patterns = True
        self.log_security_violations = True
        
        # Rate limiting (requests per minute)
        self.rate_limit_inference = int(os.getenv('RATE_LIMIT_INFERENCE', '1000'))
        self.rate_limit_training = int(os.getenv('RATE_LIMIT_TRAINING', '100'))
        
        # Resource limits
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '8192'))
        self.max_cpu_percent = int(os.getenv('MAX_CPU_PERCENT', '80'))
        self.max_execution_time = int(os.getenv('MAX_EXECUTION_TIME', '300'))  # seconds
        
        # Logging and monitoring
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.enable_audit_logging = True
        self.security_log_retention_days = 30
        
        # Network security
        self.allowed_hosts = ['localhost', '127.0.0.1', '::1']
        self.max_request_size_mb = 50
        self.require_https = os.getenv('REQUIRE_HTTPS', 'false').lower() == 'true'
        
        # Authentication and authorization  
        self.require_authentication = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
        self.session_timeout_minutes = 30
        self.max_login_attempts = 5
        
        # Encryption
        self.encrypt_models = os.getenv('ENCRYPT_MODELS', 'false').lower() == 'true'
        self.encryption_key_length = 256
        
        # Dangerous imports to monitor/block
        self.dangerous_imports = {
            'subprocess': 'high',
            'os': 'medium', 
            'sys': 'low',
            'pickle': 'high',
            'marshal': 'high',
            'shelve': 'medium',
            'socket': 'medium',
            'urllib': 'medium',
            'requests': 'low',
            'eval': 'critical',
            'exec': 'critical'
        }
        
        # Malicious patterns
        self.malicious_patterns = [
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess',
            r'os\.system',
            r'rm\s+-rf',
            r'DROP\s+TABLE',
            r'<script',
            r'javascript:',
            r'file://',
            r'ftp://',
            r'\.\./',
            r'\.\.\\',
            r'/etc/passwd',
            r'/etc/shadow',
            r'C:\\Windows\\System32'
        ]
    
    def _load_custom_config(self, config_path: Path):
        """Load custom configuration from file."""
        try:
            import json
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Update configuration with custom values
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            print(f"Warning: Failed to load custom config from {config_path}: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return warnings."""
        warnings = []
        
        # Check for insecure settings
        if not self.enable_content_filtering:
            warnings.append("Content filtering is disabled - security risk")
        
        if not self.scan_for_malicious_patterns:
            warnings.append("Malicious pattern scanning is disabled")
        
        if not self.log_security_violations:
            warnings.append("Security violation logging is disabled")
        
        if self.max_input_size_mb > 1000:
            warnings.append(f"Very large input size limit: {self.max_input_size_mb}MB")
        
        if self.max_neurons > 1000000:
            warnings.append(f"Very large neuron limit: {self.max_neurons}")
        
        if not self.require_https and self.require_authentication:
            warnings.append("Authentication over HTTP is insecure")
        
        if self.max_login_attempts > 10:
            warnings.append(f"High login attempt limit: {self.max_login_attempts}")
        
        # Check environment variables for secrets
        dangerous_env_vars = ['PASSWORD', 'SECRET', 'API_KEY', 'TOKEN']
        for env_var in os.environ:
            if any(dangerous in env_var.upper() for dangerous in dangerous_env_vars):
                warnings.append(f"Potential secret in environment variable: {env_var}")
        
        return warnings
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security configuration."""
        return {
            'input_validation': {
                'max_input_size_mb': self.max_input_size_mb,
                'max_neurons': self.max_neurons,
                'max_time_steps': self.max_time_steps,
                'content_filtering_enabled': self.enable_content_filtering
            },
            'file_security': {
                'allowed_extensions': self.allowed_file_extensions,
                'path_traversal_blocked': not self.allow_path_traversal,
                'max_path_length': self.max_path_length
            },
            'resource_limits': {
                'max_memory_mb': self.max_memory_mb,
                'max_cpu_percent': self.max_cpu_percent,
                'max_execution_time': self.max_execution_time
            },
            'logging': {
                'security_logging_enabled': self.log_security_violations,
                'audit_logging_enabled': self.enable_audit_logging,
                'log_level': self.log_level
            },
            'authentication': {
                'auth_required': self.require_authentication,
                'https_required': self.require_https,
                'session_timeout_minutes': self.session_timeout_minutes
            }
        }
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        self._load_default_config()  # Re-read environment variables
    
    def is_import_allowed(self, module_name: str) -> bool:
        """Check if import is allowed based on security policy."""
        if module_name in self.dangerous_imports:
            risk_level = self.dangerous_imports[module_name]
            # Block critical and high-risk imports in production
            if risk_level in ['critical', 'high'] and os.getenv('ENVIRONMENT', 'development') == 'production':
                return False
        return True
    
    def get_allowed_operations(self) -> List[str]:
        """Get list of allowed operations based on security policy."""
        operations = ['inference', 'model_loading', 'data_processing']
        
        if os.getenv('ENVIRONMENT', 'development') == 'development':
            operations.extend(['training', 'model_saving', 'debugging'])
        
        if self.require_authentication:
            operations = [f'authenticated_{op}' for op in operations]
        
        return operations
    
    def save_config(self, output_path: Path):
        """Save current configuration to file."""
        config_dict = {}
        
        # Get all configuration attributes
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        
        # Save to JSON file
        import json
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)


# Global security configuration instance
security_config = SecurityConfig()