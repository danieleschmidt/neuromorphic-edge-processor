"""Secure wrappers for potentially dangerous operations."""

import os
import subprocess
import tempfile
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from .security_config import security_config
from .input_validator import InputValidator, ValidationError


logger = logging.getLogger(__name__)


class SecureFileOperations:
    """Secure file operations with validation and logging."""
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """Initialize secure file operations.
        
        Args:
            validator: Input validator instance
        """
        self.validator = validator or InputValidator()
        self.allowed_paths = ['/tmp', '/var/tmp', str(Path.home())]
        
    def secure_file_read(self, file_path: Union[str, Path], max_size_mb: float = 10.0) -> bytes:
        """Securely read file with validation.
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB
            
        Returns:
            File contents as bytes
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate file path
        self.validator.validate_file_path(file_path, operation="read")
        
        file_path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        if not any(str(file_path).startswith(allowed) for allowed in self.allowed_paths):
            if not str(file_path).startswith(str(Path.cwd())):  # Allow current working directory
                raise ValidationError(
                    f"File path not in allowed directories: {file_path}",
                    "path_not_allowed",
                    {"path": str(file_path), "allowed_paths": self.allowed_paths}
                )
        
        # Check file exists and is actually a file
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}", "file_not_found")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}", "not_a_file")
        
        # Check file size
        size_mb = file_path.stat().st_size / 1024 / 1024
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB)",
                "file_too_large",
                {"size_mb": size_mb, "max_mb": max_size_mb}
            )
        
        # Read file safely
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except IOError as e:
            raise ValidationError(f"Failed to read file: {e}", "read_error")
    
    def secure_file_write(self, file_path: Union[str, Path], content: bytes, 
                         overwrite: bool = False) -> None:
        """Securely write file with validation.
        
        Args:
            file_path: Path to file
            content: Content to write
            overwrite: Whether to overwrite existing files
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate file path
        self.validator.validate_file_path(file_path, operation="write")
        
        file_path = Path(file_path).resolve()
        
        # Check if file already exists
        if file_path.exists() and not overwrite:
            raise ValidationError(
                f"File already exists and overwrite=False: {file_path}",
                "file_exists"
            )
        
        # Validate content size
        content_mb = len(content) / 1024 / 1024
        if content_mb > security_config.max_input_size_mb:
            raise ValidationError(
                f"Content too large: {content_mb:.2f}MB",
                "content_too_large",
                {"size_mb": content_mb, "limit_mb": security_config.max_input_size_mb}
            )
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file safely
        try:
            with open(file_path, 'wb') as f:
                f.write(content)
            logger.info(f"Securely wrote file: {file_path}")
        except IOError as e:
            raise ValidationError(f"Failed to write file: {e}", "write_error")


class SecureSubprocess:
    """Secure subprocess execution with validation."""
    
    def __init__(self):
        """Initialize secure subprocess wrapper."""
        self.allowed_commands = {
            'python3': ['python3'],
            'pip': ['pip', 'pip3'],
            'git': ['git'],
            'ls': ['ls'],
            'find': ['find'],
            'grep': ['grep', 'rg'],
            'cat': ['cat'],
            'head': ['head'],
            'tail': ['tail']
        }
        
    def secure_run(self, cmd: List[str], timeout: int = 30, 
                  capture_output: bool = True) -> subprocess.CompletedProcess:
        """Securely run subprocess command.
        
        Args:
            cmd: Command and arguments
            timeout: Timeout in seconds
            capture_output: Whether to capture output
            
        Returns:
            CompletedProcess result
            
        Raises:
            ValidationError: If command is not allowed
        """
        if not cmd or not isinstance(cmd, list):
            raise ValidationError("Command must be a non-empty list", "invalid_command")
        
        command_name = cmd[0]
        
        # Check if command is allowed
        allowed = False
        for allowed_cmd, variations in self.allowed_commands.items():
            if command_name in variations:
                allowed = True
                break
        
        if not allowed:
            raise ValidationError(
                f"Command not allowed: {command_name}",
                "command_not_allowed",
                {"command": command_name, "allowed": list(self.allowed_commands.keys())}
            )
        
        # Validate arguments for dangerous patterns
        for arg in cmd[1:]:
            if any(dangerous in str(arg) for dangerous in ['rm -rf', 'sudo', '&&', '||', ';', '|']):
                raise ValidationError(
                    f"Dangerous pattern in command arguments: {arg}",
                    "dangerous_argument",
                    {"argument": str(arg)}
                )
        
        # Run command securely
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=capture_output,
                shell=False,  # Never use shell=True
                check=False,
                text=True
            )
            
            logger.info(f"Securely executed command: {' '.join(cmd)}")
            return result
            
        except subprocess.TimeoutExpired:
            raise ValidationError(f"Command timed out after {timeout}s", "command_timeout")
        except Exception as e:
            raise ValidationError(f"Command execution failed: {e}", "execution_failed")


class SecureEnvironment:
    """Secure environment variable handling."""
    
    @staticmethod
    def get_secure_env(var_name: str, default: Optional[str] = None, 
                      allow_secrets: bool = False) -> Optional[str]:
        """Get environment variable with security checks.
        
        Args:
            var_name: Environment variable name
            default: Default value if not found
            allow_secrets: Whether to allow secret-like variables
            
        Returns:
            Environment variable value or default
            
        Raises:
            ValidationError: If variable appears to contain secrets
        """
        if not allow_secrets:
            dangerous_patterns = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL']
            if any(pattern in var_name.upper() for pattern in dangerous_patterns):
                logger.warning(f"Potentially sensitive environment variable: {var_name}")
                
                # In production, block access to secrets
                if os.getenv('ENVIRONMENT', 'development') == 'production':
                    raise ValidationError(
                        f"Access to secret environment variable blocked: {var_name}",
                        "secret_env_blocked"
                    )
        
        return os.getenv(var_name, default)
    
    @staticmethod
    def set_secure_env(var_name: str, value: str, temporary: bool = True) -> None:
        """Set environment variable securely.
        
        Args:
            var_name: Environment variable name
            value: Value to set
            temporary: Whether this is a temporary setting
        """
        if not temporary:
            logger.warning(f"Permanently setting environment variable: {var_name}")
        
        # Validate variable name
        if not var_name.replace('_', '').isalnum():
            raise ValidationError(
                f"Invalid environment variable name: {var_name}",
                "invalid_env_name"
            )
        
        os.environ[var_name] = value
        logger.info(f"Set environment variable: {var_name}")


class SecureImporter:
    """Secure import validation and monitoring."""
    
    def __init__(self):
        """Initialize secure importer."""
        self.dangerous_imports = security_config.dangerous_imports
        self.import_log = []
    
    def validate_import(self, module_name: str) -> bool:
        """Validate if import is allowed.
        
        Args:
            module_name: Module name to import
            
        Returns:
            True if import is allowed
            
        Raises:
            ValidationError: If import is blocked
        """
        # Check against dangerous imports
        if module_name in self.dangerous_imports:
            risk_level = self.dangerous_imports[module_name]
            
            # Log the import attempt
            self.import_log.append({
                'module': module_name,
                'risk_level': risk_level,
                'timestamp': __import__('time').time()
            })
            
            # Block critical imports in production
            if risk_level == 'critical' and os.getenv('ENVIRONMENT', 'development') == 'production':
                raise ValidationError(
                    f"Critical import blocked in production: {module_name}",
                    "import_blocked",
                    {"module": module_name, "risk_level": risk_level}
                )
            
            # Warn about high-risk imports
            if risk_level in ['high', 'critical']:
                logger.warning(f"High-risk import detected: {module_name} (risk: {risk_level})")
        
        return True
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Get import statistics.
        
        Returns:
            Dictionary with import statistics
        """
        if not self.import_log:
            return {"total_imports": 0, "risk_breakdown": {}}
        
        risk_counts = {}
        for entry in self.import_log:
            risk = entry['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        return {
            "total_imports": len(self.import_log),
            "risk_breakdown": risk_counts,
            "recent_imports": self.import_log[-10:]  # Last 10 imports
        }


# Global instances for easy access
secure_files = SecureFileOperations()
secure_subprocess = SecureSubprocess()
secure_env = SecureEnvironment()
secure_importer = SecureImporter()