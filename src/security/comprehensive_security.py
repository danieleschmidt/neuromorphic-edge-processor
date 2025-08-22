"""
Comprehensive Security Framework for Neuromorphic Systems

Multi-layered security implementation including:
- Input validation and sanitization
- Resource monitoring and protection
- Anomaly detection
- Secure execution contexts
- Audit logging and compliance
"""

import os
import sys
import time
import hashlib
import hmac
import logging
import threading
import json
import traceback
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import warnings


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    TOP_SECRET = 4


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source: str
    description: str
    details: Dict[str, Any]
    mitigated: bool = False
    mitigation_action: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    max_execution_time: float = 300.0  # 5 minutes
    allowed_file_extensions: List[str] = None
    blocked_functions: List[str] = None
    max_input_size: int = 1024 * 1024  # 1MB
    require_authentication: bool = True
    enable_audit_logging: bool = True
    rate_limit_requests: int = 100  # per minute
    sandbox_execution: bool = True
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.py', '.json', '.txt', '.md', '.yml', '.yaml']
        if self.blocked_functions is None:
            self.blocked_functions = ['eval', 'exec', '__import__', 'compile']


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns to detect
        self.malicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True',
            r'\.\./\.\.',
            r'/etc/passwd',
            r'DROP\s+TABLE',
            r'SELECT.*FROM.*WHERE',
            r'<script',
            r'javascript:',
            r'vbscript:',
        ]
    
    def validate_input(self, data: Any, input_type: str = "general") -> Tuple[bool, str]:
        """
        Validate input data for security issues.
        
        Args:
            data: Input data to validate
            input_type: Type of input (string, file, json, etc.)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Size validation
            if hasattr(data, '__len__'):
                if len(str(data)) > self.policy.max_input_size:
                    return False, f"Input size {len(str(data))} exceeds maximum {self.policy.max_input_size}"
            
            # Type-specific validation
            if input_type == "string":
                return self._validate_string(data)
            elif input_type == "file_path":
                return self._validate_file_path(data)
            elif input_type == "json":
                return self._validate_json(data)
            elif input_type == "numeric":
                return self._validate_numeric(data)
            elif input_type == "list":
                return self._validate_list(data)
            else:
                return self._validate_general(data)
                
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, f"Validation failed: {str(e)}"
    
    def _validate_string(self, data: str) -> Tuple[bool, str]:
        """Validate string input."""
        if not isinstance(data, str):
            return False, "Expected string input"
        
        # Check for malicious patterns
        import re
        for pattern in self.malicious_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return False, f"Malicious pattern detected: {pattern}"
        
        # Check encoding
        try:
            data.encode('utf-8')
        except UnicodeEncodeError:
            return False, "Invalid character encoding"
        
        return True, ""
    
    def _validate_file_path(self, path: str) -> Tuple[bool, str]:
        """Validate file path."""
        if not isinstance(path, str):
            return False, "Path must be string"
        
        # Normalize path
        normalized_path = os.path.normpath(path)
        
        # Check for path traversal
        if '..' in normalized_path:
            return False, "Path traversal detected"
        
        # Check file extension
        _, ext = os.path.splitext(normalized_path)
        if ext and ext.lower() not in self.policy.allowed_file_extensions:
            return False, f"File extension {ext} not allowed"
        
        # Check for suspicious paths
        suspicious_paths = ['/etc/', '/proc/', '/sys/', 'C:\\Windows\\', 'C:\\Program Files\\']
        for suspicious in suspicious_paths:
            if suspicious in normalized_path:
                return False, f"Access to {suspicious} not allowed"
        
        return True, ""
    
    def _validate_json(self, data: Union[str, dict]) -> Tuple[bool, str]:
        """Validate JSON input."""
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
                
            # Check JSON structure depth
            if self._get_dict_depth(parsed) > 10:
                return False, "JSON structure too deep"
            
            # Check for suspicious keys
            suspicious_keys = ['__proto__', 'constructor', 'prototype']
            if self._contains_suspicious_keys(parsed, suspicious_keys):
                return False, "Suspicious JSON keys detected"
            
            return True, ""
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    def _validate_numeric(self, data: Union[int, float]) -> Tuple[bool, str]:
        """Validate numeric input."""
        if not isinstance(data, (int, float)):
            return False, "Expected numeric input"
        
        # Check for reasonable ranges
        if abs(data) > 1e12:
            return False, "Number too large"
        
        # Check for NaN and infinity
        if isinstance(data, float):
            if data != data:  # NaN check
                return False, "NaN not allowed"
            if abs(data) == float('inf'):
                return False, "Infinity not allowed"
        
        return True, ""
    
    def _validate_list(self, data: list) -> Tuple[bool, str]:
        """Validate list input."""
        if not isinstance(data, list):
            return False, "Expected list input"
        
        # Check list size
        if len(data) > 10000:
            return False, "List too large"
        
        # Validate each element
        for i, item in enumerate(data):
            is_valid, message = self._validate_general(item)
            if not is_valid:
                return False, f"Invalid item at index {i}: {message}"
        
        return True, ""
    
    def _validate_general(self, data: Any) -> Tuple[bool, str]:
        """General validation for any data type."""
        # Check for prohibited types
        prohibited_types = [type, type(lambda: None), type(exec)]
        if type(data) in prohibited_types:
            return False, f"Prohibited data type: {type(data)}"
        
        return True, ""
    
    def _get_dict_depth(self, data: Any, current_depth: int = 0) -> int:
        """Get maximum depth of nested dictionary."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_dict_depth(value, current_depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_dict_depth(item, current_depth) for item in data)
        else:
            return current_depth
    
    def _contains_suspicious_keys(self, data: Any, suspicious_keys: List[str]) -> bool:
        """Check if data contains suspicious keys."""
        if isinstance(data, dict):
            for key in data.keys():
                if str(key) in suspicious_keys:
                    return True
                if self._contains_suspicious_keys(data[key], suspicious_keys):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_suspicious_keys(item, suspicious_keys):
                    return True
        return False


class ResourceMonitor:
    """Monitor and protect system resources."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # Resource tracking
        self.start_time = time.time()
        self.peak_memory = 0
        self.operation_count = 0
        self.resource_violations = []
        
        # Rate limiting
        self.request_timestamps = []
        self.rate_limit_window = 60  # 1 minute
        
        # Threading
        self.lock = threading.RLock()
    
    def check_memory_usage(self) -> Tuple[bool, str]:
        """Check current memory usage."""
        try:
            # Simple memory check using sys.getsizeof on local variables
            # This is a fallback when psutil is not available
            import gc
            objects = gc.get_objects()
            estimated_memory = sum(sys.getsizeof(obj) for obj in objects[:1000])  # Sample
            
            if estimated_memory > self.policy.max_memory_usage:
                return False, f"Estimated memory usage {estimated_memory} exceeds limit {self.policy.max_memory_usage}"
            
            if estimated_memory > self.peak_memory:
                self.peak_memory = estimated_memory
            
            return True, ""
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {str(e)}")
            return True, ""  # Allow operation if check fails
    
    def check_execution_time(self) -> Tuple[bool, str]:
        """Check if execution time is within limits."""
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > self.policy.max_execution_time:
            return False, f"Execution time {elapsed_time:.2f}s exceeds limit {self.policy.max_execution_time}s"
        
        return True, ""
    
    def check_rate_limit(self) -> Tuple[bool, str]:
        """Check if rate limit is exceeded."""
        with self.lock:
            current_time = time.time()
            
            # Remove old requests outside the window
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if current_time - ts < self.rate_limit_window
            ]
            
            # Check if rate limit exceeded
            if len(self.request_timestamps) >= self.policy.rate_limit_requests:
                return False, f"Rate limit exceeded: {len(self.request_timestamps)} requests in {self.rate_limit_window}s"
            
            # Add current request
            self.request_timestamps.append(current_time)
            
            return True, ""
    
    def increment_operation_count(self):
        """Increment operation counter."""
        with self.lock:
            self.operation_count += 1
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        with self.lock:
            return {
                'peak_memory': self.peak_memory,
                'operation_count': self.operation_count,
                'execution_time': time.time() - self.start_time,
                'recent_requests': len(self.request_timestamps),
                'resource_violations': len(self.resource_violations)
            }


class AnomalyDetector:
    """Detect anomalous behavior patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Baseline patterns
        self.normal_patterns = {
            'request_frequency': [],
            'operation_types': {},
            'error_rates': [],
            'resource_usage': []
        }
        
        # Anomaly thresholds
        self.thresholds = {
            'request_frequency_std': 3.0,
            'error_rate_threshold': 0.1,
            'resource_spike_factor': 2.0
        }
        
        self.lock = threading.RLock()
    
    def update_baseline(self, metric_type: str, value: float):
        """Update baseline patterns."""
        with self.lock:
            if metric_type in self.normal_patterns:
                self.normal_patterns[metric_type].append(value)
                
                # Keep only recent data
                if len(self.normal_patterns[metric_type]) > 1000:
                    self.normal_patterns[metric_type].pop(0)
    
    def detect_anomaly(self, metric_type: str, current_value: float) -> Tuple[bool, str]:
        """Detect if current value is anomalous."""
        with self.lock:
            if metric_type not in self.normal_patterns:
                return False, ""
            
            baseline_values = self.normal_patterns[metric_type]
            
            if len(baseline_values) < 10:
                # Not enough data for detection
                return False, "Insufficient baseline data"
            
            # Calculate statistics
            mean_value = sum(baseline_values) / len(baseline_values)
            variance = sum((x - mean_value) ** 2 for x in baseline_values) / len(baseline_values)
            std_dev = variance ** 0.5
            
            # Check for anomaly
            if std_dev > 0:
                z_score = abs(current_value - mean_value) / std_dev
                
                if z_score > self.thresholds['request_frequency_std']:
                    return True, f"Anomalous {metric_type}: z-score {z_score:.2f}"
            
            return False, ""


class SecurityManager:
    """Main security management system."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.logger = self._setup_security_logger()
        
        # Security components
        self.input_validator = InputValidator(self.policy)
        self.resource_monitor = ResourceMonitor(self.policy)
        self.anomaly_detector = AnomalyDetector()
        
        # Security events
        self.security_events: List[SecurityEvent] = []
        self.blocked_operations = 0
        self.security_violations = 0
        
        # Authentication (simplified)
        self.authenticated_sessions = set()
        self.session_timeout = 3600  # 1 hour
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("Security manager initialized with comprehensive protection")
    
    def _setup_security_logger(self) -> logging.Logger:
        """Setup secure audit logging."""
        logger = logging.getLogger('security_audit')
        
        if not logger.handlers:
            # Create secure log handler
            handler = logging.FileHandler('security_audit.log', mode='a')
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def validate_and_execute(
        self,
        operation: Callable,
        inputs: Any = None,
        operation_type: str = "general",
        security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> Tuple[bool, Any, str]:
        """
        Validate inputs and execute operation securely.
        
        Args:
            operation: Function to execute
            inputs: Input data to validate
            operation_type: Type of operation for context
            security_level: Required security clearance
            
        Returns:
            Tuple of (success, result, error_message)
        """
        start_time = time.time()
        
        try:
            # Check authentication if required
            if self.policy.require_authentication:
                if not self._check_authentication():
                    self._log_security_event(
                        "authentication_failure",
                        ThreatLevel.HIGH,
                        "Unauthenticated access attempt"
                    )
                    return False, None, "Authentication required"
            
            # Rate limiting check
            rate_ok, rate_msg = self.resource_monitor.check_rate_limit()
            if not rate_ok:
                self._log_security_event(
                    "rate_limit_exceeded",
                    ThreatLevel.MEDIUM,
                    rate_msg
                )
                return False, None, rate_msg
            
            # Input validation
            if inputs is not None:
                input_valid, input_msg = self.input_validator.validate_input(inputs, operation_type)
                if not input_valid:
                    self._log_security_event(
                        "input_validation_failure",
                        ThreatLevel.MEDIUM,
                        f"Invalid input: {input_msg}"
                    )
                    return False, None, f"Input validation failed: {input_msg}"
            
            # Resource checks
            memory_ok, memory_msg = self.resource_monitor.check_memory_usage()
            if not memory_ok:
                self._log_security_event(
                    "memory_limit_exceeded",
                    ThreatLevel.HIGH,
                    memory_msg
                )
                return False, None, memory_msg
            
            time_ok, time_msg = self.resource_monitor.check_execution_time()
            if not time_ok:
                self._log_security_event(
                    "execution_time_exceeded",
                    ThreatLevel.HIGH,
                    time_msg
                )
                return False, None, time_msg
            
            # Execute operation with monitoring
            self.resource_monitor.increment_operation_count()
            
            if self.policy.sandbox_execution:
                result = self._execute_sandboxed(operation, inputs)
            else:
                result = operation(inputs) if inputs is not None else operation()
            
            # Update anomaly detection
            execution_time = time.time() - start_time
            self.anomaly_detector.update_baseline('execution_time', execution_time)
            
            # Check for anomalies
            anomaly_detected, anomaly_msg = self.anomaly_detector.detect_anomaly('execution_time', execution_time)
            if anomaly_detected:
                self._log_security_event(
                    "execution_anomaly",
                    ThreatLevel.MEDIUM,
                    anomaly_msg
                )
            
            return True, result, ""
            
        except Exception as e:
            error_msg = f"Secure execution failed: {str(e)}"
            self._log_security_event(
                "execution_error",
                ThreatLevel.MEDIUM,
                error_msg,
                {"exception": str(e), "traceback": traceback.format_exc()}
            )
            return False, None, error_msg
    
    def _execute_sandboxed(self, operation: Callable, inputs: Any) -> Any:
        """Execute operation in sandboxed environment."""
        # Simple sandboxing - limit built-ins access
        original_builtins = {}
        restricted_builtins = ['eval', 'exec', '__import__', 'open', 'compile']
        
        try:
            # Temporarily remove dangerous built-ins
            for builtin_name in restricted_builtins:
                if hasattr(__builtins__, builtin_name):
                    original_builtins[builtin_name] = getattr(__builtins__, builtin_name)
                    delattr(__builtins__, builtin_name)
            
            # Execute operation
            if inputs is not None:
                result = operation(inputs)
            else:
                result = operation()
            
            return result
            
        finally:
            # Restore built-ins
            for builtin_name, builtin_func in original_builtins.items():
                setattr(__builtins__, builtin_name, builtin_func)
    
    def _check_authentication(self) -> bool:
        """Check if current session is authenticated."""
        # Simplified authentication check
        # In production, this would check actual session tokens
        session_id = getattr(threading.current_thread(), 'session_id', None)
        return session_id in self.authenticated_sessions
    
    def authenticate_session(self, credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Authenticate a session (simplified implementation)."""
        # In production, this would check against a secure credential store
        session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()
        
        # Simple credential check (replace with real authentication)
        if credentials.get('username') and credentials.get('password'):
            self.authenticated_sessions.add(session_id)
            threading.current_thread().session_id = session_id
            
            self._log_security_event(
                "authentication_success",
                ThreatLevel.LOW,
                f"Session {session_id[:8]} authenticated"
            )
            
            return True, session_id
        else:
            self._log_security_event(
                "authentication_failure",
                ThreatLevel.MEDIUM,
                "Invalid credentials provided"
            )
            return False, "Invalid credentials"
    
    def _log_security_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        event = SecurityEvent(
            event_id=hashlib.sha256(f"{time.time()}{event_type}".encode()).hexdigest()[:16],
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source=f"thread_{threading.current_thread().ident}",
            description=description,
            details=details or {}
        )
        
        with self.lock:
            self.security_events.append(event)
            
            # Keep only recent events
            if len(self.security_events) > 10000:
                self.security_events.pop(0)
            
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.security_violations += 1
        
        # Log to audit log
        self.logger.warning(
            f"[{threat_level.value.upper()}] {event_type}: {description}",
            extra={"event_id": event.event_id, "details": details}
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self.lock:
            recent_events = [
                asdict(event) for event in self.security_events[-50:]
            ]
            
            threat_counts = {}
            for event in self.security_events[-100:]:
                threat_counts[event.threat_level.value] = threat_counts.get(event.threat_level.value, 0) + 1
            
            return {
                'total_events': len(self.security_events),
                'security_violations': self.security_violations,
                'blocked_operations': self.blocked_operations,
                'recent_events': recent_events,
                'threat_distribution': threat_counts,
                'resource_stats': self.resource_monitor.get_resource_stats(),
                'authenticated_sessions': len(self.authenticated_sessions),
                'policy_summary': asdict(self.policy)
            }
    
    def export_security_report(self, filepath: str):
        """Export comprehensive security report."""
        report = {
            'report_timestamp': time.time(),
            'security_status': self.get_security_status(),
            'recommendations': self._generate_security_recommendations()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Security report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export security report: {str(e)}")
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current status."""
        recommendations = []
        
        status = self.get_security_status()
        
        if status['security_violations'] > 10:
            recommendations.append("High number of security violations detected - review access controls")
        
        if status['threat_distribution'].get('high', 0) > 5:
            recommendations.append("Multiple high-threat events - investigate potential attacks")
        
        if status['resource_stats']['peak_memory'] > self.policy.max_memory_usage * 0.8:
            recommendations.append("Memory usage approaching limits - optimize resource usage")
        
        recommendations.extend([
            "Regularly review security logs and events",
            "Update security policies based on threat landscape",
            "Implement multi-factor authentication for sensitive operations",
            "Monitor for unusual access patterns and behaviors",
            "Keep security components updated and patched"
        ])
        
        return recommendations


# Context manager for secure operations
@contextmanager
def secure_context(security_manager: SecurityManager, operation_type: str = "general"):
    """Context manager for secure operation execution."""
    try:
        # Pre-execution security checks
        memory_ok, memory_msg = security_manager.resource_monitor.check_memory_usage()
        if not memory_ok:
            raise SecurityError(f"Security check failed: {memory_msg}")
        
        yield security_manager
        
    except Exception as e:
        security_manager._log_security_event(
            "secure_context_error",
            ThreatLevel.MEDIUM,
            f"Error in secure context: {str(e)}"
        )
        raise
    
    finally:
        # Post-execution cleanup
        pass


class SecurityError(Exception):
    """Security-related exception."""
    pass


def demo_security_system():
    """Demonstrate the comprehensive security system."""
    print("Neuromorphic Security System Demo")
    print("=" * 40)
    
    # Create security manager
    policy = SecurityPolicy(
        max_memory_usage=10 * 1024 * 1024,  # 10MB for demo
        max_execution_time=10.0,
        rate_limit_requests=5
    )
    
    security_manager = SecurityManager(policy)
    
    # Test authentication
    print("\n1. Testing Authentication:")
    success, session_id = security_manager.authenticate_session({
        'username': 'demo_user',
        'password': 'demo_pass'
    })
    print(f"Authentication: {'Success' if success else 'Failed'}")
    
    # Test secure operation
    print("\n2. Testing Secure Operation:")
    def safe_operation(data):
        return sum(data) if isinstance(data, list) else 0
    
    success, result, message = security_manager.validate_and_execute(
        safe_operation,
        inputs=[1, 2, 3, 4, 5],
        operation_type="list"
    )
    
    print(f"Operation result: {result if success else f'Failed: {message}'}")
    
    # Test malicious input detection
    print("\n3. Testing Malicious Input Detection:")
    success, result, message = security_manager.validate_and_execute(
        safe_operation,
        inputs="eval(print('hello'))",
        operation_type="string"
    )
    
    print(f"Malicious input: {'Blocked' if not success else 'Not detected'}")
    print(f"Message: {message}")
    
    # Display security status
    print("\n4. Security Status:")
    status = security_manager.get_security_status()
    print(f"Total events: {status['total_events']}")
    print(f"Security violations: {status['security_violations']}")
    print(f"Threat distribution: {status['threat_distribution']}")
    
    print("\nSecurity demo completed successfully!")


if __name__ == "__main__":
    demo_security_system()