"""
Runtime Validation and Self-Testing Framework

Comprehensive validation system that works without external dependencies,
providing runtime checks, self-testing, and robust error recovery.
"""

import sys
import os
import time
import traceback
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import json


class ValidationLevel(Enum):
    """Validation severity levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class ValidationReport:
    """Detailed validation report."""
    test_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    duration: float
    timestamp: float
    suggestions: List[str]


class RuntimeValidator:
    """
    Comprehensive runtime validation and self-testing framework.
    
    Features:
    - Dependency-free validation
    - Runtime health checks
    - Performance monitoring
    - Error detection and recovery
    - Self-healing capabilities
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = self._setup_logger()
        
        # Validation results storage
        self.validation_reports: List[ValidationReport] = []
        self.validation_history: Dict[str, List[ValidationReport]] = {}
        
        # Runtime monitoring
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_validation_time = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Self-test registry
        self.registered_tests: Dict[str, Callable] = {}
        self._register_core_tests()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - VALIDATOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_core_tests(self):
        """Register core validation tests."""
        self.register_test("system_health", self._test_system_health)
        self.register_test("memory_usage", self._test_memory_usage)
        self.register_test("python_environment", self._test_python_environment)
        self.register_test("file_system", self._test_file_system)
        self.register_test("import_validation", self._test_import_validation)
        self.register_test("numerical_stability", self._test_numerical_stability)
        self.register_test("error_handling", self._test_error_handling)
        self.register_test("performance_baseline", self._test_performance_baseline)
    
    def register_test(self, name: str, test_func: Callable) -> None:
        """Register a custom validation test."""
        with self.lock:
            self.registered_tests[name] = test_func
            self.logger.info(f"Registered validation test: {name}")
    
    def validate_all(self, level: Optional[ValidationLevel] = None) -> Dict[str, Any]:
        """Run all registered validation tests."""
        level = level or self.validation_level
        start_time = time.time()
        
        self.logger.info(f"Starting comprehensive validation at level: {level.value}")
        
        results = {}
        total_tests = len(self.registered_tests)
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        errors = 0
        
        # Run each test
        for test_name, test_func in self.registered_tests.items():
            try:
                # Skip tests based on validation level
                if level == ValidationLevel.BASIC and test_name in ["exhaustive_memory", "stress_test"]:
                    continue
                
                test_start = time.time()
                report = self._run_single_test(test_name, test_func)
                test_duration = time.time() - test_start
                
                report.duration = test_duration
                results[test_name] = report
                
                # Update counters
                if report.result == ValidationResult.PASS:
                    passed_tests += 1
                elif report.result == ValidationResult.FAIL:
                    failed_tests += 1
                elif report.result == ValidationResult.WARN:
                    warnings += 1
                elif report.result == ValidationResult.ERROR:
                    errors += 1
                
                # Store in history
                with self.lock:
                    if test_name not in self.validation_history:
                        self.validation_history[test_name] = []
                    self.validation_history[test_name].append(report)
                    
                    # Keep only last 100 results per test
                    if len(self.validation_history[test_name]) > 100:
                        self.validation_history[test_name].pop(0)
                
            except Exception as e:
                error_report = ValidationReport(
                    test_name=test_name,
                    result=ValidationResult.ERROR,
                    message=f"Test execution failed: {str(e)}",
                    details={"exception": str(e), "traceback": traceback.format_exc()},
                    duration=0.0,
                    timestamp=time.time(),
                    suggestions=["Check test implementation", "Review error logs"]
                )
                results[test_name] = error_report
                errors += 1
        
        total_duration = time.time() - start_time
        self.last_validation_time = time.time()
        
        # Compute overall status
        overall_status = "PASS"
        if errors > 0:
            overall_status = "ERROR"
        elif failed_tests > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "WARN"
        
        summary = {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warnings,
            "errors": errors,
            "duration": total_duration,
            "validation_level": level.value,
            "timestamp": time.time(),
            "results": results
        }
        
        self.logger.info(
            f"Validation completed: {overall_status} "
            f"({passed_tests}/{total_tests} passed, "
            f"{warnings} warnings, {failed_tests} failed, {errors} errors) "
            f"in {total_duration:.2f}s"
        )
        
        return summary
    
    def _run_single_test(self, test_name: str, test_func: Callable) -> ValidationReport:
        """Run a single validation test."""
        start_time = time.time()
        
        try:
            result, message, details, suggestions = test_func()
            
            report = ValidationReport(
                test_name=test_name,
                result=result,
                message=message,
                details=details or {},
                duration=time.time() - start_time,
                timestamp=time.time(),
                suggestions=suggestions or []
            )
            
            return report
            
        except Exception as e:
            return ValidationReport(
                test_name=test_name,
                result=ValidationResult.ERROR,
                message=f"Test execution error: {str(e)}",
                details={"exception": str(e), "traceback": traceback.format_exc()},
                duration=time.time() - start_time,
                timestamp=time.time(),
                suggestions=["Check test implementation", "Review error logs"]
            )
    
    def _test_system_health(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test overall system health."""
        try:
            details = {}
            suggestions = []
            
            # Check Python version
            python_version = sys.version_info
            details["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version < (3, 8):
                return (
                    ValidationResult.WARN,
                    f"Python version {details['python_version']} is below recommended 3.8+",
                    details,
                    ["Upgrade to Python 3.8 or higher for better performance"]
                )
            
            # Check available memory
            try:
                import psutil
                memory = psutil.virtual_memory()
                details["memory_total"] = memory.total
                details["memory_available"] = memory.available
                details["memory_percent"] = memory.percent
                
                if memory.percent > 90:
                    suggestions.append("High memory usage detected - consider reducing batch sizes")
            except ImportError:
                details["memory_check"] = "psutil not available"
            
            # Check disk space
            try:
                import shutil
                disk_usage = shutil.disk_usage("/")
                details["disk_total"] = disk_usage.total
                details["disk_free"] = disk_usage.free
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                details["disk_percent"] = disk_percent
                
                if disk_percent > 90:
                    suggestions.append("Low disk space - consider cleanup")
            except Exception:
                details["disk_check"] = "disk usage check failed"
            
            return (
                ValidationResult.PASS,
                "System health check passed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"System health check failed: {str(e)}",
                {"error": str(e)},
                ["Check system configuration"]
            )
    
    def _test_memory_usage(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test memory usage patterns."""
        try:
            details = {}
            suggestions = []
            
            # Test memory allocation patterns
            test_arrays = []
            allocation_sizes = [1000, 10000, 100000]
            
            for size in allocation_sizes:
                try:
                    # Simple list allocation test
                    test_array = list(range(size))
                    test_arrays.append(test_array)
                    details[f"allocation_{size}"] = "success"
                except MemoryError:
                    details[f"allocation_{size}"] = "failed"
                    suggestions.append(f"Memory allocation failed at size {size}")
            
            # Test garbage collection
            del test_arrays
            
            # Check memory growth patterns
            import gc
            gc.collect()
            details["gc_objects"] = len(gc.get_objects())
            
            return (
                ValidationResult.PASS,
                "Memory usage test completed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Memory test failed: {str(e)}",
                {"error": str(e)},
                ["Check memory configuration"]
            )
    
    def _test_python_environment(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test Python environment configuration."""
        try:
            details = {}
            suggestions = []
            
            # Check Python path
            details["python_path"] = sys.path
            details["python_executable"] = sys.executable
            
            # Check standard library modules
            standard_modules = ["os", "sys", "json", "logging", "threading", "time"]
            missing_modules = []
            
            for module in standard_modules:
                try:
                    __import__(module)
                    details[f"module_{module}"] = "available"
                except ImportError:
                    missing_modules.append(module)
                    details[f"module_{module}"] = "missing"
            
            if missing_modules:
                return (
                    ValidationResult.FAIL,
                    f"Missing standard modules: {missing_modules}",
                    details,
                    ["Reinstall Python or check installation"]
                )
            
            # Check optional scientific modules
            optional_modules = ["numpy", "torch", "scipy", "pandas"]
            available_optional = []
            
            for module in optional_modules:
                try:
                    __import__(module)
                    available_optional.append(module)
                    details[f"optional_{module}"] = "available"
                except ImportError:
                    details[f"optional_{module}"] = "missing"
            
            details["available_optional_modules"] = available_optional
            
            if not available_optional:
                suggestions.append("Consider installing scientific computing packages")
            
            return (
                ValidationResult.PASS,
                "Python environment check passed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Python environment test failed: {str(e)}",
                {"error": str(e)},
                ["Check Python installation"]
            )
    
    def _test_file_system(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test file system operations."""
        try:
            details = {}
            suggestions = []
            
            # Check current directory
            current_dir = os.getcwd()
            details["current_directory"] = current_dir
            details["directory_writable"] = os.access(current_dir, os.W_OK)
            details["directory_readable"] = os.access(current_dir, os.R_OK)
            
            # Test file operations
            test_file = os.path.join(current_dir, "temp_validation_test.txt")
            
            try:
                # Write test
                with open(test_file, 'w') as f:
                    f.write("validation test")
                details["file_write"] = "success"
                
                # Read test
                with open(test_file, 'r') as f:
                    content = f.read()
                details["file_read"] = "success" if content == "validation test" else "failed"
                
                # Delete test
                os.remove(test_file)
                details["file_delete"] = "success"
                
            except Exception as e:
                details["file_operations"] = f"failed: {str(e)}"
                suggestions.append("Check file system permissions")
            
            # Check source directory structure
            src_path = os.path.join(current_dir, "src")
            if os.path.exists(src_path):
                details["src_directory"] = "exists"
                
                # Check key directories
                key_dirs = ["models", "algorithms", "utils", "monitoring", "security"]
                for dir_name in key_dirs:
                    dir_path = os.path.join(src_path, dir_name)
                    details[f"dir_{dir_name}"] = "exists" if os.path.exists(dir_path) else "missing"
            else:
                details["src_directory"] = "missing"
                suggestions.append("Source directory structure missing")
            
            return (
                ValidationResult.PASS,
                "File system test completed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"File system test failed: {str(e)}",
                {"error": str(e)},
                ["Check file system configuration"]
            )
    
    def _test_import_validation(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test import capabilities for the neuromorphic package."""
        try:
            details = {}
            suggestions = []
            
            # Test basic imports without dependencies
            basic_modules = [
                "src",
                "src.utils.config",
                "src.utils.logging",
                "src.security.security_config",
                "src.monitoring.health_monitor"
            ]
            
            successful_imports = []
            failed_imports = []
            
            for module in basic_modules:
                try:
                    __import__(module)
                    successful_imports.append(module)
                    details[f"import_{module.replace('.', '_')}"] = "success"
                except ImportError as e:
                    failed_imports.append((module, str(e)))
                    details[f"import_{module.replace('.', '_')}"] = f"failed: {str(e)}"
                except Exception as e:
                    failed_imports.append((module, f"unexpected error: {str(e)}"))
                    details[f"import_{module.replace('.', '_')}"] = f"error: {str(e)}"
            
            details["successful_imports"] = len(successful_imports)
            details["failed_imports"] = len(failed_imports)
            
            if failed_imports:
                suggestions.extend([
                    "Check for missing dependencies",
                    "Verify source code syntax",
                    "Review import paths"
                ])
                
                if len(failed_imports) > len(successful_imports):
                    return (
                        ValidationResult.FAIL,
                        f"Most imports failed ({len(failed_imports)}/{len(basic_modules)})",
                        details,
                        suggestions
                    )
                else:
                    return (
                        ValidationResult.WARN,
                        f"Some imports failed ({len(failed_imports)}/{len(basic_modules)})",
                        details,
                        suggestions
                    )
            
            return (
                ValidationResult.PASS,
                "All basic imports successful",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Import validation failed: {str(e)}",
                {"error": str(e)},
                ["Check package structure"]
            )
    
    def _test_numerical_stability(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test numerical computation stability."""
        try:
            details = {}
            suggestions = []
            
            # Basic arithmetic tests
            test_cases = [
                ("addition", lambda: 1.0 + 2.0, 3.0),
                ("multiplication", lambda: 2.0 * 3.0, 6.0),
                ("division", lambda: 6.0 / 2.0, 3.0),
                ("power", lambda: 2.0 ** 3, 8.0),
                ("sqrt", lambda: 4.0 ** 0.5, 2.0)
            ]
            
            failed_tests = []
            
            for test_name, computation, expected in test_cases:
                try:
                    result = computation()
                    if abs(result - expected) < 1e-10:
                        details[f"test_{test_name}"] = "pass"
                    else:
                        details[f"test_{test_name}"] = f"fail: got {result}, expected {expected}"
                        failed_tests.append(test_name)
                except Exception as e:
                    details[f"test_{test_name}"] = f"error: {str(e)}"
                    failed_tests.append(test_name)
            
            # Test edge cases
            edge_cases = [
                ("large_number", lambda: 1e100 * 1e100, "overflow handling"),
                ("small_number", lambda: 1e-100 / 1e100, "underflow handling"),
                ("division_by_zero", lambda: 1.0 / 0.0, "division by zero"),
                ("invalid_sqrt", lambda: (-1.0) ** 0.5, "invalid operation")
            ]
            
            for test_name, computation, description in edge_cases:
                try:
                    result = computation()
                    details[f"edge_{test_name}"] = f"result: {result}"
                except ZeroDivisionError:
                    details[f"edge_{test_name}"] = "properly caught division by zero"
                except OverflowError:
                    details[f"edge_{test_name}"] = "properly caught overflow"
                except Exception as e:
                    details[f"edge_{test_name}"] = f"caught: {type(e).__name__}"
            
            if failed_tests:
                suggestions.append("Basic numerical operations failing - check system")
                return (
                    ValidationResult.FAIL,
                    f"Numerical stability issues: {failed_tests}",
                    details,
                    suggestions
                )
            
            return (
                ValidationResult.PASS,
                "Numerical stability tests passed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Numerical stability test failed: {str(e)}",
                {"error": str(e)},
                ["Check numerical computing environment"]
            )
    
    def _test_error_handling(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test error handling capabilities."""
        try:
            details = {}
            suggestions = []
            
            # Test exception handling
            exception_tests = [
                ("value_error", ValueError, "test value error"),
                ("type_error", TypeError, "test type error"),
                ("runtime_error", RuntimeError, "test runtime error"),
                ("key_error", KeyError, "test key error")
            ]
            
            handled_exceptions = 0
            
            for test_name, exception_type, message in exception_tests:
                try:
                    raise exception_type(message)
                except exception_type as e:
                    if str(e) == message:
                        details[f"exception_{test_name}"] = "properly handled"
                        handled_exceptions += 1
                    else:
                        details[f"exception_{test_name}"] = f"message mismatch: {str(e)}"
                except Exception as e:
                    details[f"exception_{test_name}"] = f"unexpected: {type(e).__name__}"
            
            details["handled_exceptions"] = handled_exceptions
            details["total_exception_tests"] = len(exception_tests)
            
            # Test try-except-finally
            finally_executed = False
            try:
                try:
                    raise ValueError("test finally")
                except ValueError:
                    pass
                finally:
                    finally_executed = True
                    
                details["finally_block"] = "executed" if finally_executed else "not executed"
                
            except Exception as e:
                details["finally_test"] = f"failed: {str(e)}"
                suggestions.append("Exception handling mechanism issues")
            
            if handled_exceptions < len(exception_tests):
                suggestions.append("Some exception types not properly handled")
            
            return (
                ValidationResult.PASS,
                "Error handling tests completed",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Error handling test failed: {str(e)}",
                {"error": str(e)},
                ["Check exception handling implementation"]
            )
    
    def _test_performance_baseline(self) -> Tuple[ValidationResult, str, Dict[str, Any], List[str]]:
        """Test basic performance baseline."""
        try:
            details = {}
            suggestions = []
            
            # Test computation speed
            iterations = 100000
            
            # Arithmetic operations
            start_time = time.time()
            result = 0
            for i in range(iterations):
                result += i * 2
            arithmetic_time = time.time() - start_time
            details["arithmetic_ops_per_sec"] = iterations / arithmetic_time
            
            # List operations
            start_time = time.time()
            test_list = []
            for i in range(iterations // 100):
                test_list.append(i)
            list_time = time.time() - start_time
            details["list_ops_per_sec"] = (iterations // 100) / list_time
            
            # Dictionary operations
            start_time = time.time()
            test_dict = {}
            for i in range(iterations // 100):
                test_dict[i] = i * 2
            dict_time = time.time() - start_time
            details["dict_ops_per_sec"] = (iterations // 100) / dict_time
            
            # Function call overhead
            def test_function(x):
                return x * 2
            
            start_time = time.time()
            for i in range(iterations // 10):
                test_function(i)
            function_time = time.time() - start_time
            details["function_calls_per_sec"] = (iterations // 10) / function_time
            
            # Performance thresholds (very conservative)
            if details["arithmetic_ops_per_sec"] < 10000:
                suggestions.append("Arithmetic operations slower than expected")
            if details["list_ops_per_sec"] < 1000:
                suggestions.append("List operations slower than expected")
            if details["dict_ops_per_sec"] < 1000:
                suggestions.append("Dictionary operations slower than expected")
            
            return (
                ValidationResult.PASS,
                "Performance baseline established",
                details,
                suggestions
            )
            
        except Exception as e:
            return (
                ValidationResult.ERROR,
                f"Performance baseline test failed: {str(e)}",
                {"error": str(e)},
                ["Check system performance"]
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        with self.lock:
            if not self.validation_history:
                return {"message": "No validation history available"}
            
            summary = {
                "total_validations": sum(len(reports) for reports in self.validation_history.values()),
                "test_coverage": list(self.validation_history.keys()),
                "recent_results": {},
                "trends": {},
                "recommendations": []
            }
            
            # Get most recent results for each test
            for test_name, reports in self.validation_history.items():
                if reports:
                    latest = reports[-1]
                    summary["recent_results"][test_name] = {
                        "result": latest.result.value,
                        "message": latest.message,
                        "timestamp": latest.timestamp
                    }
                    
                    # Analyze trends (pass/fail rate over time)
                    recent_reports = reports[-10:]  # Last 10 results
                    pass_rate = sum(1 for r in recent_reports if r.result == ValidationResult.PASS) / len(recent_reports)
                    summary["trends"][test_name] = {
                        "pass_rate": pass_rate,
                        "trend": "improving" if pass_rate > 0.7 else "degrading" if pass_rate < 0.3 else "stable"
                    }
            
            # Generate recommendations
            failing_tests = [name for name, result in summary["recent_results"].items() 
                           if result["result"] in ["fail", "error"]]
            
            if failing_tests:
                summary["recommendations"].append(f"Address failing tests: {', '.join(failing_tests)}")
            
            degrading_tests = [name for name, trend in summary["trends"].items() 
                             if trend["trend"] == "degrading"]
            
            if degrading_tests:
                summary["recommendations"].append(f"Monitor degrading tests: {', '.join(degrading_tests)}")
            
            summary["recommendations"].extend([
                "Run validation regularly to catch issues early",
                "Monitor system resources during validation",
                "Keep validation history for trend analysis"
            ])
            
            return summary
    
    def export_report(self, filepath: str, format: str = "json") -> None:
        """Export validation report to file."""
        summary = self.get_validation_summary()
        summary["export_timestamp"] = time.time()
        summary["validation_level"] = self.validation_level.value
        
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            else:
                # Simple text format
                with open(filepath, 'w') as f:
                    f.write("NEUROMORPHIC VALIDATION REPORT\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Total Validations: {summary['total_validations']}\n")
                    f.write(f"Test Coverage: {len(summary['test_coverage'])} tests\n\n")
                    
                    f.write("Recent Results:\n")
                    for test_name, result in summary["recent_results"].items():
                        f.write(f"  {test_name}: {result['result'].upper()}\n")
                    
                    f.write("\nRecommendations:\n")
                    for rec in summary["recommendations"]:
                        f.write(f"  - {rec}\n")
            
            self.logger.info(f"Validation report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {str(e)}")


def main():
    """Run comprehensive validation."""
    validator = RuntimeValidator(ValidationLevel.COMPREHENSIVE)
    
    print("Starting Neuromorphic System Validation...")
    print("=" * 50)
    
    results = validator.validate_all()
    
    print(f"\nValidation Status: {results['overall_status']}")
    print(f"Tests: {results['passed']}/{results['total_tests']} passed")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if results['warnings'] > 0:
        print(f"Warnings: {results['warnings']}")
    if results['failed'] > 0:
        print(f"Failures: {results['failed']}")
    if results['errors'] > 0:
        print(f"Errors: {results['errors']}")
    
    # Export detailed report
    try:
        validator.export_report("validation_report.json")
        print("\nDetailed report saved to validation_report.json")
    except Exception as e:
        print(f"Could not save report: {e}")
    
    return 0 if results['overall_status'] == 'PASS' else 1


if __name__ == "__main__":
    exit(main())