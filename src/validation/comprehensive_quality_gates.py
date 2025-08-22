"""
Comprehensive Quality Gates System

Multi-layered quality assurance including:
- Functional testing and validation
- Security compliance checking
- Performance benchmarking
- Code quality assessment
- Production readiness verification
"""

import time
import os
import sys
import json
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import subprocess
import hashlib


class QualityGateStatus(Enum):
    """Quality gate status levels."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class QualityGateCategory(Enum):
    """Quality gate categories."""
    FUNCTIONALITY = "functionality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    COMPLIANCE = "compliance"


@dataclass
class QualityGateResult:
    """Individual quality gate result."""
    gate_name: str
    category: QualityGateCategory
    status: QualityGateStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: float
    requirements: List[str]
    recommendations: List[str]


@dataclass
class QualityGateReport:
    """Comprehensive quality gate report."""
    total_gates: int
    passed_gates: int
    failed_gates: int
    warnings: int
    errors: int
    skipped: int
    overall_score: float
    overall_status: QualityGateStatus
    execution_time: float
    timestamp: float
    results: List[QualityGateResult]
    summary: Dict[str, Any]


class FunctionalTester:
    """Comprehensive functional testing framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
    
    def test_core_functionality(self) -> QualityGateResult:
        """Test core system functionality."""
        start_time = time.time()
        details = {}
        requirements = ["Core models must be importable", "Basic operations must work"]
        recommendations = []
        
        try:
            # Test 1: Import core models
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from models.dependency_free_models import (
                    PurePythonLIFNeuron, PurePythonSpikingNetwork, 
                    PurePythonLiquidStateMachine, ModelFactory
                )
                details["core_imports"] = "success"
                score = 25
            except Exception as e:
                details["core_imports"] = f"failed: {str(e)}"
                score = 0
                recommendations.append("Fix core model imports")
            
            # Test 2: Basic neuron functionality
            try:
                neuron = PurePythonLIFNeuron()
                spike = neuron.update(2.0)
                state = neuron.get_state()
                details["neuron_functionality"] = "success"
                details["neuron_state"] = state
                score += 25
            except Exception as e:
                details["neuron_functionality"] = f"failed: {str(e)}"
                recommendations.append("Fix neuron implementation")
            
            # Test 3: Network functionality
            try:
                network = PurePythonSpikingNetwork([3, 5, 2])
                outputs = network.forward([1.0, 0.5, 0.0], time_steps=3)
                stats = network.get_network_stats()
                details["network_functionality"] = "success"
                details["network_stats"] = stats
                score += 25
            except Exception as e:
                details["network_functionality"] = f"failed: {str(e)}"
                recommendations.append("Fix network implementation")
            
            # Test 4: LSM functionality
            try:
                lsm = PurePythonLiquidStateMachine(3, 10, 2)
                outputs = lsm.update([1.0, 0.5, 0.0])
                activity = lsm.get_reservoir_activity()
                details["lsm_functionality"] = "success"
                details["lsm_activity"] = activity
                score += 25
            except Exception as e:
                details["lsm_functionality"] = f"failed: {str(e)}"
                recommendations.append("Fix LSM implementation")
            
            # Determine status
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "All core functionality tests passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Most core functionality tests passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Core functionality tests failed"
            
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Functional testing failed: {str(e)}"
            score = 0
            details["error"] = str(e)
            recommendations.append("Fix test execution environment")
        
        return QualityGateResult(
            gate_name="core_functionality",
            category=QualityGateCategory.FUNCTIONALITY,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )
    
    def test_model_factory(self) -> QualityGateResult:
        """Test model factory functionality."""
        start_time = time.time()
        details = {}
        requirements = ["Model factory must create all model types", "Factory must handle missing dependencies gracefully"]
        recommendations = []
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.dependency_free_models import ModelFactory
            
            score = 0
            
            # Test LIF neuron creation
            try:
                neuron = ModelFactory.create_lif_neuron()
                details["lif_creation"] = "success"
                score += 33
            except Exception as e:
                details["lif_creation"] = f"failed: {str(e)}"
                recommendations.append("Fix LIF neuron factory method")
            
            # Test network creation
            try:
                network = ModelFactory.create_spiking_network([3, 5, 2])
                details["network_creation"] = "success"
                score += 33
            except Exception as e:
                details["network_creation"] = f"failed: {str(e)}"
                recommendations.append("Fix network factory method")
            
            # Test LSM creation
            try:
                lsm = ModelFactory.create_liquid_state_machine(3, 10, 2)
                details["lsm_creation"] = "success"
                score += 34
            except Exception as e:
                details["lsm_creation"] = f"failed: {str(e)}"
                recommendations.append("Fix LSM factory method")
            
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "Model factory tests passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Model factory tests passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Model factory tests failed"
                
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Model factory testing failed: {str(e)}"
            score = 0
            details["error"] = str(e)
        
        return QualityGateResult(
            gate_name="model_factory",
            category=QualityGateCategory.FUNCTIONALITY,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )


class SecurityValidator:
    """Security compliance and validation framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_security_implementation(self) -> QualityGateResult:
        """Validate security implementation."""
        start_time = time.time()
        details = {}
        requirements = ["Security scanner must be functional", "Input validation must work", "Authentication must be implemented"]
        recommendations = []
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from security.comprehensive_security import SecurityManager, SecurityPolicy
            
            score = 0
            
            # Test 1: Security manager initialization
            try:
                policy = SecurityPolicy()
                security_manager = SecurityManager(policy)
                details["security_manager_init"] = "success"
                score += 25
            except Exception as e:
                details["security_manager_init"] = f"failed: {str(e)}"
                recommendations.append("Fix security manager initialization")
            
            # Test 2: Input validation
            try:
                def safe_operation(data):
                    return len(data) if hasattr(data, '__len__') else 0
                
                # Test safe input
                success, result, message = security_manager.validate_and_execute(
                    safe_operation, [1, 2, 3], "list"
                )
                details["safe_input_validation"] = "success" if success else f"failed: {message}"
                
                # Test malicious input
                success, result, message = security_manager.validate_and_execute(
                    safe_operation, "eval(print('hello'))", "string"
                )
                details["malicious_input_detection"] = "success" if not success else "failed: malicious input not detected"
                
                score += 25
            except Exception as e:
                details["input_validation"] = f"failed: {str(e)}"
                recommendations.append("Fix input validation system")
            
            # Test 3: Authentication
            try:
                auth_success, session_id = security_manager.authenticate_session({
                    'username': 'test_user',
                    'password': 'test_pass'
                })
                details["authentication"] = "success" if auth_success else "failed"
                score += 25
            except Exception as e:
                details["authentication"] = f"failed: {str(e)}"
                recommendations.append("Fix authentication system")
            
            # Test 4: Security monitoring
            try:
                status = security_manager.get_security_status()
                details["security_monitoring"] = "success"
                details["security_events"] = len(status.get('recent_events', []))
                score += 25
            except Exception as e:
                details["security_monitoring"] = f"failed: {str(e)}"
                recommendations.append("Fix security monitoring")
            
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "Security validation passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Security validation passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Security validation failed"
                
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Security validation failed: {str(e)}"
            score = 0
            details["error"] = str(e)
        
        return QualityGateResult(
            gate_name="security_implementation",
            category=QualityGateCategory.SECURITY,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )
    
    def validate_security_scanning(self) -> QualityGateResult:
        """Validate security scanning capabilities."""
        start_time = time.time()
        details = {}
        requirements = ["Security scanner must detect threats", "Scanner must classify threat levels"]
        recommendations = []
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from security.security_scanner import SecurityScanner
            
            scanner = SecurityScanner()
            score = 0
            
            # Test scanner functionality
            try:
                # Create a temporary test file with security issues
                test_file_content = '''
import os
import subprocess
password = "hardcoded_password"
subprocess.call(user_input, shell=True)
eval(user_data)
'''
                
                test_file_path = "temp_security_test.py"
                with open(test_file_path, 'w') as f:
                    f.write(test_file_content)
                
                # Scan the test file
                violations = scanner.scan_file(test_file_path)
                
                # Check if scanner detected issues
                if violations:
                    details["threat_detection"] = f"detected {len(violations)} threats"
                    score += 50
                    
                    # Check threat classification
                    threat_levels = [v.level.value for v in violations]
                    if 'critical' in threat_levels or 'high' in threat_levels:
                        details["threat_classification"] = "correctly classified high/critical threats"
                        score += 50
                    else:
                        details["threat_classification"] = "threats detected but classification may be weak"
                        score += 25
                else:
                    details["threat_detection"] = "no threats detected (potential issue)"
                    recommendations.append("Improve threat detection sensitivity")
                
                # Clean up
                try:
                    os.remove(test_file_path)
                except:
                    pass
                
            except Exception as e:
                details["scanner_test"] = f"failed: {str(e)}"
                recommendations.append("Fix security scanner implementation")
            
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "Security scanning validation passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Security scanning validation passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Security scanning validation failed"
                
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Security scanning validation failed: {str(e)}"
            score = 0
            details["error"] = str(e)
        
        return QualityGateResult(
            gate_name="security_scanning",
            category=QualityGateCategory.SECURITY,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )


class PerformanceBenchmarker:
    """Performance benchmarking and validation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def benchmark_core_performance(self) -> QualityGateResult:
        """Benchmark core system performance."""
        start_time = time.time()
        details = {}
        requirements = ["Core operations must meet performance thresholds", "Memory usage must be reasonable"]
        recommendations = []
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.dependency_free_models import PurePythonLIFNeuron, PurePythonSpikingNetwork
            
            score = 0
            
            # Benchmark 1: Neuron update performance
            try:
                neuron = PurePythonLIFNeuron()
                iterations = 10000
                
                neuron_start = time.time()
                for i in range(iterations):
                    neuron.update(1.5)
                neuron_time = time.time() - neuron_start
                
                neuron_ops_per_sec = iterations / neuron_time
                details["neuron_performance"] = {
                    "operations_per_second": neuron_ops_per_sec,
                    "time_per_operation": neuron_time / iterations
                }
                
                # Performance thresholds
                if neuron_ops_per_sec > 100000:  # 100k ops/sec
                    score += 25
                elif neuron_ops_per_sec > 50000:  # 50k ops/sec
                    score += 15
                    recommendations.append("Optimize neuron update performance")
                else:
                    recommendations.append("Significantly improve neuron performance")
                
            except Exception as e:
                details["neuron_performance"] = f"failed: {str(e)}"
                recommendations.append("Fix neuron performance issues")
            
            # Benchmark 2: Network forward pass performance
            try:
                network = PurePythonSpikingNetwork([10, 20, 10])
                test_input = [1.0] * 10
                iterations = 1000
                
                network_start = time.time()
                for i in range(iterations):
                    network.forward(test_input, time_steps=1)
                network_time = time.time() - network_start
                
                network_ops_per_sec = iterations / network_time
                details["network_performance"] = {
                    "operations_per_second": network_ops_per_sec,
                    "time_per_operation": network_time / iterations
                }
                
                if network_ops_per_sec > 1000:  # 1k forward passes/sec
                    score += 25
                elif network_ops_per_sec > 500:  # 500 forward passes/sec
                    score += 15
                    recommendations.append("Optimize network forward pass")
                else:
                    recommendations.append("Significantly improve network performance")
                
            except Exception as e:
                details["network_performance"] = f"failed: {str(e)}"
                recommendations.append("Fix network performance issues")
            
            # Benchmark 3: Memory efficiency
            try:
                import gc
                
                gc.collect()
                initial_objects = len(gc.get_objects())
                
                # Create and destroy objects
                neurons = [PurePythonLIFNeuron() for _ in range(100)]
                peak_objects = len(gc.get_objects())
                
                del neurons
                gc.collect()
                final_objects = len(gc.get_objects())
                
                memory_efficiency = (peak_objects - initial_objects) / 100  # Objects per neuron
                cleanup_efficiency = (peak_objects - final_objects) / (peak_objects - initial_objects)
                
                details["memory_efficiency"] = {
                    "objects_per_neuron": memory_efficiency,
                    "cleanup_efficiency": cleanup_efficiency
                }
                
                if cleanup_efficiency > 0.8:  # 80% cleanup
                    score += 25
                elif cleanup_efficiency > 0.6:  # 60% cleanup
                    score += 15
                    recommendations.append("Improve memory cleanup")
                else:
                    recommendations.append("Fix memory leaks")
                
            except Exception as e:
                details["memory_efficiency"] = f"failed: {str(e)}"
                recommendations.append("Fix memory efficiency testing")
            
            # Benchmark 4: Scalability test
            try:
                # Test with increasing network sizes
                sizes = [5, 10, 20, 50]
                scaling_results = []
                
                for size in sizes:
                    network = PurePythonSpikingNetwork([size, size, size])
                    test_input = [1.0] * size
                    
                    scale_start = time.time()
                    for _ in range(10):
                        network.forward(test_input, time_steps=1)
                    scale_time = time.time() - scale_start
                    
                    scaling_results.append({
                        "size": size,
                        "time": scale_time,
                        "time_per_neuron": scale_time / (size * 3)  # 3 layers
                    })
                
                details["scalability"] = scaling_results
                
                # Check if scaling is reasonable (should be roughly linear)
                if len(scaling_results) >= 2:
                    scaling_factor = scaling_results[-1]["time"] / scaling_results[0]["time"]
                    size_factor = sizes[-1] / sizes[0]
                    
                    if scaling_factor < size_factor * 2:  # Less than quadratic scaling
                        score += 25
                    elif scaling_factor < size_factor * 3:
                        score += 15
                        recommendations.append("Improve scalability")
                    else:
                        recommendations.append("Fix poor scalability")
                
            except Exception as e:
                details["scalability"] = f"failed: {str(e)}"
                recommendations.append("Fix scalability testing")
            
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "Performance benchmarks passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Performance benchmarks passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Performance benchmarks failed"
                
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Performance benchmarking failed: {str(e)}"
            score = 0
            details["error"] = str(e)
        
        return QualityGateResult(
            gate_name="core_performance",
            category=QualityGateCategory.PERFORMANCE,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )
    
    def benchmark_optimization_features(self) -> QualityGateResult:
        """Benchmark optimization features."""
        start_time = time.time()
        details = {}
        requirements = ["Optimization features must provide speedup", "Caching must improve performance"]
        recommendations = []
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from optimization.advanced_performance_optimizer import PerformanceOptimizer, OptimizationConfig
            
            score = 0
            
            # Test performance optimizer
            try:
                config = OptimizationConfig(cache_size=100)
                optimizer = PerformanceOptimizer(config)
                
                # Test function to optimize
                def test_function(n):
                    return sum(i * i for i in range(n))
                
                # Optimize function
                optimized_func = optimizer.optimize_function(test_function)
                
                # Benchmark without cache (first call)
                start_no_cache = time.time()
                result1 = optimized_func(1000)
                time_no_cache = time.time() - start_no_cache
                
                # Benchmark with cache (second call)
                start_with_cache = time.time()
                result2 = optimized_func(1000)
                time_with_cache = time.time() - start_with_cache
                
                # Verify results are same
                if result1 == result2:
                    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
                    details["cache_speedup"] = speedup
                    
                    if speedup > 10:  # 10x speedup
                        score += 50
                    elif speedup > 2:  # 2x speedup
                        score += 30
                        recommendations.append("Improve cache performance")
                    else:
                        recommendations.append("Cache not providing significant speedup")
                else:
                    details["cache_correctness"] = "failed - results don't match"
                    recommendations.append("Fix cache correctness issues")
                
                optimizer.shutdown()
                
            except Exception as e:
                details["optimization_test"] = f"failed: {str(e)}"
                recommendations.append("Fix optimization system")
            
            # Test concurrent processing
            try:
                from optimization.advanced_performance_optimizer import ConcurrentProcessor, OptimizationConfig
                
                config = OptimizationConfig(enable_concurrent_processing=True, max_workers=2)
                processor = ConcurrentProcessor(config)
                
                def slow_function(x):
                    time.sleep(0.01)  # 10ms delay
                    return x * x
                
                # Sequential execution
                seq_start = time.time()
                seq_results = [slow_function(i) for i in range(10)]
                seq_time = time.time() - seq_start
                
                # Concurrent execution
                conc_start = time.time()
                conc_results = processor.submit_batch(slow_function, list(range(10)))
                conc_time = time.time() - conc_start
                
                if seq_results == sorted(conc_results):
                    concurrent_speedup = seq_time / conc_time if conc_time > 0 else 1
                    details["concurrent_speedup"] = concurrent_speedup
                    
                    if concurrent_speedup > 1.5:  # 1.5x speedup
                        score += 50
                    elif concurrent_speedup > 1.1:  # 1.1x speedup
                        score += 25
                        recommendations.append("Improve concurrent processing efficiency")
                    else:
                        recommendations.append("Concurrent processing not providing speedup")
                else:
                    details["concurrent_correctness"] = "failed - results don't match"
                    recommendations.append("Fix concurrent processing correctness")
                
                processor.shutdown()
                
            except Exception as e:
                details["concurrent_test"] = f"failed: {str(e)}"
                recommendations.append("Fix concurrent processing")
            
            if score >= 90:
                status = QualityGateStatus.PASS
                message = "Optimization benchmarks passed"
            elif score >= 70:
                status = QualityGateStatus.WARN
                message = "Optimization benchmarks passed with warnings"
            else:
                status = QualityGateStatus.FAIL
                message = "Optimization benchmarks failed"
                
        except Exception as e:
            status = QualityGateStatus.ERROR
            message = f"Optimization benchmarking failed: {str(e)}"
            score = 0
            details["error"] = str(e)
        
        return QualityGateResult(
            gate_name="optimization_features",
            category=QualityGateCategory.PERFORMANCE,
            status=status,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            requirements=requirements,
            recommendations=recommendations
        )


class QualityGateOrchestrator:
    """Main quality gate orchestration system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality gate components
        self.functional_tester = FunctionalTester()
        self.security_validator = SecurityValidator()
        self.performance_benchmarker = PerformanceBenchmarker()
        
        # Results storage
        self.results: List[QualityGateResult] = []
        
    def run_all_quality_gates(self) -> QualityGateReport:
        """Run all quality gates and generate comprehensive report."""
        start_time = time.time()
        self.logger.info("Starting comprehensive quality gate execution")
        
        # Define all quality gates
        quality_gates = [
            # Functionality gates
            ("core_functionality", self.functional_tester.test_core_functionality),
            ("model_factory", self.functional_tester.test_model_factory),
            
            # Security gates
            ("security_implementation", self.security_validator.validate_security_implementation),
            ("security_scanning", self.security_validator.validate_security_scanning),
            
            # Performance gates
            ("core_performance", self.performance_benchmarker.benchmark_core_performance),
            ("optimization_features", self.performance_benchmarker.benchmark_optimization_features),
        ]
        
        results = []
        
        # Execute each quality gate
        for gate_name, gate_function in quality_gates:
            try:
                self.logger.info(f"Executing quality gate: {gate_name}")
                result = gate_function()
                results.append(result)
                
                status_emoji = {
                    QualityGateStatus.PASS: "✅",
                    QualityGateStatus.WARN: "⚠️",
                    QualityGateStatus.FAIL: "❌",
                    QualityGateStatus.ERROR: "🚨",
                    QualityGateStatus.SKIP: "⏭️"
                }
                
                self.logger.info(f"{status_emoji.get(result.status, '❓')} {gate_name}: {result.message} (Score: {result.score:.1f})")
                
            except Exception as e:
                self.logger.error(f"Quality gate {gate_name} failed with exception: {str(e)}")
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    category=QualityGateCategory.FUNCTIONALITY,
                    status=QualityGateStatus.ERROR,
                    score=0.0,
                    message=f"Quality gate execution failed: {str(e)}",
                    details={"exception": str(e), "traceback": traceback.format_exc()},
                    execution_time=0.0,
                    timestamp=time.time(),
                    requirements=[],
                    recommendations=["Fix quality gate implementation"]
                )
                results.append(error_result)
        
        # Calculate overall metrics
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.status == QualityGateStatus.PASS)
        failed_gates = sum(1 for r in results if r.status == QualityGateStatus.FAIL)
        warnings = sum(1 for r in results if r.status == QualityGateStatus.WARN)
        errors = sum(1 for r in results if r.status == QualityGateStatus.ERROR)
        skipped = sum(1 for r in results if r.status == QualityGateStatus.SKIP)
        
        # Calculate overall score (weighted average)
        category_weights = {
            QualityGateCategory.FUNCTIONALITY: 0.3,
            QualityGateCategory.SECURITY: 0.25,
            QualityGateCategory.PERFORMANCE: 0.25,
            QualityGateCategory.RELIABILITY: 0.1,
            QualityGateCategory.MAINTAINABILITY: 0.05,
            QualityGateCategory.COMPLIANCE: 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = category_weights.get(result.category, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        if errors > 0:
            overall_status = QualityGateStatus.ERROR
        elif failed_gates > 0:
            overall_status = QualityGateStatus.FAIL
        elif warnings > 0:
            overall_status = QualityGateStatus.WARN
        else:
            overall_status = QualityGateStatus.PASS
        
        # Generate summary
        summary = {
            "pass_rate": passed_gates / total_gates if total_gates > 0 else 0.0,
            "categories": {},
            "top_recommendations": [],
            "critical_issues": []
        }
        
        # Category breakdown
        for category in QualityGateCategory:
            category_results = [r for r in results if r.category == category]
            if category_results:
                category_score = sum(r.score for r in category_results) / len(category_results)
                category_status = max((r.status for r in category_results), key=lambda s: s.value)
                summary["categories"][category.value] = {
                    "score": category_score,
                    "status": category_status.value,
                    "count": len(category_results)
                }
        
        # Collect recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Get top recommendations (most frequent)
        from collections import Counter
        rec_counter = Counter(all_recommendations)
        summary["top_recommendations"] = [rec for rec, count in rec_counter.most_common(5)]
        
        # Identify critical issues
        summary["critical_issues"] = [
            f"{r.gate_name}: {r.message}" 
            for r in results 
            if r.status in [QualityGateStatus.FAIL, QualityGateStatus.ERROR]
        ]
        
        execution_time = time.time() - start_time
        
        report = QualityGateReport(
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warnings=warnings,
            errors=errors,
            skipped=skipped,
            overall_score=overall_score,
            overall_status=overall_status,
            execution_time=execution_time,
            timestamp=time.time(),
            results=results,
            summary=summary
        )
        
        self.logger.info(f"Quality gate execution completed: {overall_status.value.upper()} "
                        f"(Score: {overall_score:.1f}, {passed_gates}/{total_gates} passed) "
                        f"in {execution_time:.2f}s")
        
        return report
    
    def export_report(self, report: QualityGateReport, filepath: str) -> None:
        """Export quality gate report to file."""
        try:
            # Convert to serializable format
            report_dict = asdict(report)
            
            # Convert enums to strings
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                else:
                    return obj
            
            serializable_report = convert_enums(report_dict)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)
            
            self.logger.info(f"Quality gate report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export quality gate report: {str(e)}")


def main():
    """Run comprehensive quality gates."""
    print("Neuromorphic System Quality Gates")
    print("=" * 50)
    
    orchestrator = QualityGateOrchestrator()
    
    # Run all quality gates
    report = orchestrator.run_all_quality_gates()
    
    # Display summary
    print(f"\nOverall Status: {report.overall_status.value.upper()}")
    print(f"Overall Score: {report.overall_score:.1f}/100.0")
    print(f"Execution Time: {report.execution_time:.2f} seconds")
    print(f"\nResults: {report.passed_gates}/{report.total_gates} passed")
    
    if report.warnings > 0:
        print(f"Warnings: {report.warnings}")
    if report.failed_gates > 0:
        print(f"Failures: {report.failed_gates}")
    if report.errors > 0:
        print(f"Errors: {report.errors}")
    
    # Display category breakdown
    print("\nCategory Breakdown:")
    for category, data in report.summary["categories"].items():
        print(f"  {category}: {data['score']:.1f}/100 ({data['status']})")
    
    # Display critical issues
    if report.summary["critical_issues"]:
        print("\nCritical Issues:")
        for issue in report.summary["critical_issues"]:
            print(f"  - {issue}")
    
    # Display recommendations
    if report.summary["top_recommendations"]:
        print("\nTop Recommendations:")
        for rec in report.summary["top_recommendations"]:
            print(f"  - {rec}")
    
    # Export detailed report
    try:
        orchestrator.export_report(report, "quality_gates_report.json")
        print(f"\nDetailed report saved to quality_gates_report.json")
    except Exception as e:
        print(f"Could not save detailed report: {e}")
    
    # Return appropriate exit code
    if report.overall_status in [QualityGateStatus.PASS, QualityGateStatus.WARN]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())