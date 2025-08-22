"""
TERRAGON SDLC QUALITY GATES VALIDATION
Comprehensive quality gates implementation as specified in TERRAGON SDLC v4.0
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: str  # "PASSED", "FAILED", "SKIPPED"
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    execution_time_ms: float = 0.0
    critical: bool = False


@dataclass
class QualityGatesReport:
    """Complete quality gates validation report."""
    overall_status: str
    overall_score: float
    execution_time_ms: float
    timestamp: str
    generation: str
    gates: List[QualityGateResult]
    
    def to_dict(self):
        return asdict(self)


class QualityGatesValidator:
    """Validates all mandatory quality gates for TERRAGON SDLC."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
    
    def run_all_gates(self) -> QualityGatesReport:
        """Run all mandatory quality gates."""
        print("=" * 80)
        print("TERRAGON SDLC QUALITY GATES VALIDATION")
        print("=" * 80)
        
        # Define mandatory quality gates
        gates = [
            ("Functional Tests", self._validate_functional_tests, True),
            ("Security Validation", self._validate_security, True),
            ("Performance Benchmarks", self._validate_performance, True),
            ("Code Quality", self._validate_code_quality, False),
            ("Documentation", self._validate_documentation, False),
            ("Integration Tests", self._validate_integration, True),
            ("Resource Limits", self._validate_resource_limits, True),
            ("Deployment Readiness", self._validate_deployment_readiness, False),
        ]
        
        # Execute each gate
        for gate_name, gate_func, is_critical in gates:
            print(f"\n🔍 Executing: {gate_name}")
            print("-" * 50)
            
            start_time = time.time()
            try:
                result = gate_func()
                result.critical = is_critical
                result.execution_time_ms = (time.time() - start_time) * 1000
                self.results.append(result)
                
                status_icon = "✅" if result.status == "PASSED" else "❌" if result.status == "FAILED" else "⏭️"
                print(f"{status_icon} {gate_name}: {result.status} (Score: {result.score:.2f})")
                
            except Exception as e:
                error_result = QualityGateResult(
                    name=gate_name,
                    status="FAILED",
                    score=0.0,
                    details={"error": str(e)},
                    execution_time_ms=(time.time() - start_time) * 1000,
                    critical=is_critical
                )
                self.results.append(error_result)
                print(f"❌ {gate_name}: FAILED - {e}")
        
        # Generate final report
        return self._generate_report()
    
    def _validate_functional_tests(self) -> QualityGateResult:
        """Validate functional tests pass."""
        details = {}
        score = 0.0
        
        # Test Generation 1 (Basic Functionality)
        try:
            result = subprocess.run([
                sys.executable, 'test_basic_functionality.py'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            gen1_passed = result.returncode == 0
            details['generation_1'] = {
                'status': 'PASSED' if gen1_passed else 'FAILED',
                'output': result.stdout[-500:] if result.stdout else '',
                'errors': result.stderr[-500:] if result.stderr else ''
            }
            score += 0.4 if gen1_passed else 0.0
            
        except Exception as e:
            details['generation_1'] = {'status': 'ERROR', 'error': str(e)}
        
        # Test Generation 2 (Robust Functionality)
        try:
            result = subprocess.run([
                sys.executable, 'test_robust_functionality.py'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            gen2_passed = result.returncode == 0
            details['generation_2'] = {
                'status': 'PASSED' if gen2_passed else 'FAILED',
                'output': result.stdout[-500:] if result.stdout else '',
                'errors': result.stderr[-500:] if result.stderr else ''
            }
            score += 0.3 if gen2_passed else 0.0
            
        except Exception as e:
            details['generation_2'] = {'status': 'ERROR', 'error': str(e)}
        
        # Test Generation 3 (Optimized Functionality)
        try:
            result = subprocess.run([
                sys.executable, 'tests/test_generation3_working.py'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            gen3_passed = result.returncode == 0
            details['generation_3'] = {
                'status': 'PASSED' if gen3_passed else 'FAILED',
                'output': result.stdout[-500:] if result.stdout else '',
                'errors': result.stderr[-500:] if result.stderr else ''
            }
            score += 0.3 if gen3_passed else 0.0
            
        except Exception as e:
            details['generation_3'] = {'status': 'ERROR', 'error': str(e)}
        
        status = "PASSED" if score >= 0.7 else "FAILED"
        
        return QualityGateResult(
            name="Functional Tests",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_security(self) -> QualityGateResult:
        """Validate security measures are in place."""
        details = {}
        score = 0.0
        
        # Check for security modules
        security_files = [
            'src/security/__init__.py',
            'src/security/input_sanitizer.py',
            'src/utils/robust_error_handling.py'
        ]
        
        security_modules_exist = 0
        for file_path in security_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                security_modules_exist += 1
                details[f'security_module_{file_path}'] = 'EXISTS'
            else:
                details[f'security_module_{file_path}'] = 'MISSING'
        
        score += (security_modules_exist / len(security_files)) * 0.4
        
        # Test security validation
        try:
            # Import and test input sanitizer
            from security.input_sanitizer import InputSanitizer, global_sanitizer
            
            sanitizer = InputSanitizer()
            
            # Test basic validation
            test_input = "test_input"
            sanitized = sanitizer.sanitize_input(test_input)
            
            details['input_sanitizer'] = 'FUNCTIONAL'
            score += 0.3
            
        except Exception as e:
            details['input_sanitizer'] = f'ERROR: {str(e)}'
        
        # Check for security configuration
        try:
            from utils.robust_error_handling import RobustErrorHandler
            handler = RobustErrorHandler()
            details['error_handling'] = 'FUNCTIONAL'
            score += 0.3
        except Exception as e:
            details['error_handling'] = f'ERROR: {str(e)}'
        
        status = "PASSED" if score >= 0.7 else "FAILED"
        
        return QualityGateResult(
            name="Security Validation", 
            status=status,
            score=score,
            details=details
        )
    
    def _validate_performance(self) -> QualityGateResult:
        """Validate performance benchmarks meet requirements."""
        details = {}
        score = 0.0
        
        try:
            # Import and test optimized neuron performance
            from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams
            import numpy as np
            
            # Performance test
            params = OptimizedLIFParams(enable_jit=True)
            neuron = OptimizedLIFNeuron(params, n_neurons=100)
            
            # Warm up
            test_input = np.random.randn(100) * 1e-9
            for _ in range(5):
                neuron.forward(test_input)
            
            # Benchmark
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                result = neuron.forward(test_input)
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            throughput_hz = 1000 / avg_time_ms if avg_time_ms > 0 else 0
            
            # Get performance metrics
            metrics = neuron.get_performance_metrics()
            
            details['performance_test'] = {
                'avg_latency_ms': avg_time_ms,
                'throughput_hz': throughput_hz,
                'jit_enabled': metrics.get('jit_enabled', False),
                'iterations': iterations
            }
            
            # Performance criteria
            if avg_time_ms < 50:  # Less than 50ms average latency
                score += 0.4
            elif avg_time_ms < 100:
                score += 0.2
            
            if throughput_hz > 100:  # More than 100 Hz throughput
                score += 0.3
            elif throughput_hz > 50:
                score += 0.15
            
            if metrics.get('jit_enabled', False):
                score += 0.3
            
        except Exception as e:
            details['performance_test'] = f'ERROR: {str(e)}'
        
        status = "PASSED" if score >= 0.6 else "FAILED"
        
        return QualityGateResult(
            name="Performance Benchmarks",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_code_quality(self) -> QualityGateResult:
        """Validate code quality standards."""
        details = {}
        score = 0.0
        
        # Check for proper module structure
        required_modules = [
            'src/__init__.py',
            'src/models/__init__.py',
            'src/utils/__init__.py',
            'src/security/__init__.py'
        ]
        
        modules_exist = 0
        for module in required_modules:
            if (self.project_root / module).exists():
                modules_exist += 1
        
        score += (modules_exist / len(required_modules)) * 0.5
        details['module_structure'] = f'{modules_exist}/{len(required_modules)} modules exist'
        
        # Check for documentation strings
        try:
            from models.lif_neuron import LIFNeuron
            if LIFNeuron.__doc__:
                score += 0.25
                details['documentation'] = 'Classes have docstrings'
        except:
            details['documentation'] = 'Unable to verify docstrings'
        
        # Check for type hints (basic check)
        try:
            import inspect
            from models.lif_neuron import LIFNeuron
            sig = inspect.signature(LIFNeuron.__init__)
            if any(param.annotation != inspect.Parameter.empty for param in sig.parameters.values()):
                score += 0.25
                details['type_hints'] = 'Type hints present'
        except:
            details['type_hints'] = 'Unable to verify type hints'
        
        status = "PASSED" if score >= 0.7 else "FAILED"
        
        return QualityGateResult(
            name="Code Quality",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation completeness."""
        details = {}
        score = 0.0
        
        # Check for essential documentation files
        doc_files = ['README.md', 'API_REFERENCE.md', 'DEPLOYMENT_GUIDE.md']
        
        for doc_file in doc_files:
            if (self.project_root / doc_file).exists():
                score += 0.33
                details[doc_file] = 'EXISTS'
            else:
                details[doc_file] = 'MISSING'
        
        status = "PASSED" if score >= 0.6 else "FAILED"
        
        return QualityGateResult(
            name="Documentation",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_integration(self) -> QualityGateResult:
        """Validate integration between components."""
        details = {}
        score = 0.0
        
        try:
            # Test full system integration
            from models.lif_neuron import LIFNeuron, LIFParams
            from models.spiking_neural_network import SpikingNeuralNetwork, NetworkTopology
            from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams
            import numpy as np
            
            # Test basic model integration
            params = LIFParams()
            neuron = LIFNeuron(params, n_neurons=5)
            test_input = np.random.randn(5) * 1e-9
            result = neuron.forward(test_input)
            
            if 'spikes' in result and 'v_mem' in result:
                score += 0.3
                details['basic_model'] = 'INTEGRATION_OK'
            
            # Test optimized model integration
            opt_params = OptimizedLIFParams()
            opt_neuron = OptimizedLIFNeuron(opt_params, n_neurons=5)
            opt_result = opt_neuron.forward(test_input)
            
            if 'spikes' in opt_result and 'v_mem' in opt_result:
                score += 0.4
                details['optimized_model'] = 'INTEGRATION_OK'
            
            # Test security integration
            from security.input_sanitizer import InputSanitizer
            sanitizer = InputSanitizer()
            sanitized_input = sanitizer.sanitize_input(test_input)
            
            score += 0.3
            details['security_integration'] = 'INTEGRATION_OK'
            
        except Exception as e:
            details['integration_error'] = str(e)
        
        status = "PASSED" if score >= 0.7 else "FAILED"
        
        return QualityGateResult(
            name="Integration Tests",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_resource_limits(self) -> QualityGateResult:
        """Validate resource usage within limits."""
        details = {}
        score = 0.0
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage test
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run memory-intensive operation
            from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams
            import numpy as np
            
            neurons = []
            for i in range(10):  # Create multiple neurons
                params = OptimizedLIFParams()
                neuron = OptimizedLIFNeuron(params, n_neurons=50)
                neurons.append(neuron)
                
                # Run some operations
                for _ in range(10):
                    test_input = np.random.randn(50) * 1e-9
                    neuron.forward(test_input)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            details['memory_usage'] = {
                'initial_mb': initial_memory,
                'final_mb': final_memory,
                'increase_mb': memory_increase
            }
            
            # Memory criteria (should not exceed 500MB increase)
            if memory_increase < 200:
                score += 0.5
            elif memory_increase < 500:
                score += 0.3
            
            # CPU usage (basic check)
            cpu_percent = process.cpu_percent(interval=1)
            details['cpu_usage_percent'] = cpu_percent
            
            if cpu_percent < 80:
                score += 0.5
            elif cpu_percent < 95:
                score += 0.3
            
        except Exception as e:
            details['resource_error'] = str(e)
        
        status = "PASSED" if score >= 0.6 else "FAILED"
        
        return QualityGateResult(
            name="Resource Limits",
            status=status,
            score=score,
            details=details
        )
    
    def _validate_deployment_readiness(self) -> QualityGateResult:
        """Validate deployment readiness."""
        details = {}
        score = 0.0
        
        # Check for deployment files
        deployment_files = [
            'requirements.txt',
            'setup.py', 
            'Dockerfile',
            'deploy.sh'
        ]
        
        deployment_files_exist = 0
        for file_path in deployment_files:
            if (self.project_root / file_path).exists():
                deployment_files_exist += 1
                details[file_path] = 'EXISTS'
            else:
                details[file_path] = 'MISSING'
        
        score += (deployment_files_exist / len(deployment_files)) * 1.0
        
        status = "PASSED" if score >= 0.7 else "FAILED"
        
        return QualityGateResult(
            name="Deployment Readiness",
            status=status,
            score=score,
            details=details
        )
    
    def _generate_report(self) -> QualityGatesReport:
        """Generate comprehensive quality gates report."""
        total_time_ms = (time.time() - self.start_time) * 1000
        
        # Calculate overall score and status
        if not self.results:
            overall_score = 0.0
            overall_status = "FAILED"
        else:
            # Weight critical gates more heavily
            total_weight = 0
            weighted_score = 0
            
            for result in self.results:
                weight = 2.0 if result.critical else 1.0
                weighted_score += result.score * weight
                total_weight += weight
            
            overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Check critical gates
            critical_failures = [r for r in self.results if r.critical and r.status == "FAILED"]
            
            if critical_failures:
                overall_status = "FAILED"
            elif overall_score >= 0.8:
                overall_status = "PASSED"
            elif overall_score >= 0.6:
                overall_status = "CONDITIONAL"
            else:
                overall_status = "FAILED"
        
        return QualityGatesReport(
            overall_status=overall_status,
            overall_score=overall_score,
            execution_time_ms=total_time_ms,
            timestamp=datetime.now().isoformat(),
            generation="Generation 3 (Optimized)",
            gates=self.results
        )


def main():
    """Main entry point for quality gates validation."""
    validator = QualityGatesValidator()
    report = validator.run_all_gates()
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUALITY GATES VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"Overall Status: {report.overall_status}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Execution Time: {report.execution_time_ms:.1f}ms")
    print(f"Generation: {report.generation}")
    
    print(f"\nGate Results:")
    for gate in report.gates:
        status_icon = "✅" if gate.status == "PASSED" else "❌" if gate.status == "FAILED" else "⏭️"
        critical_mark = " 🔴" if gate.critical else ""
        print(f"  {status_icon} {gate.name}: {gate.status} ({gate.score:.2f}){critical_mark}")
    
    # Save report
    report_file = Path("quality_gates_report.json")
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Determine exit code
    if report.overall_status == "PASSED":
        print("\n🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION DEPLOYMENT")
        return 0
    elif report.overall_status == "CONDITIONAL":
        print("\n⚠️  CONDITIONAL PASS - SOME IMPROVEMENTS RECOMMENDED")
        return 0
    else:
        print("\n❌ QUALITY GATES FAILED - REMEDIATION REQUIRED")
        return 1


if __name__ == "__main__":
    sys.exit(main())