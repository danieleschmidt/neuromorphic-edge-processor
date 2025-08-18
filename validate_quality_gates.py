#!/usr/bin/env python3
"""
Quality Gates Validation Script
Comprehensive validation of the neuromorphic edge processor implementation
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {
            "timestamp": time.time(),
            "overall_status": "UNKNOWN",
            "gates": {},
            "summary": {},
            "recommendations": []
        }
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🚀 Starting Neuromorphic Edge Processor Quality Gates Validation")
        print("=" * 70)
        
        # Quality gates to validate
        gates = [
            ("Code Structure", self.validate_code_structure),
            ("Functionality", self.validate_functionality),
            ("Performance", self.validate_performance),
            ("Security", self.validate_security),
            ("Documentation", self.validate_documentation),
            ("Dependencies", self.validate_dependencies),
            ("Examples", self.validate_examples),
            ("Deployment", self.validate_deployment_readiness)
        ]
        
        total_score = 0
        max_score = 0
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Validating: {gate_name}")
            print("-" * 40)
            
            try:
                score, max_points, details = gate_func()
                total_score += score
                max_score += max_points
                
                self.results["gates"][gate_name] = {
                    "score": score,
                    "max_score": max_points,
                    "percentage": (score / max_points * 100) if max_points > 0 else 0,
                    "status": "PASS" if score >= max_points * 0.8 else "FAIL",
                    "details": details
                }
                
                status_emoji = "✅" if score >= max_points * 0.8 else "❌"
                print(f"{status_emoji} {gate_name}: {score}/{max_points} ({score/max_points*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ {gate_name}: FAILED ({e})")
                self.results["gates"][gate_name] = {
                    "score": 0,
                    "max_score": 100,
                    "percentage": 0,
                    "status": "ERROR",
                    "details": {"error": str(e)}
                }
                max_score += 100
        
        # Calculate overall results
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        if overall_percentage >= 85:
            overall_status = "EXCELLENT"
        elif overall_percentage >= 70:
            overall_status = "GOOD"
        elif overall_percentage >= 50:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        self.results["overall_status"] = overall_status
        self.results["summary"] = {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": overall_percentage,
            "passed_gates": len([g for g in self.results["gates"].values() if g["status"] == "PASS"]),
            "failed_gates": len([g for g in self.results["gates"].values() if g["status"] in ["FAIL", "ERROR"]]),
            "total_gates": len(gates)
        }
        
        self._generate_recommendations()
        self._print_summary()
        
        return self.results
    
    def validate_code_structure(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate code structure and organization."""
        score = 0
        max_score = 100
        details = {}
        
        # Check directory structure
        required_dirs = [
            "src", "src/models", "src/algorithms", "src/utils", 
            "src/optimization", "src/monitoring", "src/security",
            "tests", "examples", "benchmarks"
        ]
        
        existing_dirs = []
        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
                score += 5
        
        details["directory_structure"] = {
            "required": required_dirs,
            "existing": existing_dirs,
            "score": len(existing_dirs) * 5
        }
        
        # Check for key files
        key_files = [
            "README.md", "setup.py", "requirements.txt",
            "src/__init__.py", "src/models/__init__.py",
            "examples/neuromorphic_demo.py"
        ]
        
        existing_files = []
        for file_path in key_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                score += 5
        
        details["key_files"] = {
            "required": key_files,
            "existing": existing_files,
            "score": len(existing_files) * 5
        }
        
        # Check code quality indicators
        python_files = list(self.repo_root.glob("**/*.py"))
        details["code_metrics"] = {
            "total_python_files": len(python_files),
            "lines_of_code": self._count_lines_of_code(python_files),
            "has_docstrings": self._check_docstrings(python_files)
        }
        
        if len(python_files) > 10:
            score += 10
        if details["code_metrics"]["lines_of_code"] > 1000:
            score += 10
        
        return min(score, max_score), max_score, details
    
    def validate_functionality(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate core functionality."""
        score = 0
        max_score = 100
        details = {}
        
        # Test core demos
        demo_tests = [
            ("Neuromorphic Demo", "examples/neuromorphic_demo.py"),
            ("Health Monitoring", "examples/health_monitoring_demo.py"),
            ("Performance Optimization", "examples/performance_optimization_demo.py")
        ]
        
        working_demos = []
        for demo_name, demo_path in demo_tests:
            try:
                result = subprocess.run(
                    [sys.executable, demo_path],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    working_demos.append(demo_name)
                    score += 20
                    print(f"  ✅ {demo_name}: Working")
                else:
                    print(f"  ❌ {demo_name}: Failed ({result.stderr[:100]}...)")
                    
            except Exception as e:
                print(f"  ❌ {demo_name}: Error ({e})")
        
        details["demo_tests"] = {
            "total": len(demo_tests),
            "working": working_demos,
            "score": len(working_demos) * 20
        }
        
        # Check for model implementations
        model_files = [
            "src/models/lif_neuron.py",
            "src/models/spiking_neural_network.py", 
            "src/models/liquid_state_machine.py"
        ]
        
        existing_models = []
        for model_file in model_files:
            if (self.repo_root / model_file).exists():
                existing_models.append(model_file)
                score += 10
        
        details["model_implementations"] = {
            "required": model_files,
            "existing": existing_models,
            "score": len(existing_models) * 10
        }
        
        # Check algorithm implementations
        algo_files = list((self.repo_root / "src" / "algorithms").glob("*.py"))
        if len(algo_files) > 3:
            score += 10
        
        details["algorithms"] = {
            "count": len(algo_files),
            "files": [f.name for f in algo_files]
        }
        
        return min(score, max_score), max_score, details
    
    def validate_performance(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate performance optimizations."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for optimization modules
        optimization_features = [
            ("Caching", "src/optimization/adaptive_caching.py"),
            ("Performance Optimizer", "src/optimization/performance_optimizer.py"),
            ("Memory Optimizer", "src/optimization/memory_optimizer.py"),
            ("Concurrent Processor", "src/optimization/concurrent_processor.py")
        ]
        
        implemented_optimizations = []
        for feature_name, file_path in optimization_features:
            if (self.repo_root / file_path).exists():
                implemented_optimizations.append(feature_name)
                score += 15
                print(f"  ✅ {feature_name}: Implemented")
            else:
                print(f"  ⚠️ {feature_name}: Not found")
        
        details["optimization_features"] = {
            "total": len(optimization_features),
            "implemented": implemented_optimizations,
            "score": len(implemented_optimizations) * 15
        }
        
        # Test performance demo
        try:
            result = subprocess.run(
                [sys.executable, "examples/performance_optimization_demo.py"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                score += 25
                print(f"  ✅ Performance demo: Working")
                
                # Look for performance metrics in output
                if "speedup" in result.stdout.lower():
                    score += 15
                    print(f"  ✅ Performance metrics: Detected")
            else:
                print(f"  ❌ Performance demo: Failed")
                
        except Exception as e:
            print(f"  ❌ Performance demo: Error ({e})")
        
        details["performance_demo"] = {
            "working": score >= 25,
            "has_metrics": score >= 40
        }
        
        return min(score, max_score), max_score, details
    
    def validate_security(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate security implementations."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for security modules
        security_features = [
            ("Input Validator", "src/security/input_validator.py"),
            ("Security Manager", "src/security/security_manager.py"),
            ("Error Handling", "src/utils/error_handling.py")
        ]
        
        implemented_security = []
        for feature_name, file_path in security_features:
            if (self.repo_root / file_path).exists():
                implemented_security.append(feature_name)
                score += 20
                print(f"  ✅ {feature_name}: Implemented")
            else:
                print(f"  ⚠️ {feature_name}: Not found")
        
        details["security_features"] = {
            "total": len(security_features),
            "implemented": implemented_security,
            "score": len(implemented_security) * 20
        }
        
        # Check for security practices in code
        python_files = list(self.repo_root.glob("**/*.py"))
        security_indicators = 0
        
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(keyword in content.lower() for keyword in 
                      ['validation', 'sanitize', 'security', 'error handling']):
                    security_indicators += 1
            except:
                pass
        
        if security_indicators > 3:
            score += 20
            print(f"  ✅ Security practices: Found in code")
        
        details["security_practices"] = {
            "files_with_security_indicators": security_indicators,
            "score": 20 if security_indicators > 3 else 0
        }
        
        # Check for no obvious security issues
        security_issues = 0
        for py_file in python_files[:5]:
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(issue in content for issue in ['eval(', 'exec(', 'shell=True']):
                    security_issues += 1
            except:
                pass
        
        if security_issues == 0:
            score += 20
            print(f"  ✅ No obvious security issues")
        else:
            print(f"  ⚠️ Potential security issues detected")
        
        details["security_scan"] = {
            "issues_found": security_issues,
            "clean": security_issues == 0
        }
        
        return min(score, max_score), max_score, details
    
    def validate_documentation(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate documentation quality."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for documentation files
        doc_files = [
            ("README", "README.md"),
            ("API Reference", "API_REFERENCE.md"),
            ("Changelog", "CHANGELOG.md"),
            ("Contributing", "CONTRIBUTING.md"),
            ("Deployment Guide", "DEPLOYMENT_GUIDE.md")
        ]
        
        existing_docs = []
        for doc_name, file_path in doc_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                existing_docs.append(doc_name)
                score += 15
                print(f"  ✅ {doc_name}: Found")
                
                # Check file size (should be substantial)
                try:
                    size = full_path.stat().st_size
                    if size > 1000:  # At least 1KB
                        score += 5
                except:
                    pass
            else:
                print(f"  ⚠️ {doc_name}: Missing")
        
        details["documentation_files"] = {
            "total": len(doc_files),
            "existing": existing_docs,
            "score": len(existing_docs) * 15
        }
        
        # Check README quality
        readme_path = self.repo_root / "README.md"
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(encoding='utf-8')
                readme_score = 0
                
                if len(readme_content) > 2000:
                    readme_score += 5
                if "installation" in readme_content.lower():
                    readme_score += 5
                if "usage" in readme_content.lower():
                    readme_score += 5
                if "example" in readme_content.lower():
                    readme_score += 5
                
                score += readme_score
                details["readme_quality"] = {
                    "length": len(readme_content),
                    "has_installation": "installation" in readme_content.lower(),
                    "has_usage": "usage" in readme_content.lower(),
                    "has_examples": "example" in readme_content.lower(),
                    "score": readme_score
                }
                
            except:
                pass
        
        return min(score, max_score), max_score, details
    
    def validate_dependencies(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate dependency management."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for dependency files
        dep_files = [
            ("requirements.txt", "requirements.txt"),
            ("setup.py", "setup.py")
        ]
        
        existing_dep_files = []
        for name, file_path in dep_files:
            if (self.repo_root / file_path).exists():
                existing_dep_files.append(name)
                score += 25
                print(f"  ✅ {name}: Found")
        
        details["dependency_files"] = {
            "existing": existing_dep_files,
            "score": len(existing_dep_files) * 25
        }
        
        # Check requirements.txt quality
        req_path = self.repo_root / "requirements.txt"
        if req_path.exists():
            try:
                req_content = req_path.read_text(encoding='utf-8')
                req_lines = [line.strip() for line in req_content.split('\n') if line.strip() and not line.startswith('#')]
                
                if len(req_lines) > 5:
                    score += 15
                    print(f"  ✅ Requirements: {len(req_lines)} dependencies")
                
                # Check for version pins
                pinned_deps = [line for line in req_lines if '>=' in line or '==' in line]
                if len(pinned_deps) > len(req_lines) * 0.5:
                    score += 10
                    print(f"  ✅ Version pinning: Good")
                
                details["requirements_analysis"] = {
                    "total_deps": len(req_lines),
                    "pinned_deps": len(pinned_deps),
                    "pinning_ratio": len(pinned_deps) / max(len(req_lines), 1)
                }
                
            except:
                pass
        
        # Check for dependency organization
        if (self.repo_root / "setup.py").exists():
            try:
                setup_content = (self.repo_root / "setup.py").read_text(encoding='utf-8')
                if "extras_require" in setup_content:
                    score += 15
                    print(f"  ✅ Optional dependencies: Organized")
            except:
                pass
        
        return min(score, max_score), max_score, details
    
    def validate_examples(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate examples and demos."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for example files
        example_files = list((self.repo_root / "examples").glob("*.py"))
        
        if len(example_files) >= 3:
            score += 30
            print(f"  ✅ Example count: {len(example_files)} files")
        
        details["example_files"] = {
            "count": len(example_files),
            "files": [f.name for f in example_files]
        }
        
        # Test key examples
        key_examples = [
            "neuromorphic_demo.py",
            "health_monitoring_demo.py", 
            "performance_optimization_demo.py"
        ]
        
        working_examples = []
        for example in key_examples:
            example_path = self.repo_root / "examples" / example
            if example_path.exists():
                try:
                    # Quick syntax check
                    with open(example_path) as f:
                        compile(f.read(), example_path, 'exec')
                    working_examples.append(example)
                    score += 15
                    print(f"  ✅ {example}: Syntax valid")
                except Exception as e:
                    print(f"  ❌ {example}: Syntax error ({e})")
            else:
                print(f"  ⚠️ {example}: Missing")
        
        details["key_examples"] = {
            "total": len(key_examples),
            "working": working_examples,
            "score": len(working_examples) * 15
        }
        
        # Check for Jupyter notebooks
        notebook_files = list((self.repo_root / "examples").glob("*.ipynb"))
        if len(notebook_files) > 0:
            score += 10
            print(f"  ✅ Jupyter notebooks: {len(notebook_files)} found")
        
        details["notebooks"] = {
            "count": len(notebook_files),
            "files": [f.name for f in notebook_files]
        }
        
        return min(score, max_score), max_score, details
    
    def validate_deployment_readiness(self) -> Tuple[int, int, Dict[str, Any]]:
        """Validate deployment readiness."""
        score = 0
        max_score = 100
        details = {}
        
        # Check for deployment files
        deployment_files = [
            ("Deployment Config", "deployment_config.json"),
            ("Docker File", "Dockerfile"),
            ("Docker Compose", "docker-compose.yml"),
            ("Deployment Guide", "DEPLOYMENT_GUIDE.md")
        ]
        
        existing_deployment = []
        for name, file_path in deployment_files:
            if (self.repo_root / file_path).exists():
                existing_deployment.append(name)
                score += 15
                print(f"  ✅ {name}: Found")
        
        details["deployment_files"] = {
            "total": len(deployment_files),
            "existing": existing_deployment
        }
        
        # Check deployment directory
        deployment_dir = self.repo_root / "deployment"
        if deployment_dir.exists():
            deploy_files = list(deployment_dir.glob("*"))
            score += 20
            print(f"  ✅ Deployment directory: {len(deploy_files)} files")
            
            details["deployment_directory"] = {
                "exists": True,
                "file_count": len(deploy_files),
                "files": [f.name for f in deploy_files]
            }
        
        # Check for configuration management
        config_files = list(self.repo_root.glob("*config*.json")) + list(self.repo_root.glob("*config*.yaml"))
        if len(config_files) > 0:
            score += 20
            print(f"  ✅ Configuration files: {len(config_files)} found")
        
        details["configuration"] = {
            "config_files": len(config_files),
            "files": [f.name for f in config_files]
        }
        
        return min(score, max_score), max_score, details
    
    def _count_lines_of_code(self, python_files: List[Path]) -> int:
        """Count total lines of code."""
        total_lines = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Count non-empty, non-comment lines
                    code_lines = [line for line in lines 
                                 if line.strip() and not line.strip().startswith('#')]
                    total_lines += len(code_lines)
            except:
                pass
        return total_lines
    
    def _check_docstrings(self, python_files: List[Path]) -> bool:
        """Check if files have docstrings."""
        files_with_docstrings = 0
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                content = py_file.read_text(encoding='utf-8')
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
            except:
                pass
        return files_with_docstrings > len(python_files[:10]) * 0.5
    
    def _generate_recommendations(self):
        """Generate improvement recommendations."""
        recommendations = []
        
        for gate_name, gate_result in self.results["gates"].items():
            if gate_result["status"] in ["FAIL", "ERROR"]:
                if gate_name == "Code Structure":
                    recommendations.append("Improve code organization and add missing directories/files")
                elif gate_name == "Functionality":
                    recommendations.append("Fix failing demos and implement missing core functionality")
                elif gate_name == "Performance":
                    recommendations.append("Add performance optimizations and caching mechanisms")
                elif gate_name == "Security":
                    recommendations.append("Implement comprehensive input validation and security measures")
                elif gate_name == "Documentation":
                    recommendations.append("Add comprehensive documentation and improve README")
                elif gate_name == "Dependencies":
                    recommendations.append("Better dependency management and version pinning")
                elif gate_name == "Examples":
                    recommendations.append("Add more working examples and fix syntax errors")
                elif gate_name == "Deployment":
                    recommendations.append("Add deployment configurations and containerization")
        
        # General recommendations based on overall score
        overall_percentage = self.results["summary"]["percentage"]
        if overall_percentage < 50:
            recommendations.append("Focus on implementing core functionality and basic structure")
        elif overall_percentage < 70:
            recommendations.append("Improve documentation and add more comprehensive testing")
        elif overall_percentage < 85:
            recommendations.append("Focus on performance optimization and security hardening")
        else:
            recommendations.append("Excellent work! Consider advanced features and optimizations")
        
        self.results["recommendations"] = recommendations
    
    def _print_summary(self):
        """Print validation summary."""
        print(f"\n🏆 QUALITY GATES VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Total Score: {summary['total_score']}/{summary['max_score']} ({summary['percentage']:.1f}%)")
        print(f"Passed Gates: {summary['passed_gates']}/{summary['total_gates']}")
        
        print(f"\n📊 Gate Results:")
        for gate_name, gate_result in self.results["gates"].items():
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[gate_result["status"]]
            print(f"  {status_emoji} {gate_name:20}: {gate_result['score']:3d}/{gate_result['max_score']:3d} ({gate_result['percentage']:5.1f}%)")
        
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n🎯 Next Steps:")
        if summary['percentage'] >= 85:
            print("  • Project is production-ready!")
            print("  • Consider advanced optimizations")
            print("  • Add monitoring and analytics")
        elif summary['percentage'] >= 70:
            print("  • Address failing quality gates")
            print("  • Improve documentation coverage")
            print("  • Add comprehensive testing")
        else:
            print("  • Focus on core functionality")
            print("  • Implement basic structure")
            print("  • Add essential documentation")


def main():
    """Main validation entry point."""
    validator = QualityGateValidator()
    results = validator.run_all_validations()
    
    # Save results
    results_file = Path("quality_gates_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    overall_percentage = results["summary"]["percentage"]
    if overall_percentage >= 70:
        print("✅ Quality gates validation: PASSED")
        sys.exit(0)
    else:
        print("❌ Quality gates validation: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()