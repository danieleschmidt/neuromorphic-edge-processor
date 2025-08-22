"""Advanced security scanner for neuromorphic systems."""

import ast
import re
import hashlib
import logging
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security vulnerability levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityViolation:
    """Security violation data structure."""
    level: SecurityLevel
    category: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str


class SecurityScanner:
    """Advanced security scanner for neuromorphic computing systems."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.violations: List[SecurityViolation] = []
        self.scanned_files: Set[str] = set()
        
        # Dangerous function patterns
        self.dangerous_functions = {
            'eval': SecurityLevel.CRITICAL,
            'exec': SecurityLevel.CRITICAL,
            '__import__': SecurityLevel.HIGH,
            'compile': SecurityLevel.HIGH,
            'globals': SecurityLevel.MEDIUM,
            'locals': SecurityLevel.MEDIUM,
            'vars': SecurityLevel.MEDIUM,
            'dir': SecurityLevel.LOW,
            'getattr': SecurityLevel.MEDIUM,
            'setattr': SecurityLevel.MEDIUM,
            'hasattr': SecurityLevel.LOW,
            'delattr': SecurityLevel.MEDIUM
        }
        
        # Dangerous imports
        self.dangerous_imports = {
            'subprocess': SecurityLevel.HIGH,
            'os': SecurityLevel.MEDIUM,
            'sys': SecurityLevel.LOW,
            'pickle': SecurityLevel.HIGH,
            'marshal': SecurityLevel.HIGH,
            'shelve': SecurityLevel.MEDIUM,
            'tempfile': SecurityLevel.LOW,
            'shutil': SecurityLevel.MEDIUM,
            'socket': SecurityLevel.MEDIUM,
            'urllib': SecurityLevel.MEDIUM,
            'requests': SecurityLevel.LOW
        }
        
        # SQL injection patterns
        self.sql_patterns = [
            r"SELECT.*FROM.*WHERE.*=.*\+",
            r"INSERT.*INTO.*VALUES.*\+",
            r"UPDATE.*SET.*=.*\+",
            r"DELETE.*FROM.*WHERE.*=.*\+",
            r"DROP.*TABLE",
            r"CREATE.*TABLE",
            r"ALTER.*TABLE"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"/etc/passwd",
            r"/etc/shadow",
            r"C:\\Windows\\System32"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"system\(",
            r"popen\(",
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r"commands\.",
            r"shell=True"
        ]
    
    def scan_file(self, file_path: Path) -> List[SecurityViolation]:
        """Scan a single file for security vulnerabilities."""
        violations = []
        
        if file_path.suffix not in ['.py']:
            return violations
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Parse AST for Python files
            if file_path.suffix == '.py':
                violations.extend(self._scan_python_ast(file_path, content, lines))
            
            # Pattern-based scanning
            violations.extend(self._scan_patterns(file_path, content, lines))
            
            self.scanned_files.add(str(file_path))
            
        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")
        
        return violations
    
    def _scan_python_ast(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityViolation]:
        """Scan Python AST for security issues."""
        violations = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                # Check function calls
                if isinstance(node, ast.Call):
                    violations.extend(self._check_function_call(node, file_path, lines))
                
                # Check imports
                elif isinstance(node, ast.Import):
                    violations.extend(self._check_import(node, file_path, lines))
                
                elif isinstance(node, ast.ImportFrom):
                    violations.extend(self._check_import_from(node, file_path, lines))
                
                # Check string literals for injection patterns (support both ast.Str and ast.Constant)
                elif isinstance(node, (ast.Str, ast.Constant)) and isinstance(getattr(node, 's', getattr(node, 'value', None)), str):
                    violations.extend(self._check_string_literal(node, file_path, lines))
                
                # Check attribute access
                elif isinstance(node, ast.Attribute):
                    violations.extend(self._check_attribute_access(node, file_path, lines))
        
        except SyntaxError as e:
            violations.append(SecurityViolation(
                level=SecurityLevel.MEDIUM,
                category="syntax_error",
                description=f"Syntax error in Python file: {e}",
                file_path=str(file_path),
                line_number=getattr(e, 'lineno', 1),
                code_snippet="",
                recommendation="Fix syntax errors to enable proper security scanning"
            ))
        
        return violations
    
    def _check_function_call(self, node: ast.Call, file_path: Path, lines: List[str]) -> List[SecurityViolation]:
        """Check function calls for dangerous patterns."""
        violations = []
        
        # Get function name
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"
            else:
                func_name = node.func.attr
        
        if func_name and func_name in self.dangerous_functions:
            line_num = getattr(node, 'lineno', 1)
            code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
            
            violations.append(SecurityViolation(
                level=self.dangerous_functions[func_name],
                category="dangerous_function",
                description=f"Use of potentially dangerous function: {func_name}",
                file_path=str(file_path),
                line_number=line_num,
                code_snippet=code_snippet,
                recommendation=f"Avoid using {func_name} or implement proper input validation"
            ))
        
        return violations
    
    def _check_import(self, node: ast.Import, file_path: Path, lines: List[str]) -> List[SecurityViolation]:
        """Check import statements for dangerous modules."""
        violations = []
        
        for alias in node.names:
            module_name = alias.name.split('.')[0]  # Get base module name
            
            if module_name in self.dangerous_imports:
                line_num = getattr(node, 'lineno', 1)
                code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                
                violations.append(SecurityViolation(
                    level=self.dangerous_imports[module_name],
                    category="dangerous_import",
                    description=f"Import of potentially dangerous module: {module_name}",
                    file_path=str(file_path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    recommendation=f"Use {module_name} carefully with proper input validation"
                ))
        
        return violations
    
    def _check_import_from(self, node: ast.ImportFrom, file_path: Path, lines: List[str]) -> List[SecurityViolation]:
        """Check from-import statements for dangerous modules."""
        violations = []
        
        if node.module:
            module_name = node.module.split('.')[0]
            
            if module_name in self.dangerous_imports:
                line_num = getattr(node, 'lineno', 1)
                code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                
                violations.append(SecurityViolation(
                    level=self.dangerous_imports[module_name],
                    category="dangerous_import",
                    description=f"Import from potentially dangerous module: {module_name}",
                    file_path=str(file_path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    recommendation=f"Use {module_name} imports carefully with proper validation"
                ))
        
        return violations
    
    def _check_string_literal(self, node: Union[ast.Str, ast.Constant], file_path: Path, lines: List[str]) -> List[SecurityViolation]:
        """Check string literals for injection patterns."""
        violations = []
        
        # Handle both ast.Str (deprecated) and ast.Constant (new)
        if hasattr(node, 's'):
            string_value = node.s
        elif hasattr(node, 'value') and isinstance(node.value, str):
            string_value = node.value
        else:
            return violations
            
        if not isinstance(string_value, str):
            return violations
            
        string_value = string_value.lower()
        line_num = getattr(node, 'lineno', 1)
        code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
        
        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if re.search(pattern, string_value, re.IGNORECASE):
                violations.append(SecurityViolation(
                    level=SecurityLevel.HIGH,
                    category="sql_injection",
                    description=f"Potential SQL injection pattern detected",
                    file_path=str(file_path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    recommendation="Use parameterized queries instead of string concatenation"
                ))
        
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, string_value, re.IGNORECASE):
                violations.append(SecurityViolation(
                    level=SecurityLevel.MEDIUM,
                    category="path_traversal",
                    description=f"Potential path traversal pattern detected",
                    file_path=str(file_path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    recommendation="Validate and sanitize file paths"
                ))
        
        # Check for command injection
        for pattern in self.command_injection_patterns:
            if re.search(pattern, string_value, re.IGNORECASE):
                violations.append(SecurityViolation(
                    level=SecurityLevel.HIGH,
                    category="command_injection",
                    description=f"Potential command injection pattern detected",
                    file_path=str(file_path),
                    line_number=line_num,
                    code_snippet=code_snippet,
                    recommendation="Use subprocess with shell=False and validate inputs"
                ))
        
        return violations
    
    def _check_attribute_access(self, node: ast.Attribute, file_path: Path, lines: List[str]) -> List[SecurityViolation]:
        """Check attribute access for dangerous patterns."""
        violations = []
        
        # Check for dangerous attribute patterns like __class__, __bases__, etc.
        dangerous_attributes = ['__class__', '__bases__', '__subclasses__', '__globals__', '__dict__']
        
        if node.attr in dangerous_attributes:
            line_num = getattr(node, 'lineno', 1)
            code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
            
            violations.append(SecurityViolation(
                level=SecurityLevel.MEDIUM,
                category="dangerous_attribute",
                description=f"Access to potentially dangerous attribute: {node.attr}",
                file_path=str(file_path),
                line_number=line_num,
                code_snippet=code_snippet,
                recommendation="Avoid accessing internal Python attributes"
            ))
        
        return violations
    
    def _scan_patterns(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityViolation]:
        """Scan file content for pattern-based vulnerabilities."""
        violations = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', SecurityLevel.HIGH, "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', SecurityLevel.HIGH, "Hardcoded API key"),
            (r'secret[_-]?key\s*=\s*["\'][^"\']{16,}["\']', SecurityLevel.HIGH, "Hardcoded secret key"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', SecurityLevel.HIGH, "Hardcoded token"),
            (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', SecurityLevel.MEDIUM, "Possible base64 encoded secret")
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, level, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(SecurityViolation(
                        level=level,
                        category="hardcoded_secret",
                        description=description,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        recommendation="Use environment variables or secure configuration"
                    ))
        
        return violations
    
    def scan_directory(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan entire directory for security vulnerabilities."""
        if exclude_patterns is None:
            exclude_patterns = [
                '*.pyc', '__pycache__', '.git', '.pytest_cache', 
                '*.egg-info', 'build', 'dist', 'node_modules'
            ]
        
        all_violations = []
        file_count = 0
        
        for file_path in directory.rglob('*.py'):
            # Skip excluded patterns
            if any(file_path.match(pattern) for pattern in exclude_patterns):
                continue
            
            file_violations = self.scan_file(file_path)
            all_violations.extend(file_violations)
            file_count += 1
        
        # Categorize violations by severity
        violations_by_level = {
            SecurityLevel.CRITICAL: [],
            SecurityLevel.HIGH: [],
            SecurityLevel.MEDIUM: [],
            SecurityLevel.LOW: []
        }
        
        for violation in all_violations:
            violations_by_level[violation.level].append(violation)
        
        return {
            'total_files_scanned': file_count,
            'total_violations': len(all_violations),
            'violations_by_level': {
                level.value: len(violations) for level, violations in violations_by_level.items()
            },
            'detailed_violations': violations_by_level,
            'security_score': self._calculate_security_score(violations_by_level),
            'recommendations': self._generate_recommendations(violations_by_level)
        }
    
    def _calculate_security_score(self, violations_by_level: Dict[SecurityLevel, List[SecurityViolation]]) -> float:
        """Calculate security score based on violations."""
        weights = {
            SecurityLevel.CRITICAL: 10,
            SecurityLevel.HIGH: 5,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.LOW: 1
        }
        
        total_score = 100.0
        
        for level, violations in violations_by_level.items():
            penalty = len(violations) * weights[level]
            total_score -= penalty
        
        return max(0.0, total_score)
    
    def _generate_recommendations(self, violations_by_level: Dict[SecurityLevel, List[SecurityViolation]]) -> List[str]:
        """Generate security recommendations based on violations."""
        recommendations = []
        
        if violations_by_level[SecurityLevel.CRITICAL]:
            recommendations.append("CRITICAL: Remove all eval() and exec() calls immediately")
        
        if violations_by_level[SecurityLevel.HIGH]:
            recommendations.append("HIGH: Review and secure dangerous imports (subprocess, pickle, etc.)")
            recommendations.append("HIGH: Implement input validation for all external inputs")
        
        if violations_by_level[SecurityLevel.MEDIUM]:
            recommendations.append("MEDIUM: Use environment variables for configuration secrets")
            recommendations.append("MEDIUM: Implement proper error handling and logging")
        
        if violations_by_level[SecurityLevel.LOW]:
            recommendations.append("LOW: Review information disclosure through debugging functions")
        
        recommendations.extend([
            "Implement comprehensive input validation",
            "Use parameterized queries for database operations",
            "Sanitize all user inputs and file paths",
            "Enable security linting in CI/CD pipeline",
            "Regular security audits and penetration testing"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def generate_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate a formatted security report."""
        report = []
        report.append("=" * 60)
        report.append("NEUROMORPHIC SECURITY SCAN REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Files Scanned: {scan_results['total_files_scanned']}")
        report.append(f"Total Violations: {scan_results['total_violations']}")
        report.append(f"Security Score: {scan_results['security_score']:.1f}/100.0")
        report.append("")
        
        report.append("Violations by Severity:")
        for level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH, SecurityLevel.MEDIUM, SecurityLevel.LOW]:
            count = scan_results['violations_by_level'][level.value]
            if count > 0:
                report.append(f"  {level.value.upper()}: {count}")
        
        report.append("")
        report.append("Top Recommendations:")
        for i, rec in enumerate(scan_results['recommendations'], 1):
            report.append(f"  {i}. {rec}")
        
        return "\n".join(report)


def main():
    """Run security scanner on the neuromorphic codebase."""
    scanner = SecurityScanner()
    
    # Scan the source directory
    src_path = Path(__file__).parent.parent
    results = scanner.scan_directory(src_path)
    
    # Generate and print report
    report = scanner.generate_report(results)
    print(report)
    
    # Return exit code based on security score
    if results['security_score'] < 70:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())