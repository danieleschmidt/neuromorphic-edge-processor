"""Comprehensive validation system for neuromorphic computing operations."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import math


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"        # Fail on any validation error
    MODERATE = "moderate"    # Fail on critical errors, warn on others
    PERMISSIVE = "permissive"  # Only warn on validation errors


class ValidationSeverity(Enum):
    """Validation error severity."""
    CRITICAL = "critical"    # Must be fixed
    HIGH = "high"           # Should be fixed
    MEDIUM = "medium"       # Could be fixed
    LOW = "low"             # Informational


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    overall_passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    summary: Dict[str, int]  # Count by severity
    
    def get_failures(self, min_severity: ValidationSeverity = ValidationSeverity.LOW) -> List[ValidationResult]:
        """Get failed validations above minimum severity."""
        return [
            result for result in self.results
            if not result.passed and self._severity_rank(result.severity) >= self._severity_rank(min_severity)
        ]
    
    def _severity_rank(self, severity: ValidationSeverity) -> int:
        """Get numeric rank for severity."""
        ranks = {
            ValidationSeverity.LOW: 1,
            ValidationSeverity.MEDIUM: 2,
            ValidationSeverity.HIGH: 3,
            ValidationSeverity.CRITICAL: 4
        }
        return ranks[severity]


class TensorValidator:
    """Validator for tensor data."""
    
    @staticmethod
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> ValidationResult:
        """Validate tensor shape constraints."""
        
        if expected_shape is not None and tuple(tensor.shape) != expected_shape:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}",
                suggestions=["Check input preprocessing", "Verify model input requirements"]
            )
        
        if min_dims is not None and tensor.dim() < min_dims:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message=f"Insufficient dimensions: expected at least {min_dims}, got {tensor.dim()}",
                suggestions=["Add missing dimensions with unsqueeze()", "Check data format"]
            )
        
        if max_dims is not None and tensor.dim() > max_dims:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message=f"Too many dimensions: expected at most {max_dims}, got {tensor.dim()}",
                suggestions=["Remove extra dimensions with squeeze()", "Reshape tensor"]
            )
        
        if min_size is not None and tensor.numel() < min_size:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Tensor too small: expected at least {min_size} elements, got {tensor.numel()}",
                suggestions=["Check input data generation", "Verify batch size"]
            )
        
        if max_size is not None and tensor.numel() > max_size:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message=f"Tensor too large: expected at most {max_size} elements, got {tensor.numel()}",
                details={"memory_estimate_mb": tensor.numel() * tensor.element_size() / 1024**2},
                suggestions=["Reduce batch size", "Use tensor chunking", "Consider memory optimization"]
            )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Shape validation passed"
        )
    
    @staticmethod
    def validate_dtype(
        tensor: torch.Tensor,
        allowed_dtypes: Optional[List[torch.dtype]] = None,
        expected_dtype: Optional[torch.dtype] = None
    ) -> ValidationResult:
        """Validate tensor data type."""
        
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message=f"Data type mismatch: expected {expected_dtype}, got {tensor.dtype}",
                suggestions=[f"Convert with .to({expected_dtype})", "Check data preprocessing"]
            )
        
        if allowed_dtypes is not None and tensor.dtype not in allowed_dtypes:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Data type not allowed: {tensor.dtype} not in {allowed_dtypes}",
                suggestions=[f"Convert to one of: {allowed_dtypes}"]
            )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Data type validation passed"
        )
    
    @staticmethod
    def validate_values(
        tensor: torch.Tensor,
        check_finite: bool = True,
        check_range: Optional[Tuple[float, float]] = None,
        allow_zero: bool = True,
        check_sparsity: Optional[Tuple[float, float]] = None  # (min_sparsity, max_sparsity)
    ) -> ValidationResult:
        """Validate tensor values."""
        
        # Check for NaN and infinite values
        if check_finite:
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Tensor contains {nan_count} NaN values",
                    details={"nan_percentage": nan_count / tensor.numel() * 100},
                    suggestions=["Check data preprocessing", "Use torch.nan_to_num()", "Debug computation pipeline"]
                )
            
            if torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Tensor contains {inf_count} infinite values",
                    details={"inf_percentage": inf_count / tensor.numel() * 100},
                    suggestions=["Check for division by zero", "Add gradient clipping", "Use numerical stabilization"]
                )
        
        # Check value range
        if check_range is not None:
            min_val, max_val = check_range
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()
            
            if tensor_min < min_val or tensor_max > max_val:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Values out of range: [{tensor_min:.6f}, {tensor_max:.6f}] not in [{min_val}, {max_val}]",
                    details={"actual_range": [tensor_min, tensor_max], "expected_range": [min_val, max_val]},
                    suggestions=["Apply value clipping", "Check normalization", "Verify input scaling"]
                )
        
        # Check for zeros if not allowed
        if not allow_zero:
            zero_count = (tensor == 0).sum().item()
            if zero_count > 0:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Tensor contains {zero_count} zero values (not allowed)",
                    details={"zero_percentage": zero_count / tensor.numel() * 100},
                    suggestions=["Replace zeros with small epsilon", "Check data generation"]
                )
        
        # Check sparsity
        if check_sparsity is not None:
            min_sparsity, max_sparsity = check_sparsity
            zero_count = (tensor == 0).sum().item()
            sparsity = zero_count / tensor.numel()
            
            if sparsity < min_sparsity or sparsity > max_sparsity:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Sparsity {sparsity:.3f} not in range [{min_sparsity}, {max_sparsity}]",
                    details={"actual_sparsity": sparsity, "expected_range": [min_sparsity, max_sparsity]},
                    suggestions=["Adjust sparsity generation", "Check pruning parameters"]
                )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Value validation passed"
        )
    
    @staticmethod
    def validate_distribution(
        tensor: torch.Tensor,
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None,
        tolerance: float = 0.1
    ) -> ValidationResult:
        """Validate tensor statistical distribution."""
        
        mean = tensor.mean().item()
        std = tensor.std().item()
        
        issues = []
        
        if expected_mean is not None:
            mean_diff = abs(mean - expected_mean)
            if mean_diff > tolerance:
                issues.append(f"Mean {mean:.6f} differs from expected {expected_mean:.6f} by {mean_diff:.6f}")
        
        if expected_std is not None:
            std_diff = abs(std - expected_std)
            if std_diff > tolerance:
                issues.append(f"Std {std:.6f} differs from expected {expected_std:.6f} by {std_diff:.6f}")
        
        if issues:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.LOW,
                message=f"Distribution validation failed: {'; '.join(issues)}",
                details={"actual_mean": mean, "actual_std": std},
                suggestions=["Check normalization", "Verify data preprocessing", "Adjust distribution parameters"]
            )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Distribution validation passed",
            details={"mean": mean, "std": std}
        )


class SpikeValidator:
    """Validator for spike data."""
    
    @staticmethod
    def validate_spike_train(
        spikes: torch.Tensor,
        max_spike_rate: Optional[float] = None,
        min_spike_rate: Optional[float] = None,
        check_binary: bool = True
    ) -> ValidationResult:
        """Validate spike train properties."""
        
        # Check if spikes are binary
        if check_binary:
            unique_values = torch.unique(spikes)
            if not torch.allclose(unique_values, torch.tensor([0., 1.], device=spikes.device)[:len(unique_values)]):
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.HIGH,
                    message=f"Spike train not binary: found values {unique_values.tolist()}",
                    suggestions=["Apply thresholding", "Check spike generation", "Use binary encoding"]
                )
        
        # Calculate spike rate
        if spikes.dim() >= 2:
            # Assume last dimension is time
            spike_count = spikes.sum(dim=-1)
            time_steps = spikes.shape[-1]
            spike_rates = spike_count / time_steps  # Rate per time step
            
            mean_rate = spike_rates.mean().item()
            max_rate = spike_rates.max().item()
            
            if max_spike_rate is not None and max_rate > max_spike_rate:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Spike rate too high: {max_rate:.6f} > {max_spike_rate}",
                    details={"mean_rate": mean_rate, "max_rate": max_rate},
                    suggestions=["Reduce spike probability", "Apply rate limiting", "Check input scaling"]
                )
            
            if min_spike_rate is not None and mean_rate < min_spike_rate:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Spike rate too low: {mean_rate:.6f} < {min_spike_rate}",
                    details={"mean_rate": mean_rate, "max_rate": max_rate},
                    suggestions=["Increase spike probability", "Check input sensitivity", "Verify threshold settings"]
                )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Spike train validation passed"
        )
    
    @staticmethod
    def validate_temporal_patterns(
        spikes: torch.Tensor,
        min_pattern_length: Optional[int] = None,
        max_burst_length: Optional[int] = None
    ) -> ValidationResult:
        """Validate temporal spike patterns."""
        
        if spikes.dim() < 2:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message="Spike tensor must have at least 2 dimensions for temporal validation",
                suggestions=["Add time dimension", "Check data format"]
            )
        
        # Analyze patterns (simplified)
        issues = []
        
        if min_pattern_length is not None:
            # Check for minimum pattern length
            for neuron_idx in range(spikes.shape[0]):
                neuron_spikes = spikes[neuron_idx]
                spike_positions = torch.nonzero(neuron_spikes).squeeze()
                
                if len(spike_positions) > 1:
                    intervals = torch.diff(spike_positions)
                    min_interval = intervals.min().item()
                    
                    if min_interval < min_pattern_length:
                        issues.append(f"Neuron {neuron_idx}: minimum interval {min_interval} < {min_pattern_length}")
        
        if max_burst_length is not None:
            # Check for maximum burst length
            for neuron_idx in range(spikes.shape[0]):
                neuron_spikes = spikes[neuron_idx]
                
                # Find consecutive spike runs
                spike_runs = []
                current_run = 0
                
                for spike in neuron_spikes:
                    if spike > 0:
                        current_run += 1
                    else:
                        if current_run > 0:
                            spike_runs.append(current_run)
                            current_run = 0
                
                if current_run > 0:
                    spike_runs.append(current_run)
                
                if spike_runs and max(spike_runs) > max_burst_length:
                    issues.append(f"Neuron {neuron_idx}: burst length {max(spike_runs)} > {max_burst_length}")
        
        if issues:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Temporal pattern validation failed: {'; '.join(issues[:3])}{'...' if len(issues) > 3 else ''}",
                details={"issue_count": len(issues)},
                suggestions=["Adjust pattern generation", "Check temporal constraints", "Verify spike timing"]
            )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Temporal pattern validation passed"
        )


class ModelValidator:
    """Validator for neuromorphic models."""
    
    @staticmethod
    def validate_architecture(
        model: torch.nn.Module,
        expected_input_shape: Optional[Tuple[int, ...]] = None,
        expected_output_shape: Optional[Tuple[int, ...]] = None,
        max_parameters: Optional[int] = None
    ) -> ValidationResult:
        """Validate model architecture."""
        
        parameter_count = sum(p.numel() for p in model.parameters())
        
        if max_parameters is not None and parameter_count > max_parameters:
            return ValidationResult(
                passed=False,
                severity=ValidationSeverity.HIGH,
                message=f"Model too large: {parameter_count} parameters > {max_parameters}",
                details={"parameter_count": parameter_count, "memory_estimate_mb": parameter_count * 4 / 1024**2},
                suggestions=["Reduce model size", "Use parameter sharing", "Apply pruning"]
            )
        
        # Test forward pass if shapes provided
        if expected_input_shape is not None:
            try:
                dummy_input = torch.randn(1, *expected_input_shape)
                with torch.no_grad():
                    output = model(dummy_input)
                
                if expected_output_shape is not None:
                    actual_output_shape = tuple(output.shape[1:])  # Remove batch dimension
                    if actual_output_shape != expected_output_shape:
                        return ValidationResult(
                            passed=False,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Output shape mismatch: expected {expected_output_shape}, got {actual_output_shape}",
                            suggestions=["Check model architecture", "Verify layer configurations"]
                        )
            
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Forward pass failed: {str(e)}",
                    suggestions=["Check model definition", "Verify layer compatibility", "Debug architecture"]
                )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Architecture validation passed",
            details={"parameter_count": parameter_count}
        )
    
    @staticmethod
    def validate_weights(
        model: torch.nn.Module,
        check_initialization: bool = True,
        check_gradients: bool = True
    ) -> ValidationResult:
        """Validate model weights and gradients."""
        
        issues = []
        
        for name, param in model.named_parameters():
            # Check for NaN/Inf in weights
            if torch.isnan(param).any():
                issues.append(f"NaN values in parameter {name}")
            
            if torch.isinf(param).any():
                issues.append(f"Infinite values in parameter {name}")
            
            # Check initialization
            if check_initialization:
                param_std = param.std().item()
                if param_std < 1e-6:
                    issues.append(f"Parameter {name} appears uninitialized (std={param_std:.2e})")
                elif param_std > 10:
                    issues.append(f"Parameter {name} has large values (std={param_std:.2f})")
            
            # Check gradients if available
            if check_gradients and param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues.append(f"NaN gradients in parameter {name}")
                
                if torch.isinf(param.grad).any():
                    issues.append(f"Infinite gradients in parameter {name}")
                
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                    issues.append(f"Large gradients in parameter {name} (norm={grad_norm:.2f})")
        
        if issues:
            severity = ValidationSeverity.CRITICAL if any("NaN" in issue or "Infinite" in issue for issue in issues) else ValidationSeverity.MEDIUM
            
            return ValidationResult(
                passed=False,
                severity=severity,
                message=f"Weight validation failed: {'; '.join(issues[:3])}{'...' if len(issues) > 3 else ''}",
                details={"issue_count": len(issues)},
                suggestions=["Check weight initialization", "Apply gradient clipping", "Debug training loop"]
            )
        
        return ValidationResult(
            passed=True,
            severity=ValidationSeverity.LOW,
            message="Weight validation passed"
        )


class ComprehensiveValidator:
    """Main validator orchestrating all validation checks."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        """Initialize comprehensive validator.
        
        Args:
            level: Validation strictness level
        """
        self.level = level
        self.tensor_validator = TensorValidator()
        self.spike_validator = SpikeValidator()
        self.model_validator = ModelValidator()
    
    def validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        **validation_kwargs
    ) -> List[ValidationResult]:
        """Validate tensor with all applicable checks."""
        
        results = []
        
        # Shape validation
        if any(k in validation_kwargs for k in ['expected_shape', 'min_dims', 'max_dims', 'min_size', 'max_size']):
            shape_kwargs = {k: v for k, v in validation_kwargs.items() 
                          if k in ['expected_shape', 'min_dims', 'max_dims', 'min_size', 'max_size']}
            results.append(self.tensor_validator.validate_shape(tensor, **shape_kwargs))
        
        # Data type validation
        if any(k in validation_kwargs for k in ['allowed_dtypes', 'expected_dtype']):
            dtype_kwargs = {k: v for k, v in validation_kwargs.items() 
                          if k in ['allowed_dtypes', 'expected_dtype']}
            results.append(self.tensor_validator.validate_dtype(tensor, **dtype_kwargs))
        
        # Value validation
        if any(k in validation_kwargs for k in ['check_finite', 'check_range', 'allow_zero', 'check_sparsity']):
            value_kwargs = {k: v for k, v in validation_kwargs.items() 
                          if k in ['check_finite', 'check_range', 'allow_zero', 'check_sparsity']}
            results.append(self.tensor_validator.validate_values(tensor, **value_kwargs))
        
        # Distribution validation
        if any(k in validation_kwargs for k in ['expected_mean', 'expected_std', 'tolerance']):
            dist_kwargs = {k: v for k, v in validation_kwargs.items() 
                         if k in ['expected_mean', 'expected_std', 'tolerance']}
            results.append(self.tensor_validator.validate_distribution(tensor, **dist_kwargs))
        
        return results
    
    def validate_spikes(
        self,
        spikes: torch.Tensor,
        name: str = "spikes",
        **validation_kwargs
    ) -> List[ValidationResult]:
        """Validate spike data with all applicable checks."""
        
        results = []
        
        # Basic tensor validation
        results.extend(self.validate_tensor(spikes, name, check_finite=True, check_range=(0, 1)))
        
        # Spike-specific validation
        train_kwargs = {k: v for k, v in validation_kwargs.items() 
                       if k in ['max_spike_rate', 'min_spike_rate', 'check_binary']}
        if train_kwargs:
            results.append(self.spike_validator.validate_spike_train(spikes, **train_kwargs))
        
        # Temporal pattern validation
        pattern_kwargs = {k: v for k, v in validation_kwargs.items() 
                         if k in ['min_pattern_length', 'max_burst_length']}
        if pattern_kwargs:
            results.append(self.spike_validator.validate_temporal_patterns(spikes, **pattern_kwargs))
        
        return results
    
    def validate_model(
        self,
        model: torch.nn.Module,
        name: str = "model",
        **validation_kwargs
    ) -> List[ValidationResult]:
        """Validate model with all applicable checks."""
        
        results = []
        
        # Architecture validation
        arch_kwargs = {k: v for k, v in validation_kwargs.items() 
                      if k in ['expected_input_shape', 'expected_output_shape', 'max_parameters']}
        if arch_kwargs:
            results.append(self.model_validator.validate_architecture(model, **arch_kwargs))
        
        # Weight validation
        weight_kwargs = {k: v for k, v in validation_kwargs.items() 
                        if k in ['check_initialization', 'check_gradients']}
        if weight_kwargs or not validation_kwargs:  # Default weight validation
            results.append(self.model_validator.validate_weights(model, **weight_kwargs))
        
        return results
    
    def create_report(self, results: List[ValidationResult]) -> ValidationReport:
        """Create comprehensive validation report."""
        
        passed_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]
        
        # Count by severity
        severity_counts = {severity.value: 0 for severity in ValidationSeverity}
        for result in failed_results:
            severity_counts[result.severity.value] += 1
        
        # Determine overall pass/fail based on level
        overall_passed = True
        if self.level == ValidationLevel.STRICT:
            overall_passed = len(failed_results) == 0
        elif self.level == ValidationLevel.MODERATE:
            critical_failures = [r for r in failed_results if r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]
            overall_passed = len(critical_failures) == 0
        # PERMISSIVE always passes (warnings only)
        
        return ValidationReport(
            overall_passed=overall_passed,
            total_checks=len(results),
            passed_checks=len(passed_results),
            failed_checks=len(failed_results),
            results=results,
            summary=severity_counts
        )
    
    def validate_and_report(
        self,
        data: Union[torch.Tensor, torch.nn.Module],
        data_type: str,
        name: str = "data",
        **validation_kwargs
    ) -> ValidationReport:
        """Validate data and return comprehensive report."""
        
        if data_type == "tensor":
            results = self.validate_tensor(data, name, **validation_kwargs)
        elif data_type == "spikes":
            results = self.validate_spikes(data, name, **validation_kwargs)
        elif data_type == "model":
            results = self.validate_model(data, name, **validation_kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        report = self.create_report(results)
        
        # Handle warnings/errors based on level
        if not report.overall_passed:
            failures = report.get_failures(ValidationSeverity.MEDIUM)
            message = f"Validation failed for {name}: {len(failures)} issues found"
            
            if self.level == ValidationLevel.STRICT:
                raise ValueError(message)
            elif self.level == ValidationLevel.MODERATE:
                critical_failures = report.get_failures(ValidationSeverity.HIGH)
                if critical_failures:
                    raise ValueError(message)
                else:
                    warnings.warn(message, UserWarning)
            else:  # PERMISSIVE
                warnings.warn(message, UserWarning)
        
        return report


# Global validator instance
_global_validator = ComprehensiveValidator()


def validate_tensor(tensor: torch.Tensor, **kwargs) -> ValidationReport:
    """Validate tensor using global validator."""
    return _global_validator.validate_and_report(tensor, "tensor", **kwargs)


def validate_spikes(spikes: torch.Tensor, **kwargs) -> ValidationReport:
    """Validate spike data using global validator."""
    return _global_validator.validate_and_report(spikes, "spikes", **kwargs)


def validate_model(model: torch.nn.Module, **kwargs) -> ValidationReport:
    """Validate model using global validator."""
    return _global_validator.validate_and_report(model, "model", **kwargs)


def set_validation_level(level: ValidationLevel):
    """Set global validation level."""
    global _global_validator
    _global_validator.level = level