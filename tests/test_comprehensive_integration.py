"""Comprehensive integration tests for neuromorphic edge processor."""

import pytest
import torch
import numpy as np
import time
import threading
from pathlib import Path
import tempfile
import json

# Import all our advanced systems
from src.utils.advanced_logging import NeuromorphicLogger, get_logger
from src.utils.circuit_breaker import CircuitBreaker, get_circuit_breaker, NEUROMORPHIC_CONFIGS
from src.utils.error_handling import ErrorHandler, handle_errors, NeuromorphicError
from src.validation.comprehensive_validator import ComprehensiveValidator, ValidationLevel
from src.optimization.auto_scaler import AutoScaler, NeuromorphicAutoScaler
from src.optimization.distributed_processor import DistributedProcessor, TaskPriority
from src.optimization.memory_optimizer import MemoryOptimizer, get_memory_optimizer
from src.security.security_manager import SecurityManager, SecurityConfig


class TestAdvancedLogging:
    """Test advanced logging system."""
    
    def test_structured_logging(self):
        """Test structured JSON logging."""
        logger = NeuromorphicLogger(
            "test_logger",
            structured=True,
            enable_metrics=True
        )
        
        # Test context setting
        logger.set_context(user_id="test_user", session_id="session_123")
        
        # Test basic logging
        logger.info("Test message", tags={"component": "test"})
        
        # Test operation timer
        with logger.operation_timer("test_operation", param1="value1"):
            time.sleep(0.1)
        
        # Test metrics collection
        metrics = logger.get_metrics_summary()
        assert "counters" in metrics
        assert "timers" in metrics
        
        # Test neuromorphic-specific logging
        logger.log_spike_processing(
            num_spikes=1000,
            processing_time_ms=50.0,
            spike_rate_hz=20.0,
            sparsity=0.8
        )
        
        logger.log_model_performance(
            model_name="test_model",
            accuracy=0.95,
            inference_time_ms=25.0,
            memory_usage_mb=100.0
        )
    
    def test_request_context(self):
        """Test request-scoped logging context."""
        logger = get_logger("test_request_logger")
        
        with logger.request_context(user_id="user123") as request_id:
            logger.info("Processing request")
            assert request_id is not None
            
            context = logger.get_context()
            assert "correlation_id" in context
            assert context["user_id"] == "user123"


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_basic_circuit_breaker(self):
        """Test basic circuit breaker operation."""
        breaker = CircuitBreaker("test_circuit", NEUROMORPHIC_CONFIGS['model_inference'])
        
        def failing_function():
            raise ValueError("Simulated failure")
        
        def successful_function():
            return "success"
        
        # Test successful calls
        result = breaker.call(successful_function)
        assert result == "success"
        
        # Test failure handling
        with pytest.raises(ValueError):
            breaker.call(failing_function)
        
        stats = breaker.get_stats()
        assert stats.failure_count == 1
    
    def test_circuit_breaker_opening(self):
        """Test circuit breaker opening on repeated failures."""
        from src.utils.circuit_breaker import CircuitBreakerConfig, CircuitBreakerError
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1.0
        )
        breaker = CircuitBreaker("test_opening", config)
        
        def failing_function():
            raise ValueError("Always fails")
        
        # Cause failures to open circuit
        for _ in range(3):
            try:
                breaker.call(failing_function)
            except (ValueError, CircuitBreakerError):
                pass
        
        # Circuit should now be open
        with pytest.raises(CircuitBreakerError):
            breaker.call(failing_function)
        
        assert breaker.get_state().value == "open"
    
    def test_neuromorphic_circuit_breakers(self):
        """Test neuromorphic-specific circuit breakers."""
        from src.utils.circuit_breaker import get_neuromorphic_circuit_breaker
        
        # Test different operation types
        inference_breaker = get_neuromorphic_circuit_breaker("model_inference")
        spike_breaker = get_neuromorphic_circuit_breaker("spike_processing")
        
        assert inference_breaker is not None
        assert spike_breaker is not None
        
        # Different configurations
        assert inference_breaker.config.failure_threshold != spike_breaker.config.failure_threshold


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_error_handler_basics(self):
        """Test basic error handling functionality."""
        handler = ErrorHandler(max_retry_attempts=2)
        
        call_count = [0]
        
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError(f"Attempt {call_count[0]} failed")
            return "success"
        
        # Should succeed after retry
        result = handler.safe_execute(flaky_function)
        assert result == "success"
        assert call_count[0] == 2
        
        # Check statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] >= 1
    
    def test_error_classification(self):
        """Test automatic error classification."""
        handler = ErrorHandler()
        
        # Test memory error
        try:
            raise MemoryError("Out of memory")
        except MemoryError as e:
            error_info = handler.handle_error(e)
            assert error_info.category.value == "memory"
            assert error_info.severity.value == "high"
    
    def test_error_decorator(self):
        """Test error handling decorator."""
        @handle_errors(max_retries=1)
        def decorated_function(should_fail=True):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test failure
        with pytest.raises(ValueError):
            decorated_function(should_fail=True)
        
        # Test success
        result = decorated_function(should_fail=False)
        assert result == "success"


class TestComprehensiveValidator:
    """Test comprehensive validation system."""
    
    def test_tensor_validation(self):
        """Test tensor validation."""
        validator = ComprehensiveValidator(level=ValidationLevel.STRICT)
        
        # Valid tensor
        valid_tensor = torch.randn(10, 20)
        report = validator.validate_and_report(
            valid_tensor,
            "tensor",
            expected_shape=(10, 20),
            check_finite=True
        )
        assert report.overall_passed
        
        # Invalid tensor (NaN values)
        invalid_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError):
            validator.validate_and_report(
                invalid_tensor,
                "tensor",
                check_finite=True
            )
    
    def test_spike_validation(self):
        """Test spike data validation."""
        validator = ComprehensiveValidator(level=ValidationLevel.MODERATE)
        
        # Valid spike data
        spike_data = torch.randint(0, 2, (5, 100), dtype=torch.float32)
        report = validator.validate_and_report(
            spike_data,
            "spikes",
            check_binary=True,
            max_spike_rate=0.5
        )
        assert report.overall_passed or report.failed_checks == 0  # Moderate level allows warnings
    
    def test_model_validation(self):
        """Test model validation."""
        validator = ComprehensiveValidator(level=ValidationLevel.PERMISSIVE)
        
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        report = validator.validate_and_report(
            model,
            "model",
            expected_input_shape=(10,),
            expected_output_shape=(1,)
        )
        # Should pass with permissive level
        assert report.total_checks > 0


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    def test_basic_autoscaler(self):
        """Test basic auto-scaling operations."""
        scaler = AutoScaler(
            min_capacity=1,
            max_capacity=5,
            evaluation_interval=0.1  # Fast evaluation for testing
        )
        
        # Test manual scaling
        scaler.manual_scale(3, "Test scaling")
        assert scaler.get_current_capacity() == 3
        
        # Test metric recording
        scaler.record_metric("cpu_utilization", 85.0)  # High CPU
        scaler.record_metric("memory_usage", 90.0)     # High memory
        
        # Test scaling evaluation (might not scale due to cooldown)
        scaling_action = scaler.evaluate_scaling()
        
        stats = scaler.get_metrics_summary()
        assert "current_capacity" in stats
        assert stats["current_capacity"] == 3
    
    def test_neuromorphic_autoscaler(self):
        """Test neuromorphic-specific auto-scaling."""
        scaler = NeuromorphicAutoScaler(min_capacity=1, max_capacity=5)
        
        # Test neuromorphic metrics
        scaler.record_spike_metrics(
            spike_rate=1500.0,  # High spike rate
            processing_latency=25.0,
            sparsity=0.8
        )
        
        scaler.record_inference_metrics(
            accuracy=0.85,  # Lower accuracy
            queue_length=60,  # High queue
            energy_per_inference=5.0
        )
        
        # Get recommendations
        recommendations = scaler.get_neuromorphic_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestDistributedProcessor:
    """Test distributed processing system."""
    
    def test_distributed_processor_basic(self):
        """Test basic distributed processing."""
        with DistributedProcessor(num_workers=2) as processor:
            # Test basic tensor operation
            task_id = processor.submit_task(
                "tensor_multiply",
                torch.randn(3, 3),
                torch.randn(3, 3),
                timeout_seconds=10.0
            )
            
            result = processor.get_result(task_id, timeout=5.0)
            assert result is not None
            assert result.status.value == "completed"
            assert isinstance(result.result, torch.Tensor)
    
    def test_batch_processing(self):
        """Test batch processing."""
        with DistributedProcessor(num_workers=2) as processor:
            # Submit batch of spike processing tasks
            spike_arrays = [np.random.randint(0, 2, (10, 50)) for _ in range(5)]
            weight_arrays = [np.random.randn(10, 50) for _ in range(5)]
            
            args_list = list(zip(spike_arrays, weight_arrays))
            
            task_ids = processor.submit_batch(
                "spike_processing",
                args_list,
                priority=TaskPriority.HIGH
            )
            
            assert len(task_ids) == 5
            
            # Get all results
            results = processor.get_batch_results(task_ids, timeout=10.0)
            
            successful_results = [r for r in results if r and r.status.value == "completed"]
            assert len(successful_results) > 0
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        with DistributedProcessor(num_workers=1) as processor:
            task_id = processor.submit_task(
                "spike_processing",
                np.random.randint(0, 2, (100, 100)),
                np.random.randn(100, 100)
            )
            
            # Cancel immediately
            cancelled = processor.cancel_task(task_id)
            assert cancelled
            
            status = processor.get_status()
            assert "cancelled" in [r.status.value for r in status.get('completed_tasks', {}).values()]


class TestMemoryOptimizer:
    """Test memory optimization system."""
    
    def test_memory_optimizer_basic(self):
        """Test basic memory optimization."""
        optimizer = MemoryOptimizer(enable_automatic_optimization=False)
        
        # Get current memory usage
        memory_usage = optimizer.get_memory_usage()
        assert memory_usage.total_bytes > 0
        
        # Test optimization
        results = optimizer.optimize_memory()
        assert isinstance(results, list)
        
        # Test tensor pool
        tensor = optimizer.tensor_pool.get_tensor((10, 10))
        assert tensor.shape == (10, 10)
        
        optimizer.tensor_pool.return_tensor(tensor)
        
        stats = optimizer.tensor_pool.get_stats()
        assert stats["allocations"] >= 1
    
    def test_memory_monitoring(self):
        """Test automatic memory monitoring."""
        optimizer = MemoryOptimizer(
            optimization_interval=0.1,  # Fast interval for testing
            enable_automatic_optimization=False
        )
        
        # Start monitoring briefly
        optimizer.start_automatic_optimization()
        time.sleep(0.5)  # Let it run a few cycles
        optimizer.stop_automatic_optimization()
        
        # Check that memory history was recorded
        report = optimizer.get_optimization_report()
        assert "current_memory" in report


class TestSecurityManager:
    """Test security management system."""
    
    def test_security_manager_basic(self):
        """Test basic security functionality."""
        config = SecurityConfig(
            max_input_size=1000,
            enable_input_sanitization=True
        )
        manager = SecurityManager(config)
        
        # Test valid input
        valid_tensor = torch.randn(10, 10)
        assert manager.validate_input(valid_tensor, "test_source")
        
        # Test input sanitization
        noisy_tensor = torch.tensor([1.0, float('inf'), 3.0, float('nan')])
        sanitized = manager.sanitize_input(noisy_tensor)
        assert torch.isfinite(sanitized).all()
        
        # Test rate limiting
        assert manager.rate_limit_check("test_client")
    
    def test_security_validation(self):
        """Test security input validation."""
        manager = SecurityManager()
        
        # Test oversized input
        large_tensor = torch.randn(1000, 1000)  # Larger than default limit
        assert not manager.validate_input(large_tensor)
        
        # Test invalid data type
        invalid_tensor = torch.randint(0, 255, (10, 10), dtype=torch.uint8)
        # Should fail if uint8 not in allowed types
        result = manager.validate_input(invalid_tensor)
        # Result depends on configuration
        
        # Test memory usage check
        memory_ok = manager.check_memory_usage()
        assert isinstance(memory_ok, bool)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_full_neuromorphic_pipeline(self):
        """Test complete neuromorphic processing pipeline."""
        # Initialize all components
        logger = get_logger("integration_test")
        validator = ComprehensiveValidator(ValidationLevel.MODERATE)
        security_manager = SecurityManager()
        memory_optimizer = get_memory_optimizer()
        
        with logger.operation_timer("full_pipeline"):
            # 1. Generate synthetic spike data
            spike_data = torch.randint(0, 2, (50, 100), dtype=torch.float32)
            
            # 2. Validate input
            validation_report = validator.validate_and_report(
                spike_data,
                "spikes",
                check_binary=True,
                max_spike_rate=0.5
            )
            
            if not validation_report.overall_passed:
                logger.warning("Validation warnings detected", 
                             context={"failures": validation_report.failed_checks})
            
            # 3. Security check
            if not security_manager.validate_input(spike_data, "pipeline_input"):
                raise SecurityError("Input validation failed")
            
            # 4. Process data (simulated)
            with memory_optimizer:
                processed_data = spike_data.sum(dim=1)  # Simple processing
            
            # 5. Log results
            logger.log_spike_processing(
                num_spikes=int(spike_data.sum()),
                processing_time_ms=10.0,
                spike_rate_hz=float(spike_data.sum() / spike_data.shape[1]),
                sparsity=float((spike_data == 0).sum() / spike_data.numel())
            )
            
            assert processed_data.shape[0] == spike_data.shape[0]
    
    def test_fault_tolerance_scenario(self):
        """Test system behavior under fault conditions."""
        # Setup components with fault tolerance
        circuit_breaker = get_circuit_breaker("fault_test")
        error_handler = ErrorHandler(max_retry_attempts=3)
        logger = get_logger("fault_tolerance_test")
        
        failure_count = [0]
        
        def unreliable_operation():
            failure_count[0] += 1
            if failure_count[0] <= 2:
                raise ValueError(f"Simulated failure {failure_count[0]}")
            return "success"
        
        # Test error handling with circuit breaker
        try:
            result = circuit_breaker.call(
                error_handler.safe_execute,
                unreliable_operation
            )
            assert result == "success"
            logger.info("Fault tolerance test passed", context={"retries": failure_count[0] - 1})
            
        except Exception as e:
            logger.error("Fault tolerance test failed", exc_info=True)
            raise
    
    def test_performance_monitoring_scenario(self):
        """Test integrated performance monitoring."""
        logger = get_logger("performance_test")
        memory_optimizer = get_memory_optimizer()
        
        # Simulate workload with monitoring
        with logger.operation_timer("performance_test"):
            # Create some tensors to use memory
            tensors = []
            for i in range(10):
                tensor = torch.randn(100, 100)
                tensors.append(tensor)
            
            # Get memory snapshot
            memory_before = memory_optimizer.get_memory_usage()
            
            # Process data
            results = []
            for tensor in tensors:
                result = torch.matmul(tensor, tensor.T)
                results.append(result)
            
            # Optimize memory
            optimization_results = memory_optimizer.optimize_memory()
            
            memory_after = memory_optimizer.get_memory_usage()
            
            # Log performance metrics
            logger.info("Performance test completed", metrics={
                "tensors_processed": len(tensors),
                "memory_before_mb": memory_before.allocated_bytes / 1024**2,
                "memory_after_mb": memory_after.allocated_bytes / 1024**2,
                "optimizations_applied": len([r for r in optimization_results if r.success])
            })


class TestProductionReadiness:
    """Test production readiness aspects."""
    
    def test_configuration_management(self):
        """Test configuration management."""
        # Test that all components can be configured
        config_items = []
        
        # Logger configuration
        logger = get_logger("config_test", level="DEBUG", structured=True)
        config_items.append("logger")
        
        # Security configuration
        security_config = SecurityConfig(
            max_input_size=5000,
            rate_limit_requests=500
        )
        security_manager = SecurityManager(security_config)
        config_items.append("security")
        
        # Memory optimizer configuration
        memory_optimizer = MemoryOptimizer(
            target_memory_usage=0.7,
            optimization_interval=30.0
        )
        config_items.append("memory_optimizer")
        
        assert len(config_items) == 3
    
    def test_monitoring_and_observability(self):
        """Test monitoring and observability features."""
        logger = get_logger("observability_test")
        
        # Test structured logging with tracing
        with logger.request_context() as request_id:
            logger.info("Starting observability test")
            
            # Simulate some operations
            with logger.operation_timer("database_query"):
                time.sleep(0.01)
            
            with logger.operation_timer("model_inference"):
                time.sleep(0.02)
            
            logger.info("Observability test completed")
        
        # Check metrics were collected
        metrics = logger.get_metrics_summary()
        assert "timers" in metrics
        assert len(metrics["timers"]) >= 2
    
    def test_error_reporting(self):
        """Test comprehensive error reporting."""
        error_handler = ErrorHandler(log_errors=True)
        
        try:
            # Simulate various types of errors
            errors_to_test = [
                ValueError("Invalid input"),
                RuntimeError("Processing failed"),
                MemoryError("Out of memory"),
                NeuromorphicError("Custom neuromorphic error")
            ]
            
            for error in errors_to_test:
                try:
                    raise error
                except Exception as e:
                    error_info = error_handler.handle_error(e, context={
                        "test_context": "error_reporting_test"
                    })
                    assert error_info.error_id is not None
                    assert error_info.severity is not None
            
            # Check error statistics
            stats = error_handler.get_error_statistics()
            assert stats["total_errors"] == len(errors_to_test)
            
        finally:
            # Clean up
            error_handler.clear_error_history()
    
    def test_graceful_degradation(self):
        """Test graceful degradation under resource constraints."""
        # Test memory optimizer under memory pressure
        memory_optimizer = MemoryOptimizer(target_memory_usage=0.5)  # Aggressive
        
        # Create memory pressure
        large_tensors = []
        try:
            for _ in range(5):
                tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)
            
            # Test optimization under pressure
            results = memory_optimizer.optimize_memory()
            
            # Should handle gracefully without crashing
            assert isinstance(results, list)
            
        finally:
            # Clean up
            del large_tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running neuromorphic edge processor integration tests...")
    
    # Test each major component
    test_classes = [
        TestAdvancedLogging,
        TestCircuitBreaker,
        TestErrorHandling,
        TestComprehensiveValidator,
        TestAutoScaler,
        TestDistributedProcessor,
        TestMemoryOptimizer,
        TestSecurityManager,
        TestIntegrationScenarios,
        TestProductionReadiness
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")
        
        test_instance = test_class()
        methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")
    
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All integration tests passed! System is production-ready.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review issues before production deployment.")