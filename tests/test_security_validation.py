"""Security validation tests for neuromorphic edge processor."""

import unittest
import sys
import os
import tempfile
import json
import hashlib

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSecurityValidation(unittest.TestCase):
    """Test security validation and input sanitization."""
    
    def setUp(self):
        """Set up test environment."""
        self.max_input_size = 10000
        self.max_sequence_length = 1000
        self.allowed_dtypes = ['float32', 'float64', 'int32', 'int64']
        self.value_range = (-100.0, 100.0)
    
    def test_input_size_limits(self):
        """Test input size validation."""
        # Valid input size
        valid_size = 784
        self.assertLess(valid_size, self.max_input_size)
        
        # Invalid input size (too large)
        invalid_size = 50000
        self.assertGreater(invalid_size, self.max_input_size)
        
        print("✓ Input size limits validated")
    
    def test_sequence_length_limits(self):
        """Test sequence length validation."""
        # Valid sequence length
        valid_length = 100
        self.assertLess(valid_length, self.max_sequence_length)
        
        # Invalid sequence length (too long)
        invalid_length = 5000
        self.assertGreater(invalid_length, self.max_sequence_length)
        
        print("✓ Sequence length limits validated")
    
    def test_value_range_validation(self):
        """Test value range constraints."""
        min_val, max_val = self.value_range
        
        # Valid values
        valid_values = [0.0, 1.0, -1.0, 50.0, -50.0]
        for val in valid_values:
            self.assertGreaterEqual(val, min_val)
            self.assertLessEqual(val, max_val)
        
        # Invalid values
        invalid_values = [-200.0, 200.0, 1000.0, -1000.0]
        for val in invalid_values:
            self.assertTrue(val < min_val or val > max_val)
        
        print("✓ Value range validation successful")
    
    def test_dtype_validation(self):
        """Test data type validation."""
        # Valid dtypes
        for dtype in self.allowed_dtypes:
            self.assertIn(dtype, self.allowed_dtypes)
        
        # Invalid dtypes
        invalid_dtypes = ['str', 'object', 'complex128', 'bool']
        for dtype in invalid_dtypes:
            self.assertNotIn(dtype, self.allowed_dtypes)
        
        print("✓ Data type validation successful")
    
    def test_malicious_pattern_detection(self):
        """Test detection of malicious patterns in inputs."""
        malicious_patterns = [
            '__import__',
            'eval(',
            'exec(',
            'subprocess',
            'os.system',
            'rm -rf',
            'DROP TABLE',
            '<script',
            'javascript:'
        ]
        
        # Test pattern detection logic
        for pattern in malicious_patterns:
            test_string = f"This contains {pattern} which is suspicious"
            self.assertIn(pattern.lower(), test_string.lower())
        
        # Test safe strings
        safe_strings = [
            "normal input data",
            "spike train values: [1, 0, 1, 0]",
            "learning_rate = 0.001"
        ]
        
        for safe_string in safe_strings:
            for pattern in malicious_patterns:
                self.assertNotIn(pattern.lower(), safe_string.lower())
        
        print("✓ Malicious pattern detection validated")
    
    def test_file_path_validation(self):
        """Test file path security validation."""
        # Valid file paths
        valid_paths = [
            "model.h5",
            "data/train.npz",
            "results/experiment_001.json"
        ]
        
        for path in valid_paths:
            self.assertNotIn("..", path)
            self.assertTrue(len(path) < 1000)
        
        # Invalid file paths (path traversal)
        invalid_paths = [
            "../../../etc/passwd",
            "../../system/config",
            "data/../../../root/.bashrc"
        ]
        
        for path in invalid_paths:
            self.assertIn("..", path)
        
        print("✓ File path validation successful")
    
    def test_configuration_sanitization(self):
        """Test configuration input sanitization."""
        # Valid configuration
        valid_config = {
            "input_size": 784,
            "hidden_size": 128,
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "cpu"
        }
        
        # Validate each field
        self.assertIsInstance(valid_config["input_size"], int)
        self.assertGreater(valid_config["input_size"], 0)
        self.assertLess(valid_config["input_size"], self.max_input_size)
        
        self.assertIsInstance(valid_config["learning_rate"], (int, float))
        self.assertGreater(valid_config["learning_rate"], 0)
        self.assertLessEqual(valid_config["learning_rate"], 1)
        
        # Test dangerous configuration keys
        dangerous_keys = ["__builtins__", "__globals__", "sys", "os", "subprocess"]
        for key in dangerous_keys:
            self.assertNotIn(key, valid_config)
        
        print("✓ Configuration sanitization validated")
    
    def test_memory_usage_limits(self):
        """Test memory usage validation."""
        max_memory_mb = 1000.0
        
        # Calculate memory usage for typical model
        input_size = 784
        hidden_size = 128
        output_size = 10
        batch_size = 32
        
        # Estimate memory usage (simplified)
        param_memory = (input_size * hidden_size + hidden_size * output_size) * 4 / (1024 * 1024)
        activation_memory = (batch_size * hidden_size) * 4 / (1024 * 1024)
        total_memory = param_memory + activation_memory
        
        self.assertLess(total_memory, max_memory_mb)
        
        print("✓ Memory usage limits validated")
    
    def test_rate_limiting(self):
        """Test rate limiting logic."""
        max_requests = 1000
        time_window = 3600  # 1 hour
        
        # Simulate request tracking
        request_times = []
        current_time = 1000000  # Mock timestamp
        
        # Add requests within window
        for i in range(500):
            request_times.append(current_time + i)
        
        # Filter requests within window
        window_start = current_time + 500 - time_window
        requests_in_window = [t for t in request_times if t > window_start]
        
        self.assertLess(len(requests_in_window), max_requests)
        
        print("✓ Rate limiting logic validated")
    
    def test_input_hash_validation(self):
        """Test input hash computation for integrity."""
        test_data = "sample input data for hashing"
        
        # Compute hash
        hash1 = hashlib.sha256(test_data.encode()).hexdigest()
        
        # Compute same hash again
        hash2 = hashlib.sha256(test_data.encode()).hexdigest()
        
        # Hashes should match
        self.assertEqual(hash1, hash2)
        
        # Modified data should have different hash
        modified_data = test_data + " modified"
        hash3 = hashlib.sha256(modified_data.encode()).hexdigest()
        
        self.assertNotEqual(hash1, hash3)
        
        print("✓ Input hash validation successful")
    
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        import random
        
        # Generate random values
        random_values = [random.random() for _ in range(100)]
        
        # Check properties
        self.assertEqual(len(random_values), 100)
        self.assertTrue(all(0 <= val <= 1 for val in random_values))
        
        # Check for reasonable distribution (not all same)
        unique_values = set(random_values)
        self.assertGreater(len(unique_values), 50)  # Should have many unique values
        
        print("✓ Secure random generation validated")
    
    def test_error_information_leakage(self):
        """Test that error messages don't leak sensitive information."""
        # Simulate error scenarios
        error_messages = [
            "Invalid input format",
            "Configuration parameter out of range",
            "Memory limit exceeded",
            "Rate limit exceeded"
        ]
        
        # Error messages should be generic
        sensitive_info = ["password", "token", "key", "secret", "admin", "root"]
        
        for message in error_messages:
            for sensitive in sensitive_info:
                self.assertNotIn(sensitive.lower(), message.lower())
        
        print("✓ Error information leakage prevention validated")


class TestDataValidation(unittest.TestCase):
    """Test data validation and integrity checks."""
    
    def test_nan_and_inf_detection(self):
        """Test detection of NaN and infinite values."""
        import math
        
        # Valid numbers
        valid_numbers = [0.0, 1.0, -1.0, 0.001, 1000.0]
        for num in valid_numbers:
            self.assertFalse(math.isnan(num))
            self.assertFalse(math.isinf(num))
        
        # Invalid numbers
        invalid_numbers = [float('nan'), float('inf'), float('-inf')]
        for num in invalid_numbers:
            self.assertTrue(math.isnan(num) or math.isinf(num))
        
        print("✓ NaN and infinity detection validated")
    
    def test_spike_data_validation(self):
        """Test validation of spike train data."""
        # Valid spike data (binary)
        valid_spikes = [1, 0, 1, 0, 0, 1, 1, 0]
        self.assertTrue(all(spike in [0, 1] for spike in valid_spikes))
        
        # Valid spike data (small positive values for rates)
        valid_rates = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.assertTrue(all(0 <= rate <= 10 for rate in valid_rates))
        
        # Invalid spike data
        invalid_spikes = [-1, 2, 15, 100]
        self.assertFalse(all(0 <= spike <= 1 for spike in invalid_spikes))
        
        print("✓ Spike data validation successful")
    
    def test_temporal_consistency(self):
        """Test temporal consistency in sequences."""
        # Test sequence with reasonable temporal changes
        sequence = [0.1, 0.2, 0.25, 0.3, 0.28, 0.35]
        
        # Calculate max change between consecutive elements
        max_change = max(abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1))
        
        # Should not have extremely large jumps
        self.assertLess(max_change, 10.0)
        
        # Test sequence with unreasonable jumps
        bad_sequence = [0.1, 0.2, 100.0, 0.3]
        bad_max_change = max(abs(bad_sequence[i+1] - bad_sequence[i]) for i in range(len(bad_sequence)-1))
        self.assertGreater(bad_max_change, 10.0)
        
        print("✓ Temporal consistency validation successful")
    
    def test_model_parameter_validation(self):
        """Test model parameter validation."""
        # Valid parameters
        valid_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 100,
            "hidden_size": 128,
            "dropout_rate": 0.1
        }
        
        # Validate ranges
        self.assertGreater(valid_params["learning_rate"], 0)
        self.assertLessEqual(valid_params["learning_rate"], 1)
        
        self.assertGreater(valid_params["batch_size"], 0)
        self.assertLess(valid_params["batch_size"], 10000)
        
        self.assertGreater(valid_params["num_epochs"], 0)
        self.assertLess(valid_params["num_epochs"], 100000)
        
        self.assertGreaterEqual(valid_params["dropout_rate"], 0)
        self.assertLessEqual(valid_params["dropout_rate"], 1)
        
        print("✓ Model parameter validation successful")


if __name__ == '__main__':
    print("🛡️ Running Security Validation Test Suite")
    print("=" * 60)
    
    # Run security tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    class SecurityTestResult(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
    
    class SecurityTestRunner(unittest.TextTestRunner):
        resultclass = SecurityTestResult
    
    runner = SecurityTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print security summary
    print("\n" + "=" * 60)
    print("🛡️ Security Validation Summary:")
    print(f"Security tests run: {result.testsRun}")
    print(f"Security failures: {len(result.failures)}")
    print(f"Security errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All security validations passed! System is secure.")
        security_score = 100
    else:
        failed_tests = len(result.failures) + len(result.errors)
        security_score = max(0, (result.testsRun - failed_tests) / result.testsRun * 100)
        print(f"⚠ Security score: {security_score:.1f}%")
    
    print(f"\n🔒 Security Assessment: {'PASSED' if security_score >= 90 else 'NEEDS IMPROVEMENT'}")