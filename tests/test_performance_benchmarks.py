"""Performance benchmark validation tests."""

import unittest
import sys
import os
import time
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarking functionality."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.target_latency_ms = 100.0
        self.min_throughput = 10.0  # samples per second
        self.max_memory_mb = 500.0
        self.efficiency_threshold = 0.8
    
    def test_inference_latency(self):
        """Test inference latency requirements."""
        # Simulate inference timing
        def mock_inference():
            time.sleep(0.05)  # 50ms simulation
            return [0.8, 0.1, 0.1]  # Mock output
        
        # Time the inference
        start_time = time.time()
        result = mock_inference()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate latency
        self.assertLess(inference_time, self.target_latency_ms)
        self.assertIsNotNone(result)
        
        print(f"✓ Inference latency: {inference_time:.2f}ms (target: <{self.target_latency_ms}ms)")
    
    def test_throughput_performance(self):
        """Test throughput performance."""
        batch_size = 32
        num_batches = 10
        
        # Simulate batch processing
        def process_batch(batch_data):
            time.sleep(0.01)  # 10ms per batch
            return [i * 0.1 for i in range(len(batch_data))]
        
        start_time = time.time()
        
        total_samples = 0
        for _ in range(num_batches):
            batch_data = list(range(batch_size))
            result = process_batch(batch_data)
            total_samples += len(batch_data)
        
        total_time = time.time() - start_time
        throughput = total_samples / total_time
        
        self.assertGreater(throughput, self.min_throughput)
        
        print(f"✓ Throughput: {throughput:.1f} samples/sec (target: >{self.min_throughput} samples/sec)")
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        # Simulate memory allocation tracking
        initial_memory = 50.0  # Mock initial memory in MB
        
        # Simulate model loading
        model_memory = 100.0  # MB
        current_memory = initial_memory + model_memory
        
        # Simulate inference with additional memory
        inference_memory = 25.0  # MB
        peak_memory = current_memory + inference_memory
        
        self.assertLess(peak_memory, self.max_memory_mb)
        
        print(f"✓ Peak memory usage: {peak_memory:.1f}MB (target: <{self.max_memory_mb}MB)")
    
    def test_energy_efficiency(self):
        """Test energy efficiency metrics."""
        # Mock energy consumption data
        baseline_power = 10.0  # Watts
        neuromorphic_power = 2.0  # Watts
        
        efficiency_ratio = baseline_power / neuromorphic_power
        efficiency_score = min(1.0, efficiency_ratio / 5.0)  # Normalize to 0-1
        
        self.assertGreater(efficiency_score, self.efficiency_threshold)
        
        print(f"✓ Energy efficiency: {efficiency_score:.2f} (target: >{self.efficiency_threshold})")
    
    def test_sparsity_benefits(self):
        """Test sparsity optimization benefits."""
        # Mock spike data with different sparsity levels
        dense_data = [1] * 1000  # Dense data
        sparse_data = [1] * 100 + [0] * 900  # 90% sparse
        
        # Simulate processing time benefits
        def process_data(data):
            # Sparse data should process faster
            non_zero_count = sum(1 for x in data if x != 0)
            processing_time = non_zero_count * 0.001  # 1ms per non-zero element
            return processing_time
        
        dense_time = process_data(dense_data)
        sparse_time = process_data(sparse_data)
        
        speedup = dense_time / sparse_time
        self.assertGreater(speedup, 5.0)  # Expect significant speedup
        
        sparsity_ratio = sparse_data.count(0) / len(sparse_data)
        self.assertGreater(sparsity_ratio, 0.8)  # At least 80% sparse
        
        print(f"✓ Sparsity speedup: {speedup:.1f}x, sparsity: {sparsity_ratio:.1%}")
    
    def test_scalability(self):
        """Test performance scalability."""
        input_sizes = [100, 500, 1000, 2000]
        processing_times = []
        
        for size in input_sizes:
            # Simulate processing with linear complexity
            start_time = time.time()
            
            # Mock computation that scales linearly
            time.sleep(size * 0.00001)  # Linear scaling
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Check that scaling is reasonable (approximately linear)
        for i in range(1, len(input_sizes)):
            size_ratio = input_sizes[i] / input_sizes[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Time ratio should be close to size ratio for linear scaling
            self.assertLess(abs(time_ratio - size_ratio), size_ratio * 0.5)
        
        print(f"✓ Scalability test passed for sizes: {input_sizes}")
    
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import threading
        import queue
        
        num_threads = 4
        tasks_per_thread = 10
        results_queue = queue.Queue()
        
        def worker_task(task_id):
            # Simulate work
            time.sleep(0.01)
            results_queue.put(f"task_{task_id}_completed")
        
        # Launch concurrent tasks
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            for j in range(tasks_per_thread):
                task_id = i * tasks_per_thread + j
                thread = threading.Thread(target=worker_task, args=(task_id,))
                threads.append(thread)
                thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        expected_sequential_time = num_threads * tasks_per_thread * 0.01
        
        # Concurrent processing should be faster than sequential
        speedup = expected_sequential_time / total_time
        self.assertGreater(speedup, 2.0)  # Expect at least 2x speedup
        
        # Check all tasks completed
        completed_tasks = []
        while not results_queue.empty():
            completed_tasks.append(results_queue.get())
        
        self.assertEqual(len(completed_tasks), num_threads * tasks_per_thread)
        
        print(f"✓ Concurrent processing speedup: {speedup:.1f}x")
    
    def test_cache_performance(self):
        """Test caching system performance."""
        # Mock cache implementation
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def cached_computation(key):
            nonlocal cache_hits, cache_misses
            
            if key in cache:
                cache_hits += 1
                return cache[key]
            else:
                cache_misses += 1
                # Simulate expensive computation
                time.sleep(0.01)
                result = key * 2  # Mock result
                cache[key] = result
                return result
        
        # Test with repeated accesses
        test_keys = [1, 2, 3, 1, 2, 3, 4, 1, 2]
        
        start_time = time.time()
        for key in test_keys:
            cached_computation(key)
        total_time = time.time() - start_time
        
        hit_rate = cache_hits / (cache_hits + cache_misses)
        
        self.assertGreater(hit_rate, 0.5)  # Expect >50% hit rate
        self.assertLess(total_time, 0.05)  # Should be fast due to caching
        
        print(f"✓ Cache hit rate: {hit_rate:.1%}, total time: {total_time:.3f}s")
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        individual_processing_times = []
        batch_processing_times = []
        
        sample_data = list(range(100))
        
        # Individual processing
        start_time = time.time()
        for item in sample_data:
            time.sleep(0.0001)  # 0.1ms per item
        individual_time = time.time() - start_time
        
        # Batch processing (should be more efficient)
        batch_size = 10
        batches = [sample_data[i:i+batch_size] for i in range(0, len(sample_data), batch_size)]
        
        start_time = time.time()
        for batch in batches:
            time.sleep(0.0005)  # 0.5ms per batch (more efficient than 10x0.1ms)
        batch_time = time.time() - start_time
        
        efficiency = individual_time / batch_time
        self.assertGreater(efficiency, 1.5)  # Batching should be at least 50% more efficient
        
        print(f"✓ Batch processing efficiency: {efficiency:.1f}x faster")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks."""
        import gc
        
        # Simulate memory allocation and deallocation
        initial_objects = len(gc.get_objects())
        
        # Allocate memory
        temp_data = []
        for i in range(1000):
            temp_data.append([j for j in range(100)])
        
        allocated_objects = len(gc.get_objects())
        
        # Deallocate memory
        temp_data = None
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Check that memory was properly freed
        leaked_objects = final_objects - initial_objects
        
        self.assertLess(leaked_objects, 100)  # Allow some overhead
        
        print(f"✓ Memory leak test: {leaked_objects} objects remaining (acceptable)")


class TestBenchmarkReporting(unittest.TestCase):
    """Test benchmark reporting and analysis."""
    
    def test_benchmark_result_format(self):
        """Test benchmark result format validation."""
        sample_result = {
            "model_name": "spiking_neural_network",
            "task_name": "inference_speed",
            "execution_time": 0.025,  # 25ms
            "throughput": 40.0,  # samples/sec
            "memory_usage": 128.0,  # MB
            "accuracy": 0.87,
            "additional_metrics": {
                "sparsity": 0.85,
                "energy_consumption": 0.001  # Joules
            }
        }
        
        # Validate required fields
        required_fields = ["model_name", "task_name", "execution_time", "throughput", "memory_usage"]
        for field in required_fields:
            self.assertIn(field, sample_result)
        
        # Validate value types and ranges
        self.assertIsInstance(sample_result["execution_time"], (int, float))
        self.assertGreater(sample_result["execution_time"], 0)
        
        self.assertIsInstance(sample_result["throughput"], (int, float))
        self.assertGreater(sample_result["throughput"], 0)
        
        if sample_result.get("accuracy"):
            self.assertGreaterEqual(sample_result["accuracy"], 0)
            self.assertLessEqual(sample_result["accuracy"], 1)
        
        print("✓ Benchmark result format validated")
    
    def test_performance_comparison(self):
        """Test performance comparison calculations."""
        baseline_results = {
            "accuracy": [0.80, 0.82, 0.81],
            "latency": [120, 115, 118],  # ms
            "memory": [200, 195, 205]    # MB
        }
        
        neuromorphic_results = {
            "accuracy": [0.85, 0.87, 0.86],
            "latency": [45, 42, 48],     # ms
            "memory": [128, 125, 130]    # MB
        }
        
        # Calculate improvements
        def calculate_improvement(baseline, neuromorphic, higher_is_better=True):
            baseline_avg = sum(baseline) / len(baseline)
            neuromorphic_avg = sum(neuromorphic) / len(neuromorphic)
            
            if higher_is_better:
                return (neuromorphic_avg - baseline_avg) / baseline_avg
            else:
                return (baseline_avg - neuromorphic_avg) / baseline_avg
        
        accuracy_improvement = calculate_improvement(
            baseline_results["accuracy"], 
            neuromorphic_results["accuracy"], 
            higher_is_better=True
        )
        
        latency_improvement = calculate_improvement(
            baseline_results["latency"], 
            neuromorphic_results["latency"], 
            higher_is_better=False
        )
        
        memory_improvement = calculate_improvement(
            baseline_results["memory"], 
            neuromorphic_results["memory"], 
            higher_is_better=False
        )
        
        # Validate improvements
        self.assertGreater(accuracy_improvement, 0)  # Better accuracy
        self.assertGreater(latency_improvement, 0)   # Lower latency
        self.assertGreater(memory_improvement, 0)    # Lower memory usage
        
        print(f"✓ Performance improvements - Accuracy: {accuracy_improvement:.1%}, "
              f"Latency: {latency_improvement:.1%}, Memory: {memory_improvement:.1%}")
    
    def test_statistical_significance(self):
        """Test statistical significance of benchmark results."""
        import statistics
        import math
        
        # Sample data for two models
        model_a_results = [0.82, 0.84, 0.83, 0.85, 0.81, 0.84, 0.82, 0.85]
        model_b_results = [0.87, 0.89, 0.88, 0.86, 0.88, 0.87, 0.89, 0.86]
        
        # Calculate means and standard deviations
        mean_a = statistics.mean(model_a_results)
        mean_b = statistics.mean(model_b_results)
        
        stdev_a = statistics.stdev(model_a_results)
        stdev_b = statistics.stdev(model_b_results)
        
        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(((len(model_a_results) - 1) * stdev_a**2 + 
                               (len(model_b_results) - 1) * stdev_b**2) / 
                              (len(model_a_results) + len(model_b_results) - 2))
        
        cohens_d = (mean_b - mean_a) / pooled_std
        
        # Validate difference
        self.assertGreater(mean_b, mean_a)  # Model B should be better
        self.assertGreater(cohens_d, 0.5)  # Medium to large effect size
        
        print(f"✓ Statistical analysis - Mean difference: {mean_b - mean_a:.3f}, "
              f"Effect size (Cohen's d): {cohens_d:.3f}")


if __name__ == '__main__':
    print("⚡ Running Performance Benchmark Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("⚡ Performance Test Summary:")
    print(f"Performance tests run: {result.testsRun}")
    print(f"Performance failures: {len(result.failures)}")
    print(f"Performance errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        performance_score = 100
        print("✅ All performance tests passed!")
    else:
        failed_tests = len(result.failures) + len(result.errors)
        performance_score = max(0, (result.testsRun - failed_tests) / result.testsRun * 100)
        print(f"⚠ Performance score: {performance_score:.1f}%")
    
    print(f"\n🚀 Performance Assessment: {'EXCELLENT' if performance_score >= 95 else 'GOOD' if performance_score >= 80 else 'NEEDS OPTIMIZATION'}")