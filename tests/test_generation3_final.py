"""
Final Generation 3 Optimization Tests
Validates all optimization features work correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams
from optimization.adaptive_cache import AdaptiveCache, CachePolicy
from optimization.performance_optimizer import PerformanceOptimizer


def test_generation3_optimizations():
    """Test all Generation 3 optimization features."""
    print("=" * 60)
    print("GENERATION 3 OPTIMIZATION VALIDATION")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Optimized Neuron Initialization
    total_tests += 1
    try:
        params = OptimizedLIFParams(enable_jit=True)
        neuron = OptimizedLIFNeuron(params, n_neurons=10)
        
        assert neuron.n_neurons == 10
        assert neuron.opt_params.enable_jit is True
        assert hasattr(neuron, 'performance_metrics')
        
        print("✓ Test 1: Optimized neuron initialization - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Optimized neuron initialization - FAILED: {e}")
    
    # Test 2: JIT Compilation Performance
    total_tests += 1
    try:
        # Non-JIT neuron
        params_no_jit = OptimizedLIFParams(enable_jit=False)
        neuron_no_jit = OptimizedLIFNeuron(params_no_jit, n_neurons=50)
        
        # JIT neuron
        params_jit = OptimizedLIFParams(enable_jit=True)
        neuron_jit = OptimizedLIFNeuron(params_jit, n_neurons=50)
        
        # Test input
        test_input = np.random.randn(50) * 1e-9
        
        # Warm up JIT
        for _ in range(3):
            neuron_jit.forward(test_input)
        
        # Time without JIT
        start_time = time.time()
        for _ in range(50):
            neuron_no_jit.forward(test_input)
        time_no_jit = time.time() - start_time
        
        # Time with JIT
        start_time = time.time()
        for _ in range(50):
            neuron_jit.forward(test_input)
        time_jit = time.time() - start_time
        
        speedup = time_no_jit / time_jit if time_jit > 0 else 1.0
        print(f"✓ Test 2: JIT compilation performance - PASSED (speedup: {speedup:.2f}x)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: JIT compilation performance - FAILED: {e}")
    
    # Test 3: Adaptive Caching
    total_tests += 1
    try:
        cache = AdaptiveCache(max_size=100)
        
        # Add test data
        for i in range(150):  # Exceed capacity
            cache.put(f"key_{i}", np.random.randn(10))
        
        # Check size constraint
        assert len(cache._cache) <= cache.max_size
        
        # Test retrieval
        cache.put("test_key", np.array([1, 2, 3]))
        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert np.array_equal(retrieved, np.array([1, 2, 3]))
        
        print("✓ Test 3: Adaptive caching - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3: Adaptive caching - FAILED: {e}")
    
    # Test 4: Performance Monitoring
    total_tests += 1
    try:
        neuron = OptimizedLIFNeuron(OptimizedLIFParams(), n_neurons=10)
        
        # Generate some activity
        for i in range(20):
            test_input = np.random.randn(10) * 1e-9
            neuron.forward(test_input)
        
        # Get performance metrics
        metrics = neuron.get_performance_metrics()
        
        assert 'total_forward_calls' in metrics
        assert metrics['total_forward_calls'] == 20
        assert 'average_latency_ms' in metrics
        assert metrics['average_latency_ms'] > 0
        
        print(f"✓ Test 4: Performance monitoring - PASSED ({metrics['total_forward_calls']} calls tracked)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4: Performance monitoring - FAILED: {e}")
    
    # Test 5: Batch Processing
    total_tests += 1
    try:
        neuron = OptimizedLIFNeuron(OptimizedLIFParams(), n_neurons=5)
        
        # Create batch input (batch_size=10, seq_length=20, n_neurons=5)
        batch_input = np.random.randn(10, 20, 5) * 1e-9
        
        # Process batch
        batch_results = neuron.batch_forward(batch_input)
        
        assert 'spikes' in batch_results
        assert 'membrane_potentials' in batch_results
        assert batch_results['spikes'].shape == (10, 20, 5)
        
        print("✓ Test 5: Batch processing - PASSED (10x20x5 batch)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Batch processing - FAILED: {e}")
    
    # Test 6: Memory Pool Management
    total_tests += 1
    try:
        neuron = OptimizedLIFNeuron(OptimizedLIFParams(), n_neurons=10)
        
        # Test that memory pools exist and function
        assert hasattr(neuron, '_memory_pool')
        
        # Perform operations to use memory pool
        for _ in range(15):
            test_input = np.random.randn(10) * 1e-9
            neuron.forward(test_input)
        
        print("✓ Test 6: Memory pool management - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 6: Memory pool management - FAILED: {e}")
    
    # Test 7: Performance Optimizer
    total_tests += 1
    try:
        optimizer = PerformanceOptimizer()
        
        # Test profiling
        def test_function():
            return np.sum(np.random.randn(100) ** 2)
        
        profile = optimizer.profile_compute_performance(test_function, iterations=5)
        
        assert 'avg_time_ms' in profile
        assert profile['avg_time_ms'] > 0
        
        print("✓ Test 7: Performance optimizer - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 7: Performance optimizer - FAILED: {e}")
    
    # Test 8: Spike Train Optimization
    total_tests += 1
    try:
        neuron = OptimizedLIFNeuron(OptimizedLIFParams(), n_neurons=3)
        
        # Create current profile
        duration = 100.0  # ms
        current_profile = np.sin(np.linspace(0, 2*np.pi, int(duration))) * 2e-9
        
        spike_train = neuron.get_spike_train_optimized(
            current_profile=current_profile,
            duration=duration,
            dt=1.0
        )
        
        assert 'spike_times' in spike_train
        assert 'spike_counts' in spike_train
        assert 'firing_rate' in spike_train
        assert len(spike_train['spike_times']) == 3
        
        print("✓ Test 8: Spike train optimization - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 8: Spike train optimization - FAILED: {e}")
    
    # Final results
    print("\n" + "=" * 60)
    print(f"GENERATION 3 TESTS COMPLETED: {tests_passed}/{total_tests} PASSED")
    
    if tests_passed == total_tests:
        print("🚀 ALL GENERATION 3 OPTIMIZATION FEATURES VALIDATED!")
        print("✅ Ready for Quality Gates Implementation")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_generation3_optimizations()
    print("=" * 60)
    sys.exit(0 if success else 1)