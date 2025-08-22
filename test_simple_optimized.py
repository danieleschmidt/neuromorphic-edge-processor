"""Simple test for optimized LIF neuron."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams

def test_basic_functionality():
    """Test basic optimized neuron functionality."""
    print("Testing optimized LIF neuron...")
    
    # Create neuron
    params = OptimizedLIFParams(enable_jit=True)
    neuron = OptimizedLIFNeuron(params, n_neurons=5)
    
    print(f"✓ Created neuron with {neuron.n_neurons} neurons")
    
    # Test forward pass
    test_input = np.array([1e-9, 2e-9, 0.5e-9, 1.5e-9, 0.8e-9])
    result = neuron.forward(test_input)
    
    print(f"✓ Forward pass completed")
    print(f"  - Result keys: {list(result.keys())}")
    
    # Test performance stats
    stats = neuron.get_performance_metrics()
    print(f"✓ Performance stats: {stats}")
    
    print("All basic tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()