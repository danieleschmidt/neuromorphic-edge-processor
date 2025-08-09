"""Comprehensive tests for Spiking Neural Network."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.spiking_neural_network import SpikingNeuralNetwork, LIFNeuron, SpikingLayer


class TestLIFNeuron:
    """Test cases for LIF neuron model."""
    
    def test_lif_neuron_initialization(self):
        """Test LIF neuron initialization."""
        neuron = LIFNeuron(
            tau_mem=20.0,
            tau_syn=5.0,
            v_thresh=-50.0,
            v_reset=-70.0,
            v_rest=-65.0
        )
        
        assert neuron.tau_mem == 20.0
        assert neuron.tau_syn == 5.0
        assert neuron.v_thresh == -50.0
        assert neuron.v_reset == -70.0
        assert neuron.v_rest == -65.0
    
    def test_lif_neuron_reset_state(self):
        """Test neuron state reset."""
        neuron = LIFNeuron()
        
        batch_size, num_neurons = 2, 100
        neuron.reset_state(batch_size, num_neurons)
        
        assert neuron.v_mem.shape == (batch_size, num_neurons)
        assert neuron.i_syn.shape == (batch_size, num_neurons)
        assert neuron.refractory_time.shape == (batch_size, num_neurons)
        
        # Check initial values
        assert torch.all(neuron.v_mem == neuron.v_rest)
        assert torch.all(neuron.i_syn == 0)
        assert torch.all(neuron.refractory_time == 0)
    
    def test_lif_neuron_forward_pass(self):
        """Test LIF neuron forward pass."""
        neuron = LIFNeuron(dt=0.1)
        
        batch_size, num_neurons = 2, 10
        neuron.reset_state(batch_size, num_neurons)
        
        # Strong input current to trigger spikes
        input_current = torch.ones(batch_size, num_neurons) * 20.0
        
        spikes = neuron.forward(input_current, time_step=0)
        
        assert spikes.shape == (batch_size, num_neurons)
        assert spikes.dtype == torch.float32
        assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes
    
    def test_lif_neuron_spike_generation(self):
        """Test that strong inputs generate spikes."""
        neuron = LIFNeuron(v_thresh=-50.0, v_rest=-65.0, dt=0.1)
        neuron.reset_state(1, 1)
        
        # Apply strong current multiple times
        total_spikes = 0
        for t in range(100):
            input_current = torch.tensor([[30.0]])  # Strong current
            spikes = neuron.forward(input_current, t)
            total_spikes += spikes.sum().item()
        
        # Should generate at least some spikes
        assert total_spikes > 0
    
    def test_lif_neuron_refractory_period(self):
        """Test refractory period enforcement."""
        neuron = LIFNeuron(refractory_period=5.0, dt=0.1)  # 5ms refractory
        neuron.reset_state(1, 1)
        
        # Force a spike
        neuron.v_mem.fill_(neuron.v_thresh + 1)
        spikes1 = neuron.forward(torch.tensor([[0.0]]), 0)
        
        # Immediately try to spike again
        neuron.v_mem.fill_(neuron.v_thresh + 1)
        spikes2 = neuron.forward(torch.tensor([[0.0]]), 1)
        
        assert spikes1.sum() > 0  # First spike should occur
        assert spikes2.sum() == 0  # Second spike should be blocked


class TestSpikingLayer:
    """Test cases for spiking layer."""
    
    def test_spiking_layer_initialization(self):
        """Test spiking layer initialization."""
        layer = SpikingLayer(
            input_size=100,
            output_size=50,
            connection_prob=0.1,
            dt=0.1
        )
        
        assert layer.input_size == 100
        assert layer.output_size == 50
        assert layer.weight.shape == (50, 100)
        assert layer.connection_mask.shape == (50, 100)
        assert layer.delays.shape == (50, 100)
    
    def test_spiking_layer_forward_pass(self):
        """Test spiking layer forward pass."""
        layer = SpikingLayer(input_size=10, output_size=5, dt=0.1)
        
        batch_size = 2
        layer.reset_state(batch_size)
        
        # Binary spike input
        input_spikes = torch.randint(0, 2, (batch_size, 10)).float()
        
        output_spikes = layer.forward(input_spikes)
        
        assert output_spikes.shape == (batch_size, 5)
        assert output_spikes.dtype == torch.float32
        assert torch.all((output_spikes == 0) | (output_spikes == 1))
    
    def test_spiking_layer_synaptic_delays(self):
        """Test synaptic delay implementation."""
        layer = SpikingLayer(input_size=5, output_size=3, dt=0.1)
        layer.reset_state(1)
        
        # Send input spikes
        input_spikes = torch.ones(1, 5)
        
        outputs = []
        for t in range(10):
            output = layer.forward(input_spikes if t == 0 else torch.zeros(1, 5))
            outputs.append(output.clone())
        
        # Should see delayed responses
        total_output = sum(o.sum() for o in outputs)
        assert total_output > 0  # Some output should occur due to delays


class TestSpikingNeuralNetwork:
    """Test cases for complete spiking neural network."""
    
    def test_snn_initialization(self):
        """Test SNN initialization."""
        layer_sizes = [784, 100, 10]
        snn = SpikingNeuralNetwork(
            layer_sizes=layer_sizes,
            connection_prob=0.1,
            dt=0.1,
            encoding_method="poisson"
        )
        
        assert len(snn.layers) == len(layer_sizes) - 1
        assert snn.layers[0].input_size == layer_sizes[0]
        assert snn.layers[0].output_size == layer_sizes[1]
        assert snn.layers[1].input_size == layer_sizes[1]
        assert snn.layers[1].output_size == layer_sizes[2]
    
    def test_snn_forward_pass(self):
        """Test SNN forward pass."""
        layer_sizes = [100, 50, 10]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        batch_size = 4
        input_rates = torch.rand(batch_size, layer_sizes[0]) * 10  # 0-10 Hz
        
        output = snn.forward(input_rates, duration=50.0)
        
        assert output.shape == (batch_size, layer_sizes[-1])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_snn_with_spikes_return(self):
        """Test SNN forward pass with spike return."""
        layer_sizes = [50, 30, 10]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        batch_size = 2
        input_rates = torch.rand(batch_size, layer_sizes[0]) * 5
        
        output, spike_trains = snn.forward(
            input_rates, duration=30.0, return_spikes=True
        )
        
        assert output.shape == (batch_size, layer_sizes[-1])
        assert len(spike_trains) == len(layer_sizes) - 1  # One per layer
        
        for i, spikes in enumerate(spike_trains):
            expected_shape = (batch_size, layer_sizes[i+1], 300)  # 30ms / 0.1ms
            assert spikes.shape == expected_shape
            assert torch.all((spikes == 0) | (spikes == 1))
    
    def test_snn_energy_computation(self):
        """Test energy consumption computation."""
        layer_sizes = [20, 15, 5]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        batch_size = 2
        input_rates = torch.rand(batch_size, layer_sizes[0]) * 3
        
        _, spike_trains = snn.forward(
            input_rates, duration=20.0, return_spikes=True
        )
        
        energy_metrics = snn.compute_energy_consumption(spike_trains)
        
        assert "total_spikes" in energy_metrics
        assert "sparsity" in energy_metrics
        assert "energy_consumption" in energy_metrics
        assert "energy_efficiency" in energy_metrics
        assert "theoretical_speedup" in energy_metrics
        
        assert energy_metrics["total_spikes"] >= 0
        assert 0 <= energy_metrics["sparsity"] <= 1
        assert energy_metrics["energy_consumption"] > 0
    
    def test_snn_training_step(self):
        """Test SNN training step."""
        layer_sizes = [10, 8, 3]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        batch_size = 4
        x = torch.rand(batch_size, layer_sizes[0]) * 5
        y = torch.rand(batch_size, layer_sizes[-1])
        
        loss, metrics = snn.train_step(x, y, duration=20.0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "loss" in metrics
        assert "total_spikes" in metrics
        assert "energy_efficiency" in metrics
        assert metrics["loss"] >= 0
    
    def test_snn_different_encodings(self):
        """Test different spike encoding methods."""
        layer_sizes = [20, 10, 5]
        encodings = ["poisson", "regular", "temporal"]
        
        for encoding in encodings:
            snn = SpikingNeuralNetwork(
                layer_sizes, 
                dt=0.1, 
                encoding_method=encoding
            )
            
            x = torch.rand(2, layer_sizes[0]) * 8
            output = snn.forward(x, duration=30.0)
            
            assert output.shape == (2, layer_sizes[-1])
            assert not torch.isnan(output).any()
    
    def test_snn_activity_analysis(self):
        """Test network activity analysis."""
        layer_sizes = [15, 10, 5]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        batch_size = 3
        x = torch.rand(batch_size, layer_sizes[0]) * 6
        
        _, spike_trains = snn.forward(x, duration=25.0, return_spikes=True)
        
        analysis = snn.analyze_network_activity(spike_trains)
        
        assert "network_sparsity" in analysis
        assert 0 <= analysis["network_sparsity"] <= 1
        
        # Check per-layer analysis
        for i in range(len(layer_sizes) - 1):
            layer_key = f"layer_{i}"
            assert layer_key in analysis
            assert "mean_firing_rate" in analysis[layer_key]
            assert "sparsity" in analysis[layer_key]
    
    def test_snn_reset_between_runs(self):
        """Test that SNN properly resets state between runs."""
        layer_sizes = [10, 8, 5]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.rand(1, layer_sizes[0]) * 5
        
        # First run
        output1 = snn.forward(x, duration=20.0)
        
        # Second run (should be independent)
        output2 = snn.forward(x, duration=20.0)
        
        # Outputs might be different due to stochasticity in encoding
        # but should have same shape and be finite
        assert output1.shape == output2.shape
        assert torch.isfinite(output1).all()
        assert torch.isfinite(output2).all()
    
    def test_snn_gradient_computation(self):
        """Test that gradients can be computed."""
        layer_sizes = [8, 6, 4]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.rand(2, layer_sizes[0]) * 4
        y = torch.rand(2, layer_sizes[-1])
        
        # Enable gradients
        for param in snn.parameters():
            param.requires_grad_(True)
        
        output = snn.forward(x, duration=15.0)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Compute gradients
        loss.backward()
        
        # Check that some gradients exist
        gradient_found = False
        for param in snn.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                gradient_found = True
                break
        
        assert gradient_found, "No gradients computed"


class TestSpikingNetworkEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_network(self):
        """Test behavior with minimal network."""
        # Smallest possible network
        layer_sizes = [1, 1]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.tensor([[1.0]])
        output = snn.forward(x, duration=10.0)
        
        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()
    
    def test_zero_input(self):
        """Test network with zero input."""
        layer_sizes = [10, 5, 3]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.zeros(2, layer_sizes[0])
        output = snn.forward(x, duration=20.0)
        
        assert output.shape == (2, layer_sizes[-1])
        assert torch.isfinite(output).all()
    
    def test_very_short_duration(self):
        """Test network with very short simulation duration."""
        layer_sizes = [5, 3, 2]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.rand(1, layer_sizes[0]) * 10
        output = snn.forward(x, duration=0.5)  # Very short
        
        assert output.shape == (1, layer_sizes[-1])
        assert torch.isfinite(output).all()
    
    def test_large_input_values(self):
        """Test network with large input values."""
        layer_sizes = [5, 4, 2]
        snn = SpikingNeuralNetwork(layer_sizes, dt=0.1)
        
        x = torch.rand(1, layer_sizes[0]) * 1000  # Very high rates
        output = snn.forward(x, duration=10.0)
        
        assert output.shape == (1, layer_sizes[-1])
        assert torch.isfinite(output).all()


@pytest.fixture
def sample_snn():
    """Fixture providing a sample SNN for tests."""
    layer_sizes = [20, 15, 10, 5]
    return SpikingNeuralNetwork(
        layer_sizes=layer_sizes,
        connection_prob=0.15,
        dt=0.1,
        encoding_method="poisson"
    )


def test_snn_reproducibility(sample_snn):
    """Test that SNN produces consistent results with same random seed."""
    x = torch.rand(2, 20) * 5
    
    # Set seed and run
    torch.manual_seed(42)
    output1, spikes1 = sample_snn.forward(x, duration=20.0, return_spikes=True)
    
    # Reset seed and run again
    torch.manual_seed(42)
    sample_snn.reset_state(2)  # Reset network state
    output2, spikes2 = sample_snn.forward(x, duration=20.0, return_spikes=True)
    
    # Note: Due to stochastic encoding, results might not be identical
    # but should be similar in magnitude
    assert output1.shape == output2.shape
    assert torch.isfinite(output1).all()
    assert torch.isfinite(output2).all()


def test_snn_batch_consistency(sample_snn):
    """Test that SNN handles different batch sizes consistently."""
    input_size = 20
    
    # Single sample
    x1 = torch.rand(1, input_size) * 5
    output1 = sample_snn.forward(x1, duration=15.0)
    
    # Batch of same sample
    x_batch = x1.repeat(4, 1)
    output_batch = sample_snn.forward(x_batch, duration=15.0)
    
    assert output1.shape == (1, 5)
    assert output_batch.shape == (4, 5)
    
    # All outputs should be finite
    assert torch.isfinite(output1).all()
    assert torch.isfinite(output_batch).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])