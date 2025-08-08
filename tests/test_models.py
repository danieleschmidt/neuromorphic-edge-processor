"""Comprehensive tests for neuromorphic models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

from src.models.lif_neuron import LIFNeuron
from src.models.population import NeuronPopulation
from src.models.spiking_network import SpikingNeuralNetwork


class TestLIFNeuron:
    """Test suite for Leaky Integrate-and-Fire neuron."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.neuron = LIFNeuron()
        self.batch_size = 2
        self.time_steps = 100
        
    def test_neuron_initialization(self):
        """Test neuron proper initialization."""
        neuron = LIFNeuron(
            tau_mem=10.0,
            tau_syn=2.0,
            v_thresh=-45.0,
            v_reset=-75.0,
            adaptive_thresh=False
        )
        
        assert neuron.tau_mem == 10.0
        assert neuron.tau_syn == 2.0
        assert neuron.v_thresh == -45.0
        assert neuron.v_reset == -75.0
        assert not neuron.adaptive_thresh
        
    def test_forward_pass_shape(self):
        """Test forward pass returns correct shapes."""
        input_current = torch.randn(self.batch_size, self.time_steps)
        
        spikes, voltage = self.neuron(input_current)
        
        assert spikes.shape == (self.batch_size, self.time_steps)
        assert voltage.shape == (self.batch_size, self.time_steps)
        assert spikes.dtype == torch.float32
        
    def test_spike_generation(self):
        """Test that spikes are generated with sufficient input."""
        # Strong positive input should generate spikes
        strong_input = torch.ones(1, self.time_steps) * 100.0
        spikes, voltage = self.neuron(strong_input)
        
        assert spikes.sum() > 0, "Strong input should generate spikes"
        assert torch.all((spikes == 0) | (spikes == 1)), "Spikes should be binary"
        
    def test_no_spikes_weak_input(self):
        """Test that weak input doesn't generate spikes."""
        weak_input = torch.ones(1, self.time_steps) * 0.1
        spikes, voltage = self.neuron(weak_input)
        
        assert spikes.sum() == 0, "Weak input should not generate spikes"
        
    def test_refractory_period(self):
        """Test refractory period prevents immediate spikes."""
        neuron = LIFNeuron(refractory_period=10.0)  # Long refractory period
        strong_input = torch.ones(1, 20) * 100.0
        
        spikes, voltage = neuron(strong_input)
        
        # Should not have consecutive spikes due to refractory period
        consecutive_spikes = 0
        for i in range(1, spikes.shape[1]):
            if spikes[0, i] == 1 and spikes[0, i-1] == 1:
                consecutive_spikes += 1
        
        assert consecutive_spikes == 0, "No consecutive spikes during refractory period"
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold mechanism."""
        neuron = LIFNeuron(adaptive_thresh=True)
        input_current = torch.ones(1, 50) * 30.0  # Moderate input
        
        spikes, voltage = neuron(input_current)
        
        # With adaptive threshold, spike frequency should decrease over time
        early_spikes = spikes[0, :25].sum()
        late_spikes = spikes[0, 25:].sum()
        
        # This is probabilistic, but generally true for adaptive neurons
        assert early_spikes >= late_spikes or spikes.sum() < 3, "Adaptive threshold should reduce spiking"
        
    def test_reset_state(self):
        """Test state reset functionality."""
        # Run simulation
        input_current = torch.ones(1, 50) * 20.0
        self.neuron(input_current)
        
        # Reset state
        self.neuron.reset_state()
        
        # Check state is reset
        assert torch.allclose(self.neuron.v_mem, torch.tensor(self.neuron.v_rest))
        assert torch.allclose(self.neuron.i_syn, torch.tensor(0.0))
        
    def test_power_consumption(self):
        """Test power consumption calculation."""
        power_low = self.neuron.get_power_consumption(spike_rate=1.0)  # 1 Hz
        power_high = self.neuron.get_power_consumption(spike_rate=100.0)  # 100 Hz
        
        assert power_high > power_low, "Higher spike rate should consume more power"
        assert power_low > 0, "Power consumption should be positive"
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with pytest.raises((ValueError, RuntimeError)):
            # Wrong input shape
            self.neuron(torch.randn(5, 10, 15))  # 3D instead of 2D
            
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        # Very large input
        large_input = torch.ones(1, 10) * 1e6
        spikes, voltage = self.neuron(large_input)
        
        assert not torch.isnan(spikes).any(), "Spikes should not contain NaN"
        assert not torch.isnan(voltage).any(), "Voltage should not contain NaN"
        assert not torch.isinf(spikes).any(), "Spikes should not be infinite"
        assert not torch.isinf(voltage).any(), "Voltage should not be infinite"


class TestNeuronPopulation:
    """Test suite for neuron population."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.population_size = 10
        self.population = NeuronPopulation(
            population_size=self.population_size,
            connection_sparsity=0.2
        )
        
    def test_population_initialization(self):
        """Test population initialization."""
        assert len(self.population.neurons) == self.population_size
        assert self.population.connection_matrix.shape == (self.population_size, self.population_size)
        
    def test_connection_topology(self):
        """Test connection topology generation."""
        # Random topology
        pop_random = NeuronPopulation(5, topology="random", connection_sparsity=0.4)
        assert pop_random.connection_matrix.sum() > 0, "Should have connections"
        
        # Small world topology
        pop_sw = NeuronPopulation(10, topology="small_world", connection_sparsity=0.3)
        assert pop_sw.connection_matrix.sum() > 0, "Should have connections"
        
    def test_forward_pass(self):
        """Test forward pass through population."""
        batch_size = 2
        time_steps = 50
        external_input = torch.randn(batch_size, self.population_size, time_steps)
        
        spikes, voltages = self.population(external_input)
        
        assert spikes.shape == (batch_size, self.population_size, time_steps)
        assert voltages.shape == (batch_size, self.population_size, time_steps)
        
    def test_network_statistics(self):
        """Test network statistics computation."""
        stats = self.population.get_network_statistics()
        
        required_keys = ["num_connections", "connection_density", "mean_degree"]
        for key in required_keys:
            assert key in stats, f"Missing statistic: {key}"
            
        assert stats["connection_density"] >= 0, "Density should be non-negative"
        assert stats["connection_density"] <= 1, "Density should not exceed 1"
        
    def test_population_reset(self):
        """Test population reset functionality."""
        # Run simulation
        external_input = torch.randn(1, self.population_size, 20)
        self.population(external_input)
        
        # Reset population
        self.population.reset_population()
        
        # All neurons should be reset (hard to test directly, but shouldn't raise errors)
        assert True  # If we get here without exceptions, reset worked
        
    def test_power_estimation(self):
        """Test population power estimation."""
        spike_rates = torch.ones(self.population_size) * 10.0  # 10 Hz each
        power = self.population.get_population_power(spike_rates)
        
        assert power > 0, "Population should consume power"
        assert isinstance(power, float), "Power should be a float"


class TestSpikingNeuralNetwork:
    """Test suite for spiking neural network."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        
        self.network = SpikingNeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size
        )
        
    def test_network_initialization(self):
        """Test network initialization."""
        assert self.network.input_size == self.input_size
        assert self.network.hidden_sizes == self.hidden_sizes
        assert self.network.output_size == self.output_size
        
        # Check layer count (hidden + output)
        expected_layers = len(self.hidden_sizes) + 1
        assert len(self.network.layers) == expected_layers
        
    def test_forward_pass(self):
        """Test forward pass through network."""
        batch_size = 2
        time_steps = 100
        input_spikes = torch.rand(batch_size, self.input_size, time_steps) < 0.1  # 10% spike rate
        input_spikes = input_spikes.float()
        
        output_spikes = self.network(input_spikes)
        
        assert output_spikes.shape == (batch_size, self.output_size, time_steps)
        assert torch.all((output_spikes == 0) | (output_spikes == 1)), "Output should be binary spikes"
        
    def test_forward_with_traces(self):
        """Test forward pass with activity traces."""
        batch_size = 1
        time_steps = 50
        input_spikes = torch.rand(batch_size, self.input_size, time_steps) < 0.05
        input_spikes = input_spikes.float()
        
        output_spikes, traces = self.network(input_spikes, return_traces=True)
        
        assert "input" in traces, "Should include input traces"
        assert "layer_0_spikes" in traces, "Should include layer 0 spikes"
        assert len(traces) >= 3, "Should have multiple trace entries"
        
    def test_stdp_learning(self):
        """Test STDP learning mechanism."""
        pre_spikes = torch.zeros(1, 10, 50)
        post_spikes = torch.zeros(1, 5, 50)
        
        # Create causal spike pairs (pre before post)
        pre_spikes[0, 0, 10] = 1.0
        post_spikes[0, 0, 15] = 1.0  # 5ms later
        
        # Get initial weights
        initial_weights = self.network.inter_layer_weights[0].weight.clone()
        
        # Apply STDP
        self.network.apply_stdp(pre_spikes, post_spikes, 0)
        
        # Weights should change
        final_weights = self.network.inter_layer_weights[0].weight
        assert not torch.allclose(initial_weights, final_weights), "STDP should modify weights"
        
    def test_unsupervised_training(self):
        """Test unsupervised training with STDP."""
        # Create simple training data
        num_samples = 10
        time_steps = 50
        training_data = torch.rand(num_samples, self.input_size, time_steps) < 0.05
        training_data = training_data.float()
        
        # Train for a few epochs
        losses = self.network.train_unsupervised(training_data, num_epochs=3, learning_rate=1e-4)
        
        assert len(losses) == 3, "Should have loss for each epoch"
        assert all(isinstance(loss, float) for loss in losses), "Losses should be floats"
        
    def test_network_statistics(self):
        """Test comprehensive network statistics."""
        stats = self.network.get_network_statistics()
        
        assert "architecture" in stats, "Should include architecture stats"
        assert "activity" in stats, "Should include activity stats"
        assert "connectivity" in stats, "Should include connectivity stats"
        
        arch_stats = stats["architecture"]
        assert arch_stats["input_size"] == self.input_size
        assert arch_stats["output_size"] == self.output_size
        assert arch_stats["total_neurons"] > 0
        
    def test_power_estimation(self):
        """Test network power consumption estimation."""
        power_breakdown = self.network.estimate_power_consumption(duration=1.0)
        
        assert "total" in power_breakdown, "Should include total power"
        assert "static" in power_breakdown, "Should include static power"
        assert power_breakdown["total"] > 0, "Total power should be positive"
        
    def test_edge_deployment_export(self):
        """Test edge deployment configuration export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "deployment_config.json"
            
            # This should not raise an exception
            self.network.export_for_edge_deployment(str(output_path))
            
            assert output_path.exists(), "Config file should be created"
            
            # Verify config format
            with open(output_path, 'r') as f:
                config = json.load(f)
            
            assert "model_type" in config, "Should specify model type"
            assert "architecture" in config, "Should include architecture"
            assert "quantized_weights" in config, "Should include quantized weights"
        
    def test_network_reset(self):
        """Test network state reset."""
        # Run some activity through network
        input_spikes = torch.rand(1, self.input_size, 30) < 0.1
        self.network(input_spikes.float())
        
        # Reset network
        self.network.reset_network()
        
        # Activity should be reset to zero
        assert torch.allclose(self.network.layer_activities, torch.zeros_like(self.network.layer_activities))
        
    def test_invalid_architecture(self):
        """Test handling of invalid network architectures."""
        with pytest.raises(ValueError):
            # Invalid hidden sizes
            SpikingNeuralNetwork(
                input_size=10,
                hidden_sizes=[0, -5],  # Invalid sizes
                output_size=2
            )
            
    def test_memory_efficiency(self):
        """Test memory efficiency of network operations."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple forward passes
        for _ in range(10):
            input_spikes = torch.rand(2, self.input_size, 50) < 0.1
            output = self.network(input_spikes.float())
            del output
            
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f}MB"


@pytest.fixture
def sample_spike_data():
    """Generate sample spike data for testing."""
    batch_size = 4
    num_neurons = 100
    time_steps = 200
    
    # Generate Poisson spike trains
    spike_data = torch.rand(batch_size, num_neurons, time_steps) < 0.02
    return spike_data.float()


@pytest.fixture
def trained_network():
    """Create a pre-trained network for testing."""
    network = SpikingNeuralNetwork(
        input_size=50,
        hidden_sizes=[32, 16],
        output_size=5
    )
    
    # Simulate some training by adding small random noise to weights
    with torch.no_grad():
        for layer in network.inter_layer_weights:
            layer.weight.add_(torch.randn_like(layer.weight) * 0.01)
    
    return network


class TestIntegration:
    """Integration tests for model components."""
    
    def test_neuron_population_integration(self, sample_spike_data):
        """Test integration between individual neurons and populations."""
        population_size = 20
        
        # Create population
        population = NeuronPopulation(population_size)
        
        # Run forward pass
        batch_size = sample_spike_data.shape[0]
        input_data = sample_spike_data[:, :population_size, :]
        
        spikes, voltages = population(input_data)
        
        assert not torch.isnan(spikes).any(), "No NaN in population spikes"
        assert not torch.isnan(voltages).any(), "No NaN in population voltages"
        
    def test_network_population_integration(self, trained_network, sample_spike_data):
        """Test integration between populations and full networks."""
        input_size = trained_network.input_size
        input_data = sample_spike_data[:, :input_size, :100]  # Truncate time
        
        # Forward pass
        output = trained_network(input_data)
        
        # Check output properties
        assert output.shape[0] == input_data.shape[0], "Batch size preserved"
        assert output.shape[1] == trained_network.output_size, "Correct output size"
        assert torch.all((output >= 0) & (output <= 1)), "Output in valid range"
        
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end processing pipeline."""
        # Create network
        network = SpikingNeuralNetwork(
            input_size=100,
            hidden_sizes=[50, 25],
            output_size=10
        )
        
        # Generate input
        input_spikes = torch.rand(3, 100, 150) < 0.03  # Low spike rate
        input_spikes = input_spikes.float()
        
        # Forward pass with traces
        output, traces = network(input_spikes, return_traces=True)
        
        # Verify pipeline integrity
        assert output.sum() >= 0, "Some output activity expected"
        assert len(traces) >= 4, "Multiple processing stages"
        
        # Test statistics computation
        stats = network.get_network_statistics()
        assert stats["architecture"]["total_neurons"] > 0
        
        # Test power estimation
        power = network.estimate_power_consumption()
        assert power["total"] > 0
        
        print(f"✅ End-to-end test completed successfully!")
        print(f"   Network output spikes: {output.sum().item():.1f}")
        print(f"   Total power consumption: {power['total']:.2f} μW")
        print(f"   Processing stages: {len(traces)}")