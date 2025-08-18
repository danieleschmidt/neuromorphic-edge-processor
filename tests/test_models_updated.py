"""Comprehensive test suite for neuromorphic models."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.models.lif_neuron import LIFNeuron, JAXLIFNeuron
from src.models.spiking_neural_network import SpikingNeuralNetwork, SparseLinear, STDPLearning
from src.models.liquid_state_machine import LiquidStateMachine, SpikingReadout


class TestLIFNeuron:
    """Test suite for LIF neuron implementation."""
    
    @pytest.fixture
    def neuron(self):
        """Create a standard LIF neuron for testing."""
        return LIFNeuron(
            tau_mem=20.0,
            tau_syn=5.0,
            threshold=1.0,
            reset_potential=0.0,
            refractory_period=2,
            device="cpu"
        )
    
    @pytest.fixture
    def batch_input(self):
        """Create test input current."""
        batch_size, time_steps = 4, 100
        return torch.randn(batch_size, time_steps) * 0.5 + 1.2
    
    def test_neuron_initialization(self, neuron):
        """Test neuron initialization."""
        assert neuron.tau_mem == 20.0
        assert neuron.tau_syn == 5.0
        assert neuron.threshold == 1.0
        assert neuron.reset_potential == 0.0
        assert neuron.refractory_period == 2
        
        # Check computed parameters
        assert abs(neuron.alpha_mem - np.exp(-1.0 / 20.0)) < 1e-6
        assert abs(neuron.alpha_syn - np.exp(-1.0 / 5.0)) < 1e-6
    
    def test_single_spike_generation(self, neuron):
        """Test single spike generation."""
        # Strong input should generate spike
        strong_input = torch.tensor([[2.0]])
        spikes, states = neuron(strong_input)
        
        assert spikes.shape == (1, 1)
        assert spikes[0, 0] == 1.0  # Should spike
        
        # Check membrane potential was reset
        assert states['membrane_potential'][0, 0] == neuron.reset_potential
    
    def test_subthreshold_response(self, neuron):
        """Test subthreshold membrane dynamics."""
        # Weak input should not generate spike
        weak_input = torch.tensor([[0.1]])
        spikes, states = neuron(weak_input)
        
        assert spikes[0, 0] == 0.0  # Should not spike
        assert states['membrane_potential'][0, 0] > 0.0  # But should increase
        assert states['membrane_potential'][0, 0] < neuron.threshold
    
    def test_refractory_period(self, neuron):
        """Test refractory period implementation."""
        # Generate spike then apply strong input during refractory
        strong_input = torch.tensor([[2.0, 2.0, 2.0]])
        spikes, states = neuron(strong_input)
        
        assert spikes[0, 0] == 1.0  # First spike
        assert spikes[0, 1] == 0.0  # Refractory
        assert spikes[0, 2] == 0.0  # Still refractory
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold mechanism."""
        neuron = LIFNeuron(adaptive_threshold=True, tau_adapt=100.0)
        
        # Multiple spikes should increase threshold
        strong_input = torch.tensor([[2.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
        spikes, states = neuron(strong_input)
        
        # Check that threshold adaptation occurred
        initial_threshold = states['threshold'][0, 0]
        later_threshold = states['threshold'][0, 3]
        assert later_threshold > initial_threshold
    
    def test_batch_processing(self, neuron, batch_input):
        """Test batch processing capability."""
        batch_size = batch_input.shape[0]
        neuron.reset_state(batch_size)
        
        spikes, states = neuron(batch_input)
        
        assert spikes.shape == batch_input.shape
        assert states['membrane_potential'].shape == batch_input.shape
        assert states['synaptic_current'].shape == batch_input.shape
        
        # Check that different batch items can have different responses
        spike_counts = torch.sum(spikes, dim=1)
        assert len(torch.unique(spike_counts)) > 1  # Different responses
    
    def test_energy_computation(self, neuron, batch_input):
        """Test energy consumption calculation."""
        spikes, _ = neuron(batch_input)
        energy = neuron.compute_energy_consumption(spikes)
        
        assert isinstance(energy, float)
        assert energy > 0  # Should consume some energy
        
        # More spikes should consume more energy
        high_spikes = torch.ones_like(spikes)
        high_energy = neuron.compute_energy_consumption(high_spikes)
        assert high_energy > energy
    
    def test_firing_rate_calculation(self, neuron, batch_input):
        """Test firing rate calculation."""
        spikes, _ = neuron(batch_input)
        firing_rate = neuron.get_firing_rate(spikes)
        
        assert firing_rate.shape[0] == batch_input.shape[0]
        assert firing_rate.shape[1] == batch_input.shape[1]
        assert torch.all(firing_rate >= 0)  # Non-negative firing rates


class TestSpikingNeuralNetwork:
    """Test suite for Spiking Neural Network."""
    
    @pytest.fixture
    def snn(self):
        """Create a test SNN."""
        return SpikingNeuralNetwork(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=5,
            device="cpu"
        )
    
    @pytest.fixture
    def spike_input(self):
        """Create test spike input."""
        batch_size, time_steps, input_size = 3, 50, 10
        spikes = torch.rand(batch_size, time_steps, input_size) < 0.1
        return spikes.float()
    
    def test_snn_initialization(self, snn):
        """Test SNN initialization."""
        assert snn.input_size == 10
        assert snn.hidden_sizes == [20, 15]
        assert snn.output_size == 5
        assert len(snn.layers) == 3  # 2 hidden + 1 output
        assert len(snn.synapses) == 3
    
    def test_snn_forward_pass(self, snn, spike_input):
        """Test SNN forward pass."""
        result = snn(spike_input)
        
        assert 'output_spikes' in result
        assert 'firing_rates' in result
        assert 'energy_consumption' in result
        
        output_spikes = result['output_spikes']
        assert output_spikes.shape == (3, 50, 5)  # batch, time, output
        assert torch.all((output_spikes == 0) | (output_spikes == 1))  # Binary
    
    def test_snn_state_tracking(self, snn, spike_input):
        """Test SNN state tracking."""
        result = snn(spike_input, return_states=True)
        
        assert 'layer_states' in result
        assert 'layer_spikes' in result
        
        layer_spikes = result['layer_spikes']
        assert len(layer_spikes) == 4  # input + 3 layers


class TestLiquidStateMachine:
    """Test Liquid State Machine implementation."""
    
    @pytest.fixture
    def lsm(self):
        """Create test LSM."""
        return LiquidStateMachine(
            input_size=8,
            reservoir_size=50,
            output_size=3,
            connectivity=0.1,
            device="cpu"
        )
    
    @pytest.fixture
    def lsm_input(self):
        """Create test input for LSM."""
        batch_size, time_steps, input_size = 2, 30, 8
        spikes = torch.rand(batch_size, time_steps, input_size) < 0.15
        return spikes.float()
    
    def test_lsm_initialization(self, lsm):
        """Test LSM initialization."""
        assert lsm.input_size == 8
        assert lsm.reservoir_size == 50
        assert lsm.output_size == 3
        assert len(lsm.reservoir) == 50
        
        # Check reservoir weights properties
        reservoir_weights = lsm.reservoir_weights
        assert reservoir_weights.shape == (50, 50)
        
        # Check sparsity
        sparsity = (reservoir_weights == 0).float().mean().item()
        assert sparsity > 0.8  # Should be sparse
    
    def test_lsm_forward_pass(self, lsm, lsm_input):
        """Test LSM forward pass."""
        result = lsm(lsm_input)
        
        assert 'outputs' in result
        assert 'final_output' in result
        assert 'reservoir_spikes' in result
        
        outputs = result['outputs']
        assert outputs.shape == (2, 30, 3)  # batch, time, output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])