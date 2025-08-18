#!/usr/bin/env python3
"""
Neuromorphic Edge Processor Demo
Demonstrates core functionality without external dependencies
"""

import sys
import os
import math
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Simple tensor implementation for demo purposes
class SimpleTensor:
    def __init__(self, data, shape=None):
        if isinstance(data, list):
            self.data = data
            if shape is None:
                self.shape = self._calculate_shape(data)
            else:
                self.shape = shape
        else:
            self.data = [data]
            self.shape = (1,) if shape is None else shape
    
    def _calculate_shape(self, data):
        if isinstance(data[0], list):
            return (len(data), len(data[0]))
        return (len(data),)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    def sum(self):
        if len(self.shape) == 1:
            return sum(self.data)
        return sum(sum(row) if isinstance(row, list) else row for row in self.data)


class SimpleLIFNeuron:
    """Simplified LIF neuron for demonstration."""
    
    def __init__(self, tau_mem=20.0, tau_syn=5.0, threshold=1.0, reset_potential=0.0):
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset_potential = reset_potential
        
        # Membrane dynamics
        self.alpha_mem = math.exp(-1.0 / tau_mem)
        self.alpha_syn = math.exp(-1.0 / tau_syn)
        
        # State variables
        self.v_mem = 0.0
        self.i_syn = 0.0
        self.refractory_counter = 0
        
    def step(self, input_current):
        """Single time step simulation."""
        # Update synaptic current
        self.i_syn = self.alpha_syn * self.i_syn + input_current
        
        # Update membrane potential (if not refractory)
        if self.refractory_counter == 0:
            self.v_mem = self.alpha_mem * self.v_mem + self.i_syn
        
        # Check for spike
        spike = 0
        if self.v_mem >= self.threshold and self.refractory_counter == 0:
            spike = 1
            self.v_mem = self.reset_potential
            self.refractory_counter = 2  # 2ms refractory period
        
        # Update refractory counter
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            
        return spike, self.v_mem
    
    def simulate(self, input_currents):
        """Simulate over time series."""
        spikes = []
        voltages = []
        
        for current in input_currents:
            spike, voltage = self.step(current)
            spikes.append(spike)
            voltages.append(voltage)
            
        return spikes, voltages


class SimpleSpikingNetwork:
    """Simplified spiking neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create neurons
        self.hidden_neurons = [SimpleLIFNeuron() for _ in range(hidden_size)]
        self.output_neurons = [SimpleLIFNeuron() for _ in range(output_size)]
        
        # Create random weights
        random.seed(42)  # For reproducible results
        self.input_weights = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] 
                             for _ in range(hidden_size)]
        self.hidden_weights = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] 
                              for _ in range(output_size)]
    
    def forward(self, input_spikes):
        """Forward pass through network."""
        time_steps = len(input_spikes[0]) if isinstance(input_spikes[0], list) else len(input_spikes)
        
        hidden_spikes = [[] for _ in range(self.hidden_size)]
        output_spikes = [[] for _ in range(self.output_size)]
        
        for t in range(time_steps):
            # Get input at time t
            if isinstance(input_spikes[0], list):
                current_input = [input_spikes[i][t] for i in range(self.input_size)]
            else:
                current_input = [input_spikes[t] if t < len(input_spikes) else 0]
                current_input = current_input * self.input_size  # Broadcast
            
            # Hidden layer
            for i, neuron in enumerate(self.hidden_neurons):
                # Calculate input current
                current = sum(w * inp for w, inp in zip(self.input_weights[i], current_input))
                spike, _ = neuron.step(current)
                hidden_spikes[i].append(spike)
            
            # Output layer
            current_hidden = [spikes[-1] for spikes in hidden_spikes]
            for i, neuron in enumerate(self.output_neurons):
                # Calculate input current
                current = sum(w * inp for w, inp in zip(self.hidden_weights[i], current_hidden))
                spike, _ = neuron.step(current)
                output_spikes[i].append(spike)
        
        return output_spikes


def demonstrate_lif_neuron():
    """Demonstrate LIF neuron functionality."""
    print("🧠 LIF Neuron Demonstration")
    print("=" * 40)
    
    # Create neuron
    neuron = SimpleLIFNeuron(tau_mem=20.0, threshold=1.0)
    
    # Test 1: Subthreshold response
    print("\n📊 Test 1: Subthreshold input")
    weak_input = [0.1] * 50
    spikes, voltages = neuron.simulate(weak_input)
    print(f"Input: {weak_input[0]} (constant)")
    print(f"Spikes generated: {sum(spikes)}")
    print(f"Final voltage: {voltages[-1]:.3f}")
    
    # Reset neuron
    neuron.v_mem = 0.0
    neuron.i_syn = 0.0
    
    # Test 2: Suprathreshold response
    print("\n📊 Test 2: Suprathreshold input")
    strong_input = [1.5] * 20
    spikes, voltages = neuron.simulate(strong_input)
    print(f"Input: {strong_input[0]} (constant)")
    print(f"Spikes generated: {sum(spikes)}")
    print(f"Spike times: {[i for i, s in enumerate(spikes) if s == 1]}")
    
    # Reset neuron
    neuron.v_mem = 0.0
    neuron.i_syn = 0.0
    
    # Test 3: Varying input
    print("\n📊 Test 3: Varying input pattern")
    varying_input = [1.2 * math.sin(0.1 * i) + 0.8 for i in range(100)]
    spikes, voltages = neuron.simulate(varying_input)
    print(f"Input: sinusoidal pattern")
    print(f"Total spikes: {sum(spikes)}")
    print(f"Spike rate: {sum(spikes) / len(spikes) * 1000:.1f} Hz")


def demonstrate_spiking_network():
    """Demonstrate spiking neural network."""
    print("\n\n🌐 Spiking Neural Network Demonstration")
    print("=" * 50)
    
    # Create network
    network = SimpleSpikingNetwork(input_size=3, hidden_size=5, output_size=2)
    
    # Test 1: Simple spike input
    print("\n📊 Test 1: Simple spike pattern")
    input_pattern = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]] * 5  # 20 time steps
    input_spikes = [[row[i] for row in input_pattern] for i in range(3)]
    
    output_spikes = network.forward(input_spikes)
    
    print(f"Input pattern length: {len(input_pattern)} time steps")
    print(f"Network architecture: {network.input_size} → {network.hidden_size} → {network.output_size}")
    
    for i, spikes in enumerate(output_spikes):
        spike_count = sum(spikes)
        spike_times = [t for t, s in enumerate(spikes) if s == 1]
        print(f"Output neuron {i}: {spike_count} spikes at times {spike_times}")
    
    # Test 2: Random input
    print("\n📊 Test 2: Random spike input")
    random.seed(123)
    time_steps = 50
    input_spikes = [[1 if random.random() < 0.1 else 0 for _ in range(time_steps)] 
                   for _ in range(3)]
    
    output_spikes = network.forward(input_spikes)
    
    input_rate = sum(sum(channel) for channel in input_spikes) / (3 * time_steps)
    output_rate = sum(sum(channel) for channel in output_spikes) / (2 * time_steps)
    
    print(f"Input spike rate: {input_rate:.3f}")
    print(f"Output spike rate: {output_rate:.3f}")


def demonstrate_energy_computation():
    """Demonstrate energy consumption calculation."""
    print("\n\n⚡ Energy Consumption Demonstration")
    print("=" * 45)
    
    # Energy model parameters
    base_power = 0.1e-12  # 0.1 pW base power per neuron
    spike_energy = 1.0e-12  # 1 pJ per spike
    
    def compute_energy(spike_train, duration_ms):
        """Compute energy consumption."""
        num_spikes = sum(spike_train)
        static_energy = base_power * duration_ms * 1e-3  # Convert to seconds
        dynamic_energy = spike_energy * num_spikes
        return static_energy + dynamic_energy
    
    # Test different activity levels
    duration = 100  # ms
    test_cases = [
        ("Low activity", [1 if random.random() < 0.01 else 0 for _ in range(duration)]),
        ("Medium activity", [1 if random.random() < 0.05 else 0 for _ in range(duration)]),
        ("High activity", [1 if random.random() < 0.2 else 0 for _ in range(duration)]),
    ]
    
    for name, spikes in test_cases:
        energy = compute_energy(spikes, duration)
        spike_rate = sum(spikes) / duration * 1000  # Hz
        print(f"{name:15}: {spike_rate:5.1f} Hz, {energy*1e12:6.2f} pJ")


def demonstrate_liquid_state_machine():
    """Demonstrate simplified liquid state machine concept."""
    print("\n\n🌊 Liquid State Machine Concept")
    print("=" * 40)
    
    # Create reservoir neurons
    reservoir_size = 10
    reservoir = [SimpleLIFNeuron(tau_mem=random.uniform(15, 25), 
                                threshold=random.uniform(0.8, 1.2)) 
                for _ in range(reservoir_size)]
    
    # Random connectivity
    random.seed(42)
    connections = [[random.uniform(-0.3, 0.3) if random.random() < 0.2 else 0 
                   for _ in range(reservoir_size)] for _ in range(reservoir_size)]
    
    # Input weights
    input_weights = [random.uniform(-1, 1) for _ in range(reservoir_size)]
    
    def lsm_step(input_current, reservoir_states):
        """Single LSM step."""
        new_states = []
        
        for i, neuron in enumerate(reservoir):
            # Input + recurrent connections
            total_input = input_current * input_weights[i]
            total_input += sum(connections[i][j] * reservoir_states[j] 
                             for j in range(reservoir_size))
            
            spike, voltage = neuron.step(total_input)
            new_states.append(spike)
        
        return new_states
    
    # Simulate LSM
    print("📊 LSM Response to Input Pattern")
    input_pattern = [2.0 if i % 10 < 3 else 0.1 for i in range(50)]  # Periodic bursts
    
    reservoir_states = [0] * reservoir_size
    all_states = []
    
    for inp in input_pattern:
        reservoir_states = lsm_step(inp, reservoir_states)
        all_states.append(reservoir_states[:])
    
    # Analyze reservoir activity
    total_activity = sum(sum(states) for states in all_states)
    active_neurons = sum(1 for i in range(reservoir_size) 
                        if sum(states[i] for states in all_states) > 0)
    
    print(f"Input pattern: periodic bursts")
    print(f"Reservoir size: {reservoir_size} neurons")
    print(f"Total activity: {total_activity} spikes")
    print(f"Active neurons: {active_neurons}/{reservoir_size}")
    print(f"Average activity: {total_activity/len(all_states)/reservoir_size:.3f} spikes/step/neuron")


def main():
    """Main demonstration function."""
    print("🚀 Neuromorphic Edge Processor - Core Functionality Demo")
    print("=" * 60)
    print("This demo shows the basic principles without external dependencies")
    
    # Run demonstrations
    demonstrate_lif_neuron()
    demonstrate_spiking_network()
    demonstrate_energy_computation()
    demonstrate_liquid_state_machine()
    
    print("\n\n✅ Demo Complete!")
    print("=" * 20)
    print("🧠 LIF neurons: ✓ Spike generation, membrane dynamics")
    print("🌐 Networks: ✓ Multi-layer spike processing")
    print("⚡ Energy: ✓ Ultra-low power consumption")
    print("🌊 LSM: ✓ Reservoir computing principles")
    print("\nNeuromorphic computing demonstrates:")
    print("• Event-driven processing")
    print("• Sparse activity patterns")
    print("• Ultra-low energy consumption")
    print("• Temporal pattern recognition")


if __name__ == "__main__":
    main()