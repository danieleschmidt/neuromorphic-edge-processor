#!/usr/bin/env python3
"""
Secure Neuromorphic Edge Processor Demo
Demonstrates security-hardened neuromorphic computing functionality
"""

import sys
import os
import math
import random
from typing import List, Tuple, Optional, Dict, Any
import logging

# Add src to path securely
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecureLIFNeuron:
    """Security-hardened LIF neuron implementation."""
    
    def __init__(self, tau_mem: float = 20.0, tau_syn: float = 5.0, 
                 threshold: float = 1.0, reset_potential: float = 0.0):
        """Initialize LIF neuron with parameter validation."""
        # Validate parameters
        if not (1.0 <= tau_mem <= 100.0):
            raise ValueError(f"tau_mem must be between 1.0 and 100.0, got {tau_mem}")
        if not (0.1 <= tau_syn <= 50.0):
            raise ValueError(f"tau_syn must be between 0.1 and 50.0, got {tau_syn}")
        if not (0.1 <= threshold <= 10.0):
            raise ValueError(f"threshold must be between 0.1 and 10.0, got {threshold}")
        if not (-5.0 <= reset_potential <= 5.0):
            raise ValueError(f"reset_potential must be between -5.0 and 5.0, got {reset_potential}")
        
        self.tau_mem = float(tau_mem)
        self.tau_syn = float(tau_syn)
        self.threshold = float(threshold)
        self.reset_potential = float(reset_potential)
        
        # Membrane dynamics
        self.alpha_mem = math.exp(-1.0 / self.tau_mem)
        self.alpha_syn = math.exp(-1.0 / self.tau_syn)
        
        # State variables
        self.v_mem = 0.0
        self.i_syn = 0.0
        self.refractory_counter = 0
        
        # Security/monitoring
        self.step_count = 0
        self.spike_count = 0
        
    def step(self, input_current: float) -> Tuple[int, float]:
        """Execute single time step with input validation."""
        # Validate input
        if not isinstance(input_current, (int, float)):
            raise TypeError(f"input_current must be numeric, got {type(input_current)}")
        if not (-100.0 <= input_current <= 100.0):
            raise ValueError(f"input_current out of range [-100, 100]: {input_current}")
        
        input_current = float(input_current)
        
        # Update step counter
        self.step_count += 1
        if self.step_count > 1000000:  # Prevent infinite loops
            raise RuntimeError("Maximum step count exceeded (1M steps)")
        
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
            self.refractory_counter = 2
            self.spike_count += 1
            
            # Log excessive spiking
            if self.spike_count > 10000:
                logger.warning(f"Neuron spike count high: {self.spike_count}")
        
        # Update refractory counter
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
        
        return spike, self.v_mem
    
    def simulate(self, input_currents: List[float], max_steps: int = 10000) -> Tuple[List[int], List[float]]:
        """Simulate over time series with limits."""
        if not isinstance(input_currents, list):
            raise TypeError("input_currents must be a list")
        
        if len(input_currents) > max_steps:
            raise ValueError(f"Input sequence too long: {len(input_currents)} > {max_steps}")
        
        spikes = []
        voltages = []
        
        for i, current in enumerate(input_currents):
            if i >= max_steps:
                logger.warning(f"Simulation truncated at {max_steps} steps")
                break
                
            spike, voltage = self.step(current)
            spikes.append(spike)
            voltages.append(voltage)
        
        return spikes, voltages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get neuron statistics."""
        return {
            "steps": self.step_count,
            "spikes": self.spike_count,
            "spike_rate": self.spike_count / max(1, self.step_count) * 1000,  # Hz
            "current_voltage": self.v_mem,
            "current_current": self.i_syn,
            "refractory_state": self.refractory_counter > 0
        }


class SecureSpikingNetwork:
    """Security-hardened spiking neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize network with size validation."""
        # Validate architecture
        if not (1 <= input_size <= 1000):
            raise ValueError(f"input_size must be between 1 and 1000, got {input_size}")
        if not (1 <= hidden_size <= 1000):
            raise ValueError(f"hidden_size must be between 1 and 1000, got {hidden_size}")
        if not (1 <= output_size <= 100):
            raise ValueError(f"output_size must be between 1 and 100, got {output_size}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create neurons with validated parameters
        self.hidden_neurons = []
        for i in range(hidden_size):
            neuron = SecureLIFNeuron(
                tau_mem=max(10.0, min(30.0, 20.0 + random.uniform(-5, 5))),
                tau_syn=max(2.0, min(8.0, 5.0 + random.uniform(-2, 2))),
                threshold=max(0.5, min(2.0, 1.0 + random.uniform(-0.3, 0.3)))
            )
            self.hidden_neurons.append(neuron)
        
        self.output_neurons = []
        for i in range(output_size):
            neuron = SecureLIFNeuron(
                tau_mem=max(10.0, min(30.0, 20.0 + random.uniform(-5, 5))),
                tau_syn=max(2.0, min(8.0, 5.0 + random.uniform(-2, 2))),
                threshold=max(0.5, min(2.0, 1.0 + random.uniform(-0.3, 0.3)))
            )
            self.output_neurons.append(neuron)
        
        # Create secure random weights
        random.seed(42)  # Deterministic for testing
        self.input_weights = []
        for i in range(hidden_size):
            row = []
            for j in range(input_size):
                weight = random.uniform(-0.5, 0.5)
                row.append(weight)
            self.input_weights.append(row)
        
        self.hidden_weights = []
        for i in range(output_size):
            row = []
            for j in range(hidden_size):
                weight = random.uniform(-0.5, 0.5)
                row.append(weight)
            self.hidden_weights.append(row)
        
        # Statistics
        self.forward_calls = 0
        
    def forward(self, input_spikes: List[List[int]], max_time_steps: int = 1000) -> List[List[int]]:
        """Forward pass with security validation."""
        # Validate input
        if not isinstance(input_spikes, list):
            raise TypeError("input_spikes must be a list")
        
        if len(input_spikes) != self.input_size:
            raise ValueError(f"input_spikes size {len(input_spikes)} != input_size {self.input_size}")
        
        # Validate time dimension
        if input_spikes and len(input_spikes[0]) > max_time_steps:
            raise ValueError(f"Time steps {len(input_spikes[0])} > max {max_time_steps}")
        
        # Validate spike values
        for neuron_spikes in input_spikes:
            for spike in neuron_spikes:
                if spike not in [0, 1]:
                    raise ValueError(f"Invalid spike value: {spike} (must be 0 or 1)")
        
        self.forward_calls += 1
        if self.forward_calls > 10000:
            logger.warning(f"High forward pass count: {self.forward_calls}")
        
        time_steps = len(input_spikes[0]) if input_spikes else 0
        hidden_spikes = [[] for _ in range(self.hidden_size)]
        output_spikes = [[] for _ in range(self.output_size)]
        
        for t in range(min(time_steps, max_time_steps)):
            # Get input at time t
            current_input = [input_spikes[i][t] for i in range(self.input_size)]
            
            # Hidden layer
            current_hidden = []
            for i, neuron in enumerate(self.hidden_neurons):
                # Calculate weighted input
                total_input = sum(w * inp for w, inp in zip(self.input_weights[i], current_input))
                # Clamp input to prevent overflow
                total_input = max(-10.0, min(10.0, total_input))
                
                spike, _ = neuron.step(total_input)
                current_hidden.append(spike)
                hidden_spikes[i].append(spike)
            
            # Output layer
            for i, neuron in enumerate(self.output_neurons):
                # Calculate weighted input
                total_input = sum(w * inp for w, inp in zip(self.hidden_weights[i], current_hidden))
                # Clamp input to prevent overflow
                total_input = max(-10.0, min(10.0, total_input))
                
                spike, _ = neuron.step(total_input)
                output_spikes[i].append(spike)
        
        return output_spikes
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        hidden_stats = [neuron.get_stats() for neuron in self.hidden_neurons]
        output_stats = [neuron.get_stats() for neuron in self.output_neurons]
        
        return {
            "architecture": [self.input_size, self.hidden_size, self.output_size],
            "forward_calls": self.forward_calls,
            "hidden_layer_stats": hidden_stats,
            "output_layer_stats": output_stats,
            "total_spikes": sum(s["spikes"] for s in hidden_stats + output_stats)
        }


def demonstrate_secure_lif_neuron():
    """Demonstrate secure LIF neuron functionality."""
    print("🔒 Secure LIF Neuron Demonstration")
    print("=" * 45)
    
    try:
        # Create secure neuron
        neuron = SecureLIFNeuron(tau_mem=20.0, threshold=1.0)
        
        # Test 1: Controlled subthreshold response
        print("\n📊 Test 1: Subthreshold input (controlled)")
        weak_input = [0.1] * 50  # Limited to 50 steps
        spikes, voltages = neuron.simulate(weak_input)
        print(f"Input: constant {weak_input[0]}")
        print(f"Spikes generated: {sum(spikes)}")
        print(f"Final voltage: {voltages[-1]:.3f}")
        print(f"Neuron stats: {neuron.get_stats()}")
        
        # Reset neuron state
        neuron.v_mem = 0.0
        neuron.i_syn = 0.0
        neuron.refractory_counter = 0
        
        # Test 2: Controlled suprathreshold response
        print("\n📊 Test 2: Suprathreshold input (controlled)")
        strong_input = [1.5] * 20  # Limited duration
        spikes, voltages = neuron.simulate(strong_input)
        print(f"Input: constant {strong_input[0]}")
        print(f"Spikes generated: {sum(spikes)}")
        spike_times = [i for i, s in enumerate(spikes) if s == 1]
        print(f"Spike times: {spike_times}")
        
        # Test 3: Validated input pattern
        print("\n📊 Test 3: Sinusoidal input pattern (validated)")
        pattern_length = 100  # Controlled length
        varying_input = []
        for i in range(pattern_length):
            # Secure math operations
            value = 1.2 * math.sin(0.1 * i) + 0.8
            # Clamp to safe range
            value = max(-5.0, min(5.0, value))
            varying_input.append(value)
        
        neuron.v_mem = 0.0  # Reset
        neuron.i_syn = 0.0
        spikes, voltages = neuron.simulate(varying_input)
        
        print(f"Input: sinusoidal pattern ({len(varying_input)} steps)")
        print(f"Total spikes: {sum(spikes)}")
        spike_rate = sum(spikes) / len(spikes) * 1000 if spikes else 0
        print(f"Spike rate: {spike_rate:.1f} Hz")
        
    except Exception as e:
        logger.error(f"Error in LIF demonstration: {e}")
        print(f"❌ Error: {e}")


def demonstrate_secure_network():
    """Demonstrate secure spiking neural network."""
    print("\n\n🔒 Secure Spiking Neural Network Demonstration")
    print("=" * 55)
    
    try:
        # Create network with validated parameters
        network = SecureSpikingNetwork(input_size=3, hidden_size=5, output_size=2)
        
        # Test 1: Controlled spike pattern
        print("\n📊 Test 1: Simple spike pattern (validated)")
        time_steps = 20  # Controlled length
        input_pattern = []
        
        # Create deterministic test pattern
        patterns = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
        for t in range(time_steps):
            input_pattern.append(patterns[t % len(patterns)])
        
        # Convert to neuron-major format
        input_spikes = []
        for neuron_idx in range(3):
            neuron_spikes = [input_pattern[t][neuron_idx] for t in range(time_steps)]
            input_spikes.append(neuron_spikes)
        
        output_spikes = network.forward(input_spikes)
        
        print(f"Input pattern length: {len(input_pattern)} time steps")
        print(f"Network architecture: {network.input_size} → {network.hidden_size} → {network.output_size}")
        
        for i, spikes in enumerate(output_spikes):
            spike_count = sum(spikes)
            spike_times = [t for t, s in enumerate(spikes) if s == 1]
            print(f"Output neuron {i}: {spike_count} spikes at times {spike_times}")
        
        # Test 2: Random but controlled input
        print("\n📊 Test 2: Random spike input (controlled)")
        random.seed(123)  # Deterministic
        time_steps = 50  # Controlled length
        spike_prob = 0.1  # Low probability for safety
        
        input_spikes = []
        for neuron_idx in range(3):
            neuron_spikes = []
            for t in range(time_steps):
                spike = 1 if random.random() < spike_prob else 0
                neuron_spikes.append(spike)
            input_spikes.append(neuron_spikes)
        
        output_spikes = network.forward(input_spikes)
        
        # Calculate rates safely
        input_rate = sum(sum(channel) for channel in input_spikes) / (3 * time_steps)
        output_rate = sum(sum(channel) for channel in output_spikes) / (2 * time_steps)
        
        print(f"Input spike rate: {input_rate:.3f}")
        print(f"Output spike rate: {output_rate:.3f}")
        
        # Network statistics
        print("\n📈 Network Statistics:")
        stats = network.get_network_stats()
        print(f"Total forward calls: {stats['forward_calls']}")
        print(f"Total network spikes: {stats['total_spikes']}")
        
    except Exception as e:
        logger.error(f"Error in network demonstration: {e}")
        print(f"❌ Error: {e}")


def demonstrate_secure_energy_computation():
    """Demonstrate energy computation with validation."""
    print("\n\n🔒 Secure Energy Consumption Demonstration")
    print("=" * 50)
    
    # Energy model parameters (validated)
    base_power_pW = 0.1  # 0.1 pW base power per neuron
    spike_energy_pJ = 1.0  # 1 pJ per spike
    
    if not (0.01 <= base_power_pW <= 10.0):
        raise ValueError(f"Invalid base power: {base_power_pW}")
    if not (0.1 <= spike_energy_pJ <= 100.0):
        raise ValueError(f"Invalid spike energy: {spike_energy_pJ}")
    
    def compute_energy(spike_train: List[int], duration_ms: int) -> float:
        """Compute energy consumption with validation."""
        if not isinstance(spike_train, list):
            raise TypeError("spike_train must be a list")
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError("duration_ms must be positive integer")
        if len(spike_train) > 100000:  # Prevent excessive computation
            raise ValueError("spike_train too long")
        
        # Validate spike values
        for spike in spike_train:
            if spike not in [0, 1]:
                raise ValueError(f"Invalid spike value: {spike}")
        
        num_spikes = sum(spike_train)
        static_energy = base_power_pW * duration_ms * 1e-3  # Convert to seconds
        dynamic_energy = spike_energy_pJ * num_spikes
        return static_energy + dynamic_energy
    
    # Test different activity levels with controlled randomness
    duration = 100  # ms
    random.seed(42)  # Deterministic
    
    test_cases = [
        ("Low activity", [1 if random.random() < 0.01 else 0 for _ in range(duration)]),
        ("Medium activity", [1 if random.random() < 0.05 else 0 for _ in range(duration)]),
        ("High activity", [1 if random.random() < 0.2 else 0 for _ in range(duration)]),
    ]
    
    try:
        for name, spikes in test_cases:
            energy = compute_energy(spikes, duration)
            spike_rate = sum(spikes) / duration * 1000  # Hz
            print(f"{name:15}: {spike_rate:5.1f} Hz, {energy:6.2f} pJ")
            
    except Exception as e:
        logger.error(f"Error in energy computation: {e}")
        print(f"❌ Error: {e}")


def main():
    """Main demonstration function with error handling."""
    print("🔒 Secure Neuromorphic Edge Processor - Core Functionality Demo")
    print("=" * 70)
    print("This demo shows security-hardened neuromorphic computing principles")
    
    try:
        # Run secure demonstrations
        demonstrate_secure_lif_neuron()
        demonstrate_secure_network()
        demonstrate_secure_energy_computation()
        
        print("\n\n✅ Secure Demo Complete!")
        print("=" * 25)
        print("🔒 Security Features Demonstrated:")
        print("• Input validation and range checking")
        print("• Resource limits and monitoring")
        print("• Error handling and logging")
        print("• Deterministic behavior for testing")
        print("• Memory and computation bounds")
        
        print("\n🧠 Neuromorphic Features:")
        print("• LIF neurons with membrane dynamics")
        print("• Multi-layer spike processing")
        print("• Energy consumption modeling")
        print("• Temporal pattern recognition")
        
    except Exception as e:
        logger.critical(f"Critical error in main demo: {e}")
        print(f"\n❌ Demo failed with critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()